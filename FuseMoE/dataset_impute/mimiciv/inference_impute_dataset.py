import os
import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_mm import MultiModalImputationDataset, collate_mm
from csdi_mm_moe import CSDI_MultiModal_MoE


def load_samples(path: str):
    assert os.path.exists(path), f"not found: {path}"
    ext = os.path.splitext(path)[1].lower()

    if ext in [".pt", ".pth"]:
        obj = torch.load(path, map_location="cpu")
    elif ext in [".pkl", ".pickle"]:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    elif ext == ".npy":
        obj = np.load(path, allow_pickle=True).tolist()
    else:
        raise ValueError(f"unsupported file extension: {ext}")

    if isinstance(obj, dict) and "samples" in obj:
        obj = obj["samples"]
    if not isinstance(obj, (list, tuple)):
        raise ValueError(f"loaded object must be list/tuple of dict, got {type(obj)}")

    return list(obj)


@torch.no_grad()
def impute_batch_full(model, batch, n_samples: int = 20, agg: str = "median"):
    """
    对 batch 做“全量插补”：
      cond_mask = observed_mask （已观测位置作为条件）
      输出：imputed_data (B, K, L) 以及 moe_w (B,3)
    """
    model.eval()

    # 复用你 CSDI_MultiModal_MoE 里的 process_data / moe / get_side_info_mm
    (observed_data, observed_mask, observed_tp,
     gt_mask, hist_mask, seq_len,
     text_vec, cxr_vec, ecg_vec,
     text_missing, cxr_missing, ecg_missing) = model.process_data(batch)

    # ✅ 全量插补：把所有观测点当条件（不把 gt_mask 扣掉）
    cond_mask = observed_mask

    fused_ctx, moe_w = model.moe(
        text_vec, cxr_vec, ecg_vec,
        text_missing, cxr_missing, ecg_missing
    )
    side_info = model.get_side_info_mm(observed_tp, cond_mask, fused_ctx)

    # samples: (B, n_samples, K, L)
    samples = model.impute(observed_data, cond_mask, side_info, n_samples)

    if agg == "median":
        pred = samples.median(dim=1).values  # (B, K, L)
    elif agg == "mean":
        pred = samples.mean(dim=1)           # (B, K, L)
    else:
        raise ValueError("agg must be median or mean")

    # ✅ 用观测值覆盖（保证插补不改动原有观测点）
    imputed = observed_data * observed_mask + pred * (1.0 - observed_mask)

    return imputed, observed_data, observed_mask, seq_len, moe_w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True, help="checkpoint path (model/ckpt_csdi_mm_moe.pt)")
    parser.add_argument("--output_path", type=str, required=True, help="output pkl path")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=20, help="diffusion sample count")
    parser.add_argument("--agg", type=str, default="median", choices=["median", "mean"])
    parser.add_argument("--tiny", type=int, default=None, help="only first N samples for smoke test")
    args = parser.parse_args()

    # device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print("device:", device)

    # load ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = ckpt["config"]

    # load samples
    samples = load_samples(args.data)
    if args.tiny is not None:
        samples = samples[:args.tiny]
        print(f"[tiny] using first {len(samples)} samples")

    # dataset/dataloader
    ds = MultiModalImputationDataset(
        samples,
        text_dim=config["data"]["text_dim"],
        ecg_dim=config["data"]["ecg_dim"],
        cxr_dim=config["data"]["cxr_dim"],
        eval_mask_ratio=config["data"].get("eval_mask_ratio", 0.0),
        seed=config["train"]["seed"],
        mode="test",
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_mm,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # build model + load weights
    model = CSDI_MultiModal_MoE(config=config, device=device, target_dim=30).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # output samples (copy + add new key)
    out_samples = samples  # 直接在原 list 上加字段（也可以 deep copy，看你需求）

    seen = 0
    pbar = tqdm(dl, desc="imputing", leave=True)
    for batch in pbar:
        imputed, observed_data, observed_mask, seq_len, moe_w = impute_batch_full(
            model, batch, n_samples=args.n_samples, agg=args.agg
        )

        imputed = imputed.detach().cpu().numpy()         # (B, K, Lpad)
        observed_mask = observed_mask.detach().cpu().numpy()
        seq_len = seq_len.detach().cpu().numpy()
        moe_w = moe_w.detach().cpu().numpy()             # (B,3)

        B = imputed.shape[0]
        for i in range(B):
            idx = seen + i
            L = int(seq_len[i])

            # model tensor layout: (K, L) -> back to (L, K)
            imputed_i = imputed[i, :, :L].T  # (L,30)

            # 给每条样本新增字段
            out_samples[idx]["irg_ts_imputed"] = imputed_i.astype(np.float32)

            # 把 MoE 权重
            out_samples[idx]["moe_w_text_cxr_ecg"] = moe_w[i].astype(np.float32)

        seen += B

    assert seen == len(out_samples), f"processed {seen} != {len(out_samples)}"

    # save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out_samples, f)

    print(f"Saved imputed dataset to: {args.out}")
    print("Example keys of first sample:", list(out_samples[0].keys()))
    print("Example shapes:",
          "irg_ts:", np.asarray(out_samples[0]["irg_ts"]).shape,
          "irg_ts_imputed:", np.asarray(out_samples[0]["irg_ts_imputed"]).shape,
          "mask:", np.asarray(out_samples[0]["irg_ts_mask"]).shape)


if __name__ == "__main__":
    main()
