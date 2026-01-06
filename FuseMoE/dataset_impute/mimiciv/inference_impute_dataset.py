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
def impute_batch_full(model, batch, n_samples: int = 20):
    """
    全量插补：
      cond_mask = observed_mask
      聚合方式：median（固定）
    """
    model.eval()

    (observed_data, observed_mask, observed_tp,
     gt_mask, hist_mask, seq_len,
     text_vec, cxr_vec, ecg_vec,
     text_missing, cxr_missing, ecg_missing) = model.process_data(batch)

    # ✅ 所有观测点作为条件
    cond_mask = observed_mask

    fused_ctx, moe_w = model.moe(
        text_vec, cxr_vec, ecg_vec,
        text_missing, cxr_missing, ecg_missing
    )
    side_info = model.get_side_info_mm(observed_tp, cond_mask, fused_ctx)

    # (B, n_samples, K, L)
    samples = model.impute(observed_data, cond_mask, side_info, n_samples)

    # ✅ 固定用中位数
    pred = samples.median(dim=1).values  # (B, K, L)

    # 观测点不变，缺失点用预测
    imputed = observed_data * observed_mask + pred * (1.0 - observed_mask)

    return imputed, observed_mask, seq_len, moe_w


def main():
    ##python inference_impute_dataset.py \
#   --data_path /playpen-shared/kechengli/workspace/dataset/mimiciv_pkl/test_ihm-48-cxr-notes-ecg_stays.pkl \
#   --model_path /playpen-shared/kechengli/workspace/Fusemoe/FuseMoE/dataset_impute/mimiciv/models/csdi_moe.pt \
#   --output_path .output_imputed_data/tmp_imputed_64.pkl \
#   --gpu 0 \
#   --batch_size 16 \
#   --n_samples 100 \
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--tiny", type=int, default=None)
    args = parser.parse_args()

    # device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print("device:", device)

    # load checkpoint
    ckpt = torch.load(args.model_path, map_location="cpu")
    config = ckpt["config"]

    # load data
    samples = load_samples(args.data_path)
    if args.tiny is not None:
        samples = samples[:args.tiny]
        print(f"[tiny] using first {len(samples)} samples")

    # dataset / dataloader（不使用 worker）
    ds = MultiModalImputationDataset(
        samples,
        text_dim=config["data"]["text_dim"],
        ecg_dim=config["data"]["ecg_dim"],
        cxr_dim=config["data"]["cxr_dim"],
        eval_mask_ratio=0.0,
        seed=config["train"]["seed"],
        mode="test",
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,          # ✅ 明确为 0
        collate_fn=collate_mm,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    # model
    model = CSDI_MultiModal_MoE(
        config=config, device=device, target_dim=30
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    out_samples = samples
    seen = 0

    pbar = tqdm(dl, desc="imputing", leave=True)
    for batch in pbar:
        imputed, observed_mask, seq_len, moe_w = impute_batch_full(
            model, batch, n_samples=args.n_samples
        )

        imputed = imputed.cpu().numpy()       # (B,K,Lpad)
        observed_mask = observed_mask.cpu().numpy()
        seq_len = seq_len.cpu().numpy()
        moe_w = moe_w.cpu().numpy()            # (B,3)

        B = imputed.shape[0]
        for i in range(B):
            idx = seen + i
            L = int(seq_len[i])

            # (K,L) -> (L,K)
            imputed_i = imputed[i, :, :L].T

            out_samples[idx]["irg_ts_imputed"] = imputed_i.astype(np.float32)
            out_samples[idx]["moe_w_text_cxr_ecg"] = moe_w[i].astype(np.float32)

        seen += B

    assert seen == len(out_samples), f"processed {seen} != {len(out_samples)}"

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(out_samples, f)

    print(f"Saved imputed dataset to: {args.output_path}")
    print("Example keys:", list(out_samples[0].keys()))
    print("Example shapes:",
          "irg_ts:", np.asarray(out_samples[0]["irg_ts"]).shape,
          "irg_ts_imputed:", np.asarray(out_samples[0]["irg_ts_imputed"]).shape)


if __name__ == "__main__":
    main()
