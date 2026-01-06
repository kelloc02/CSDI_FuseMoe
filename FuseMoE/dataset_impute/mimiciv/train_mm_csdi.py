import os
import math
import argparse
import random
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

from config_mm import get_config
from dataset_mm import MultiModalImputationDataset, collate_mm
from csdi_mm_moe import CSDI_MultiModal_MoE

## 命令行：python train_mm_csdi.py   --data /playpen-shared/kechengli/workspace/dataset/mimiciv_pkl/test_ihm-48-cxr-notes-ecg_stays.pkl   --epochs 50   --batch_size 16   --gpu 0
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
def eval_one_epoch(model, loader, device, n_samples=20, desc="val"):
    model.eval()
    total_mse = 0.0
    total_cnt = 0.0

    pbar = tqdm(loader, desc=f"[{desc}] sampling", leave=False)
    for batch in pbar:
        samples, observed_data, target_mask, observed_mask, tp, moe_w = model.evaluate(
            batch, n_samples=n_samples
        )

        pred = samples.median(dim=1).values  # (B,K,L)

        err = (pred - observed_data) ** 2
        mse = (err * target_mask).sum().item()
        cnt = target_mask.sum().item()

        total_mse += mse
        total_cnt += cnt

        rmse_running = math.sqrt(total_mse / max(total_cnt, 1.0))
        pbar.set_postfix(rmse=f"{rmse_running:.4f}")

    rmse = math.sqrt(total_mse / max(total_cnt, 1.0))
    return rmse


def save_curves(out_dir, train_loss_step, train_loss_epoch, val_rmse_epoch, moe_w_epoch):
    os.makedirs(out_dir, exist_ok=True)

    # 1) step-level training loss
    plt.figure()
    plt.plot(train_loss_step, linewidth=1)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Training Loss (step-level)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_loss_step.png"))
    plt.close()

    # 2) epoch-level training loss
    plt.figure()
    plt.plot(train_loss_epoch, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Train Loss")
    plt.title("Training Loss (epoch-level)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_loss_epoch.png"))
    plt.close()

    # 3) val RMSE
    plt.figure()
    plt.plot(val_rmse_epoch, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Val RMSE")
    plt.title("Validation RMSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_rmse.png"))
    plt.close()

    # 4) MoE weights per epoch (mean over steps)
    if moe_w_epoch is not None and len(moe_w_epoch) > 0:
        moe_w_epoch = np.asarray(moe_w_epoch)  # (E,3)
        plt.figure()
        plt.plot(moe_w_epoch[:, 0], marker="o", label="text")
        plt.plot(moe_w_epoch[:, 1], marker="o", label="cxr")
        plt.plot(moe_w_epoch[:, 2], marker="o", label="ecg")
        plt.xlabel("Epoch")
        plt.ylabel("Mean gate weight")
        plt.title("MoE Gate Weights (epoch mean)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "moe_weights_epoch.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="path to samples (.pt/.pkl/.npy), list[dict]")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gpu", type=int, default=0, help="gpu index to use (default 0)")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--tiny", type=int, default=None, help="use only first N samples for smoke test")
    parser.add_argument("--plot_dir", type=str, default="plots", help="where to save curve plots")
    args = parser.parse_args()

    config = get_config()
    if args.epochs is not None:
        config["train"]["max_epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["train"]["lr"] = args.lr
    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers

    # 指定 GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    set_seed(config["train"]["seed"])

    samples = load_samples(args.data)
    if args.tiny is not None:
        samples = samples[: args.tiny]
        print(f"[tiny] using first {len(samples)} samples")

    n = len(samples)
    assert n >= 10, f"dataset too small: {n}"

    # 80/10/10 split（小数据时保证 val 至少 1 条）
    n_train = int(n * 0.8)
    n_val = max(1, int(n * 0.1))
    n_train = max(1, n_train)
    if n_train + n_val >= n:
        n_train = max(1, n - n_val - 1)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    print(f"split: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    print(f"device: {device}")

    ds_train = MultiModalImputationDataset(
        train_samples,
        text_dim=config["data"]["text_dim"],
        ecg_dim=config["data"]["ecg_dim"],
        cxr_dim=config["data"]["cxr_dim"],
        eval_mask_ratio=config["data"]["eval_mask_ratio"],
        seed=config["train"]["seed"],
        mode="train",
    )
    ds_val = MultiModalImputationDataset(
        val_samples,
        text_dim=config["data"]["text_dim"],
        ecg_dim=config["data"]["ecg_dim"],
        cxr_dim=config["data"]["cxr_dim"],
        eval_mask_ratio=config["data"]["eval_mask_ratio"],
        seed=config["train"]["seed"] + 1,
        mode="val",
    )

    use_cuda = (device.type == "cuda")
    dl_train = DataLoader(
        ds_train,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        collate_fn=collate_mm,
        drop_last=True,
        pin_memory=use_cuda,
        persistent_workers=(config["train"]["num_workers"] > 0),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        collate_fn=collate_mm,
        drop_last=False,
        pin_memory=use_cuda,
        persistent_workers=(config["train"]["num_workers"] > 0),
    )

    model = CSDI_MultiModal_MoE(config=config, device=device, target_dim=30).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    train_loss_step = []
    train_loss_epoch = []
    val_rmse_epoch = []
    moe_w_epoch = []   # (E,3) epoch-mean

    best_val = float("inf")
    global_step = 0

    for epoch in range(1, config["train"]["max_epochs"] + 1):
        model.train()

        epoch_losses = []
        epoch_moe_ws = []

        pbar = tqdm(dl_train, desc=f"[train] epoch {epoch}/{config['train']['max_epochs']}", leave=True)
        for batch in pbar:
            loss, moe_w = model(batch, is_train=1)

            # record
            loss_item = float(loss.item())
            train_loss_step.append(loss_item)
            epoch_losses.append(loss_item)

            # moe_w: (B,3) -> mean over batch
            w_mean = moe_w.mean(dim=0).detach().cpu().numpy()  # (3,)
            epoch_moe_ws.append(w_mean)

            optim.zero_grad(set_to_none=True)
            loss.backward()

            if config["train"]["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])

            optim.step()
            global_step += 1

            pbar.set_postfix(
                loss=f"{loss_item:.4f}",
                w_text=f"{w_mean[0]:.2f}",
                w_cxr=f"{w_mean[1]:.2f}",
                w_ecg=f"{w_mean[2]:.2f}",
            )

        avg_epoch_loss = float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else float("nan")
        train_loss_epoch.append(avg_epoch_loss)

        epoch_w = np.mean(np.stack(epoch_moe_ws, axis=0), axis=0) if len(epoch_moe_ws) > 0 else np.array([np.nan, np.nan, np.nan])
        moe_w_epoch.append(epoch_w)

        val_rmse = eval_one_epoch(model, dl_val, device, n_samples=config["eval"]["n_samples"], desc="val")
        val_rmse_epoch.append(float(val_rmse))
        print(f"[epoch {epoch}] train_loss={avg_epoch_loss:.6f} val_RMSE={val_rmse:.6f}")

        if val_rmse < best_val:
            best_val = val_rmse
            ckpt = {
                "model": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_rmse": val_rmse,
            }
            torch.save(ckpt, config["train"]["save_path"])
            print(f"  saved best -> {config['train']['save_path']}")

        save_curves(args.plot_dir, train_loss_step, train_loss_epoch, val_rmse_epoch, moe_w_epoch)

    print("training done. best_val_RMSE=", best_val)
    print(f"Saved plots to ./{args.plot_dir}/")


if __name__ == "__main__":
    main()
