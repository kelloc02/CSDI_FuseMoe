import numpy as np
import torch
from torch.utils.data import Dataset


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def sanitize_array(x, dtype=np.float32, nan=0.0, posinf=0.0, neginf=0.0):
    x = _to_numpy(x).astype(dtype)
    x = np.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    return x


def _pad_or_trunc_1d(vec: np.ndarray, out_dim: int, dtype=np.float32):
    vec = sanitize_array(vec, dtype=dtype).reshape(-1)
    if vec.shape[0] == out_dim:
        return vec
    if vec.shape[0] > out_dim:
        return vec[:out_dim]
    out = np.zeros((out_dim,), dtype=dtype)
    out[: vec.shape[0]] = vec
    return out


def mean_pool_list_of_vecs(vec_list, out_dim: int, dtype=np.float32):
    """
    side info pooling
    """
    if vec_list is None or len(vec_list) == 0:
        return np.zeros((out_dim,), dtype=dtype)

    # (N,D)
    arr = np.stack([sanitize_array(v, dtype=dtype).reshape(-1) for v in vec_list], axis=0)
    pooled = arr.mean(axis=0)  # (D,)
    pooled = np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0).astype(dtype)
    return _pad_or_trunc_1d(pooled, out_dim=out_dim, dtype=dtype)


class MultiModalImputationDataset(Dataset):
    def __init__(
        self,
        samples,
        text_dim=768,
        ecg_dim=256,
        cxr_dim=1024,
        eval_mask_ratio=0.2,
        seed=1234,
        mode="train",   # train|val|test
    ):
        self.samples = samples
        self.text_dim = int(text_dim)
        self.ecg_dim = int(ecg_dim)
        self.cxr_dim = int(cxr_dim)
        self.eval_mask_ratio = float(eval_mask_ratio)
        self.mode = mode
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.samples)

    def _make_gt_mask(self, observed_mask: np.ndarray):
        """
        observed_mask: (L,30) 0/1
        gt_mask: eval 用 cond_mask（从 observed 再扣一部分作为 target）
        """
        # 0/1
        observed_mask = sanitize_array(observed_mask, dtype=np.float32)
        observed_mask = (observed_mask > 0.5).astype(np.float32)

        if self.mode == "train":
            return observed_mask.copy()

        obs = observed_mask.copy()
        idx = np.argwhere(obs > 0.5)
        if len(idx) == 0:
            return obs

        m = int(round(len(idx) * self.eval_mask_ratio))
        if m <= 0:
            return obs

        choose = self.rng.choice(len(idx), size=m, replace=False)
        gt = obs.copy()
        gt[idx[choose, 0], idx[choose, 1]] = 0.0
        return gt

    def __getitem__(self, i):
        s = self.samples[i]

        # core time-series 
        irg_ts = sanitize_array(s["irg_ts"], dtype=np.float32)            # (L,30)
        irg_ts_mask = sanitize_array(s["irg_ts_mask"], dtype=np.float32)  # (L,30)
        ts_tt = sanitize_array(s["ts_tt"], dtype=np.float32).reshape(-1)  # (L,)

        L = irg_ts.shape[0]
        assert irg_ts.shape[1] == 30, f"expected 30 features, got {irg_ts.shape}"
        assert irg_ts_mask.shape == irg_ts.shape, "mask shape mismatch"
        assert ts_tt.shape[0] == L, "ts_tt length mismatch"

        # 0/1
        irg_ts_mask = (irg_ts_mask > 0.5).astype(np.float32)

        irg_ts = irg_ts * irg_ts_mask

        ts_tt = np.maximum(ts_tt, 0.0).astype(np.float32)

        text_missing = int(s.get("text_missing", 1))
        cxr_missing  = int(s.get("cxr_missing", 1))
        ecg_missing  = int(s.get("ecg_missing", 1))

        # text_embeddings: list[(768,)]
        text_vec = mean_pool_list_of_vecs(s.get("text_embeddings", []), out_dim=self.text_dim)
        # ecg_feats: list[(256,)]
        ecg_vec = mean_pool_list_of_vecs(s.get("ecg_feats", []), out_dim=self.ecg_dim)
        # cxr_feats: list[(1024,)]
        cxr_vec = mean_pool_list_of_vecs(s.get("cxr_feats", []), out_dim=self.cxr_dim)

        text_vec = np.nan_to_num(text_vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        ecg_vec  = np.nan_to_num(ecg_vec,  nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        cxr_vec  = np.nan_to_num(cxr_vec,  nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        gt_mask = self._make_gt_mask(irg_ts_mask)
        hist_mask = irg_ts_mask.copy()

        out = {
            "seq_len": torch.tensor(L, dtype=torch.long),

            "irg_ts": torch.from_numpy(irg_ts),             # (L,30)
            "irg_ts_mask": torch.from_numpy(irg_ts_mask),   # (L,30)
            "ts_tt": torch.from_numpy(ts_tt),               # (L,)
            "gt_mask": torch.from_numpy(gt_mask),           # (L,30)
            "hist_mask": torch.from_numpy(hist_mask),       # (L,30)

            "text_vec": torch.from_numpy(text_vec),         # (768,)
            "cxr_vec": torch.from_numpy(cxr_vec),           # (1024,)
            "ecg_vec": torch.from_numpy(ecg_vec),           # (256,)

            "text_missing": torch.tensor(text_missing, dtype=torch.long),
            "cxr_missing": torch.tensor(cxr_missing, dtype=torch.long),
            "ecg_missing": torch.tensor(ecg_missing, dtype=torch.long),

            "label": torch.tensor(int(s.get("label", 0)), dtype=torch.long),
            "name": torch.tensor(int(s.get("name", 0)), dtype=torch.long),
            "hadm_id": torch.tensor(int(s.get("hadm_id", 0)), dtype=torch.long),
            "stay_id": torch.tensor(int(s.get("stay_id", 0)), dtype=torch.long),
        }
        return out


def collate_mm(batch):
    """
      irg_ts: (B,maxL,30)
      masks : (B,maxL,30)
      ts_tt : (B,maxL)
      seq_len: (B,)
    """
    B = len(batch)
    seq_len = torch.stack([b["seq_len"] for b in batch], dim=0)  # (B,)
    maxL = int(seq_len.max().item())
    K = 30

    def pad_2d(key, pad_value=0.0):
        out = torch.full((B, maxL, K), float(pad_value), dtype=torch.float32)
        for i, b in enumerate(batch):
            L = int(b["seq_len"].item())
            out[i, :L] = b[key].float()
        return out

    def pad_1d(key, pad_value=0.0):
        out = torch.full((B, maxL), float(pad_value), dtype=torch.float32)
        for i, b in enumerate(batch):
            L = int(b["seq_len"].item())
            out[i, :L] = b[key].float()
        return out

    out = {
        "seq_len": seq_len,

        "irg_ts": pad_2d("irg_ts", 0.0),
        "irg_ts_mask": pad_2d("irg_ts_mask", 0.0),
        "gt_mask": pad_2d("gt_mask", 0.0),
        "hist_mask": pad_2d("hist_mask", 0.0),

        "ts_tt": pad_1d("ts_tt", 0.0),

        "text_vec": torch.stack([b["text_vec"].float() for b in batch], dim=0),
        "cxr_vec": torch.stack([b["cxr_vec"].float() for b in batch], dim=0),
        "ecg_vec": torch.stack([b["ecg_vec"].float() for b in batch], dim=0),

        "text_missing": torch.stack([b["text_missing"] for b in batch], dim=0),
        "cxr_missing": torch.stack([b["cxr_missing"] for b in batch], dim=0),
        "ecg_missing": torch.stack([b["ecg_missing"] for b in batch], dim=0),

        "label": torch.stack([b["label"] for b in batch], dim=0),
        "name": torch.stack([b["name"] for b in batch], dim=0),
        "hadm_id": torch.stack([b["hadm_id"] for b in batch], dim=0),
        "stay_id": torch.stack([b["stay_id"] for b in batch], dim=0),
    }
    return out
