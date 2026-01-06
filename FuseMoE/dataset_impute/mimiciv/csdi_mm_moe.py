import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from main_model import CSDI_base


class ModalityMoEFuser(nn.Module):
    """
    输入：text/cxr/ecg 各一个向量 + missing flag
    输出：融合向量 (B, ctx_dim) + 权重 (B,3)
    """
    def __init__(self, text_dim: int, cxr_dim: int, ecg_dim: int,
                 ctx_dim: int, hidden: int = 256):
        super().__init__()
        self.ctx_dim = ctx_dim

        self.proj_text = nn.Sequential(nn.Linear(text_dim, hidden), nn.SiLU(), nn.Linear(hidden, ctx_dim))
        self.proj_cxr  = nn.Sequential(nn.Linear(cxr_dim,  hidden), nn.SiLU(), nn.Linear(hidden, ctx_dim))
        self.proj_ecg  = nn.Sequential(nn.Linear(ecg_dim,  hidden), nn.SiLU(), nn.Linear(hidden, ctx_dim))

        # gate 输入：3个ctx + 3个missing flag
        self.gate = nn.Sequential(
            nn.Linear(ctx_dim * 3 + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, text_vec, cxr_vec, ecg_vec, text_missing, cxr_missing, ecg_missing):
        # vec: (B,D)  missing: (B,)
        text_ctx = self.proj_text(text_vec)  # (B,ctx)
        cxr_ctx  = self.proj_cxr(cxr_vec)
        ecg_ctx  = self.proj_ecg(ecg_vec)

        miss = torch.stack([text_missing, cxr_missing, ecg_missing], dim=-1).float().clamp(0, 1)  # (B,3)
        gate_in = torch.cat([text_ctx, cxr_ctx, ecg_ctx, miss], dim=-1)  # (B, ctx*3+3)
        logits = self.gate(gate_in)  # (B,3)

        # ✅ 缺失模态：权重“直接为0”（用超大负数实现，softmax≈0，且数值稳定）
        logits = logits - 1e4 * miss
        w = F.softmax(logits, dim=-1)  # (B,3)

        fused = w[:, 0:1] * text_ctx + w[:, 1:2] * cxr_ctx + w[:, 2:3] * ecg_ctx
        return fused, w


class CSDI_MultiModal_MoE(CSDI_base):
    """
    对 irg_ts (L,30) 插补，text/cxr/ecg 作为条件，通过 MoE 融合后注入 side_info。
    """
    def __init__(self, config, device, target_dim: int = 30):
        self.device = device
        self.target_dim = target_dim

        self.ctx_dim = int(config["multimodal"]["ctx_dim"])
        self.ctx_hidden = int(config["multimodal"]["ctx_hidden"])

        self.text_dim = int(config["data"]["text_dim"])
        self.ecg_dim = int(config["data"]["ecg_dim"])
        self.cxr_dim = int(config["data"]["cxr_dim"])

        # 先走原始 CSDI_base 初始化（会创建 diffmodel，但 side_dim 还没加 ctx）
        cfg = copy.deepcopy(config)
        super().__init__(target_dim=target_dim, config=cfg, device=device)

        # 原本的 side_dim（由 time/feature embedding + cond mask 组成）
        base_side_dim = self.emb_total_dim
        new_side_dim = base_side_dim + self.ctx_dim

        # 重建 diffusion 子模型：只改 side_dim
        from diff_models import diff_CSDI
        cfg_diff = copy.deepcopy(cfg["diffusion"])
        cfg_diff["side_dim"] = new_side_dim

        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_CSDI(cfg_diff, input_dim).to(self.device)

        # MoE fuser
        self.moe = ModalityMoEFuser(
            text_dim=self.text_dim,
            cxr_dim=self.cxr_dim,
            ecg_dim=self.ecg_dim,
            ctx_dim=self.ctx_dim,
            hidden=self.ctx_hidden,
        ).to(self.device)

    def process_data(self, batch):
        # (B, L, 30) -> (B, 30, L)
        observed_data = batch["irg_ts"].to(self.device).float().permute(0, 2, 1)
        observed_mask = batch["irg_ts_mask"].to(self.device).float().permute(0, 2, 1)

        observed_tp = batch["ts_tt"].to(self.device).float()  # (B, L)

        gt_mask = batch["gt_mask"].to(self.device).float().permute(0, 2, 1)
        hist_mask = batch["hist_mask"].to(self.device).float().permute(0, 2, 1)

        # 变长：保留 seq_len
        seq_len = batch["seq_len"].to(self.device).long()

        text_vec = batch["text_vec"].to(self.device).float()
        cxr_vec  = batch["cxr_vec"].to(self.device).float()
        ecg_vec  = batch["ecg_vec"].to(self.device).float()

        text_missing = batch["text_missing"].to(self.device).float().view(-1)
        cxr_missing  = batch["cxr_missing"].to(self.device).float().view(-1)
        ecg_missing  = batch["ecg_missing"].to(self.device).float().view(-1)

        return (observed_data, observed_mask, observed_tp,
                gt_mask, hist_mask, seq_len,
                text_vec, cxr_vec, ecg_vec,
                text_missing, cxr_missing, ecg_missing)

    def get_side_info_mm(self, observed_tp, cond_mask, fused_ctx):
        """
        base side_info: (B, base_side_dim, K, L)
        fused_ctx: (B, ctx_dim) -> broadcast (B, ctx_dim, K, L) -> concat
        """
        base_side = super().get_side_info(observed_tp, cond_mask)  # (B, base, K, L)
        B, _, K, L = base_side.shape
        ctx = fused_ctx[:, :, None, None].expand(B, self.ctx_dim, K, L)
        return torch.cat([base_side, ctx], dim=1)

    def forward(self, batch, is_train=1):
        (observed_data, observed_mask, observed_tp,
         gt_mask, hist_mask, seq_len,
         text_vec, cxr_vec, ecg_vec,
         text_missing, cxr_missing, ecg_missing) = self.process_data(batch)

        # cond_mask：训练随机遮盖；eval 使用 gt_mask
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=hist_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)

        fused_ctx, moe_w = self.moe(
            text_vec, cxr_vec, ecg_vec,
            text_missing, cxr_missing, ecg_missing
        )
        side_info = self.get_side_info_mm(observed_tp, cond_mask, fused_ctx)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        loss = loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

        return loss, moe_w.detach()

    @torch.no_grad()
    def evaluate(self, batch, n_samples: int):
        (observed_data, observed_mask, observed_tp,
         gt_mask, _, seq_len,
         text_vec, cxr_vec, ecg_vec,
         text_missing, cxr_missing, ecg_missing) = self.process_data(batch)

        cond_mask = gt_mask
        target_mask = observed_mask - cond_mask

        fused_ctx, moe_w = self.moe(
            text_vec, cxr_vec, ecg_vec,
            text_missing, cxr_missing, ecg_missing
        )
        side_info = self.get_side_info_mm(observed_tp, cond_mask, fused_ctx)

        samples = self.impute(observed_data, cond_mask, side_info, n_samples)  # (B, n_samples, K, L)

        # ✅ 变长：把 padding 区域彻底屏蔽（不参与评估/指标）
        for i in range(len(seq_len)):
            L = int(seq_len[i].item())
            target_mask[i, :, L:] = 0
            observed_mask[i, :, L:] = 0

        return samples, observed_data, target_mask, observed_mask, observed_tp, moe_w
