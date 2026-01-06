def get_config():
    """
    只要 main_model.py / diff_models.py 的 CSDI_base/diff_CSDI 接口不变，这个就能用。
    """
    config = {
        "model": {
            "timeemb": 128,
            "featureemb": 64,
            "is_unconditional": False,
            "target_strategy": "random",  # random|mix|pattern(若你原CSDI支持)
        },
        "diffusion": {
            "num_steps": 50,
            "schedule": "linear",
            "beta_start": 1e-4,
            "beta_end": 2e-2,

            "channels": 64,
            "layers": 4,
            "nheads": 8,
            "is_linear": False,
            "diffusion_embedding_dim": 128,
            # side_dim 会在模型里自动设置，并额外+ctx_dim
        },
        "multimodal": {
            "ctx_dim": 128,        # 融合条件向量维度 -> broadcast 到 (B,ctx,K,L) 拼进 side_info
            "ctx_hidden": 256,     # MoE gate hidden
        },
        "data": {
            "text_dim": 768,
            "ecg_dim": 256,
            "cxr_dim": 1024,       # ✅ 统一到 1024（适配你样本2）
            "eval_mask_ratio": 0.2 # val/test 时从观测里再扣掉一部分作为 target
        },
        "train": {
            "seed": 1234,
            "batch_size": 32,
            "lr": 1e-4,
            "weight_decay": 1e-6,
            "max_epochs": 50,
            "grad_clip": 1.0,
            "num_workers": 2,
            "log_every": 50,
            "save_path": "models/csdi_moe.pt",
        },
        "eval": {
            "n_samples": 20,
        },
    }
    return config
