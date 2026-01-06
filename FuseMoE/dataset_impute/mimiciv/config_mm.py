def get_config():
    config = {
        "model": {
            "timeemb": 128,
            "featureemb": 64,
            "is_unconditional": False,
            "target_strategy": "random", 
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
            "ctx_dim": 128,       
            "ctx_hidden": 256,     # MoE gate hidden
        },
        "data": {
            "text_dim": 768,
            "ecg_dim": 256,
            "cxr_dim": 1024,   
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
            # "save_path": "models/train_ihm-48-cxr-notes-ecg-missingInd_stays_csdi_moe.pt",
        },
        "eval": {
            "n_samples": 20,
        },
    }
    return config
