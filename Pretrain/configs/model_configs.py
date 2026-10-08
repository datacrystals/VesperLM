"""
VesperLLM model configurations keyed by active-parameter size.

Backwards compatibility:
  - "small_v2"  -> "470m"
  - "1b_scaled" -> "1b"
  - "big"       -> "4b"
"""

MODEL_CONFIGS = {
    # ------------------------------------------------------------------
    # 470 M active params (legacy alias: small_v2)
    # ------------------------------------------------------------------
    "470m": {
        # Architecture
        "dim": 1024,
        "n_layers": 10,
        "n_heads": 8,
        "n_kv_heads": 2,
        "hidden_dim": 1280,
        "num_experts": 8,
        "top_k": 2,
        "max_seq_len": 8192,

        # Training & Batching
        "micro_batch_size": 1,
        "target_accumulation_steps": 128,
        "grad_checkpoint": True,

        # Optimizer Dynamics
        "beta1": 0.9,
        "beta2_token_half_life": 10_000_000,
        "max_lr": 2e-4,
        "min_lr": 3e-5,
        "aux_weight": 0.1,

        # Scheduling
        "seq_len_start": 128,
        "seq_len_warmup": 800,
        "warmup_steps": 300,
        "total_steps": 4000,

        # Evaluation
        "eval_interval": 100,
        "val_eval_steps": 50,
    },

    # ------------------------------------------------------------------
    # 1 B active params (legacy alias: 1b_scaled)
    # ------------------------------------------------------------------
    "1b": {
        # Architecture
        "dim": 1152,
        "n_layers": 12,
        "n_heads": 12,
        "n_kv_heads": 2,
        "hidden_dim": 1600,
        "num_experts": 10,
        "top_k": 2,
        "max_seq_len": 2048,

        # Training & Batching
        "micro_batch_size": 1,
        "target_accumulation_steps": 128,

        # Optimizer Dynamics
        "beta1": 0.9,
        "beta2_token_half_life": 12_000_000,
        "max_lr": 1.3e-4,
        "min_lr": 1.5e-5,
        "aux_weight": 0.1,

        # Scheduling
        "seq_len_start": 128,
        "seq_len_warmup": 5000,
        "warmup_steps": 1500,
        "total_steps": 25000,

        # Evaluation
        "eval_interval": 100,
        "val_eval_steps": 75,
    },

    # ------------------------------------------------------------------
    # 2 B active params
    # ------------------------------------------------------------------
    "2b": {
        # Architecture
        "dim": 1536,
        "n_layers": 18,
        "n_heads": 12,
        "n_kv_heads": 4,
        "hidden_dim": 2560,
        "num_experts": 8,
        "top_k": 2,
        "max_seq_len": 2048,

        # Training & Batching
        "micro_batch_size": 1,
        "target_accumulation_steps": 128,

        # Optimizer Dynamics
        "beta1": 0.9,
        "beta2_token_half_life": 14_000_000,
        "max_lr": 1e-4,
        "min_lr": 1e-5,
        "aux_weight": 0.05,

        # Scheduling
        "seq_len_start": 128,
        "seq_len_warmup": 6000,
        "warmup_steps": 2000,
        "total_steps": 35000,

        # Evaluation
        "eval_interval": 500,
        "val_eval_steps": 75,
    },

    # ------------------------------------------------------------------
    # 4 B active params (legacy alias: big)
    # ------------------------------------------------------------------
    "4b": {
        # Architecture
        "dim": 2048,
        "n_layers": 22,
        "n_heads": 16,
        "n_kv_heads": 4,
        "hidden_dim": 3328,
        "num_experts": 8,
        "top_k": 2,
        "max_seq_len": 2048,

        # Training & Batching
        "micro_batch_size": 1,
        "target_accumulation_steps": 128,

        # Optimizer Dynamics
        "beta1": 0.9,
        "beta2_token_half_life": 16_000_000,
        "max_lr": 8e-5,
        "min_lr": 8e-6,
        "aux_weight": 0.01,

        # Scheduling
        "seq_len_start": 128,
        "seq_len_warmup": 8000,
        "warmup_steps": 2500,
        "total_steps": 50000,

        # Evaluation
        "eval_interval": 1000,
        "val_eval_steps": 50,
    },

    # ------------------------------------------------------------------
    # 8 B active params
    # ------------------------------------------------------------------
    "8b": {
        # Architecture
        "dim": 2560,
        "n_layers": 26,
        "n_heads": 20,
        "n_kv_heads": 4,
        "hidden_dim": 4096,
        "num_experts": 8,
        "top_k": 2,
        "max_seq_len": 2048,

        # Training & Batching
        "micro_batch_size": 1,
        "target_accumulation_steps": 128,

        # Optimizer Dynamics
        "beta1": 0.9,
        "beta2_token_half_life": 18_000_000,
        "max_lr": 6e-5,
        "min_lr": 6e-6,
        "aux_weight": 0.01,

        # Scheduling
        "seq_len_start": 128,
        "seq_len_warmup": 10000,
        "warmup_steps": 3000,
        "total_steps": 60000,

        # Evaluation
        "eval_interval": 1000,
        "val_eval_steps": 50,
    },
    # ------------------------------------------------------------------
    # ~30 M active params — tiny proof-of-pipeline agent model
    # ------------------------------------------------------------------
    "tiny_agent": {
        # Architecture
        "dim": 384,
        "n_layers": 6,
        "n_heads": 6,
        "n_kv_heads": 2,
        "hidden_dim": 1024,
        "num_experts": 4,
        "top_k": 2,
        "max_seq_len": 2048,

        # Training & Batching
        "micro_batch_size": 6,
        "grad_checkpoint": False,
        "target_accumulation_steps": 128,

        # Optimizer Dynamics
        "beta1": 0.9,
        "beta2_token_half_life": 1_000_000,
        "max_lr": 6e-4,
        "min_lr": 6e-5,
        "aux_weight": 0.1,

        # Scheduling
        "seq_len_start": 128,
        "seq_len_warmup": 1000,
        "warmup_steps": 200,
        "total_steps": 2000,

        # Evaluation
        "eval_interval": 100,
        "val_eval_steps": 50,
    },

    # ------------------------------------------------------------------
    # ~80 M active params — scaled-up tiny agent model
    # (~118 M total, ~81 M active: embedding 33.5M + 6 GLA layers
    # ~10.5M each + 2 full-attn layers ~10.1M each; 2 of 4 MoE
    # experts are inactive per token)
    # ------------------------------------------------------------------
    "tiny_agent_v2": {
        # Architecture
        "dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "n_kv_heads": 2,
        "hidden_dim": 1536,
        "num_experts": 4,
        "top_k": 2,
        "max_seq_len": 2048,
        "linear_type": "gla",  # "gla" (default) or "mamba2" (fla SSD path)

        # Training & Batching
        "micro_batch_size": 6,
        "grad_checkpoint": False,
        "target_accumulation_steps": 128,

        # Optimizer Dynamics
        "beta1": 0.9,
        "beta2_token_half_life": 1_000_000,
        "max_lr": 6e-4,
        "min_lr": 6e-5,
        "aux_weight": 0.1,

        # Scheduling
        "seq_len_start": 128,
        "seq_len_warmup": 1000,
        "warmup_steps": 200,
        "total_steps": 6000,

        # Evaluation
        "eval_interval": 100,
        "val_eval_steps": 50,
    },
    # ------------------------------------------------------------------
    # ~81 M active params — Vesper-K lineage smoke config
    # tiny_agent_v2 with KDA linear layers + MLA full layers
    # ------------------------------------------------------------------
    "tiny_agent_k": {
        # Architecture
        "dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "n_kv_heads": 2,
        "hidden_dim": 1536,
        "num_experts": 4,
        "top_k": 2,
        "max_seq_len": 2048,
        "linear_type": "kda",
        "full_type": "mla",
        "kda_head_dim": 64,
        "kv_lora_rank": 256,
        "v_head_dim": 128,

        # Training & Batching
        "micro_batch_size": 6,
        "grad_checkpoint": False,
        "target_accumulation_steps": 128,

        # Optimizer Dynamics
        "beta1": 0.9,
        "beta2_token_half_life": 1_000_000,
        "max_lr": 6e-4,
        "min_lr": 6e-5,
        "aux_weight": 0.1,

        # Scheduling
        "seq_len_start": 128,
        "seq_len_warmup": 1000,
        "warmup_steps": 200,
        "total_steps": 6000,

        # Evaluation
        "eval_interval": 100,
        "val_eval_steps": 50,
    },

    # ------------------------------------------------------------------
    # 470 M active params — Vesper-K lineage (KDA + MLA) for MI300X
    # ------------------------------------------------------------------
    "470m_k": {
        # Architecture
        "dim": 1024,
        "n_layers": 10,
        "n_heads": 8,
        "n_kv_heads": 2,
        "hidden_dim": 1280,
        "num_experts": 8,
        "top_k": 2,
        "max_seq_len": 8192,
        "linear_type": "kda",
        "full_type": "mla",
        "kda_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,

        # Training & Batching
        "micro_batch_size": 1,
        "target_accumulation_steps": 128,
        "grad_checkpoint": True,

        # Optimizer Dynamics
        "beta1": 0.9,
        "beta2_token_half_life": 10_000_000,
        "max_lr": 2e-4,
        "min_lr": 3e-5,
        "aux_weight": 0.1,

        # Scheduling
        "seq_len_start": 128,
        "seq_len_warmup": 800,
        "warmup_steps": 300,
        "total_steps": 4000,

        # Evaluation
        "eval_interval": 100,
        "val_eval_steps": 50,
    },
}

# Backwards-compatibility aliases
_CONFIG_ALIASES = {
    "small_v2": "470m",
    "1b_scaled": "1b",
    "big": "4b",
}


def get_model_config(name):
    """Fetch a config by name, resolving legacy aliases automatically."""
    resolved = _CONFIG_ALIASES.get(name, name)
    if resolved not in MODEL_CONFIGS:
        raise KeyError(
            f"Unknown config '{name}' (resolved: '{resolved}'). "
            f"Available: {list_configs()}"
        )
    # Return a shallow copy so callers can mutate safely
    return MODEL_CONFIGS[resolved].copy()


def list_configs():
    """Return all available primary config names."""
    return list(MODEL_CONFIGS.keys())
