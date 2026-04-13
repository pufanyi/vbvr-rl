"""Training configuration."""

from typing import Literal

from pydantic import BaseModel, field_validator


class TrainConfig(BaseModel):
    # Model
    model_path: str = "storage/models/Wan2.2-I2V-A14B-Diffusers"

    # Data
    dataset_json: str = "data/train.json"
    latent_webdataset_dir: str | None = None  # dir of precomputed latent tar shards; skips VAE/T5 loading
    dataset_size: int | None = None  # total samples in dataset (required for webdataset / IterableDataset)
    num_frames: int | None = None  # override dataset JSON config; default 81
    max_area: int | None = None  # override dataset JSON config; default 480*832
    height: int | None = None  # override dataset JSON config; fixed height
    width: int | None = None  # override dataset JSON config; fixed width
    fps: int | None = None  # override dataset JSON config; default 16
    num_workers: int = 4
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Training
    output_dir: str = "storage/checkpoints"
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    save_steps: int = 500
    log_steps: int = 10
    seed: int = 42
    ema_decay: float = 0.0  # 0 = disabled; typical value: 0.9999

    # Optimizer
    optimizer: Literal["adamw", "muon"] = "adamw"
    # AdamW
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_fused: bool = True
    # Muon
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: Literal["original", "match_rms_adamw"] | None = None
    muon_fallback_lr: float | None = None  # AdamW lr for non-2D params; None = use learning_rate

    # Which components to train
    train_experts: Literal["both", "high", "low"] = "both"
    train_text_encoder: bool = False
    gradient_checkpointing: bool = True

    # FSDP2 mixed precision
    param_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    reduce_dtype: Literal["float32", "bfloat16"] = "float32"

    # Liger Kernel (fused Triton kernels)
    use_liger_kernel: bool = False

    # torch.compile
    torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str | None = None  # e.g. "reduce-overhead", "max-autotune"

    # LoRA (set lora_rank > 0 to enable)
    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Checkpoint
    resume_from: str | None = None
    auto_resume: bool = True  # auto-detect latest checkpoint in output_dir
    # None = auto (reset when resume_from is set, keep when auto-resuming from output_dir)
    reset_dataloader: bool | None = None

    # Logging
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    # GRPO (set grpo_group_size > 0 to enable Flow-GRPO training)
    grpo_group_size: int | None = None  # G: number of samples per prompt. None = SFT mode
    grpo_sample_batch_size: int = 1  # how many G samples to batch together (tune for GPU memory)
    grpo_num_sampling_steps: int = 10  # T: denoising steps during SDE sampling
    grpo_clip_range: float = 1e-3  # PPO clipping epsilon
    grpo_kl_coeff: float = 0.004  # beta: KL penalty coefficient against reference policy
    grpo_sde_noise_scale: float = 0.7  # a in sigma_t = a * sqrt(t / (1-t))
    grpo_sde_sigma_min: float = 0.0  # noise floor for SDE std
    grpo_sde_sigma_max: float = 1.0  # noise ceiling for SDE std
    grpo_adv_clip_max: float = 5.0  # clamp advantages to [-max, max]
    grpo_reward_fn: str = "neg_loss"  # reward function name
    grpo_cfg_scale: float = 1.0  # classifier-free guidance scale during sampling

    # COS (Chain-of-Step) piecewise flow matching
    cos_tau_sigma: list[float] = [0.5]  # piecewise boundaries in sigma space (descending); len = num_intermediates
    cos_boundary_noise_std: float = 0.02  # Gaussian perturbation std for x_tau in low stage
    cos_use_standard_formula: bool = False  # ablation: use standard sigma formula per segment (discontinuous)
    cos_path_type: Literal[
        "linear", "cosine", "cubic_hermite", "smooth_blend", "quadratic_bezier", "target_linear", "target_cosine"
    ] = "linear"
    cos_smooth_blend_delta: float = 0.05  # half-width of blending window (only for smooth_blend path)

    # Trainer selection
    trainer: Literal["i2v", "cos", "grpo", "dancegrpo"] = "i2v"

    @field_validator("cos_tau_sigma", mode="before")
    @classmethod
    def _wrap_tau_sigma(cls, v):
        if isinstance(v, (int, float)):
            return [float(v)]
        return v

    # DanceGRPO (paper-inspired variant of GRPO)
    dancegrpo_share_group_init_noise: bool = True
    dancegrpo_timestep_selection_ratio: float = 1.0

    @field_validator("dancegrpo_timestep_selection_ratio")
    @classmethod
    def _validate_dancegrpo_timestep_selection_ratio(cls, v: float):
        if not (0.0 < v <= 1.0):
            raise ValueError(f"dancegrpo_timestep_selection_ratio must be in (0, 1], got {v}")
        return v

    # HSDP: shard within node, replicate across nodes.
    # Keeps NCCL IB traffic to intra-node NVLink; cross-node uses only gradient
    # all-reduce (much less locked-memory / IB registration pressure).
    hsdp: bool = True  # Hybrid Sharded Data Parallel; single-node auto-falls back to plain FSDP
    hsdp_replicate_backend: Literal["nccl", "gloo"] | None = None  # None = auto (prefer nccl)

    # Expert parallel: split MoE experts across GPU sub-groups
    expert_parallel: bool = False  # each expert gets world_size/2 GPUs with independent FSDP
    expert_parallel_data_mode: Literal["duplicate", "split"] = "duplicate"
    # duplicate: each expert group iterates a full copy of the dataset (old SFT behavior)
    # split: shard data across all ranks so expert-parallel uses full global throughput
