"""Training configuration."""

from typing import Literal

from pydantic import BaseModel, field_validator


class TrainConfig(BaseModel):
    """Base training configuration shared by all trainers."""

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
    num_workers: int = 16
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
    skip_nonfinite_gradients: bool = True
    detect_anomaly: bool = False  # debug only; very slow
    warmup_steps: int = 100
    save_steps: int = 500
    log_steps: int = 10
    seed: int = 42
    ema_decay: float = 0.0  # 0 = disabled; typical value: 0.9999
    distributed_timeout_minutes: int = 240

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

    # FSDP2
    fsdp: bool = True  # False = no sharding, manual gradient all-reduce (faster for LoRA / small models)
    param_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    reduce_dtype: Literal["float32", "bfloat16"] = "float32"
    transformer_load_dtype: Literal["auto", "bfloat16", "float32"] = "auto"
    # auto: full fine-tuning loads trainable transformers in fp32 for fp32 Adam states;
    # LoRA keeps the frozen base in bf16 unless explicitly overridden.
    prompt_dropout: float = 0.0  # whole-prompt dropout for CFG/unconditional branch training

    # Liger Kernel (fused Triton kernels)
    use_liger_kernel: bool = False

    # torch.compile
    torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str | None = None  # e.g. "reduce-overhead", "max-autotune"
    # cuDNN SDPA backward can return NaN for Wan low-noise training on H100/PyTorch 2.11.
    disable_cudnn_sdp: bool = True
    # Explicit Diffusers attention backend, e.g. "_native_flash". None uses Diffusers/PyTorch dispatch.
    attention_backend: str | None = None

    # LoRA (set lora_rank > 0 to enable)
    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Checkpoint
    resume_from: str | None = None
    auto_resume: bool = True  # auto-detect latest checkpoint in output_dir
    # None = auto (reset when resume_from is set, keep when auto-resuming from output_dir)
    reset_dataloader: bool | None = None
    # Local rank within the DCP process group. None keeps PyTorch's default
    # coordinator rank 0; negative values count from the end.
    checkpoint_dcp_coordinator_rank: int | None = None
    checkpoint_dcp_thread_count: int = 4
    checkpoint_dcp_sync_files: bool = True

    # Logging
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    # HSDP: shard within node, replicate across nodes.
    # Keeps NCCL IB traffic to intra-node NVLink; cross-node uses only gradient
    # all-reduce (much less locked-memory / IB registration pressure).
    hsdp: bool = False  # Hybrid Sharded Data Parallel; shard within node, replicate across nodes
    hsdp_replicate_backend: Literal["nccl", "gloo"] | None = None  # None = auto (prefer nccl)

    # Expert parallel: split MoE experts across GPU sub-groups
    expert_parallel: bool = False  # each expert gets world_size/2 GPUs with independent FSDP
    expert_parallel_data_mode: Literal["duplicate", "split"] = "duplicate"
    # duplicate: each expert group iterates a full copy of the dataset (old SFT behavior)
    # split: shard data across all ranks so expert-parallel uses full global throughput

    @field_validator("prompt_dropout")
    @classmethod
    def _validate_prompt_dropout(cls, v: float):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"prompt_dropout must be in [0, 1], got {v}")
        return v


class SFTConfig(TrainConfig):
    """SFT training configuration."""

    # Trainer selection
    trainer: Literal["i2v", "cos"] = "i2v"

    # COS (Chain-of-Step) piecewise flow matching
    cos_tau_sigma: list[float] = [0.5]  # piecewise boundaries in sigma space (descending); len = num_intermediates
    cos_boundary_noise_std: float = 0.02  # Gaussian perturbation std for x_tau in low stage
    cos_use_standard_formula: bool = False  # ablation: use standard sigma formula per segment (discontinuous)
    cos_path_type: Literal[
        "linear", "cosine", "cubic_hermite", "smooth_blend", "quadratic_bezier", "target_linear", "target_cosine"
    ] = "linear"
    cos_smooth_blend_delta: float = 0.05  # half-width of blending window (only for smooth_blend path)

    @field_validator("cos_tau_sigma", mode="before")
    @classmethod
    def _wrap_tau_sigma(cls, v):
        if isinstance(v, (int, float)):
            return [float(v)]
        return v


class CorrectionConfig(TrainConfig):
    """On-policy correction training configuration.

    Supervised FM loss on the standard (ε, x_GT) line, plus a correction term
    trained on the line (ε, x̂) where x̂ is an EMA-teacher rollout from ε.
    The correction target is (x_σ - x_GT) / σ, which re-aims the velocity at
    the true GT even when the teacher's generation drifts away.

    Expert parallel is unsupported here: the teacher rollout is sequential
    and keeps a whole batch at one σ per step, so only one MoE expert is
    active at a time — EP would idle half the GPUs.
    """

    # Loss balance
    correction_weight: float = 0.1  # λ on the correction term

    # Teacher rollout
    correction_num_teacher_steps: int = 4
    correction_use_sde: bool = True  # False -> deterministic ODE
    correction_sde_sigma_max: float = 1.0
    correction_cfg_scale: float = 1.0

    # σ sampling clip for constructing training points (avoid 1/σ blow-up)
    correction_sigma_lo: float = 0.05
    correction_sigma_hi: float = 0.9

    # Amortize rollout cost: only run the correction term every N optimizer micro-steps
    correction_every_n_steps: int = 1


class RLConfig(TrainConfig):
    """RL training configuration. Defaults to HSDP (falls back to plain FSDP on single node)."""

    hsdp: bool = True

    # Smoke / bounded RL runs. Defaults preserve the normal epoch-based loop.
    max_steps: int | None = None
    save_epoch_checkpoints: bool = True

    # Trainer selection
    trainer: Literal["grpo", "dancegrpo"] = "grpo"

    # GRPO (set grpo_group_size > 0 to enable Flow-GRPO training)
    grpo_group_size: int | None = None  # G: number of samples per prompt. None = SFT mode
    grpo_sample_batch_size: int = 1  # how many G samples to batch together (tune for GPU memory)
    grpo_num_sampling_steps: int = 10  # T: denoising steps during SDE sampling
    grpo_clip_range: float = 1e-3  # PPO clipping epsilon
    grpo_kl_coeff: float = 0.004  # beta: KL penalty coefficient against reference policy
    grpo_sde_noise_scale: float = 0.3  # DanceGRPO eta / exploration noise level
    grpo_sde_sigma_min: float = 0.0  # noise floor for SDE std
    grpo_sde_sigma_max: float = 1.0  # noise ceiling for SDE std
    grpo_adv_clip_max: float = 5.0  # clamp advantages to [-max, max]
    grpo_reward_fn: str = "neg_loss"  # reward function name
    grpo_cfg_scale: float = 1.0  # classifier-free guidance scale during sampling

    # DanceGRPO (paper-inspired variant of GRPO)
    dancegrpo_share_group_init_noise: bool = True
    dancegrpo_timestep_selection_ratio: float = 0.6

    @field_validator("dancegrpo_timestep_selection_ratio")
    @classmethod
    def _validate_dancegrpo_timestep_selection_ratio(cls, v: float):
        if not (0.0 < v <= 1.0):
            raise ValueError(f"dancegrpo_timestep_selection_ratio must be in (0, 1], got {v}")
        return v

    # ------------------------------------------------------------------
    # Maze reward (grpo_reward_fn: "maze")
    # ------------------------------------------------------------------
    # Number of frames to VAE-decode per rollout (cost is B·G·K decodes).
    maze_reward_num_frames: int = 6
    # Component weights — raw reward = w_traj·r_traj + w_onpath·r_onpath + w_goal·r_goal
    # (GRPO z-scores within group, so absolute scale doesn't matter, only ratios).
    maze_reward_w_traj: float = 1.0
    maze_reward_w_onpath: float = 0.5
    maze_reward_w_goal: float = 1.0
    # r_goal: 1 if end-frame ball is within this many cells of goal, else 0.
    maze_reward_goal_cells: float = 0.75

    # ------------------------------------------------------------------
    # Maze growing-line reward (grpo_reward_fn: "maze_line")
    # ------------------------------------------------------------------
    # Scores the generated red/yellow/blue/pink path-line mask against the GT
    # decoded line mask, plus a simple "line reaches goal" term.
    maze_line_reward_num_frames: int = 6
    maze_line_reward_color_threshold: float = 90.0
    maze_line_reward_color_temperature: float = 12.0
    maze_line_reward_w_mask: float = 1.0
    maze_line_reward_w_goal: float = 0.5
    maze_line_reward_goal_cells: float = 1.0

    # ------------------------------------------------------------------
    # Maze tracker/eval reward (grpo_reward_fn: "maze_tracker")
    # ------------------------------------------------------------------
    # Mirrors src.eval.maze_tracker_score overall:
    # 0.35*traj + 0.25*on_path + 0.25*goal + 0.15*progress.
    maze_tracker_reward_num_frames: int = 21
    maze_tracker_reward_search_radius: int = 96
    maze_tracker_reward_color_slack: float = 28.0
    maze_tracker_reward_goal_tolerance_cells: float = 1.0
    maze_tracker_reward_max_mean_error_cells: float = 4.0
    maze_tracker_reward_w_traj: float = 0.35
    maze_tracker_reward_w_onpath: float = 0.25
    maze_tracker_reward_w_goal: float = 0.25
    maze_tracker_reward_w_progress: float = 0.15

    # ------------------------------------------------------------------
    # VBVR rule reward (grpo_reward_fn: "vbvr_rule")
    # ------------------------------------------------------------------
    # Decode generated/GT latents to temporary mp4 files and score with the
    # vendored VBVR EvalKit task-specific rule evaluator.
    vbvr_reward_device: str = "cpu"
    vbvr_reward_fps: int = 16
    vbvr_reward_decode_batch_size: int = 1
    vbvr_reward_task_specific_only: bool = True
    vbvr_reward_tmp_dir: str = "storage/tmp/vbvr_rule_reward"
    vbvr_reward_keep_tmp: bool = False
    vbvr_reward_unsupported_score: float = 0.0
