"""Base GRPO trainer with shared Flow-GRPO algorithm infrastructure.

Provides reference policy management, SDE sampling schedule, reward/advantage
computation, and the outer training loop.  Concrete subclasses implement
``_grpo_step(batch)`` with their specific policy-gradient update logic.
"""

import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger

from src.trainer.base_rl_trainer import BaseRLTrainer
from src.trainer.config import RLConfig
from src.trainer.flops import MFUMonitor, compute_wan_seq_len, estimate_wan_forward_flops, get_gpu_peak_flops_bf16
from src.trainer.rewards import build_reward
from src.trainer.utils import cosine_lr, format_eta, shard_transformer, to_model_pixels


def _repeat_meta(meta: dict[str, Any], cur_S: int) -> dict[str, Any]:
    """Interleave-replicate every per-sample metadata field to match a chunk's group size.

    Mirrors the ``condition.repeat_interleave(cur_S, dim=0)`` pattern used on
    the training inputs so reward functions see per-sample metadata aligned
    with the flattened ``(B, cur_S)`` rollout batch.
    """
    repeated: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, torch.Tensor):
            repeated[k] = v.repeat_interleave(cur_S, dim=0)
        elif isinstance(v, list):
            repeated[k] = [item for item in v for _ in range(cur_S)]
        else:
            repeated[k] = v
    return repeated


class BaseGRPOTrainer(BaseRLTrainer):
    """Shared Flow-GRPO infrastructure.

    Subclasses must implement ``_grpo_step(batch) -> dict[str, float]``
    which performs the sampling, reward, advantage, and policy-gradient
    update for a single batch.
    """

    def __init__(self, cfg: RLConfig):
        assert cfg.grpo_group_size is not None and cfg.grpo_group_size > 0, (
            "grpo_group_size must be > 0 for GRPO training"
        )
        super().__init__(cfg)
        if self.expert_parallel:
            high_steps, low_steps = self._validate_sampling_schedule(cfg.grpo_num_sampling_steps)
            logger.info(
                "Flow-GRPO EP schedule: T={} split into high={} low={}",
                cfg.grpo_num_sampling_steps,
                high_steps,
                low_steps,
            )
        logger.info(
            "Flow-GRPO | G={} T={} expert_parallel={}",
            cfg.grpo_group_size,
            cfg.grpo_num_sampling_steps,
            cfg.expert_parallel,
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _get_expert_parallel_sampler_seed(self, cfg: RLConfig) -> int:
        """Keep high/low expert groups on the same batches for cooperative GRPO."""
        return cfg.seed

    def _pre_fsdp_setup(self, cfg: RLConfig) -> None:
        """Create frozen reference policy copies for full fine-tuning.

        Copies are made via CPU to avoid holding 4 full transformers on GPU
        simultaneously (which can OOM before FSDP sharding kicks in).
        """
        self.is_lora = cfg.lora_rank > 0
        self.ref_transformers: dict[str, torch.nn.Module] = {}
        if not self.is_lora:
            logger.info("Full fine-tuning mode: creating frozen reference policy copies (via CPU)")
            for name, m in [("transformer", self.model.transformer), ("transformer_2", self.model.transformer_2)]:
                if m is not None:
                    ref = deepcopy(m.cpu()).requires_grad_(False).eval().to(self.device)
                    m.to(self.device)  # move training model back to GPU
                    self.ref_transformers[name] = ref
                    logger.info("Reference {} created", name)

    def _setup_fsdp(self, cfg: RLConfig) -> list[torch.nn.Module]:
        sync_modules = super()._setup_fsdp(cfg)
        if cfg.fsdp:
            for _name, ref in self.ref_transformers.items():
                shard_transformer(ref, self.mesh, self.mp_policy)
        return sync_modules

    # ------------------------------------------------------------------
    # MFU
    # ------------------------------------------------------------------

    def _post_init(self, cfg: RLConfig) -> None:
        self.mfu_monitor = self._setup_mfu(cfg)
        self.reward_fn = build_reward(cfg.grpo_reward_fn, self, cfg)
        logger.info("Reward function: {}", cfg.grpo_reward_fn)

    def _setup_mfu(self, cfg: RLConfig) -> MFUMonitor | None:
        """Pre-compute FLOPs per GRPO step and create MFU monitor.

        A GRPO step has multiple phases with different forward counts:
          Sampling:  G × T × (2 if CFG else 1)  forwards (no grad)
          Reward:    G                           forwards (no grad)
          Policy:    G × T_replay               forwards (with grad → 3× FLOPs)
          Reference: G × T_replay               forwards (no grad, if kl > 0)
        """
        gpu_peak = get_gpu_peak_flops_bf16()
        if gpu_peak is None:
            return None

        ref_t = self.model.transformer or self.model.transformer_2
        if ref_t is None:
            return None

        # Sequence length
        seq_len: int | None = None
        if hasattr(self.dataset, "_configs"):
            est_cfg = self.dataset._configs[0]
            h = est_cfg.fixed_height or int(est_cfg.max_area**0.5)
            w = est_cfg.fixed_width or int(est_cfg.max_area**0.5)
            t_cfg = ref_t.config
            seq_len = compute_wan_seq_len(
                est_cfg.num_frames,
                h,
                w,
                patch_size=tuple(t_cfg.patch_size),
                vae_temporal_factor=self.model.vae_scale_factor_temporal,
                vae_spatial_factor=self.model.vae_scale_factor_spatial,
            )
        if seq_len is None:
            logger.info("MFU monitor: skipped (no resolution info available)")
            return None

        t_cfg = ref_t.config
        fwd_flops = estimate_wan_forward_flops(
            num_layers=t_cfg.num_layers,
            num_heads=t_cfg.num_attention_heads,
            head_dim=t_cfg.attention_head_dim,
            ffn_dim=t_cfg.ffn_dim,
            seq_len=seq_len,
        )

        G = cfg.grpo_group_size
        T = cfg.grpo_num_sampling_steps
        if cfg.trainer == "dancegrpo":
            T_candidates = max(1, T - 1)
            T_replay = max(1, int(T_candidates * cfg.dancegrpo_timestep_selection_ratio))
        else:
            T_replay = T
        cfg_mult = 2 if cfg.grpo_cfg_scale > 1.0 else 1

        n_no_grad = G * T * cfg_mult + G  # sampling + reward
        n_with_grad = G * T_replay * cfg_mult  # policy update
        if cfg.grpo_kl_coeff > 0:
            n_no_grad += G * T_replay * cfg_mult  # reference forwards

        flops_per_step = (n_no_grad * fwd_flops) + (n_with_grad * 3 * fwd_flops)

        logger.info(
            "MFU monitor: seq_len={} fwd={:.2e} FLOPs | "
            "no_grad={} with_grad={} → step={:.2e} FLOPs | GPU={} ({:.0f} TFLOPS)",
            seq_len,
            fwd_flops,
            n_no_grad,
            n_with_grad,
            flops_per_step,
            torch.cuda.get_device_name(0),
            gpu_peak / 1e12,
        )
        return MFUMonitor(flops_per_step, gpu_peak)

    # ------------------------------------------------------------------
    # Abstract: subclass must implement
    # ------------------------------------------------------------------

    def _grpo_step(self, batch: dict) -> dict[str, float]:
        """Execute one GRPO step. Must be implemented by subclass."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Batch encoding
    # ------------------------------------------------------------------

    # Keys consumed by the core RL loop. Anything else the dataset emits is
    # passed through to the reward as ``meta`` (e.g. ``maze_*`` tensors).
    _CORE_BATCH_KEYS = frozenset(
        {
            "prompt_embeds",
            "video_latents",
            "condition",
            "prompt",
            "videos",
            "image",
            "index",
        }
    )

    def _encode_batch_inputs(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Encode one GRPO batch, supporting both raw and precomputed paths.

        Returns ``(prompt_embeds, gt_video_latents, condition, meta)`` where
        ``meta`` holds any non-core tensors the dataset passed through — used
        by rewards that need per-sample side information (e.g. the maze
        reward reads ``maze_frame_positions_pix`` from here).
        """
        if "prompt_embeds" in batch:
            prompt_embeds = batch["prompt_embeds"].to(self.device)
            video_latents = batch["video_latents"]
            if isinstance(video_latents, list):
                gt_video_latents = video_latents[-1].to(self.device)
            else:
                gt_video_latents = video_latents.to(self.device)
            condition = batch["condition"].to(self.device)
        else:
            prompt_embeds = self.model.encode_text(batch["prompt"], self.device)
            video = to_model_pixels(batch["videos"][-1], self.device)
            image = to_model_pixels(batch["image"], self.device)
            gt_video_latents = self.model.encode_video(video)
            condition = self.model.prepare_condition(image, video.shape[2], video.shape[-2], video.shape[-1])

        meta: dict[str, Any] = {}
        for key, value in batch.items():
            if key in self._CORE_BATCH_KEYS:
                continue
            if isinstance(value, torch.Tensor):
                meta[key] = value.to(self.device)
            else:
                meta[key] = value
        return prompt_embeds, gt_video_latents, condition, meta

    # ------------------------------------------------------------------
    # Sampling schedule
    # ------------------------------------------------------------------

    def _local_expert_name(self) -> str:
        if self.expert_parallel:
            return "high" if self.expert_group == 0 else "low"
        if self.model.transformer is not None and self.model.transformer_2 is None:
            return "high"
        if self.model.transformer is None and self.model.transformer_2 is not None:
            return "low"
        return "both"

    def _build_sampling_schedule(
        self,
        num_sampling_steps: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, list[float], int]:
        """Build the GRPO sigma schedule and return the high-expert prefix length."""
        if device is None:
            device = torch.device("cpu")
        t_values = torch.linspace(1.0, 0.0, num_sampling_steps + 1, device=device, dtype=torch.float32)
        shift = self.model.flow_shift
        sigmas = shift * t_values / (1.0 + (shift - 1.0) * t_values)
        timesteps = (sigmas[:-1] * self.model.num_train_timesteps).cpu().tolist()
        high_flags = [timestep >= self.model.boundary_timestep for timestep in timesteps]
        high_step_count = sum(high_flags)
        return sigmas, timesteps, high_step_count

    def _validate_sampling_schedule(self, num_sampling_steps: int) -> tuple[int, int]:
        """Ensure expert-parallel GRPO only sees a single high->low handoff."""
        _, timesteps, high_step_count = self._build_sampling_schedule(num_sampling_steps)
        low_step_count = num_sampling_steps - high_step_count
        flags = [timestep >= self.model.boundary_timestep for timestep in timesteps]
        expected = [True] * high_step_count + [False] * low_step_count
        if flags != expected:
            raise ValueError(
                "expert_parallel GRPO requires a single high-prefix/low-suffix schedule; "
                f"got {flags} for T={num_sampling_steps}"
            )
        if high_step_count <= 0 or low_step_count <= 0:
            raise ValueError(
                "expert_parallel GRPO requires both experts to appear in the sampling schedule; "
                f"got high={high_step_count}, low={low_step_count}. "
                "Increase grpo_num_sampling_steps or disable expert_parallel."
            )
        return high_step_count, low_step_count

    # ------------------------------------------------------------------
    # Expert routing
    # ------------------------------------------------------------------

    def _get_local_transformer(self, timestep_val: float) -> torch.nn.Module:
        """Return the local expert, rejecting cross-expert timesteps in EP mode."""
        if not self.expert_parallel:
            return self.model._get_expert_for_timestep(timestep_val)

        is_high = timestep_val >= self.model.boundary_timestep
        expected_group = 0 if is_high else 1
        if self.expert_group != expected_group:
            raise RuntimeError(
                f"expert_parallel rank {self.rank} ({self._local_expert_name()}) received timestep {timestep_val} "
                "for the remote expert"
            )

        transformer = self.model.transformer if is_high else self.model.transformer_2
        if transformer is None:
            raise RuntimeError(f"Missing local {self._local_expert_name()} expert on rank {self.rank}")
        return transformer

    def _get_local_ref_transformer(self, timestep_val: float) -> torch.nn.Module:
        """Return the local frozen reference module in expert-parallel mode."""
        if not self.expert_parallel:
            raise RuntimeError("_get_local_ref_transformer is only valid in expert_parallel mode")

        is_high = timestep_val >= self.model.boundary_timestep
        key = "transformer" if is_high else "transformer_2"
        expected_group = 0 if is_high else 1
        if self.expert_group != expected_group:
            raise RuntimeError(
                f"Reference forward for timestep {timestep_val} routed to remote expert on rank {self.rank}"
            )
        if key not in self.ref_transformers:
            raise RuntimeError(f"Missing local reference module '{key}' on rank {self.rank}")
        return self.ref_transformers[key]

    # ------------------------------------------------------------------
    # Local SDE sampling segment
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_local_sampling_segment(
        self,
        start_latent: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        sigmas: torch.Tensor,
        timestep_vals: list[float],
        step_start: int,
        step_end: int,
        cfg: RLConfig,
    ) -> dict:
        """Run only the local expert's segment of the SDE sampling schedule."""
        latent = start_latent
        all_latents = [latent]
        all_log_probs = []
        local_timesteps = []

        for step_idx in range(step_start, step_end):
            sigma = sigmas[step_idx].item()
            sigma_prev = sigmas[step_idx + 1].item()
            timestep_val = timestep_vals[step_idx]

            transformer = self._get_local_transformer(timestep_val)
            model_input = torch.cat([latent, condition], dim=1)
            timestep_tensor = torch.tensor([timestep_val], device=latent.device, dtype=torch.bfloat16).expand(
                latent.shape[0]
            )
            model_output = transformer(
                hidden_states=model_input,
                timestep=timestep_tensor,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            if cfg.grpo_cfg_scale > 1.0:
                uncond_embeds = torch.zeros_like(prompt_embeds)
                uncond_output = transformer(
                    hidden_states=model_input,
                    timestep=timestep_tensor,
                    encoder_hidden_states=uncond_embeds,
                    return_dict=False,
                )[0]
                model_output = uncond_output + cfg.grpo_cfg_scale * (model_output - uncond_output)

            noise = torch.randn_like(latent)
            latent, _prev_mean, log_prob = self.model._sde_step(
                sample=latent,
                model_output=model_output,
                sigma=sigma,
                sigma_prev=sigma_prev,
                sde_noise_scale=cfg.grpo_sde_noise_scale,
                sigma_min=cfg.grpo_sde_sigma_min,
                sigma_max=cfg.grpo_sde_sigma_max,
                noise=noise,
            )
            all_latents.append(latent)
            all_log_probs.append(log_prob)
            local_timesteps.append(timestep_val)

        return {
            "latents": all_latents,
            "log_probs": all_log_probs,
            "timesteps": local_timesteps,
            "sigmas": sigmas[step_start : step_end + 1].clone(),
        }

    # ------------------------------------------------------------------
    # Advantage
    # ------------------------------------------------------------------

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Group-relative advantage normalization."""
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True) + 1e-4
        advantages = (rewards - mean) / std
        return advantages.clamp(-self.cfg.grpo_adv_clip_max, self.cfg.grpo_adv_clip_max)

    # ------------------------------------------------------------------
    # Reference policy forward
    # ------------------------------------------------------------------

    def _ref_forward(
        self,
        latent: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timestep_val: float,
    ) -> torch.Tensor:
        """Forward pass through reference policy. Returns velocity prediction."""
        B = latent.shape[0]
        device = latent.device
        model_input = torch.cat([latent, condition], dim=1)
        timestep_tensor = torch.tensor([timestep_val], device=device, dtype=torch.bfloat16).expand(B)

        if self.is_lora:
            transformer = (
                self._get_local_transformer(timestep_val)
                if self.expert_parallel
                else self.model._get_expert_for_timestep(timestep_val)
            )
            transformer.disable_adapters()
            try:
                out = transformer(
                    hidden_states=model_input,
                    timestep=timestep_tensor,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
            finally:
                transformer.enable_adapters()
            return out

        if self.expert_parallel:
            ref = self._get_local_ref_transformer(timestep_val)
        else:
            if timestep_val >= self.model.boundary_timestep:
                ref = self.ref_transformers.get("transformer")
            else:
                ref = self.ref_transformers.get("transformer_2")
            if ref is None:
                ref = next(iter(self.ref_transformers.values()))
        return ref(
            hidden_states=model_input,
            timestep=timestep_tensor,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        global_step = self.train_state.step
        start_epoch = self.train_state.epoch
        start_batch_idx = self.train_state.batch_idx
        train_start_time = time.monotonic()
        train_start_step = global_step
        stop_training = False

        for epoch in range(start_epoch, cfg.num_epochs):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)

            enum_start = start_batch_idx if epoch == start_epoch else 0
            last_batch_idx = enum_start - 1
            for batch_idx, batch in enumerate(self.dataloader, start=enum_start):
                last_batch_idx = batch_idx
                if cfg.max_steps is not None and global_step >= cfg.max_steps:
                    stop_training = True
                    break

                metrics = self._grpo_step(batch)

                self._all_reduce_gradients()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm).item()

                lr = cosine_lr(global_step, cfg.warmup_steps, self.total_steps, cfg.learning_rate)
                for opt in self.optimizers:
                    for pg in opt.param_groups:
                        pg["lr"] = lr
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                if self.ema is not None:
                    self.ema.update()

                if self.mfu_monitor is not None:
                    self.mfu_monitor.step()

                global_step += 1

                if self.expert_parallel and global_step % cfg.log_steps == 0 and self.dp_rank == 0:
                    metric_keys = ["policy_loss_low", "kl_loss_low", "reward_mean", "reward_std", "advantage_mean"]
                    if self.expert_group == 1:
                        buf = torch.tensor(
                            [metrics.get(k, 0.0) for k in metric_keys] + [grad_norm],
                            dtype=torch.float32,
                            device=self.device,
                        )
                        dist.send(buf, dst=self._expert_log_peer)
                    elif self.rank == 0:
                        buf = torch.zeros(len(metric_keys) + 1, dtype=torch.float32, device=self.device)
                        dist.recv(buf, src=self._expert_log_peer)
                        self._remote_grpo_ep_metrics = {
                            k: v for k, v in zip(metric_keys + ["grad_norm_low"], buf.tolist(), strict=True)
                        }

                if self.rank == 0 and global_step % cfg.log_steps == 0:
                    mfu = self.mfu_monitor.flush() if self.mfu_monitor is not None else None
                    mfu_str = f"{mfu:.1%}" if mfu is not None else "-"

                    elapsed = time.monotonic() - train_start_time
                    steps_done = global_step - train_start_step
                    if steps_done > 0:
                        secs_per_step = elapsed / steps_done
                        eta_str = format_eta(secs_per_step * (self.total_steps - global_step))
                        speed_str = f"{secs_per_step:.2f}"
                    else:
                        eta_str, speed_str = "?", "?"

                    if hasattr(self.dataloader.dataset, "__len__"):
                        batches = len(self.dataloader)
                    elif cfg.dataset_size is not None:
                        dp = self.dp_size if self._expert_parallel_duplicates_data(cfg) else self.world_size
                        dataset_size = (
                            self._effective_dataset_size
                            if getattr(self, "_effective_dataset_size", None) is not None
                            else cfg.dataset_size
                        )
                        batches = dataset_size // (dp * cfg.batch_size)
                    else:
                        batches = None
                    fractional_epoch = epoch + (batch_idx + 1) / batches if batches else float(epoch)
                    if self.expert_parallel:
                        merged_metrics = dict(metrics)
                        if hasattr(self, "_remote_grpo_ep_metrics"):
                            merged_metrics.update(self._remote_grpo_ep_metrics)
                            del self._remote_grpo_ep_metrics

                        logger.info(
                            "step={}/{} epoch={:.2f} reward={:.4f}+/-{:.4f} lr={:.2e} mfu={} eta={} ({} s/it)",
                            global_step,
                            self.total_steps,
                            fractional_epoch,
                            merged_metrics.get("reward_mean", 0.0),
                            merged_metrics.get("reward_std", 0.0),
                            lr,
                            mfu_str,
                            eta_str,
                            speed_str,
                        )
                        logger.info(
                            "  expert=high: policy_loss={:.4f} kl_loss={:.4f} grad_norm={:.4f}",
                            merged_metrics.get("policy_loss_high", 0.0),
                            merged_metrics.get("kl_loss_high", 0.0),
                            grad_norm,
                        )
                        logger.info(
                            "  expert=low:  policy_loss={:.4f} kl_loss={:.4f} grad_norm={:.4f} advantage_mean={:.4f}",
                            merged_metrics.get("policy_loss_low", 0.0),
                            merged_metrics.get("kl_loss_low", 0.0),
                            merged_metrics.get("grad_norm_low", 0.0),
                            merged_metrics.get("advantage_mean", 0.0),
                        )

                        if self.use_wandb:
                            import wandb

                            ep_log_metrics = {
                                "grpo/policy_loss_high": merged_metrics.get("policy_loss_high", 0.0),
                                "grpo/kl_loss_high": merged_metrics.get("kl_loss_high", 0.0),
                                "grpo/policy_loss_low": merged_metrics.get("policy_loss_low", 0.0),
                                "grpo/kl_loss_low": merged_metrics.get("kl_loss_low", 0.0),
                                "grpo/reward_mean": merged_metrics.get("reward_mean", 0.0),
                                "grpo/reward_std": merged_metrics.get("reward_std", 0.0),
                                "grpo/advantage_mean": merged_metrics.get("advantage_mean", 0.0),
                                "train/lr": lr,
                                "train/grad_norm": grad_norm,
                                "train/grad_norm_high": grad_norm,
                                "train/grad_norm_low": merged_metrics.get("grad_norm_low", 0.0),
                                "train/epoch": fractional_epoch,
                            }
                            if mfu is not None:
                                ep_log_metrics["train/mfu"] = mfu
                            wandb.log(ep_log_metrics, step=global_step)
                    else:
                        logger.info(
                            "step={}/{} epoch={:.2f} policy_loss={:.4f} kl_loss={:.4f} reward={:.4f}+/-{:.4f} "
                            "lr={:.2e} grad_norm={:.4f} mfu={} eta={} ({} s/it)",
                            global_step,
                            self.total_steps,
                            fractional_epoch,
                            metrics["policy_loss"],
                            metrics["kl_loss"],
                            metrics["reward_mean"],
                            metrics["reward_std"],
                            lr,
                            grad_norm,
                            mfu_str,
                            eta_str,
                            speed_str,
                        )

                        if self.use_wandb:
                            import wandb

                            log_metrics = {
                                "grpo/policy_loss": metrics["policy_loss"],
                                "grpo/kl_loss": metrics["kl_loss"],
                                "grpo/reward_mean": metrics["reward_mean"],
                                "grpo/reward_std": metrics["reward_std"],
                                "grpo/advantage_mean": metrics["advantage_mean"],
                                "train/lr": lr,
                                "train/grad_norm": grad_norm,
                                "train/epoch": fractional_epoch,
                            }
                            if mfu is not None:
                                log_metrics["train/mfu"] = mfu
                            wandb.log(log_metrics, step=global_step)

                if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                    self.train_state.step = global_step
                    self.train_state.epoch = epoch
                    self.train_state.batch_idx = batch_idx + 1
                    self._save_checkpoint(output_dir / f"checkpoint-{global_step}")

                if cfg.max_steps is not None and global_step >= cfg.max_steps:
                    stop_training = True
                    break

            if stop_training:
                self.train_state.step = global_step
                self.train_state.epoch = epoch
                self.train_state.batch_idx = max(last_batch_idx + 1, 0)
                if cfg.save_epoch_checkpoints:
                    self._save_checkpoint(output_dir / f"checkpoint-{global_step}")
                logger.info("Reached max_steps={} at step={}.", cfg.max_steps, global_step)
                break

            self.train_state.step = global_step
            self.train_state.epoch = epoch + 1
            self.train_state.batch_idx = 0
            if cfg.save_epoch_checkpoints:
                self._save_checkpoint(output_dir / f"checkpoint-epoch{epoch}")
            logger.info("Epoch {} done.", epoch)

        if self.use_wandb:
            import wandb

            wandb.finish()
        dist.destroy_process_group()


def _compute_ref_mean(
    latent: torch.Tensor,
    ref_output: torch.Tensor,
    sigma: float,
    sigma_prev: float,
    std_dev_t: float,
    dt: float,
) -> torch.Tensor:
    """Compute the transition mean under the reference policy."""
    if sigma > 1e-8:
        return (
            latent * (1.0 + std_dev_t**2 / (2.0 * sigma) * dt)
            + ref_output * (1.0 + std_dev_t**2 * (1.0 - sigma) / (2.0 * sigma)) * dt
        )
    return latent + ref_output * dt
