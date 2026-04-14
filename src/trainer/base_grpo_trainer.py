"""Base GRPO trainer with shared Flow-GRPO algorithm infrastructure.

Provides reference policy management, SDE sampling schedule, reward/advantage
computation, and the outer training loop.  Concrete subclasses implement
``_grpo_step(batch)`` with their specific policy-gradient update logic.
"""

import math
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from src.trainer.base_rl_trainer import BaseRLTrainer
from src.trainer.config import TrainConfig
from src.trainer.utils import cosine_lr, format_eta, shard_transformer, to_model_pixels


class BaseGRPOTrainer(BaseRLTrainer):
    """Shared Flow-GRPO infrastructure.

    Subclasses must implement ``_grpo_step(batch) -> dict[str, float]``
    which performs the sampling, reward, advantage, and policy-gradient
    update for a single batch.
    """

    def __init__(self, cfg: TrainConfig):
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

    def _get_expert_parallel_sampler_seed(self, cfg: TrainConfig) -> int:
        """Keep high/low expert groups on the same batches for cooperative GRPO."""
        return cfg.seed

    def _pre_fsdp_setup(self, cfg: TrainConfig) -> None:
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

    def _setup_fsdp(self, cfg: TrainConfig) -> list[torch.nn.Module]:
        sync_modules = super()._setup_fsdp(cfg)
        if cfg.fsdp:
            for _name, ref in self.ref_transformers.items():
                shard_transformer(ref, self.mesh, self.mp_policy)
        return sync_modules

    # ------------------------------------------------------------------
    # Abstract: subclass must implement
    # ------------------------------------------------------------------

    def _grpo_step(self, batch: dict) -> dict[str, float]:
        """Execute one GRPO step. Must be implemented by subclass."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Batch encoding
    # ------------------------------------------------------------------

    def _encode_batch_inputs(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode prompts/images/videos once for a GRPO step."""
        prompt_embeds = self.model.encode_text(batch["prompt"], self.device)
        video = to_model_pixels(batch["videos"][-1], self.device)
        image = to_model_pixels(batch["image"], self.device)
        gt_video_latents = self.model.encode_video(video)
        condition = self.model.prepare_condition(image, video.shape[2], video.shape[-2], video.shape[-1])
        return prompt_embeds, gt_video_latents, condition

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
        shift = 5.0
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
                f"got high={high_step_count}, low={low_step_count}. Increase grpo_num_sampling_steps or disable expert_parallel."
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
        cfg: TrainConfig,
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
            timestep_tensor = torch.tensor([timestep_val], device=latent.device, dtype=torch.bfloat16).expand(latent.shape[0])
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
    # Reward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_reward_neg_loss(
        self,
        generated_latents: torch.Tensor,
        gt_video_latents: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        indices: torch.Tensor | None = None,
        expert_filter: str | None = None,
    ) -> torch.Tensor:
        """Reward = -flow_matching_loss against ground truth video."""
        B = gt_video_latents.shape[0]
        device = gt_video_latents.device
        shifted_sigmas, shifted_timesteps, _bsmntw = self.model._get_training_buffers(device)

        if indices is None:
            if expert_filter == "high":
                indices = torch.randint(0, self.model.boundary_idx, (B,), device=device)
            elif expert_filter == "low":
                indices = torch.randint(self.model.boundary_idx, self.model.num_train_timesteps, (B,), device=device)
            else:
                indices = torch.randint(0, self.model.num_train_timesteps, (B,), device=device)

        sigmas = shifted_sigmas.index_select(0, indices).view(B, 1, 1, 1, 1)
        timesteps = shifted_timesteps.index_select(0, indices)

        noise = torch.randn_like(gt_video_latents)
        noisy = sigmas * noise + (1.0 - sigmas) * gt_video_latents
        target = noise - gt_video_latents
        model_input = torch.cat([noisy, condition], dim=1)

        rewards = torch.zeros(B, device=device, dtype=torch.float32)
        # FSDP requires all ranks to participate in forward (all_gather).
        # When selected is empty, run a dummy forward so ranks stay in sync.
        # Without FSDP, we can skip the forward entirely when no samples match.
        need_dummy_forward = self.cfg.fsdp
        if self.model.transformer is not None and expert_filter in (None, "high"):
            selected = (timesteps >= self.model.boundary_timestep).nonzero(as_tuple=False).flatten()
            if selected.numel() > 0:
                sel_input = model_input.index_select(0, selected)
                sel_ts = timesteps.index_select(0, selected)
                sel_pe = prompt_embeds.index_select(0, selected)
            elif need_dummy_forward:
                sel_input = model_input[:1]
                sel_ts = timesteps[:1]
                sel_pe = prompt_embeds[:1]
            else:
                sel_input = None
            if sel_input is not None:
                pred = self.model.transformer(
                    hidden_states=sel_input,
                    timestep=sel_ts,
                    encoder_hidden_states=sel_pe,
                    return_dict=False,
                )[0]
                if selected.numel() > 0:
                    per_sample_loss = F.mse_loss(
                        pred.float(),
                        target.index_select(0, selected).float(),
                        reduction="none",
                    )
                    per_sample_loss = per_sample_loss.mean(dim=list(range(1, per_sample_loss.ndim)))
                    rewards.index_copy_(0, selected, -per_sample_loss)
        if self.model.transformer_2 is not None and expert_filter in (None, "low"):
            selected = (timesteps < self.model.boundary_timestep).nonzero(as_tuple=False).flatten()
            if selected.numel() > 0:
                sel_input = model_input.index_select(0, selected)
                sel_ts = timesteps.index_select(0, selected)
                sel_pe = prompt_embeds.index_select(0, selected)
            elif need_dummy_forward:
                sel_input = model_input[:1]
                sel_ts = timesteps[:1]
                sel_pe = prompt_embeds[:1]
            else:
                sel_input = None
            if sel_input is not None:
                pred = self.model.transformer_2(
                    hidden_states=sel_input,
                    timestep=sel_ts,
                    encoder_hidden_states=sel_pe,
                    return_dict=False,
                )[0]
                if selected.numel() > 0:
                    per_sample_loss = F.mse_loss(
                        pred.float(),
                        target.index_select(0, selected).float(),
                        reduction="none",
                    )
                    per_sample_loss = per_sample_loss.mean(dim=list(range(1, per_sample_loss.ndim)))
                    rewards.index_copy_(0, selected, -per_sample_loss)
        return rewards

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
            transformer = self._get_local_transformer(timestep_val) if self.expert_parallel else self.model._get_expert_for_timestep(timestep_val)
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

        for epoch in range(start_epoch, cfg.num_epochs):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)

            enum_start = start_batch_idx if epoch == start_epoch else 0
            for batch_idx, batch in enumerate(self.dataloader, start=enum_start):
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
                    elapsed = time.monotonic() - train_start_time
                    steps_done = global_step - train_start_step
                    if steps_done > 0:
                        secs_per_step = elapsed / steps_done
                        eta_str = format_eta(secs_per_step * (self.total_steps - global_step))
                        speed_str = f"{secs_per_step:.2f}"
                    else:
                        eta_str, speed_str = "?", "?"

                    fractional_epoch = epoch + (batch_idx + 1) / len(self.dataloader)
                    if self.expert_parallel:
                        merged_metrics = dict(metrics)
                        if hasattr(self, "_remote_grpo_ep_metrics"):
                            merged_metrics.update(self._remote_grpo_ep_metrics)
                            del self._remote_grpo_ep_metrics

                        logger.info(
                            "step={}/{} epoch={:.2f} reward={:.4f}+/-{:.4f} lr={:.2e} eta={} ({} s/it)",
                            global_step,
                            self.total_steps,
                            fractional_epoch,
                            merged_metrics.get("reward_mean", 0.0),
                            merged_metrics.get("reward_std", 0.0),
                            lr,
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

                            wandb.log(
                                {
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
                                },
                                step=global_step,
                            )
                    else:
                        logger.info(
                            "step={}/{} epoch={:.2f} policy_loss={:.4f} kl_loss={:.4f} reward={:.4f}+/-{:.4f} "
                            "lr={:.2e} grad_norm={:.4f} eta={} ({} s/it)",
                            global_step,
                            self.total_steps,
                            fractional_epoch,
                            metrics["policy_loss"],
                            metrics["kl_loss"],
                            metrics["reward_mean"],
                            metrics["reward_std"],
                            lr,
                            grad_norm,
                            eta_str,
                            speed_str,
                        )

                        if self.use_wandb:
                            import wandb

                            wandb.log(
                                {
                                    "grpo/policy_loss": metrics["policy_loss"],
                                    "grpo/kl_loss": metrics["kl_loss"],
                                    "grpo/reward_mean": metrics["reward_mean"],
                                    "grpo/reward_std": metrics["reward_std"],
                                    "grpo/advantage_mean": metrics["advantage_mean"],
                                    "train/lr": lr,
                                    "train/grad_norm": grad_norm,
                                    "train/epoch": fractional_epoch,
                                },
                                step=global_step,
                            )

                if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                    self.train_state.step = global_step
                    self.train_state.epoch = epoch
                    self.train_state.batch_idx = batch_idx + 1
                    self._save_checkpoint(output_dir / f"checkpoint-{global_step}")

            self.train_state.step = global_step
            self.train_state.epoch = epoch + 1
            self.train_state.batch_idx = 0
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
