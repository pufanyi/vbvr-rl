"""Wan2.2 I2V Flow-GRPO Trainer with FSDP2 + DCP.

Implements Flow-GRPO (arXiv:2505.05470): online RL for flow matching models
by converting ODE sampling to SDE for tractable log-probability computation.

Training loop:
  1. Sampling phase: SDE-generate G videos per prompt (no_grad)
  2. Reward phase: compute rewards for generated videos
  3. Advantage phase: per-prompt group normalization
  4. Training phase: policy gradient with clipped importance ratio + KL penalty
"""

import math
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from src.trainer.base_trainer import BaseTrainer
from src.trainer.config import TrainConfig
from src.trainer.utils import cosine_lr, format_eta, shard_transformer, to_model_pixels


class GRPOTrainer(BaseTrainer):
    """Flow-GRPO trainer for Wan2.2 I2V models.

    Supports both LoRA (reference = base model via disable_adapter) and
    full fine-tuning (reference = frozen deepcopy of initial model).
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
        logger.info("Flow-GRPO | G={} T={} expert_parallel={}", cfg.grpo_group_size, cfg.grpo_num_sampling_steps, cfg.expert_parallel)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _get_expert_parallel_sampler_seed(self, cfg: TrainConfig) -> int:
        """Keep high/low expert groups on the same batches for cooperative GRPO."""
        return cfg.seed

    def _pre_fsdp_setup(self, cfg: TrainConfig) -> None:
        """Create frozen reference policy copies for full fine-tuning."""
        self.is_lora = cfg.lora_rank > 0
        self.ref_transformers: dict[str, torch.nn.Module] = {}
        if not self.is_lora:
            logger.info("Full fine-tuning mode: creating frozen reference policy copies")
            if self.model.transformer is not None:
                self.ref_transformers["transformer"] = deepcopy(self.model.transformer).requires_grad_(False).eval()
            if self.model.transformer_2 is not None:
                self.ref_transformers["transformer_2"] = deepcopy(self.model.transformer_2).requires_grad_(False).eval()

    def _setup_fsdp(self, cfg: TrainConfig) -> list[torch.nn.Module]:
        sync_modules = super()._setup_fsdp(cfg)
        # Also shard frozen reference transformers
        for _name, ref in self.ref_transformers.items():
            shard_transformer(ref, self.mesh, self.mp_policy)
        return sync_modules

    def _compute_total_steps(self) -> int:
        # GRPO: each batch = one optimizer step (no gradient_accumulation_steps splitting)
        return self.cfg.num_epochs * len(self.dataloader)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_batch_inputs(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode prompts/images/videos once for a GRPO step."""
        prompt_embeds = self.model.encode_text(batch["prompt"], self.device)
        video = to_model_pixels(batch["videos"][-1], self.device)
        image = to_model_pixels(batch["image"], self.device)
        gt_video_latents = self.model.encode_video(video)
        condition = self.model.prepare_condition(image, video.shape[2], video.shape[-2], video.shape[-1])
        return prompt_embeds, gt_video_latents, condition

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
    # Reward functions
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
        if self.model.transformer is not None and expert_filter in (None, "high"):
            selected = (timesteps >= self.model.boundary_timestep).nonzero(as_tuple=False).flatten()
            if selected.numel() > 0:
                pred = self.model.transformer(
                    hidden_states=model_input.index_select(0, selected),
                    timestep=timesteps.index_select(0, selected),
                    encoder_hidden_states=prompt_embeds.index_select(0, selected),
                    return_dict=False,
                )[0]
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
                pred = self.model.transformer_2(
                    hidden_states=model_input.index_select(0, selected),
                    timestep=timesteps.index_select(0, selected),
                    encoder_hidden_states=prompt_embeds.index_select(0, selected),
                    return_dict=False,
                )[0]
                per_sample_loss = F.mse_loss(
                    pred.float(),
                    target.index_select(0, selected).float(),
                    reduction="none",
                )
                per_sample_loss = per_sample_loss.mean(dim=list(range(1, per_sample_loss.ndim)))
                rewards.index_copy_(0, selected, -per_sample_loss)
        return rewards

    # ------------------------------------------------------------------
    # Advantage computation
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
    # Core GRPO step
    # ------------------------------------------------------------------

    def _grpo_step(self, batch: dict) -> dict[str, float]:
        if self.expert_parallel:
            return self._grpo_step_expert_parallel(batch)
        return self._grpo_step_single_group(batch)

    def _grpo_step_single_group(self, batch: dict) -> dict[str, float]:
        """Original GRPO path when all experts live on every rank."""
        cfg = self.cfg
        G = cfg.grpo_group_size
        S = cfg.grpo_sample_batch_size
        T = cfg.grpo_num_sampling_steps
        device = self.device

        prompt_embeds, gt_video_latents, condition = self._encode_batch_inputs(batch)
        B = gt_video_latents.shape[0]

        for m in [self.model.transformer, self.model.transformer_2]:
            if m is not None:
                m.eval()

        all_chunk_trajs = []
        reward_chunks = []

        for g_start in range(0, G, S):
            cur_S = min(S, G - g_start)
            cond_s = condition.repeat_interleave(cur_S, dim=0)
            pe_s = prompt_embeds.repeat_interleave(cur_S, dim=0)
            gt_s = gt_video_latents.repeat_interleave(cur_S, dim=0)

            traj = self.model.sde_generate(
                condition=cond_s,
                prompt_embeds=pe_s,
                num_sampling_steps=T,
                sde_noise_scale=cfg.grpo_sde_noise_scale,
                sigma_min=cfg.grpo_sde_sigma_min,
                sigma_max=cfg.grpo_sde_sigma_max,
                cfg_scale=cfg.grpo_cfg_scale,
            )

            reward_flat = self._compute_reward_neg_loss(
                traj["latents"][-1],
                gt_s,
                cond_s,
                pe_s,
            )
            reward_chunks.append(reward_flat.view(B, cur_S))

            del traj["noises"]
            traj["latents"] = [x.to("cpu", non_blocking=True) for x in traj["latents"]]
            traj["log_probs"] = [x.to("cpu", non_blocking=True) for x in traj["log_probs"]]
            all_chunk_trajs.append((traj, cur_S))

        rewards = torch.cat(reward_chunks, dim=1)

        if self.world_size > 1:
            all_ranks_rewards = [torch.zeros_like(rewards) for _ in range(self.world_size)]
            dist.all_gather(all_ranks_rewards, rewards)
            gathered_rewards = torch.cat(all_ranks_rewards, dim=0)
            global_mean = gathered_rewards.mean()
            global_std = gathered_rewards.std() + 1e-4
            advantages = ((rewards - global_mean) / global_std).clamp(-cfg.grpo_adv_clip_max, cfg.grpo_adv_clip_max)
        else:
            advantages = self._compute_advantages(rewards)

        for m in [self.model.transformer, self.model.transformer_2]:
            if m is not None:
                m.train()

        total_policy_loss = 0.0
        total_kl_loss = 0.0
        num_chunks = len(all_chunk_trajs)
        total_accum_steps = T * num_chunks

        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

        g_offset = 0
        for chunk_idx, (traj, cur_S) in enumerate(all_chunk_trajs):
            BS = B * cur_S
            cond_s = condition.repeat_interleave(cur_S, dim=0).detach()
            pe_s = prompt_embeds.repeat_interleave(cur_S, dim=0).detach()
            adv_chunk = advantages[:, g_offset : g_offset + cur_S].reshape(BS)
            g_offset += cur_S

            traj["latents"] = [x.to(device, non_blocking=True) for x in traj["latents"]]
            traj["log_probs"] = [x.to(device, non_blocking=True) for x in traj["log_probs"]]

            for t_idx in range(T):
                is_last = chunk_idx == num_chunks - 1 and t_idx == T - 1
                self._set_requires_gradient_sync(is_last)

                sigma = traj["sigmas"][t_idx].item()
                sigma_prev = traj["sigmas"][t_idx + 1].item()
                latent = traj["latents"][t_idx].detach()
                next_latent = traj["latents"][t_idx + 1].detach()
                old_log_prob = traj["log_probs"][t_idx].detach()
                timestep_val = traj["timesteps"][t_idx]

                timestep_tensor = torch.tensor([timestep_val], device=device, dtype=torch.bfloat16).expand(BS)
                transformer = self.model._get_expert_for_timestep(timestep_val)
                model_input = torch.cat([latent, cond_s], dim=1)

                model_output = transformer(
                    hidden_states=model_input,
                    timestep=timestep_tensor,
                    encoder_hidden_states=pe_s,
                    return_dict=False,
                )[0]

                dt = sigma_prev - sigma
                std_dev_t = cfg.grpo_sde_sigma_min + (cfg.grpo_sde_sigma_max - cfg.grpo_sde_sigma_min) * sigma
                noise_scale = std_dev_t * math.sqrt(max(-dt, 0.0))

                if sigma > 1e-8:
                    prev_mean = (
                        latent * (1.0 + std_dev_t**2 / (2.0 * sigma) * dt)
                        + model_output * (1.0 + std_dev_t**2 * (1.0 - sigma) / (2.0 * sigma)) * dt
                    )
                else:
                    prev_mean = latent + model_output * dt

                if noise_scale > 1e-8:
                    new_log_prob = (
                        -((next_latent - prev_mean) ** 2) / (2.0 * noise_scale**2)
                        - math.log(noise_scale)
                        - 0.5 * math.log(2.0 * math.pi)
                    )
                    new_log_prob = new_log_prob.mean(dim=list(range(1, new_log_prob.ndim)))
                else:
                    new_log_prob = torch.zeros(BS, device=device)

                ratio = torch.exp(new_log_prob - old_log_prob)
                unclipped = -adv_chunk * ratio
                clipped = -adv_chunk * ratio.clamp(1.0 - cfg.grpo_clip_range, 1.0 + cfg.grpo_clip_range)
                policy_loss = torch.max(unclipped, clipped).mean()

                kl_loss = torch.tensor(0.0, device=device)
                if cfg.grpo_kl_coeff > 0:
                    with torch.no_grad():
                        ref_output = self._ref_forward(latent, cond_s, pe_s, timestep_val)
                    if noise_scale > 1e-8:
                        ref_mean = _compute_ref_mean(latent, ref_output, sigma, sigma_prev, std_dev_t, dt)
                        kl_loss = ((prev_mean - ref_mean) ** 2).mean(dim=list(range(1, prev_mean.ndim))).mean()
                        kl_loss = kl_loss / (2.0 * noise_scale**2)

                loss = (policy_loss + cfg.grpo_kl_coeff * kl_loss) / total_accum_steps
                loss.backward()

                total_policy_loss += policy_loss.item()
                total_kl_loss += kl_loss.item()

            del traj["latents"], traj["log_probs"]

        return {
            "policy_loss": total_policy_loss / (T * num_chunks),
            "kl_loss": total_kl_loss / (T * num_chunks),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "advantage_mean": advantages.mean().item(),
        }

    def _grpo_step_expert_parallel(self, batch: dict) -> dict[str, float]:
        """Cooperative dual-expert GRPO where high/low experts split the trajectory."""
        cfg = self.cfg
        G = cfg.grpo_group_size
        S = cfg.grpo_sample_batch_size
        T = cfg.grpo_num_sampling_steps
        device = self.device

        prompt_embeds, gt_video_latents, condition = self._encode_batch_inputs(batch)
        B = gt_video_latents.shape[0]
        sigmas, timestep_vals, high_step_count = self._build_sampling_schedule(T, device=device)
        low_step_count = T - high_step_count
        if high_step_count <= 0 or low_step_count <= 0:
            raise RuntimeError("expert_parallel GRPO must have both high and low sampling steps")

        local_start = 0 if self.expert_group == 0 else high_step_count
        local_end = high_step_count if self.expert_group == 0 else T
        local_name = self._local_expert_name()

        for m in [self.model.transformer, self.model.transformer_2]:
            if m is not None:
                m.eval()

        all_chunk_trajs = []
        total_chunk_rewards = [] if self.expert_group == 1 else None

        for g_start in range(0, G, S):
            cur_S = min(S, G - g_start)
            BS = B * cur_S
            cond_s = condition.repeat_interleave(cur_S, dim=0)
            pe_s = prompt_embeds.repeat_interleave(cur_S, dim=0)
            gt_s = gt_video_latents.repeat_interleave(cur_S, dim=0)

            if self.expert_group == 0:
                start_latent = torch.randn_like(gt_s)
                traj = self._run_local_sampling_segment(
                    start_latent,
                    cond_s,
                    pe_s,
                    sigmas,
                    timestep_vals,
                    step_start=local_start,
                    step_end=local_end,
                    cfg=cfg,
                )
                dist.send(traj["latents"][-1].contiguous(), dst=self.peer_rank)
            else:
                boundary_latent = torch.empty_like(gt_s)
                dist.recv(boundary_latent, src=self.peer_rank)
                traj = self._run_local_sampling_segment(
                    boundary_latent,
                    cond_s,
                    pe_s,
                    sigmas,
                    timestep_vals,
                    step_start=local_start,
                    step_end=local_end,
                    cfg=cfg,
                )

            traj["latents"] = [x.to("cpu", non_blocking=True) for x in traj["latents"]]
            traj["log_probs"] = [x.to("cpu", non_blocking=True) for x in traj["log_probs"]]
            traj["sigmas"] = traj["sigmas"].to("cpu")
            all_chunk_trajs.append((traj, cur_S))

        if self.expert_group == 1:
            for traj, cur_S in all_chunk_trajs:
                BS = B * cur_S
                cond_s = condition.repeat_interleave(cur_S, dim=0)
                pe_s = prompt_embeds.repeat_interleave(cur_S, dim=0)
                gt_s = gt_video_latents.repeat_interleave(cur_S, dim=0)

                final_latent = traj["latents"][-1].to(device)
                reward_indices = torch.randint(0, self.model.num_train_timesteps, (BS,), device=device)
                reward_low = self._compute_reward_neg_loss(
                    final_latent,
                    gt_s,
                    cond_s,
                    pe_s,
                    indices=reward_indices,
                    expert_filter="low",
                )
                dist.send(final_latent.contiguous(), dst=self.peer_rank)
                dist.send(reward_indices.contiguous(), dst=self.peer_rank)
                reward_high = torch.empty(BS, device=device, dtype=torch.float32)
                dist.recv(reward_high, src=self.peer_rank)
                total_chunk_rewards.append((reward_low + reward_high).view(B, cur_S))

            rewards = torch.cat(total_chunk_rewards, dim=1)
            rewards_cpu = rewards.cpu()
            all_dp_rewards = [torch.zeros_like(rewards_cpu) for _ in range(self.dp_size)]
            dist.all_gather(all_dp_rewards, rewards_cpu, group=self._cpu_dp_pg)
            gathered_rewards = torch.cat(all_dp_rewards, dim=0)
            global_mean = gathered_rewards.mean()
            global_std = gathered_rewards.std() + 1e-4
            advantages_cpu = ((rewards_cpu - global_mean) / global_std).clamp(
                -cfg.grpo_adv_clip_max, cfg.grpo_adv_clip_max
            )
            dist.send(advantages_cpu.contiguous(), group=self._cpu_world_pg, group_dst=self.peer_rank)
            advantages = advantages_cpu.to(device)
        else:
            for _traj, cur_S in all_chunk_trajs:
                BS = B * cur_S
                cond_s = condition.repeat_interleave(cur_S, dim=0)
                pe_s = prompt_embeds.repeat_interleave(cur_S, dim=0)
                gt_s = gt_video_latents.repeat_interleave(cur_S, dim=0)

                final_latent = torch.empty_like(gt_s)
                reward_indices = torch.empty(BS, device=device, dtype=torch.long)
                dist.recv(final_latent, src=self.peer_rank)
                dist.recv(reward_indices, src=self.peer_rank)
                reward_high = self._compute_reward_neg_loss(
                    final_latent,
                    gt_s,
                    cond_s,
                    pe_s,
                    indices=reward_indices,
                    expert_filter="high",
                )
                dist.send(reward_high.contiguous(), dst=self.peer_rank)

            advantages_cpu = torch.empty((B, G), dtype=torch.float32)
            dist.recv(advantages_cpu, group=self._cpu_world_pg, group_src=self.peer_rank)
            advantages = advantages_cpu.to(device)
            rewards = None

        for m in [self.model.transformer, self.model.transformer_2]:
            if m is not None:
                m.train()

        total_policy_loss = 0.0
        total_kl_loss = 0.0
        local_step_count = 0
        num_chunks = len(all_chunk_trajs)
        total_accum_steps = T * num_chunks

        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

        g_offset = 0
        for chunk_idx, (traj, cur_S) in enumerate(all_chunk_trajs):
            BS = B * cur_S
            cond_s = condition.repeat_interleave(cur_S, dim=0).detach()
            pe_s = prompt_embeds.repeat_interleave(cur_S, dim=0).detach()
            adv_chunk = advantages[:, g_offset : g_offset + cur_S].reshape(BS)
            g_offset += cur_S

            traj["latents"] = [x.to(device) for x in traj["latents"]]
            traj["log_probs"] = [x.to(device) for x in traj["log_probs"]]

            local_T = len(traj["timesteps"])
            for t_idx in range(local_T):
                is_last = chunk_idx == num_chunks - 1 and t_idx == local_T - 1
                self._set_requires_gradient_sync(is_last)

                sigma = traj["sigmas"][t_idx].item()
                sigma_prev = traj["sigmas"][t_idx + 1].item()
                latent = traj["latents"][t_idx].detach()
                next_latent = traj["latents"][t_idx + 1].detach()
                old_log_prob = traj["log_probs"][t_idx].detach()
                timestep_val = traj["timesteps"][t_idx]

                timestep_tensor = torch.tensor([timestep_val], device=device, dtype=torch.bfloat16).expand(BS)
                transformer = self._get_local_transformer(timestep_val)
                model_input = torch.cat([latent, cond_s], dim=1)

                model_output = transformer(
                    hidden_states=model_input,
                    timestep=timestep_tensor,
                    encoder_hidden_states=pe_s,
                    return_dict=False,
                )[0]

                dt = sigma_prev - sigma
                std_dev_t = cfg.grpo_sde_sigma_min + (cfg.grpo_sde_sigma_max - cfg.grpo_sde_sigma_min) * sigma
                noise_scale = std_dev_t * math.sqrt(max(-dt, 0.0))

                if sigma > 1e-8:
                    prev_mean = (
                        latent * (1.0 + std_dev_t**2 / (2.0 * sigma) * dt)
                        + model_output * (1.0 + std_dev_t**2 * (1.0 - sigma) / (2.0 * sigma)) * dt
                    )
                else:
                    prev_mean = latent + model_output * dt

                if noise_scale > 1e-8:
                    new_log_prob = (
                        -((next_latent - prev_mean) ** 2) / (2.0 * noise_scale**2)
                        - math.log(noise_scale)
                        - 0.5 * math.log(2.0 * math.pi)
                    )
                    new_log_prob = new_log_prob.mean(dim=list(range(1, new_log_prob.ndim)))
                else:
                    new_log_prob = torch.zeros(BS, device=device)

                ratio = torch.exp(new_log_prob - old_log_prob)
                unclipped = -adv_chunk * ratio
                clipped = -adv_chunk * ratio.clamp(1.0 - cfg.grpo_clip_range, 1.0 + cfg.grpo_clip_range)
                policy_loss = torch.max(unclipped, clipped).mean()

                kl_loss = torch.tensor(0.0, device=device)
                if cfg.grpo_kl_coeff > 0:
                    with torch.no_grad():
                        ref_output = self._ref_forward(latent, cond_s, pe_s, timestep_val)
                    if noise_scale > 1e-8:
                        ref_mean = _compute_ref_mean(latent, ref_output, sigma, sigma_prev, std_dev_t, dt)
                        kl_loss = ((prev_mean - ref_mean) ** 2).mean(dim=list(range(1, prev_mean.ndim))).mean()
                        kl_loss = kl_loss / (2.0 * noise_scale**2)

                loss = (policy_loss + cfg.grpo_kl_coeff * kl_loss) / total_accum_steps
                loss.backward()

                total_policy_loss += policy_loss.item()
                total_kl_loss += kl_loss.item()
                local_step_count += 1

            del traj["latents"], traj["log_probs"]

        metrics = {
            f"policy_loss_{local_name}": total_policy_loss / max(local_step_count, 1),
            f"kl_loss_{local_name}": total_kl_loss / max(local_step_count, 1),
        }
        if self.expert_group == 1 and rewards is not None:
            metrics["reward_mean"] = rewards.mean().item()
            metrics["reward_std"] = rewards.std().item()
            metrics["advantage_mean"] = advantages.mean().item()
        return metrics

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        global_step = self.train_state.step
        start_epoch = self.train_state.epoch
        train_start_time = time.monotonic()
        train_start_step = global_step

        for epoch in range(start_epoch, cfg.num_epochs):
            self.sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(self.dataloader):
                metrics = self._grpo_step(batch)

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
                        )
                        dist.send(buf, group=self._expert_log_pg, group_dst=0)
                    elif self.rank == 0:
                        buf = torch.zeros(len(metric_keys) + 1, dtype=torch.float32)
                        dist.recv(buf, group=self._expert_log_pg, group_src=1)
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
