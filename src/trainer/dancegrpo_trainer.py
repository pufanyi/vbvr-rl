"""DanceGRPO trainer for Wan I2V.

This is a paper-inspired variant of the existing GRPO trainer that keeps the
current reward implementation while adopting two key DanceGRPO ideas:

1. Samples from the same prompt group share the same initial x_T noise.
2. Only a subset of denoising timesteps are replayed during policy updates.

The current implementation intentionally stays on the standard GRPO execution
path and does not support MoE expert-parallel mode yet.
"""

import math

import torch
import torch.distributed as dist
from loguru import logger

from src.trainer.base_grpo_trainer import BaseGRPOTrainer, _compute_ref_mean, _repeat_meta


class DanceGRPOTrainer(BaseGRPOTrainer):
    """Paper-inspired GRPO variant with shared group noise and timestep selection."""

    def __init__(self, cfg):
        super().__init__(cfg)
        if self.expert_parallel:
            raise NotImplementedError(
                "DanceGRPOTrainer currently supports standard FSDP/HSDP paths only; "
                "expert_parallel remains specific to the custom GRPO trainer."
            )
        logger.info(
            "DanceGRPO | shared_group_init_noise={} timestep_selection_ratio={:.2f}",
            cfg.dancegrpo_share_group_init_noise,
            cfg.dancegrpo_timestep_selection_ratio,
        )

    def _sample_group_initial_latents(self, condition: torch.Tensor, cur_s: int) -> torch.Tensor | None:
        """Sample one x_T per prompt and broadcast it across the whole group."""
        if not self.cfg.dancegrpo_share_group_init_noise:
            return None
        latent_shape = (condition.shape[0], condition.shape[1] - 4, *condition.shape[2:])
        shared = torch.randn(latent_shape, device=condition.device, dtype=torch.bfloat16)
        return shared.repeat_interleave(cur_s, dim=0)

    def _select_training_timesteps(self, num_sampling_steps: int) -> list[int]:
        """Subsample replay timesteps following DanceGRPO's timestep selection idea.

        The selection is generated on rank 0 and broadcast to all ranks so that
        every rank replays the same timesteps.  This is required because
        ``_get_expert_for_timestep`` routes to different FSDP-wrapped modules
        depending on the timestep value — if ranks diverge, the per-parameter
        all-gather in FSDP2 will deadlock.
        """
        keep = max(1, math.ceil(num_sampling_steps * self.cfg.dancegrpo_timestep_selection_ratio))
        if keep >= num_sampling_steps:
            return list(range(num_sampling_steps))
        if self.rank == 0:
            selected = torch.randperm(num_sampling_steps, device=self.device)[:keep].sort().values
        else:
            selected = torch.empty(keep, dtype=torch.long, device=self.device)
        dist.broadcast(selected, src=0)
        return selected.tolist()

    def _grpo_step(self, batch: dict) -> dict[str, float]:
        """DanceGRPO-style GRPO step on the standard single-group path."""
        cfg = self.cfg
        G = cfg.grpo_group_size
        S = cfg.grpo_sample_batch_size
        T = cfg.grpo_num_sampling_steps
        device = self.device

        prompt_embeds, gt_video_latents, condition, meta = self._encode_batch_inputs(batch)
        B = gt_video_latents.shape[0]
        selected_t_idxs = self._select_training_timesteps(T)
        if self.rank == 0:
            logger.info(
                "DanceGRPO replay timesteps: keep {}/{} steps ({})",
                len(selected_t_idxs),
                T,
                selected_t_idxs,
            )

        for m in [self.model.transformer, self.model.transformer_2]:
            if m is not None:
                m.eval()

        all_chunk_trajs = []
        reward_chunks = []

        for g_start in range(0, G, S):
            cur_s = min(S, G - g_start)
            cond_s = condition.repeat_interleave(cur_s, dim=0)
            pe_s = prompt_embeds.repeat_interleave(cur_s, dim=0)
            gt_s = gt_video_latents.repeat_interleave(cur_s, dim=0)
            meta_s = _repeat_meta(meta, cur_s)
            initial_latent = self._sample_group_initial_latents(condition, cur_s)

            traj = self.model.sde_generate(
                condition=cond_s,
                prompt_embeds=pe_s,
                num_sampling_steps=T,
                sde_noise_scale=cfg.grpo_sde_noise_scale,
                sigma_min=cfg.grpo_sde_sigma_min,
                sigma_max=cfg.grpo_sde_sigma_max,
                cfg_scale=cfg.grpo_cfg_scale,
                initial_latent=initial_latent,
            )

            reward_flat = self.reward_fn(
                traj["latents"][-1],
                gt_s,
                cond_s,
                pe_s,
                meta=meta_s,
            )
            reward_chunks.append(reward_flat.view(B, cur_s))

            del traj["noises"]
            traj["latents"] = [x.to("cpu", non_blocking=True) for x in traj["latents"]]
            traj["log_probs"] = [x.to("cpu", non_blocking=True) for x in traj["log_probs"]]
            all_chunk_trajs.append((traj, cur_s))

        rewards = torch.cat(reward_chunks, dim=1)
        advantages = self._compute_advantages(rewards)

        for m in [self.model.transformer, self.model.transformer_2]:
            if m is not None:
                m.train()

        total_policy_loss = 0.0
        total_kl_loss = 0.0
        num_chunks = len(all_chunk_trajs)
        total_accum_steps = len(selected_t_idxs) * num_chunks

        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

        g_offset = 0
        for chunk_idx, (traj, cur_s) in enumerate(all_chunk_trajs):
            BS = B * cur_s
            cond_s = condition.repeat_interleave(cur_s, dim=0).detach()
            pe_s = prompt_embeds.repeat_interleave(cur_s, dim=0).detach()
            adv_chunk = advantages[:, g_offset : g_offset + cur_s].reshape(BS)
            g_offset += cur_s

            traj["latents"] = [x.to(device, non_blocking=True) for x in traj["latents"]]
            traj["log_probs"] = [x.to(device, non_blocking=True) for x in traj["log_probs"]]

            for replay_idx, t_idx in enumerate(selected_t_idxs):
                is_last = chunk_idx == num_chunks - 1 and replay_idx == len(selected_t_idxs) - 1
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
            "policy_loss": total_policy_loss / max(total_accum_steps, 1),
            "kl_loss": total_kl_loss / max(total_accum_steps, 1),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "advantage_mean": advantages.mean().item(),
        }
