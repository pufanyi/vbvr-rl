"""DanceGRPO trainer for Wan I2V.

This is a paper-inspired variant of the existing GRPO trainer that keeps the
current reward implementation while adopting two key DanceGRPO ideas:

1. Samples from the same prompt group share the same initial x_T noise.
2. Only a subset of denoising timesteps are replayed during policy updates.

The current implementation intentionally stays on the standard GRPO execution
path and does not support MoE expert-parallel mode yet.
"""

import torch
import torch.distributed as dist
from loguru import logger

from src.trainer.base_grpo_trainer import BaseGRPOTrainer, _repeat_meta


class DanceGRPOTrainer(BaseGRPOTrainer):
    """Paper-inspired GRPO variant with shared group noise and timestep selection."""

    def __init__(self, cfg):
        super().__init__(cfg)
        if self.expert_parallel:
            raise NotImplementedError(
                "DanceGRPOTrainer currently supports standard FSDP/HSDP paths only; "
                "expert_parallel remains specific to the custom GRPO trainer."
            )
        if cfg.grpo_group_size < 2:
            raise ValueError("DanceGRPO requires grpo_group_size >= 2 for group-relative advantages")
        if cfg.grpo_num_sampling_steps < 2:
            raise ValueError("DanceGRPO requires grpo_num_sampling_steps >= 2")
        if cfg.grpo_sde_noise_scale <= 0:
            raise ValueError("DanceGRPO requires grpo_sde_noise_scale > 0")
        logger.info(
            "DanceGRPO | shared_group_init_noise={} timestep_selection_ratio={:.2f}",
            cfg.dancegrpo_share_group_init_noise,
            cfg.dancegrpo_timestep_selection_ratio,
        )

    def _sample_group_initial_latents(self, condition: torch.Tensor) -> torch.Tensor | None:
        """Sample one x_T per prompt for reuse across the whole group."""
        if not self.cfg.dancegrpo_share_group_init_noise:
            return None
        latent_shape = (condition.shape[0], condition.shape[1] - 4, *condition.shape[2:])
        return torch.randn(latent_shape, device=condition.device, dtype=torch.bfloat16)

    def _select_training_timesteps(self, num_sampling_steps: int) -> list[int]:
        """Subsample replay timesteps following DanceGRPO's timestep selection idea.

        The selection is generated on rank 0 and broadcast to all ranks so that
        every rank replays the same timesteps.  This is required because
        ``_get_expert_for_timestep`` routes to different FSDP-wrapped modules
        depending on the timestep value — if ranks diverge, the per-parameter
        all-gather in FSDP2 will deadlock.
        """
        candidate_steps = max(1, num_sampling_steps - 1)
        keep = max(1, int(candidate_steps * self.cfg.dancegrpo_timestep_selection_ratio))
        if keep >= candidate_steps:
            return list(range(candidate_steps))
        if self.rank == 0:
            selected = torch.randperm(candidate_steps, device=self.device)[:keep].sort().values
        else:
            selected = torch.empty(keep, dtype=torch.long, device=self.device)
        dist.broadcast(selected, src=0)
        return selected.tolist()

    def _policy_forward(
        self,
        transformer: torch.nn.Module,
        latent: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timestep_val: float,
    ) -> torch.Tensor:
        """Forward the trainable policy with optional CFG, matching rollout."""
        B = latent.shape[0]
        device = latent.device
        timestep_tensor = torch.tensor([timestep_val], device=device, dtype=torch.bfloat16).expand(B)
        model_input = torch.cat([latent, condition], dim=1)
        model_output = transformer(
            hidden_states=model_input,
            timestep=timestep_tensor,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        if self.cfg.grpo_cfg_scale <= 1.0:
            return model_output

        uncond_output = transformer(
            hidden_states=model_input,
            timestep=timestep_tensor,
            encoder_hidden_states=torch.zeros_like(prompt_embeds),
            return_dict=False,
        )[0]
        return uncond_output.to(torch.float32) + self.cfg.grpo_cfg_scale * (
            model_output.to(torch.float32) - uncond_output.to(torch.float32)
        )

    def _ref_forward_cfg(
        self,
        latent: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timestep_val: float,
    ) -> torch.Tensor:
        """Forward the reference policy with the same CFG semantics as rollout."""
        ref_output = self._ref_forward(latent, condition, prompt_embeds, timestep_val)
        if self.cfg.grpo_cfg_scale <= 1.0:
            return ref_output

        uncond_ref_output = self._ref_forward(latent, condition, torch.zeros_like(prompt_embeds), timestep_val)
        return uncond_ref_output.to(torch.float32) + self.cfg.grpo_cfg_scale * (
            ref_output.to(torch.float32) - uncond_ref_output.to(torch.float32)
        )

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
        shared_initial_latent = self._sample_group_initial_latents(condition)

        for g_start in range(0, G, S):
            cur_s = min(S, G - g_start)
            cond_s = condition.repeat_interleave(cur_s, dim=0)
            pe_s = prompt_embeds.repeat_interleave(cur_s, dim=0)
            gt_s = gt_video_latents.repeat_interleave(cur_s, dim=0)
            meta_s = _repeat_meta(meta, cur_s)
            initial_latent = (
                shared_initial_latent.repeat_interleave(cur_s, dim=0)
                if shared_initial_latent is not None
                else None
            )

            traj = self.model.sde_generate(
                condition=cond_s,
                prompt_embeds=pe_s,
                num_sampling_steps=T,
                sde_noise_scale=cfg.grpo_sde_noise_scale,
                sigma_min=cfg.grpo_sde_sigma_min,
                sigma_max=cfg.grpo_sde_sigma_max,
                cfg_scale=cfg.grpo_cfg_scale,
                initial_latent=initial_latent,
                sde_formula="dancegrpo",
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
        metric_normalizer = max(len(selected_t_idxs) * G, 1)

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

                transformer = self.model._get_expert_for_timestep(timestep_val)
                model_output = self._policy_forward(transformer, latent, cond_s, pe_s, timestep_val)
                prev_mean, noise_scale = self.model._dancegrpo_transition_mean(
                    sample=latent,
                    model_output=model_output,
                    sigma=sigma,
                    sigma_prev=sigma_prev,
                    eta=cfg.grpo_sde_noise_scale,
                )
                new_log_prob = self.model._gaussian_transition_log_prob(next_latent, prev_mean, noise_scale)

                ratio = torch.exp(new_log_prob - old_log_prob)
                unclipped = -adv_chunk * ratio
                clipped = -adv_chunk * ratio.clamp(1.0 - cfg.grpo_clip_range, 1.0 + cfg.grpo_clip_range)
                policy_loss = torch.max(unclipped, clipped).mean()

                kl_loss = torch.tensor(0.0, device=device)
                if cfg.grpo_kl_coeff > 0:
                    with torch.no_grad():
                        ref_output = self._ref_forward_cfg(latent, cond_s, pe_s, timestep_val)
                    if noise_scale > 1e-8:
                        ref_mean, _ = self.model._dancegrpo_transition_mean(
                            sample=latent,
                            model_output=ref_output,
                            sigma=sigma,
                            sigma_prev=sigma_prev,
                            eta=cfg.grpo_sde_noise_scale,
                        )
                        kl_loss = ((prev_mean - ref_mean) ** 2).mean(dim=list(range(1, prev_mean.ndim))).mean()
                        kl_loss = kl_loss / (2.0 * noise_scale**2)

                loss = (policy_loss + cfg.grpo_kl_coeff * kl_loss) * (cur_s / metric_normalizer)
                loss.backward()

                total_policy_loss += policy_loss.item() * cur_s
                total_kl_loss += kl_loss.item() * cur_s

            del traj["latents"], traj["log_probs"]

        return {
            "policy_loss": total_policy_loss / metric_normalizer,
            "kl_loss": total_kl_loss / metric_normalizer,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "advantage_mean": advantages.mean().item(),
        }
