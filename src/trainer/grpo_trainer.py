"""Wan2.2 I2V Flow-GRPO Trainer.

Implements the standard Flow-GRPO policy-gradient step with support for
both single-group (all experts on every rank) and expert-parallel
(cooperative dual-expert) modes.

All shared GRPO infrastructure (reference policies, reward, advantage,
training loop) lives in BaseGRPOTrainer.
"""

import math

import torch
import torch.distributed as dist

from src.trainer.base_grpo_trainer import BaseGRPOTrainer, _compute_ref_mean


class GRPOTrainer(BaseGRPOTrainer):
    """Flow-GRPO trainer with single-group and expert-parallel step implementations."""

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
            all_dp_rewards = [torch.zeros_like(rewards) for _ in range(self.dp_size)]
            dist.all_gather(all_dp_rewards, rewards, group=self._dp_pg)
            gathered_rewards = torch.cat(all_dp_rewards, dim=0)
            global_mean = gathered_rewards.mean()
            global_std = gathered_rewards.std() + 1e-4
            advantages = ((rewards - global_mean) / global_std).clamp(
                -cfg.grpo_adv_clip_max, cfg.grpo_adv_clip_max
            )
            dist.send(advantages.contiguous(), dst=self.peer_rank)
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

            advantages = torch.empty((B, G), device=device, dtype=torch.float32)
            dist.recv(advantages, src=self.peer_rank)
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
