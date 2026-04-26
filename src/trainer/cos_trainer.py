"""Wan2.2 I2V COS (Chain-of-Step) Trainer with FSDP2 + DCP.

Piecewise flow matching: noise -> search_state -> final_state.
Extends the standard I2V SFT trainer with a two-stage training path
that teaches the model to first develop a coarse search-like structure
(high-noise stage) and then refine it into the final execution video
(low-noise stage).

Each available MoE expert gets a dedicated sigma sample every step,
guaranteeing both experts are always trained (when train_experts='both').

Expert parallel mode (expert_parallel=True) splits GPU groups so each
group only loads and trains one expert, with independent data iteration.
"""

import time
from pathlib import Path

import torch
from loguru import logger

from src.trainer.base_trainer import BaseTrainer
from src.trainer.config import TrainConfig
from src.trainer.flops import MFUMonitor, compute_wan_seq_len, estimate_wan_forward_flops, get_gpu_peak_flops_bf16
from src.trainer.utils import cosine_lr, format_eta, to_model_pixels


class COSTrainer(BaseTrainer):
    def _post_init(self, cfg: TrainConfig) -> None:
        self.mfu_monitor = self._setup_mfu()
        self._cos_debug_accum: dict[str, float] = {}
        self._cos_debug_count: int = 0

    def _setup_mfu(self) -> MFUMonitor | None:
        """Pre-compute FLOPs and create MFU monitor. Returns None if GPU is unrecognized."""
        gpu_peak = get_gpu_peak_flops_bf16()
        if gpu_peak is None:
            return None

        bi = self.model.boundary_idx
        N = self.model.num_train_timesteps
        experts = []
        if self.model.transformer is not None:
            prob = bi / N if self._effective_train_experts == "both" else 1.0
            experts.append((prob, self.model.transformer))
        if self.model.transformer_2 is not None:
            prob = (N - bi) / N if self._effective_train_experts == "both" else 1.0
            experts.append((prob, self.model.transformer_2))

        # Determine seq_len: from dataset config or latent shape
        latent_seq_len: int | None = None
        if hasattr(self.dataset, "_configs"):
            est_cfg = self.dataset._configs[0]
            if est_cfg.fixed_height is not None and est_cfg.fixed_width is not None:
                est_h, est_w = est_cfg.fixed_height, est_cfg.fixed_width
            else:
                est_h = est_w = int(est_cfg.max_area**0.5)
            ref_t = experts[0][1] if experts else None
            if ref_t is not None:
                t_cfg = ref_t.config
                latent_seq_len = compute_wan_seq_len(
                    est_cfg.num_frames,
                    est_h,
                    est_w,
                    patch_size=tuple(t_cfg.patch_size),
                    vae_temporal_factor=self.model.vae_scale_factor_temporal,
                    vae_spatial_factor=self.model.vae_scale_factor_spatial,
                )
        else:
            # Latent dataset: peek at one sample to get (C, T', H', W')
            try:
                sample = next(iter(self.dataset))
                latent = sample["video_latents"]  # (C, T', H', W')
                if isinstance(latent, list):
                    latent = latent[-1]
                _, t_lat, h_lat, w_lat = latent.shape
                ref_t = experts[0][1] if experts else None
                if ref_t is not None:
                    p_t, p_h, p_w = ref_t.config.patch_size
                    latent_seq_len = (t_lat // p_t) * (h_lat // p_h) * (w_lat // p_w)
            except Exception as e:
                logger.info("MFU monitor: skipped (cannot peek latent sample: {})", e)

        if latent_seq_len is None:
            logger.info("MFU monitor: skipped (no resolution info available)")
            return None

        weighted_fwd_flops = 0.0
        for prob, t in experts:
            t_cfg = t.config
            fwd = estimate_wan_forward_flops(
                num_layers=t_cfg.num_layers,
                num_heads=t_cfg.num_attention_heads,
                head_dim=t_cfg.attention_head_dim,
                ffn_dim=t_cfg.ffn_dim,
                seq_len=latent_seq_len,
            )
            weighted_fwd_flops += prob * fwd

        # With dual expert: 2 forward passes per step (one per expert)
        n_expert_passes = len(experts)
        flops_per_step = (
            3 * weighted_fwd_flops * n_expert_passes * self.cfg.batch_size * self.cfg.gradient_accumulation_steps
        )

        logger.info(
            "MFU monitor: seq_len={}, forward={:.2e} FLOPs/sample, step={:.2e} FLOPs, GPU={} ({:.0f} TFLOPS bf16)",
            latent_seq_len,
            weighted_fwd_flops,
            flops_per_step,
            torch.cuda.get_device_name(0),
            gpu_peak / 1e12,
        )
        return MFUMonitor(flops_per_step, gpu_peak)

    # ------------------------------------------------------------------
    # Training loop
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

        logger.info(
            "COS training: tau_sigma={}, boundary_noise_std={}, expert_parallel={}",
            cfg.cos_tau_sigma,
            cfg.cos_boundary_noise_std,
            cfg.expert_parallel,
        )

        for epoch in range(start_epoch, cfg.num_epochs):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            for opt in self.optimizers:
                opt.zero_grad(set_to_none=True)

            enum_start = start_batch_idx if epoch == start_epoch else 0
            for batch_idx, batch in enumerate(self.dataloader, start=enum_start):
                is_last_micro_step = (batch_idx + 1) % cfg.gradient_accumulation_steps == 0
                self._set_requires_gradient_sync(is_last_micro_step)
                loss, debug = self._train_step(batch)
                scaled_loss = loss / cfg.gradient_accumulation_steps
                scaled_loss.backward()

                for k, v in debug.items():
                    self._cos_debug_accum[k] = self._cos_debug_accum.get(k, 0.0) + v
                self._cos_debug_count += 1

                if is_last_micro_step:
                    self._all_reduce_gradients()
                    self._last_grad_norm = torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm).item()

                    lr = cosine_lr(global_step, cfg.warmup_steps, self.total_steps, cfg.learning_rate)
                    for opt in self.optimizers:
                        base = getattr(opt, "_base_lr", cfg.learning_rate)
                        opt_lr = cosine_lr(global_step, cfg.warmup_steps, self.total_steps, base)
                        for pg in opt.param_groups:
                            pg["lr"] = opt_lr
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    if self.ema is not None:
                        self.ema.update()
                    global_step += 1
                    if self.mfu_monitor is not None:
                        self.mfu_monitor.step()

                    # -- Expert-parallel metric exchange --
                    # Both group leaders participate: low sends its stats to rank 0.
                    if self.expert_parallel and global_step % cfg.log_steps == 0 and self.dp_rank == 0:
                        import torch.distributed as dist

                        _ep_keys = ["loss", "target_norm", "sigma_mean", "n_cos_high", "n_cos_low"]
                        if self.expert_group == 1:
                            # Low group leader: send averaged metrics to rank 0
                            cnt = max(self._cos_debug_count, 1)
                            avg_local = {k: v / cnt for k, v in self._cos_debug_accum.items()}
                            buf = torch.tensor(
                                [avg_local.get(f"{k}_low", 0.0) for k in _ep_keys],
                                dtype=torch.float32,
                                device=self.device,
                            )
                            dist.send(buf, dst=self._expert_log_peer)
                            self._cos_debug_accum.clear()
                            self._cos_debug_count = 0
                        elif self.expert_group == 0:
                            buf = torch.zeros(len(_ep_keys), dtype=torch.float32, device=self.device)
                            dist.recv(buf, src=self._expert_log_peer)
                            self._remote_expert_avg = {
                                f"{k}_low": v for k, v in zip(_ep_keys, buf.tolist(), strict=True)
                            }

                    if self.rank == 0 and global_step % cfg.log_steps == 0:
                        mfu = self.mfu_monitor.flush() if self.mfu_monitor is not None else None
                        mfu_str = f"{mfu:.1%}" if mfu is not None else "-"

                        elapsed = time.monotonic() - train_start_time
                        steps_done = global_step - train_start_step
                        if steps_done > 0:
                            secs_per_step = elapsed / steps_done
                            eta_secs = secs_per_step * (self.total_steps - global_step)
                            eta_str = format_eta(eta_secs)
                            s_it_str = f"{secs_per_step:.1f}"
                        else:
                            eta_str = "?"
                            s_it_str = "?"

                        cnt = max(self._cos_debug_count, 1)
                        avg = {k: v / cnt for k, v in self._cos_debug_accum.items()}

                        # Merge remote expert metrics if available
                        if self.expert_parallel and hasattr(self, "_remote_expert_avg"):
                            avg.update(self._remote_expert_avg)
                            del self._remote_expert_avg

                        fractional_epoch = epoch + (batch_idx + 1) / len(self.dataloader)
                        logger.info(
                            "step={}/{} epoch={:.2f} loss={:.4f} lr={:.2e} grad_norm={:.4f} mfu={} eta={} ({} s/it)",
                            global_step,
                            self.total_steps,
                            fractional_epoch,
                            loss.item(),
                            lr,
                            self._last_grad_norm,
                            mfu_str,
                            eta_str,
                            s_it_str,
                        )

                        # Per-expert debug lines
                        for en in ["high", "low"]:
                            lk = f"loss_{en}"
                            if lk in avg:
                                logger.info(
                                    "  expert={}: loss={:.4f} tnorm={:.2f} sigma={:.3f} cos_h={:.0f} cos_l={:.0f}",
                                    en,
                                    avg[lk],
                                    avg.get(f"target_norm_{en}", 0),
                                    avg.get(f"sigma_mean_{en}", 0),
                                    avg.get(f"n_cos_high_{en}", 0),
                                    avg.get(f"n_cos_low_{en}", 0),
                                )

                        if self.use_wandb:
                            import wandb

                            log_metrics = {
                                "train/loss": loss.item(),
                                "train/lr": lr,
                                "train/epoch": fractional_epoch,
                                "train/grad_norm": self._last_grad_norm,
                            }
                            for en in ["high", "low"]:
                                lk = f"loss_{en}"
                                if lk in avg:
                                    log_metrics[f"cos/loss_{en}"] = avg[lk]
                                    log_metrics[f"cos/target_norm_{en}"] = avg.get(f"target_norm_{en}", 0)
                                    log_metrics[f"cos/sigma_mean_{en}"] = avg.get(f"sigma_mean_{en}", 0)
                            if mfu is not None:
                                log_metrics["train/mfu"] = mfu
                            wandb.log(log_metrics, step=global_step)

                        self._cos_debug_accum.clear()
                        self._cos_debug_count = 0

                    if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                        self.train_state.step = global_step
                        self.train_state.epoch = epoch
                        self.train_state.batch_idx = batch_idx + 1
                        self._save_checkpoint(output_dir / f"checkpoint-{global_step}")

            # End-of-epoch save
            self.train_state.step = global_step
            self.train_state.epoch = epoch + 1
            self.train_state.batch_idx = 0
            self._save_checkpoint(output_dir / f"checkpoint-epoch{epoch}")
            logger.info("Epoch {} done.", epoch)

        if self.use_wandb:
            import wandb

            wandb.finish()
        import torch.distributed as dist

        dist.destroy_process_group()

    def _train_step(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """Single forward pass: encode frozen inputs, compute COS piecewise loss.

        Supports two modes:
        - Raw data: batch["videos"] is a list of N batched tensors (one per step in
          the chain). videos[0..N-2] = intermediate waypoints; videos[N-1] = final target.
        - Precomputed: batch has "prompt_embeds", "condition", and "video_latents"
          where video_latents is a list of latent tensors for the full chain.
        """
        cfg = self.cfg

        if "prompt_embeds" in batch:
            prompt_embeds = batch["prompt_embeds"].to(self.device)
            condition = batch["condition"].to(self.device)
            raw_video_latents = batch["video_latents"]
            if not isinstance(raw_video_latents, list):
                raise ValueError(
                    "COS latent training requires all chain latents. Re-run precompute with --encode_all_videos."
                )
            video_latents = [latent.to(self.device) for latent in raw_video_latents]
        else:
            prompt_embeds = self.model.encode_text(batch["prompt"], self.device)
            image = to_model_pixels(batch["image"], self.device)

            videos = batch["videos"]
            final_video = to_model_pixels(videos[-1], self.device)
            condition = self.model.prepare_condition(
                image, final_video.shape[2], final_video.shape[-2], final_video.shape[-1]
            )
            video_latents = [self.model.encode_video(to_model_pixels(v, self.device)) for v in videos]

        return self.model.compute_cos_loss(
            video_latents=video_latents,
            condition=condition,
            prompt_embeds=prompt_embeds,
            taus=cfg.cos_tau_sigma,
            boundary_noise_std=cfg.cos_boundary_noise_std,
            path_type=cfg.cos_path_type,
            smooth_blend_delta=cfg.cos_smooth_blend_delta,
            prompt_dropout=cfg.prompt_dropout,
        )
