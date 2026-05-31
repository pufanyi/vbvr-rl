"""Wan2.2 I2V trainer with on-policy correction loss.

This trainer is a sibling of :class:`I2VTrainer` — not an RL trainer — because
the loss is supervised MSE, not policy-gradient. What sets it apart is that
each training step first runs a short EMA-teacher SDE/ODE rollout to obtain
x̂ = rollout(ε, c), then trains on

    x_σ     = σ · ε + (1 - σ) · x̂
    target  = (x_σ - x_GT) / σ          # re-aim velocity at real GT

mixed with the standard FM loss at a configurable weight λ.

Why not inherit from I2VTrainer? The sequential teacher rollout keeps the
whole batch at one σ per step, so Expert Parallel would idle the other
expert's GPUs. This trainer therefore explicitly forbids EP, and recomputes
MFU to account for the K extra (no-grad) forward passes per optimizer step.
"""

import contextlib
import time
from pathlib import Path

import torch
from loguru import logger

from src.trainer.base_trainer import BaseTrainer
from src.trainer.config import CorrectionConfig
from src.trainer.flops import MFUMonitor, compute_wan_seq_len, estimate_wan_forward_flops, get_gpu_peak_flops_bf16
from src.trainer.utils import cosine_lr, format_eta, to_model_pixels


class I2VCorrectionTrainer(BaseTrainer):
    """SFT + on-policy correction for Wan2.2 I2V."""

    cfg: CorrectionConfig

    def __init__(self, cfg: CorrectionConfig):
        if cfg.expert_parallel:
            raise NotImplementedError(
                "I2VCorrectionTrainer does not support expert_parallel=True: the "
                "teacher rollout is sequential and keeps the whole batch at one σ "
                "per step, which would idle half the GPUs. Set expert_parallel=False."
            )
        super().__init__(cfg)

    def _post_init(self, cfg: CorrectionConfig) -> None:
        if self.ema is None:
            logger.warning(
                "I2VCorrectionTrainer: EMA is disabled (ema_decay<=0). The teacher "
                "rollout will use the live student weights, which creates a drifting "
                "target. Strongly recommend setting ema_decay >= 0.999."
            )
        self._correction_micro_step = 0
        self.mfu_monitor = self._setup_mfu()

    # ------------------------------------------------------------------
    # MFU setup — mirrors I2VTrainer but accounts for K teacher-rollout forwards
    # ------------------------------------------------------------------

    def _setup_mfu(self) -> MFUMonitor | None:
        gpu_peak = get_gpu_peak_flops_bf16()
        if gpu_peak is None:
            return None

        bi = self.model.boundary_idx
        N = self.model.num_train_timesteps
        experts: list[tuple[float, object]] = []
        # A single-transformer model (e.g. 5B TI2V) routes every sample through the
        # one transformer; only a true dual-expert model (A14B) splits samples between
        # experts by timestep, so apply the boundary-fraction weighting only then.
        both = self._effective_train_experts == "both"
        has_high = self.model.transformer is not None
        has_low = self.model.transformer_2 is not None
        if has_high:
            prob = (bi / N) if (both and has_low) else 1.0
            experts.append((prob, self.model.transformer))
        if has_low:
            prob = ((N - bi) / N) if (both and has_high) else 1.0
            experts.append((prob, self.model.transformer_2))

        # Resolve latent seq_len (same logic as I2VTrainer._setup_mfu)
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
            try:
                sample = next(iter(self.dataset))
                latent = sample["video_latents"]
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

        weighted_fwd = 0.0
        for prob, t in experts:
            t_cfg = t.config
            fwd = estimate_wan_forward_flops(
                num_layers=t_cfg.num_layers,
                num_heads=t_cfg.num_attention_heads,
                head_dim=t_cfg.attention_head_dim,
                ffn_dim=t_cfg.ffn_dim,
                seq_len=latent_seq_len,
            )
            weighted_fwd += prob * fwd

        cfg = self.cfg
        # Per optimizer step:
        #   - standard FM loss: 1 forward + 1 backward ≈ 3 forward
        #   - correction loss (fires every N micro-steps): K forward (no-grad) + 1 forward + 1 backward
        #     amortized over N micro-steps → (K + 3) / N extra forwards per micro-step
        corr_extra_per_micro = (cfg.correction_num_teacher_steps + 3) / max(cfg.correction_every_n_steps, 1)
        total_fwd_per_micro = 3 + corr_extra_per_micro
        flops_per_step = total_fwd_per_micro * weighted_fwd * cfg.batch_size * cfg.gradient_accumulation_steps

        logger.info(
            "MFU monitor: seq_len={}, fwd={:.2e} FLOPs/sample, step={:.2e} FLOPs "
            "(fm=3× + correction≈{:.2f}×), GPU={} ({:.0f} TFLOPS bf16)",
            latent_seq_len,
            weighted_fwd,
            flops_per_step,
            corr_extra_per_micro,
            torch.cuda.get_device_name(0),
            gpu_peak / 1e12,
        )
        return MFUMonitor(flops_per_step, gpu_peak)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _unpack_batch(self, batch: dict):
        if "prompt_embeds" in batch:
            prompt_embeds = batch["prompt_embeds"].to(self.device)
            video_latents = batch["video_latents"].to(self.device)
            condition = batch["condition"].to(self.device)
        else:
            prompt_embeds = self.model.encode_text(batch["prompt"], self.device)
            video = to_model_pixels(batch["videos"][-1], self.device)
            image = to_model_pixels(batch["image"], self.device)
            video_latents = self.model.encode_video(video)
            condition = self.model.prepare_condition(image, video.shape[2], video.shape[-2], video.shape[-1])
        return video_latents, condition, prompt_embeds

    def _train_step(self, batch: dict, is_last_micro_step: bool) -> torch.Tensor:
        """Forward + backward for one micro-batch.

        FM and correction are backward'd separately so their activation graphs
        don't live simultaneously — peak memory drops from (FM + corr) down to
        max(FM, corr). Gradient sync is suppressed during FM's backward when a
        correction backward will follow, so FSDP still syncs exactly once per
        micro-step (on whichever backward is last).

        Returns a detached scalar total loss for logging; backward is done
        internally, so the caller should NOT call ``.backward()`` again.
        """
        cfg = self.cfg
        accum = cfg.gradient_accumulation_steps
        video_latents, condition, prompt_embeds = self._unpack_batch(batch)

        fire_correction = (self._correction_micro_step % cfg.correction_every_n_steps) == 0
        self._correction_micro_step += 1
        do_correction = fire_correction and cfg.correction_weight > 0.0

        # ---- FM loss (forward + backward) ----
        # Only the last backward of this micro-step syncs gradients.
        self._set_requires_gradient_sync(is_last_micro_step and not do_correction)
        loss_fm = self.model.compute_loss(
            video_latents,
            condition,
            prompt_embeds,
            prompt_dropout=cfg.prompt_dropout,
        )
        (loss_fm / accum).backward()
        loss_fm_val = loss_fm.detach()
        del loss_fm

        # ---- Correction loss (forward + backward) ----
        loss_corr_val: torch.Tensor | None = None
        effective_lambda = 0.0
        if do_correction:
            effective_lambda = cfg.correction_weight * cfg.correction_every_n_steps
            self._set_requires_gradient_sync(is_last_micro_step)
            teacher_ctx = self.ema.swap_to_shadow() if self.ema is not None else contextlib.nullcontext()
            with teacher_ctx:
                loss_corr = self.model.compute_correction_loss(
                    video_latents=video_latents,
                    condition=condition,
                    prompt_embeds=prompt_embeds,
                    num_teacher_steps=cfg.correction_num_teacher_steps,
                    use_sde=cfg.correction_use_sde,
                    sde_sigma_max=cfg.correction_sde_sigma_max,
                    sigma_clip=(cfg.correction_sigma_lo, cfg.correction_sigma_hi),
                    cfg_scale=cfg.correction_cfg_scale,
                )
            ((effective_lambda * loss_corr) / accum).backward()
            loss_corr_val = loss_corr.detach()
            del loss_corr

        if self.rank == 0:
            self._last_fm_loss = loss_fm_val.item()
            self._last_corr_loss = loss_corr_val.item() if loss_corr_val is not None else None

        total = loss_fm_val
        if loss_corr_val is not None:
            total = total + effective_lambda * loss_corr_val
        return total

    # ------------------------------------------------------------------
    # Train loop (structurally identical to I2VTrainer.train — we can't
    # just inherit because we are a BaseTrainer sibling, and we want to
    # log fm / correction components separately)
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

        self._last_fm_loss: float | None = None
        self._last_corr_loss: float | None = None

        for epoch in range(start_epoch, cfg.num_epochs):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            for opt in self.optimizers:
                opt.zero_grad(set_to_none=True)

            enum_start = start_batch_idx if epoch == start_epoch else 0
            for batch_idx, batch in enumerate(self.dataloader, start=enum_start):
                is_last_micro_step = (batch_idx + 1) % cfg.gradient_accumulation_steps == 0
                # _train_step runs forward + backward for both the FM and the
                # correction loss internally (split backward to cap peak memory)
                # and manages gradient sync. Returns a detached total loss.
                loss = self._train_step(batch, is_last_micro_step)

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

                        if hasattr(self.dataloader.dataset, "__len__"):
                            batches = len(self.dataloader)
                        elif cfg.dataset_size is not None:
                            dp = self.dp_size if self._expert_parallel_duplicates_data(cfg) else self.world_size
                            batches = cfg.dataset_size // (dp * cfg.batch_size)
                        else:
                            batches = None
                        fractional_epoch = epoch + (batch_idx + 1) / batches if batches else float(epoch)

                        corr_str = f"{self._last_corr_loss:.4f}" if self._last_corr_loss is not None else "-"
                        logger.info(
                            "step={}/{} epoch={:.2f} loss={:.4f} fm={:.4f} corr={} "
                            "lr={:.2e} grad_norm={:.4f} mfu={} eta={} ({} s/it)",
                            global_step,
                            self.total_steps,
                            fractional_epoch,
                            loss.item(),
                            self._last_fm_loss if self._last_fm_loss is not None else loss.item(),
                            corr_str,
                            lr,
                            self._last_grad_norm,
                            mfu_str,
                            eta_str,
                            s_it_str,
                        )

                        if self.use_wandb:
                            import wandb

                            log_metrics = {
                                "train/loss": loss.item(),
                                "train/lr": lr,
                                "train/epoch": fractional_epoch,
                                "train/grad_norm": self._last_grad_norm,
                            }
                            if self._last_fm_loss is not None:
                                log_metrics["train/loss_fm"] = self._last_fm_loss
                            if self._last_corr_loss is not None:
                                log_metrics["train/loss_correction"] = self._last_corr_loss
                            if mfu is not None:
                                log_metrics["train/mfu"] = mfu
                            wandb.log(log_metrics, step=global_step)

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
        import torch.distributed as dist

        dist.destroy_process_group()
