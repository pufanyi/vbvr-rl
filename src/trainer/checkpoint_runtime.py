"""Shared checkpoint save / load / discovery logic for SFT and RL trainers.

This is a mixin, not a standalone utility.  It relies on attributes that the
host trainer (BaseTrainer or BaseRLTrainer) is expected to provide — see the
docstring on :class:`CheckpointRuntimeMixin` for the full contract.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from src.trainer.checkpoint import (
    _get_checkpoint_fqns,
    _should_plain_to_lora_remap,
    get_checkpoint_optimizer_keys,
    load_dcp_into_pipeline,
)


class CheckpointRuntimeMixin:
    """Checkpoint lifecycle methods shared between SFT and RL trainers.

    The host trainer must provide the following attributes/methods::

        self.cfg              — TrainConfig
        self.model            — WanI2VForTraining (with .transformer, .transformer_2, .text_encoder, .lora_config)
        self.ema              — EMA | None
        self.train_state      — TrainState
        self.dataloader       — StatefulDataLoader
        self.rank             — int (global rank)
        self.world_size       — int
        self.expert_parallel  — bool
        self.expert_group     — int (-1 when non-EP)
        self.dp_rank          — int
        self.dp_size          — int
        self._dp_pg           — ProcessGroup | None
        self._reset_on_load   — bool
        self._barrier()       — sync all ranks
        self._build_optimizers(cfg) — returns (params, optimizers, opt_te, opt_1, opt_2, fb_te, fb_1, fb_2)
        self._compute_total_steps() — returns int
        self._checkpoint_rank()     — int
        self._checkpoint_process_group_size() — int
        self._optimizer_checkpoint_entries() — list[(name, model, optimizer)]
        self._save_optimizer_shards(path)
        self._load_optimizer_shards(path) -> list[str]
    """

    # ------------------------------------------------------------------
    # Checkpoint discovery
    # ------------------------------------------------------------------

    def _find_latest_checkpoint(self) -> str | None:
        """Scan *output_dir* for the most recent valid checkpoint.

        Detects **both** flat and expert-parallel layouts regardless of the
        current trainer's ``expert_parallel`` setting so that cross-mode
        auto-resume (e.g. output_dir has EP checkpoints but the current run
        is non-EP) can still discover and load them.
        """
        output_dir = Path(self.cfg.output_dir)
        if not output_dir.exists():
            return None

        candidates: list[tuple[int, Path]] = []
        for d in output_dir.iterdir():
            if not d.is_dir():
                continue

            # Accept flat checkpoints (.metadata at root) …
            is_ckpt = (d / ".metadata").exists()
            # … and EP checkpoints (high/ or low/ subdirs with .metadata).
            if not is_ckpt:
                is_ckpt = any(
                    (d / expert / ".metadata").exists()
                    for expert in ("high", "low")
                )
            if not is_ckpt:
                continue

            name = d.name
            if name.startswith("checkpoint-epoch"):
                try:
                    int(name.removeprefix("checkpoint-epoch"))
                    candidates.append((int(d.stat().st_mtime_ns), d))
                except ValueError:
                    continue
            elif name.startswith("checkpoint-"):
                try:
                    int(name.removeprefix("checkpoint-"))
                    candidates.append((int(d.stat().st_mtime_ns), d))
                except ValueError:
                    continue

        if not candidates:
            return None
        candidates.sort()
        latest = candidates[-1][1]
        logger.info("Auto-resume: found checkpoint {}", latest)
        return str(latest)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: Path):
        state: dict = {"train_state": self.train_state.checkpoint_view(include_optimizers=False)}
        if self.ema is not None:
            state["ema"] = self.ema

        if self.expert_parallel:
            expert_name = "high" if self.expert_group == 0 else "low"
            save_path = path / expert_name
            dcp.save(state, checkpoint_id=str(save_path), process_group=self._dp_pg)
            self._save_optimizer_shards(save_path)
            torch.save(self.dataloader.state_dict(), save_path / f"dataloader_rank{self.dp_rank}.pt")
            if self.dp_rank == 0 and self.model.lora_config is not None:
                self.model.save_lora(str(save_path / "lora"))
            if self.dp_rank == 0:
                logger.info("Saved checkpoint to {} (expert={})", save_path, expert_name)
        else:
            dcp.save(state, checkpoint_id=str(path))
            self._save_optimizer_shards(path)
            torch.save(self.dataloader.state_dict(), path / f"dataloader_rank{self.rank}.pt")
            if self.rank == 0 and self.model.lora_config is not None:
                self.model.save_lora(str(path / "lora"))
            if self.rank == 0:
                logger.info("Saved DCP checkpoint to {}", path)
        self._barrier()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: str):
        ckpt_path = Path(path)
        is_ep_checkpoint = any(
            (ckpt_path / expert / ".metadata").exists() for expert in ("high", "low")
        )
        is_flat_checkpoint = (ckpt_path / ".metadata").exists()

        # ---- Cross-layout transitions ----
        if not self.expert_parallel and is_ep_checkpoint and not is_flat_checkpoint:
            self._load_ep_into_non_ep(path)
            return

        if self.expert_parallel and is_flat_checkpoint and not is_ep_checkpoint:
            self._load_flat_into_ep(path)
            return

        # ---- Same-layout resume ----
        self._load_same_layout(path, is_ep_checkpoint)

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def _load_ep_into_non_ep(self, path: str):
        """Load an expert-parallel checkpoint into a non-EP trainer (weights only)."""
        if not self._reset_on_load:
            raise ValueError(
                "Cannot fully resume non-expert-parallel training from an expert-parallel checkpoint. "
                "Set reset_dataloader: true, disable auto_resume, or load with expert_parallel: true."
            )

        logger.info(
            "Loading expert-parallel checkpoint {} into non-expert-parallel run as weights-only init",
            path,
        )
        load_dcp_into_pipeline(self.model, path, use_ema=False)
        if self.ema is not None:
            self.ema.reinitialize()
            logger.info("EMA reinitialized from loaded model weights")

        self._reset_training_state()
        logger.info(
            "Initialized from expert-parallel checkpoint with reset optimizer state, total_steps={}",
            self.total_steps,
        )

    def _load_flat_into_ep(self, path: str):
        """Load a flat (non-EP) checkpoint into an EP trainer (weights only).

        Each expert group loads the full flat checkpoint and keeps only the
        weights relevant to its own transformer.  Optimizer and dataloader
        state are always reset because the parallelism topology changed.
        """
        if not self._reset_on_load:
            raise ValueError(
                "Cannot fully resume expert-parallel training from a flat checkpoint. "
                "Set reset_dataloader: true, disable auto_resume, or load with expert_parallel: true."
            )

        logger.info(
            "Loading flat checkpoint {} into expert-parallel run as weights-only init",
            path,
        )
        load_dcp_into_pipeline(self.model, path, use_ema=False)
        if self.ema is not None:
            self.ema.reinitialize()
            logger.info("EMA reinitialized from loaded model weights")

        self._reset_training_state()
        logger.info(
            "Initialized from flat checkpoint with reset optimizer state, total_steps={}",
            self.total_steps,
        )

    def _detect_lora_mismatch(self, load_path: str) -> str | None:
        """Detect LoRA ↔ plain weight mismatches between checkpoint and current model.

        Returns:
            "plain_to_lora" — checkpoint has plain weights, model expects LoRA base_layer
            "lora_to_plain" — checkpoint has LoRA base_layer weights, model has plain weights
            "lora_rank_mismatch" — both are LoRA but shapes differ
            None — no mismatch (or no LoRA involved)
        """
        checkpoint_fqns = _get_checkpoint_fqns(load_path)

        for name, model_module in [
            ("transformer", self.model.transformer),
            ("transformer_2", self.model.transformer_2),
        ]:
            if model_module is None:
                continue
            model_state = get_model_state_dict(model_module)
            model_has_lora = any(".base_layer." in k for k in model_state)
            ckpt_has_lora = any(
                f"train_state.{name}." in fqn and ".base_layer." in fqn
                for fqn in checkpoint_fqns
            )

            if not model_has_lora and not ckpt_has_lora:
                continue

            if model_has_lora and not ckpt_has_lora:
                # Plain checkpoint → LoRA model
                if _should_plain_to_lora_remap(name, model_state, checkpoint_fqns):
                    return "plain_to_lora"

            if ckpt_has_lora and not model_has_lora:
                return "lora_to_plain"

            if model_has_lora and ckpt_has_lora:
                # Both LoRA — check for rank mismatch by comparing lora_A shapes
                for fqn in checkpoint_fqns:
                    if f"train_state.{name}." in fqn and ".lora_A." in fqn:
                        # Extract the corresponding model key
                        model_key = fqn.removeprefix(f"train_state.{name}.")
                        if model_key in model_state:
                            # Shapes match — this key is fine
                            pass
                        else:
                            # Key exists in checkpoint but not model (rank change may cause different key structure)
                            return "lora_rank_mismatch"
                        break

        return None

    def _load_same_layout(self, path: str, is_ep_checkpoint: bool):
        """Standard resume: same EP/non-EP layout between checkpoint and trainer."""
        # Resolve path and DCP process group for expert parallel
        if self.expert_parallel:
            expert_name = "high" if self.expert_group == 0 else "low"
            ep_path = Path(path) / expert_name
            load_path = str(ep_path) if ep_path.exists() else path
            dl_rank = self.dp_rank
            dcp_kwargs: dict = {"process_group": self._dp_pg}
        else:
            load_path = path
            dl_rank = self.rank
            dcp_kwargs = {}

        logger.info("Resuming from {} ...", load_path)

        # ---- Detect LoRA mismatches before DCP load ----
        lora_mismatch = self._detect_lora_mismatch(load_path)

        if lora_mismatch == "lora_to_plain":
            raise ValueError(
                f"Checkpoint at {load_path} contains LoRA weights but the current model "
                f"has lora_rank=0 (no LoRA). To load a LoRA checkpoint into a full-rank model, "
                f"first merge the LoRA weights or set lora_rank to match the checkpoint."
            )

        if lora_mismatch == "lora_rank_mismatch":
            if not self._reset_on_load:
                raise ValueError(
                    f"Checkpoint at {load_path} has LoRA weights with a different rank than "
                    f"the current model. Set reset_dataloader: true to load base weights only, "
                    f"or use the same lora_rank as the checkpoint."
                )
            logger.warning(
                "LoRA rank mismatch detected. Loading base weights only from {}, "
                "LoRA adapters will be freshly initialized.",
                load_path,
            )
            load_dcp_into_pipeline(self.model, load_path, use_ema=False)
            if self.ema is not None:
                self.ema.reinitialize()
            self._reset_training_state()
            logger.info("Initialized from LoRA checkpoint (rank mismatch) with base weights only")
            return

        if lora_mismatch == "plain_to_lora":
            logger.info(
                "Plain checkpoint detected, loading into LoRA model via base_layer remap from {}",
                load_path,
            )
            load_dcp_into_pipeline(self.model, load_path, use_ema=False)
            if self.ema is not None:
                self.ema.reinitialize()
                logger.info("EMA reinitialized from loaded model weights")
            self._reset_training_state()
            logger.info(
                "Initialized from plain checkpoint into LoRA model, total_steps={}",
                self.total_steps,
            )
            return

        # ---- Standard DCP load (no LoRA mismatch) ----
        checkpoint_optimizer_keys = get_checkpoint_optimizer_keys(load_path)
        load_train_state = self.train_state.checkpoint_view(
            include_optimizers=True,
            optimizer_keys=checkpoint_optimizer_keys,
        )
        state: dict = {"train_state": load_train_state}
        has_legacy_ema = (Path(load_path) / "ema").is_dir()
        if self.ema is not None and not has_legacy_ema:
            state["ema"] = self.ema
        try:
            dcp.load(state, checkpoint_id=load_path, **dcp_kwargs)
        except Exception:
            if "ema" in state:
                logger.warning("Failed to load EMA from DCP, retrying without EMA")
                dcp.load({"train_state": load_train_state}, checkpoint_id=load_path, **dcp_kwargs)
            else:
                raise
        self.train_state.step = load_train_state.step
        self.train_state.epoch = load_train_state.epoch
        self.train_state.batch_idx = load_train_state.batch_idx
        if self.ema is not None and "ema" not in state:
            self.ema.reinitialize()
            logger.warning("EMA not in DCP checkpoint, reinitialized from loaded model weights")
        if self._reset_on_load:
            self._reset_training_state()
            logger.info("reset_dataloader=True: reset step/epoch/optimizer, total_steps={}", self.total_steps)
        else:
            restored_optimizers = self._load_optimizer_shards(Path(load_path))
            if restored_optimizers:
                logger.info("Restored optimizer shards: {}", ", ".join(restored_optimizers))
            elif not checkpoint_optimizer_keys:
                logger.warning(
                    "Checkpoint {} has no optimizer state in DCP and no optimizer shard files; "
                    "continuing with freshly initialized optimizer state",
                    load_path,
                )
            dl_state_path = Path(load_path) / f"dataloader_rank{dl_rank}.pt"
            if dl_state_path.exists():
                self.dataloader.load_state_dict(torch.load(dl_state_path, weights_only=False))
                logger.info("Restored dataloader state from {}", dl_state_path)
        logger.info(
            "Resumed at step={} epoch={} batch_idx={}",
            self.train_state.step,
            self.train_state.epoch,
            self.train_state.batch_idx,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_training_state(self):
        """Reset step counters, rebuild optimizers, and recompute total_steps.

        Used after any load that is incompatible with full optimizer/dataloader
        resume (cross-layout, LoRA mismatch, or explicit reset_dataloader=True).
        """
        self.train_state.step = 0
        self.train_state.epoch = 0
        self.train_state.batch_idx = 0
        (
            self.params, self.optimizers,
            self.optimizer_te, self.optimizer_1, self.optimizer_2,
            self.fallback_te, self.fallback_1, self.fallback_2,
        ) = self._build_optimizers(self.cfg)
        self.train_state.optimizer_te = self.optimizer_te
        self.train_state.optimizer_1 = self.optimizer_1
        self.train_state.optimizer_2 = self.optimizer_2
        self.train_state.fallback_te = self.fallback_te
        self.train_state.fallback_1 = self.fallback_1
        self.train_state.fallback_2 = self.fallback_2
        self.total_steps = self._compute_total_steps()
