"""Checkpoint save/load orchestration — unified ``high/`` + ``low/`` layout.

Every checkpoint on disk has the same shape regardless of whether the trainer
was flat or expert-parallel when it wrote it::

    checkpoint-N/
      ├─ high/                          # transformer (+ shared scalars)
      │   ├─ .metadata + *.distcp       # DCP: train_state (filter=text_encoder+transformer)
      │   │                             #      + ema  (filter=text_encoder+transformer)
      │   ├─ optimizer_transformer_rank{R}.pt
      │   ├─ optimizer_text_encoder_rank{R}.pt   # if trained, duplicated in low/
      │   ├─ dataloader_rank{R}.pt               # duplicated in low/
      │   └─ lora/transformer/{adapter_model.safetensors, adapter_config.json}
      └─ low/                           # transformer_2 (+ shared scalars — duplicated)
          └─ ...                        # symmetric to high/

Shared scalars (``step``, ``epoch``, ``batch_idx``, RNG, and ``text_encoder``
weights when trained) are duplicated across both subdirs.  Duplicates are
tiny relative to transformer weights.  The benefit: **a flat trainer and an
EP trainer write and read exactly the same layout**, so cross-layout resume
just works.

Legacy flat checkpoints (top-level ``.metadata`` with no ``high``/``low``
subdirs) are still loadable as a fallback; they are never *written* again.

Host trainer contract — the mixin reads:

    self.cfg, self.model, self.ema, self.train_state, self.dataloader,
    self.rank, self.mesh,
    self.expert_parallel, self.expert_group, self.dp_rank, self._dp_pg,
    self._reset_on_load, self.optimizer_te, self.optimizer_1, self.optimizer_2,
    self.total_steps,

and calls:

    self._barrier()
    self._build_optimizers(cfg)
    self._compute_total_steps()
    self._checkpoint_rank() -> int
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

from src.trainer.checkpoint import (
    MODEL_KEYS,
    extract_init_weights,
    gather_full_state_dict,
    load_optimizer_shard,
    read_dcp_to_flat_dict,
    remap_for_current_model,
    save_optimizer_shard,
    write_peft_lora_adapter,
)


@dataclass
class _SubdirEntry:
    """One ``high/`` or ``low/`` subdir that this rank should save or load."""

    subdir: str  # "high" | "low"
    transformer_key: str  # "transformer" | "transformer_2"
    transformer_model: torch.nn.Module | None
    transformer_optimizer: torch.optim.Optimizer | None
    shard_rank: int
    dcp_kwargs: dict  # e.g. {"process_group": dp_pg} for EP


class CheckpointRuntimeMixin:
    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _find_latest_checkpoint(self) -> str | None:
        out = Path(self.cfg.output_dir)
        if not out.exists():
            return None
        candidates: list[tuple[int, Path]] = []
        for d in out.iterdir():
            if not d.is_dir() or not _is_checkpoint_dir(d) or not _has_valid_step_suffix(d.name):
                continue
            candidates.append((d.stat().st_mtime_ns, d))
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
        path.mkdir(parents=True, exist_ok=True)
        entries = list(self._save_plan())
        if not entries:
            raise RuntimeError("Nothing to save — no transformer is attached to this trainer.")
        for entry in entries:
            self._save_subdir(path, entry)
        self._barrier()
        if entries[0].shard_rank == 0:
            logger.info(
                "Saved checkpoint to {} (subdirs: {})",
                path,
                ", ".join(e.subdir for e in entries),
            )

    def _save_plan(self):
        if self.expert_parallel:
            sub = "high" if self.expert_group == 0 else "low"
            key = "transformer" if sub == "high" else "transformer_2"
            model = self.model.transformer if sub == "high" else self.model.transformer_2
            optimizer = self.optimizer_1 if sub == "high" else self.optimizer_2
            yield _SubdirEntry(
                subdir=sub,
                transformer_key=key,
                transformer_model=model,
                transformer_optimizer=optimizer,
                shard_rank=self.dp_rank,
                dcp_kwargs={"process_group": self._dp_pg},
            )
            return
        # Flat: save each attached transformer to its own subdir, all ranks
        # participate in each DCP collective on the default world pg.
        for sub, key, model, optimizer in [
            ("high", "transformer", self.model.transformer, self.optimizer_1),
            ("low", "transformer_2", self.model.transformer_2, self.optimizer_2),
        ]:
            if model is None:
                continue
            yield _SubdirEntry(
                subdir=sub,
                transformer_key=key,
                transformer_model=model,
                transformer_optimizer=optimizer,
                shard_rank=self.rank,
                dcp_kwargs={},
            )

    def _save_subdir(self, root: Path, entry: _SubdirEntry) -> None:
        sub_path = root / entry.subdir
        sub_path.mkdir(parents=True, exist_ok=True)
        filter_keys = self._subdir_filter_keys(entry.transformer_key)

        # 1. DCP — train_state + EMA, restricted to this subdir's models.
        self.train_state.set_save_filter(filter_keys)
        if self.ema is not None:
            self.ema.set_save_filter(filter_keys)
        try:
            state: dict = {"train_state": self.train_state}
            if self.ema is not None:
                state["ema"] = self.ema
            dcp.save(state, checkpoint_id=str(sub_path), **entry.dcp_kwargs)
        finally:
            self.train_state.set_save_filter(None)
            if self.ema is not None:
                self.ema.set_save_filter(None)

        # 2. Optimizer shards — transformer for this subdir, plus text_encoder
        #    when trained (duplicated across both subdirs).
        if not self._hsdp_skip_optimizer_shard():
            save_optimizer_shard(
                sub_path / f"optimizer_{entry.transformer_key}_rank{entry.shard_rank}.pt",
                entry.transformer_model,
                entry.transformer_optimizer,
            )
            if self.cfg.train_text_encoder and self.model.text_encoder is not None:
                save_optimizer_shard(
                    sub_path / f"optimizer_text_encoder_rank{entry.shard_rank}.pt",
                    self.model.text_encoder,
                    self.optimizer_te,
                )

        # 3. Dataloader — per shard rank (duplicated across subdirs).
        torch.save(self.dataloader.state_dict(), sub_path / f"dataloader_rank{entry.shard_rank}.pt")

        # 4. LoRA adapter — gather is collective, writer is rank 0.
        if self.model.lora_config is not None and entry.transformer_model is not None:
            full_sd = gather_full_state_dict(entry.transformer_model)
            if entry.shard_rank == 0:
                write_peft_lora_adapter(
                    sub_path / "lora" / entry.transformer_key,
                    entry.transformer_model,
                    full_sd,
                )

    def _subdir_filter_keys(self, transformer_key: str) -> frozenset[str]:
        """Keys this subdir is responsible for (transformer + text_encoder if trained)."""
        keys = {transformer_key}
        if self.cfg.train_text_encoder and self.model.text_encoder is not None:
            keys.add("text_encoder")
        return frozenset(keys)

    def _hsdp_skip_optimizer_shard(self) -> bool:
        """Under HSDP, only the first replica writes optimizer shards."""
        return self.mesh is not None and self.mesh.ndim == 2 and self.mesh.get_local_rank("replicate") > 0

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: str):
        ckpt = Path(path)
        entries = list(self._load_plan(ckpt))
        if not entries:
            raise ValueError(
                f"Checkpoint at {path} is missing both high/.metadata and low/.metadata "
                f"(and no legacy top-level .metadata). Cannot resume."
            )

        mode = "init" if self._reset_on_load else "resume"
        logger.info(
            "{} from {} ({}) ...",
            "Initializing" if mode == "init" else "Resuming",
            ckpt,
            "legacy flat root" if entries[0].subdir == "" else "subdirs: " + ", ".join(e.subdir for e in entries),
        )
        for entry in entries:
            self._load_subdir(entry)

        if self._reset_on_load:
            # Weight-only init: EMA tracks the newly-initialised model weights.
            if self.ema is not None:
                self.ema.reinitialize()
                logger.info("EMA reinitialized from loaded model weights.")
            self._reset_training_state()
            logger.info(
                "reset_dataloader=True: reset step/epoch/optimizer, total_steps={}",
                self.total_steps,
            )

        logger.info(
            "Resumed at step={} epoch={} batch_idx={}",
            self.train_state.step,
            self.train_state.epoch,
            self.train_state.batch_idx,
        )

    def _load_plan(self, ckpt: Path):
        """Build the list of subdirs (or a single legacy-root entry) to load from."""
        high_ok = (ckpt / "high" / ".metadata").exists()
        low_ok = (ckpt / "low" / ".metadata").exists()

        if not high_ok and not low_ok:
            # Legacy flat layout: single DCP at the root holding every model.
            if not (ckpt / ".metadata").exists():
                return
            yield _LoadEntry(
                subdir="",
                path=ckpt,
                filter_keys=None,  # load everything that matches what's in TrainState
                shard_rank=self._checkpoint_rank(),
                dcp_kwargs={"process_group": self._dp_pg} if self.expert_parallel else {},
                transformer_key=None,
            )
            return

        # Unified layout: iterate high/ then low/.  EP trainers only load their
        # own group's subdir; flat trainers load both when both are present.
        for sub, key in [("high", "transformer"), ("low", "transformer_2")]:
            sub_path = ckpt / sub
            if not (sub_path / ".metadata").exists():
                continue
            if self.expert_parallel:
                want = "high" if self.expert_group == 0 else "low"
                if sub != want:
                    continue
            if getattr(self.model, key) is None:
                # Trainer doesn't actually own this transformer; skip.
                logger.warning("Skipping {} (trainer has no {} attached).", sub_path, key)
                continue
            yield _LoadEntry(
                subdir=sub,
                path=sub_path,
                filter_keys=self._subdir_filter_keys(key),
                shard_rank=self.dp_rank if self.expert_parallel else self.rank,
                dcp_kwargs={"process_group": self._dp_pg} if self.expert_parallel else {},
                transformer_key=key,
            )

    def _load_subdir(self, entry: _LoadEntry) -> None:
        if self._reset_on_load:
            self._load_for_init(entry)
        else:
            self._load_for_resume(entry)

    # ------------------------------------------------------------------
    # Resume — full state (model + optimizer + dataloader + EMA + counters).
    # ------------------------------------------------------------------

    def _load_for_resume(self, entry: _LoadEntry) -> None:
        # 1. DCP — train_state + EMA.
        self.train_state.set_save_filter(entry.filter_keys)
        if self.ema is not None:
            self.ema.set_save_filter(entry.filter_keys)
        try:
            state: dict = {"train_state": self.train_state}
            if self.ema is not None:
                state["ema"] = self.ema
            try:
                dcp.load(state, checkpoint_id=str(entry.path), **entry.dcp_kwargs)
            except Exception as e:
                if "ema" in state:
                    logger.warning(
                        "Failed to load EMA from {} ({}); retrying without it.",
                        entry.path,
                        type(e).__name__,
                    )
                    dcp.load({"train_state": self.train_state}, checkpoint_id=str(entry.path), **entry.dcp_kwargs)
                    self.ema.reinitialize()
                    logger.warning("EMA reinitialized from loaded model weights.")
                else:
                    raise
        finally:
            self.train_state.set_save_filter(None)
            if self.ema is not None:
                self.ema.set_save_filter(None)

        # 2. Optimizer shards + dataloader.
        restored = self._load_optimizer_shards_for(entry)
        if restored:
            logger.info("Restored optimizer shards from {}: {}", entry.path, ", ".join(restored))
        elif entry.transformer_key is not None:
            logger.warning(
                "No optimizer shards found in {} for {}; optimizer state starts fresh.",
                entry.path,
                entry.transformer_key,
            )
        dl_state_path = entry.path / f"dataloader_rank{entry.shard_rank}.pt"
        if dl_state_path.exists():
            self.dataloader.load_state_dict(torch.load(dl_state_path, weights_only=False))
            logger.info("Restored dataloader state from {}", dl_state_path)

    # ------------------------------------------------------------------
    # Init — weight-only load (prefer EMA shadow; auto plain → LoRA remap).
    # ------------------------------------------------------------------

    def _load_for_init(self, entry: _LoadEntry) -> None:
        """Weight-only init from a previous checkpoint.

        Rank 0 materialises the entire DCP into a CPU flat dict once (~28 GB
        for a 14B bf16 model — one-shot, acceptable for startup), extracts
        and remaps weights per model, then all ranks reshard via
        ``set_model_state_dict(broadcast_from_rank0=True)`` — the canonical
        DCP pattern for going from a full CPU dict to an FSDP2-wrapped model.

        EMA is *not* loaded here; it will be reinitialised once at the end of
        ``_load_checkpoint`` to track the newly-initialised model weights.
        Optimizer and dataloader state are not loaded either — they are reset
        via ``_reset_training_state()`` also at the end of ``_load_checkpoint``.
        """
        # Which models to init from this subdir.
        if entry.transformer_key is not None:
            model_specs: list[tuple[str, torch.nn.Module | None]] = [
                (entry.transformer_key, getattr(self.model, entry.transformer_key)),
            ]
        else:
            # Legacy flat root — a single DCP with all models.
            model_specs = [
                ("transformer", self.model.transformer),
                ("transformer_2", self.model.transformer_2),
            ]
        # text_encoder (when trained) is duplicated in both subdirs, so loading
        # it twice across high/low is idempotent — same CPU tensors broadcast.
        if self.cfg.train_text_encoder and self.model.text_encoder is not None:
            model_specs.append(("text_encoder", self.model.text_encoder))

        flat: dict[str, torch.Tensor] | None = None
        if self.rank == 0:
            flat = read_dcp_to_flat_dict(entry.path)

        try:
            for model_key, model in model_specs:
                if model is None:
                    continue
                self._broadcast_init_weights(flat, model_key, model, entry.path)
        finally:
            # Release the full CPU dict before the next subdir's pass.
            del flat

    def _broadcast_init_weights(
        self,
        flat: dict[str, torch.Tensor] | None,
        model_key: str,
        model: torch.nn.Module,
        source_path: Path,
    ) -> None:
        """Extract + remap on rank 0, then broadcast-reshard into ``model``."""
        remapped: dict[str, torch.Tensor] = {}
        source_tag: str | None = None

        if self.rank == 0 and flat is not None:
            try:
                weights, source_tag = extract_init_weights(flat, model_key)
                remapped = remap_for_current_model(weights, model)
            except RuntimeError as e:
                # Model simply isn't in this subdir's DCP (e.g. transformer_2
                # not in a high/ subdir) — skip it silently.
                logger.debug("No {} data in {}: {}", model_key, source_path, e)
                remapped, source_tag = {}, None

        # Collective: every rank participates, even with an empty dict.
        set_model_state_dict(
            model,
            model_state_dict=remapped,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                strict=False,  # tolerate lora_A/B absent from plain source
            ),
        )
        if self.rank == 0 and source_tag is not None:
            logger.info("Initialized {} from {} shadows at {}", model_key, source_tag, source_path)

    def _load_optimizer_shards_for(self, entry: _LoadEntry) -> list[str]:
        """Load any optimizer shards present in this subdir (new layout).

        Shard names are ``optimizer_<name>_rank{R}.pt``.  We look for
        ``transformer`` / ``transformer_2`` / ``text_encoder`` variants; absent
        files are skipped silently.  Legacy checkpoints (where optimizers were
        stored inside DCP) have no shard files here, so this returns ``[]``
        and the trainer runs with freshly initialised optimizer state.
        """
        restored: list[str] = []
        candidates: list[tuple[str, torch.nn.Module | None, torch.optim.Optimizer | None]] = []
        if entry.transformer_key == "transformer":
            candidates.append(("transformer", self.model.transformer, self.optimizer_1))
        elif entry.transformer_key == "transformer_2":
            candidates.append(("transformer_2", self.model.transformer_2, self.optimizer_2))
        else:
            # Legacy root: try all three names.
            candidates.extend(
                [
                    ("transformer", self.model.transformer, self.optimizer_1),
                    ("transformer_2", self.model.transformer_2, self.optimizer_2),
                ]
            )
        if self.cfg.train_text_encoder and self.model.text_encoder is not None:
            candidates.append(("text_encoder", self.model.text_encoder, self.optimizer_te))

        for name, model, optimizer in candidates:
            path = entry.path / f"optimizer_{name}_rank{entry.shard_rank}.pt"
            if load_optimizer_shard(path, model, optimizer):
                restored.append(name)
        return restored

    # ------------------------------------------------------------------
    # Reset helper
    # ------------------------------------------------------------------

    def _reset_training_state(self):
        self.train_state.step = 0
        self.train_state.epoch = 0
        self.train_state.batch_idx = 0
        (
            self.params,
            self.optimizers,
            self.optimizer_te,
            self.optimizer_1,
            self.optimizer_2,
            self.fallback_te,
            self.fallback_1,
            self.fallback_2,
        ) = self._build_optimizers(self.cfg)
        self.total_steps = self._compute_total_steps()


@dataclass
class _LoadEntry:
    subdir: str  # "high" | "low" | "" (legacy root)
    path: Path
    filter_keys: frozenset[str] | None  # which models to load (None = no filter)
    shard_rank: int
    dcp_kwargs: dict
    transformer_key: str | None  # transformer | transformer_2 | None (legacy)


def _is_checkpoint_dir(d: Path) -> bool:
    return (d / ".metadata").exists() or (d / "high" / ".metadata").exists() or (d / "low" / ".metadata").exists()


def _has_valid_step_suffix(name: str) -> bool:
    if name.startswith("checkpoint-epoch"):
        suffix = name.removeprefix("checkpoint-epoch")
    elif name.startswith("checkpoint-"):
        suffix = name.removeprefix("checkpoint-")
    else:
        return False
    try:
        int(suffix)
        return True
    except ValueError:
        return False


# Re-export so call sites that import MODEL_KEYS from checkpoint_runtime still work.
__all__ = ["CheckpointRuntimeMixin", "MODEL_KEYS"]
