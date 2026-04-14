"""DCP (Distributed Checkpoint) state management for FSDP2 training."""

import os
from functools import lru_cache
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful


def _save_component(
    sd: dict,
    key: str,
    model,
    optimizer,
    *,
    include_optimizer: bool,
    save_model_without_optimizer: bool,
):
    """Save a model state, optionally alongside its optimizer state."""
    if model is None:
        return
    if include_optimizer and optimizer is not None:
        m_sd, o_sd = get_state_dict(model, optimizer)
        sd[key] = m_sd
        sd[f"optimizer_{key}"] = o_sd
        return
    if not save_model_without_optimizer:
        return
    sd[key] = get_model_state_dict(model)


def _load_component(state_dict: dict, key: str, model, optimizer):
    """Load a model state, plus optimizer state when present."""
    if model is None or key not in state_dict:
        return
    optimizer_key = f"optimizer_{key}"
    if optimizer is not None and optimizer_key in state_dict:
        set_state_dict(
            model,
            optimizer,
            model_state_dict=state_dict[key],
            optim_state_dict=state_dict[optimizer_key],
        )
        return
    set_model_state_dict(model, model_state_dict=state_dict[key])


class TrainState(Stateful):
    """Wraps models + optimizers + RNG states for DCP save/load via the Stateful protocol.

    Each FSDP-sharded module has its own optimizer so that get_state_dict can
    correctly map parameter IDs to FQNs.
    """

    def __init__(
        self,
        text_encoder: torch.nn.Module | None = None,
        transformer: torch.nn.Module | None = None,
        transformer_2: torch.nn.Module | None = None,
        optimizer_te: torch.optim.Optimizer | None = None,
        optimizer_1: torch.optim.Optimizer | None = None,
        optimizer_2: torch.optim.Optimizer | None = None,
        fallback_te: torch.optim.Optimizer | None = None,
        fallback_1: torch.optim.Optimizer | None = None,
        fallback_2: torch.optim.Optimizer | None = None,
        include_optimizers: bool = True,
        optimizer_keys: set[str] | frozenset[str] | None = None,
        step: int = 0,
        epoch: int = 0,
        batch_idx: int = 0,
    ):
        self.text_encoder = text_encoder
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self.optimizer_te = optimizer_te
        self.optimizer_1 = optimizer_1
        self.optimizer_2 = optimizer_2
        self.fallback_te = fallback_te
        self.fallback_1 = fallback_1
        self.fallback_2 = fallback_2
        self.include_optimizers = include_optimizers
        self.optimizer_keys = None if optimizer_keys is None else set(optimizer_keys)
        self.step = step
        self.epoch = epoch
        self.batch_idx = batch_idx

    def checkpoint_view(
        self,
        *,
        include_optimizers: bool,
        optimizer_keys: set[str] | frozenset[str] | None = None,
    ) -> "TrainState":
        """Create a save/load view over the same live modules and optimizers."""
        return TrainState(
            text_encoder=self.text_encoder,
            transformer=self.transformer,
            transformer_2=self.transformer_2,
            optimizer_te=self.optimizer_te,
            optimizer_1=self.optimizer_1,
            optimizer_2=self.optimizer_2,
            fallback_te=self.fallback_te,
            fallback_1=self.fallback_1,
            fallback_2=self.fallback_2,
            include_optimizers=include_optimizers,
            optimizer_keys=optimizer_keys,
            step=self.step,
            epoch=self.epoch,
            batch_idx=self.batch_idx,
        )

    def _include_optimizer_key(self, optimizer_key: str) -> bool:
        if not self.include_optimizers:
            return False
        if self.optimizer_keys is None:
            return True
        return optimizer_key in self.optimizer_keys

    def state_dict(self):
        sd = {"step": self.step, "epoch": self.epoch, "batch_idx": self.batch_idx}
        _save_component(
            sd,
            "text_encoder",
            self.text_encoder,
            self.optimizer_te,
            include_optimizer=self._include_optimizer_key("optimizer_text_encoder"),
            save_model_without_optimizer=True,
        )
        _save_component(
            sd,
            "transformer",
            self.transformer,
            self.optimizer_1,
            include_optimizer=self._include_optimizer_key("optimizer_transformer"),
            save_model_without_optimizer=True,
        )
        _save_component(
            sd,
            "transformer_2",
            self.transformer_2,
            self.optimizer_2,
            include_optimizer=self._include_optimizer_key("optimizer_transformer_2"),
            save_model_without_optimizer=True,
        )
        # Fallback optimizers (e.g. AdamW for non-2D params under Muon)
        _save_component(
            sd,
            "text_encoder_fallback",
            self.text_encoder,
            self.fallback_te,
            include_optimizer=self._include_optimizer_key("optimizer_text_encoder_fallback"),
            save_model_without_optimizer=False,
        )
        _save_component(
            sd,
            "transformer_fallback",
            self.transformer,
            self.fallback_1,
            include_optimizer=self._include_optimizer_key("optimizer_transformer_fallback"),
            save_model_without_optimizer=False,
        )
        _save_component(
            sd,
            "transformer_2_fallback",
            self.transformer_2,
            self.fallback_2,
            include_optimizer=self._include_optimizer_key("optimizer_transformer_2_fallback"),
            save_model_without_optimizer=False,
        )
        # RNG states for reproducibility on resume
        sd["rng_cpu"] = torch.random.get_rng_state()
        sd["rng_cuda"] = torch.cuda.get_rng_state()
        return sd

    def load_state_dict(self, state_dict):
        _load_component(state_dict, "text_encoder", self.text_encoder, self.optimizer_te)
        _load_component(state_dict, "transformer", self.transformer, self.optimizer_1)
        _load_component(state_dict, "transformer_2", self.transformer_2, self.optimizer_2)
        # Fallback optimizers — missing in old checkpoints, just skip
        _load_component(state_dict, "text_encoder_fallback", self.text_encoder, self.fallback_te)
        _load_component(state_dict, "transformer_fallback", self.transformer, self.fallback_1)
        _load_component(state_dict, "transformer_2_fallback", self.transformer_2, self.fallback_2)
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]
        self.batch_idx = state_dict.get("batch_idx", 0)
        # Restore RNG states
        if "rng_cpu" in state_dict:
            torch.random.set_rng_state(state_dict["rng_cpu"])
        if "rng_cuda" in state_dict:
            torch.cuda.set_rng_state(state_dict["rng_cuda"])


def save_optimizer_shard(
    path: Path,
    model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
    *,
    process_group_size: int | None = None,
) -> bool:
    """Save one optimizer shard outside DCP to keep the DCP save plan small."""
    if model is None or optimizer is None:
        return False

    payload = {
        "format_version": 1,
        "process_group_size": process_group_size,
        "optimizer_state": get_optimizer_state_dict(model, optimizer),
    }
    torch.save(payload, path)
    return True


def load_optimizer_shard(
    path: Path,
    model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
    *,
    expected_process_group_size: int | None = None,
) -> bool:
    """Load one optimizer shard saved by save_optimizer_shard()."""
    if model is None or optimizer is None or not path.exists():
        return False

    payload = torch.load(path, map_location="cpu", weights_only=False)
    optimizer_state = payload
    if isinstance(payload, dict) and "optimizer_state" in payload:
        saved_group_size = payload.get("process_group_size")
        if (
            expected_process_group_size is not None
            and saved_group_size is not None
            and saved_group_size != expected_process_group_size
        ):
            logger.warning(
                "Skipping optimizer shard {}: saved process_group_size={}, current process_group_size={}",
                path,
                saved_group_size,
                expected_process_group_size,
            )
            return False
        optimizer_state = payload["optimizer_state"]

    set_optimizer_state_dict(model, optimizer, optimizer_state)
    return True


def _detect_checkpoint_layout(checkpoint_path: str) -> dict[str, str]:
    """Detect checkpoint layout — flat or expert-parallel.

    Returns a mapping from expert name to the DCP directory containing it.
    Flat:  {"transformer": path, "transformer_2": path}
    EP:    {"transformer": path/high, "transformer_2": path/low}
    """
    p = Path(checkpoint_path)
    high_dir = p / "high"
    low_dir = p / "low"
    if (high_dir / ".metadata").exists() or (low_dir / ".metadata").exists():
        result = {}
        if (high_dir / ".metadata").exists():
            result["transformer"] = str(high_dir)
        if (low_dir / ".metadata").exists():
            result["transformer_2"] = str(low_dir)
        return result
    return {"transformer": checkpoint_path, "transformer_2": checkpoint_path}


@lru_cache(maxsize=32)
def _get_checkpoint_fqns(dcp_path: str) -> frozenset[str]:
    """Read DCP metadata once and return all saved FQNs for this path."""
    reader = FileSystemReader(dcp_path)
    metadata = reader.read_metadata()
    return frozenset(metadata.state_dict_metadata.keys())


def get_checkpoint_optimizer_keys(dcp_path: str) -> frozenset[str]:
    """Return top-level optimizer entries present in a checkpoint."""
    optimizer_keys: set[str] = set()
    for fqn in _get_checkpoint_fqns(dcp_path):
        if not fqn.startswith("train_state."):
            continue
        component = fqn.removeprefix("train_state.").split(".", 1)[0]
        if component.startswith("optimizer_"):
            optimizer_keys.add(component)
    return frozenset(optimizer_keys)


def _should_plain_to_lora_remap(name: str, model_state: dict[str, torch.Tensor], checkpoint_fqns: set[str]) -> bool:
    """Detect whether a plain checkpoint should be remapped into LoRA base_layer keys.

    Current model keys may contain:
      blocks.0.attn1.to_q.base_layer.weight
    while a plain checkpoint contains:
      train_state.transformer.blocks.0.attn1.to_q.weight
    """
    if not any(".base_layer." in key for key in model_state):
        return False

    direct_matches = 0
    plain_matches = 0
    for model_key in model_state:
        if ".lora_" in model_key:
            continue
        if f"train_state.{name}.{model_key}" in checkpoint_fqns:
            direct_matches += 1

        plain_key = model_key.replace(".base_layer.", ".")
        if f"train_state.{name}.{plain_key}" in checkpoint_fqns:
            plain_matches += 1

    if plain_matches == 0:
        return False
    if direct_matches == 0:
        return True
    return plain_matches > direct_matches


def load_dcp_into_pipeline(pipe, checkpoint_path: str, use_ema: bool = False) -> None:
    """Load a DCP training checkpoint into a diffusers pipeline for inference.

    Supports both flat checkpoints and expert-parallel checkpoints (high/ low/ subdirs).
    Handles resharding automatically.

    Args:
        pipe: A WanImageToVideoPipeline (or similar) with .transformer / .transformer_2.
        checkpoint_path: Path to the DCP checkpoint directory.
        use_ema: If True, load EMA shadow weights instead of model weights.
    """
    needs_cleanup = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo")
        needs_cleanup = True

    try:
        layout = _detect_checkpoint_layout(checkpoint_path)
        transformers: dict[str, tuple[torch.nn.Module, str]] = {}
        if getattr(pipe, "transformer", None) is not None and "transformer" in layout:
            transformers["transformer"] = (pipe.transformer, layout["transformer"])
        if getattr(pipe, "transformer_2", None) is not None and "transformer_2" in layout:
            transformers["transformer_2"] = (pipe.transformer_2, layout["transformer_2"])

        for name, (model, dcp_path) in transformers.items():
            if use_ema:
                _load_ema_single(name, model, dcp_path)
            else:
                _load_weights_single(name, model, dcp_path)
    finally:
        if needs_cleanup:
            dist.destroy_process_group()


def _load_weights_single(name: str, model: torch.nn.Module, dcp_path: str) -> None:
    """Load model weights for a single expert from a DCP checkpoint."""
    model_state = get_model_state_dict(model)
    checkpoint_fqns = _get_checkpoint_fqns(dcp_path)

    if _should_plain_to_lora_remap(name, model_state, checkpoint_fqns):
        logger.warning(
            "Detected plain checkpoint keys for {} at {}. Loading into LoRA base_layer weights via remap.",
            name,
            dcp_path,
        )

        remapped_state: dict[str, torch.Tensor] = {}
        key_mapping: dict[str, str] = {}
        for model_key, tensor in model_state.items():
            if ".lora_" in model_key:
                continue
            checkpoint_key = model_key.replace(".base_layer.", ".")
            remapped_state[checkpoint_key] = torch.empty_like(tensor)
            key_mapping[checkpoint_key] = model_key

        state = {"train_state": {name: remapped_state}}
        dcp.load(state, checkpoint_id=dcp_path)

        merged_state = dict(model_state)
        for checkpoint_key, tensor in state["train_state"][name].items():
            merged_state[key_mapping[checkpoint_key]] = tensor

        set_model_state_dict(model, model_state_dict=merged_state)
        logger.info("Loaded {} weights from {} with plain->LoRA base remap", name, dcp_path)
        return

    state: dict = {"train_state": {name: model_state}}
    dcp.load(state, checkpoint_id=dcp_path)
    set_model_state_dict(model, model_state_dict=state["train_state"][name])
    logger.info("Loaded {} weights from {}", name, dcp_path)


def _load_ema_single(name: str, model: torch.nn.Module, dcp_path: str) -> None:
    """Load EMA shadow weights for a single expert from a DCP checkpoint."""
    shadow: dict[str, torch.Tensor] = {}
    for pname, p in model.named_parameters():
        shadow[f"{name}.{pname}"] = torch.empty_like(p, device="cpu")

    state: dict = {"ema": {"shadow": shadow, "decay": torch.tensor(0.0)}}
    dcp.load(state, checkpoint_id=dcp_path)

    prefix = f"{name}."
    sd = {k.removeprefix(prefix): v for k, v in shadow.items() if k.startswith(prefix)}
    model.load_state_dict(sd)
    logger.info("Loaded {} EMA weights from {}", name, dcp_path)
