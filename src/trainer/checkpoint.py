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
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful


def _save_pair(sd: dict, key: str, model, optimizer):
    """Save model + optimizer state_dict pair into sd[key] and sd[key_optim]."""
    if model is not None and optimizer is not None:
        m_sd, o_sd = get_state_dict(model, optimizer)
        sd[key] = m_sd
        sd[f"optimizer_{key}"] = o_sd


def _load_pair(state_dict: dict, key: str, model, optimizer):
    """Load model + optimizer state_dict pair from state_dict."""
    if model is not None and optimizer is not None and key in state_dict:
        set_state_dict(
            model,
            optimizer,
            model_state_dict=state_dict[key],
            optim_state_dict=state_dict[f"optimizer_{key}"],
        )


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
        self.step = step
        self.epoch = epoch
        self.batch_idx = batch_idx

    def state_dict(self):
        sd = {"step": self.step, "epoch": self.epoch, "batch_idx": self.batch_idx}
        _save_pair(sd, "text_encoder", self.text_encoder, self.optimizer_te)
        _save_pair(sd, "transformer", self.transformer, self.optimizer_1)
        _save_pair(sd, "transformer_2", self.transformer_2, self.optimizer_2)
        # Fallback optimizers (e.g. AdamW for non-2D params under Muon)
        _save_pair(sd, "text_encoder_fallback", self.text_encoder, self.fallback_te)
        _save_pair(sd, "transformer_fallback", self.transformer, self.fallback_1)
        _save_pair(sd, "transformer_2_fallback", self.transformer_2, self.fallback_2)
        # RNG states for reproducibility on resume
        sd["rng_cpu"] = torch.random.get_rng_state()
        sd["rng_cuda"] = torch.cuda.get_rng_state()
        return sd

    def load_state_dict(self, state_dict):
        _load_pair(state_dict, "text_encoder", self.text_encoder, self.optimizer_te)
        _load_pair(state_dict, "transformer", self.transformer, self.optimizer_1)
        _load_pair(state_dict, "transformer_2", self.transformer_2, self.optimizer_2)
        # Fallback optimizers — missing in old checkpoints, just skip
        _load_pair(state_dict, "text_encoder_fallback", self.text_encoder, self.fallback_te)
        _load_pair(state_dict, "transformer_fallback", self.transformer, self.fallback_1)
        _load_pair(state_dict, "transformer_2_fallback", self.transformer_2, self.fallback_2)
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]
        self.batch_idx = state_dict.get("batch_idx", 0)
        # Restore RNG states
        if "rng_cpu" in state_dict:
            torch.random.set_rng_state(state_dict["rng_cpu"])
        if "rng_cuda" in state_dict:
            torch.cuda.set_rng_state(state_dict["rng_cuda"])


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
