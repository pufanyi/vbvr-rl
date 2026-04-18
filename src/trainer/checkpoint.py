"""Checkpoint primitives for FSDP2 + DCP training.

Disk layout per checkpoint directory (`R` = shard rank within the DCP group):

    checkpoint-N/
      ├─ .metadata + __*_*.distcp           # DCP: train_state (model state +
      │                                     #      step counters + RNG) + ema
      ├─ optimizer_<name>_rank{R}.pt        # one .pt per shard rank, outside DCP
      ├─ dataloader_rank{R}.pt              # per shard rank
      └─ lora/                              # only when lora_rank > 0
          ├─ transformer/{adapter_model.safetensors, adapter_config.json}
          └─ transformer_2/{adapter_model.safetensors, adapter_config.json}

For expert-parallel runs the same layout sits inside `high/` (transformer) and
`low/` (transformer_2) subdirectories.

Resume requires the **same layout** (flat ↔ EP transitions raise). For weight-
only initialisation from an incompatible checkpoint, run
`scripts/convert_dcp_to_lora.py` and load the resulting safetensors via
`pipe.load_lora_weights(...)`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful


def _is_fsdp_wrapped(model: torch.nn.Module) -> bool:
    """Detect FSDP2 (fully_shard) wrapping via the DTensor ._local_tensor attr."""
    return any(hasattr(p, "_local_tensor") for p in model.parameters())


# ---------------------------------------------------------------------------
# TrainState — model weights + step counters + RNG.  Optimizers are NOT here;
# they live in `optimizer_<name>_rank{R}.pt` files alongside the DCP shards.
# ---------------------------------------------------------------------------


MODEL_KEYS = ("text_encoder", "transformer", "transformer_2")


class TrainState(Stateful):
    """DCP `Stateful` wrapper for model weights and lightweight scalar state.

    ``set_save_filter`` restricts which of the model-state keys
    (``text_encoder``, ``transformer``, ``transformer_2``) are included in
    ``state_dict()``.  This is used by the unified ``high/`` and ``low/``
    layout so that each subdir holds exactly one transformer.  ``None`` (the
    default) means "include every model that is non-None".
    """

    def __init__(
        self,
        text_encoder: torch.nn.Module | None,
        transformer: torch.nn.Module | None,
        transformer_2: torch.nn.Module | None,
        step: int = 0,
        epoch: int = 0,
        batch_idx: int = 0,
    ):
        self.text_encoder = text_encoder
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self.step = step
        self.epoch = epoch
        self.batch_idx = batch_idx
        self._save_filter: frozenset[str] | None = None

    def set_save_filter(self, keys: set[str] | frozenset[str] | None) -> None:
        self._save_filter = None if keys is None else frozenset(keys)

    def _included(self, key: str) -> bool:
        return self._save_filter is None or key in self._save_filter

    @staticmethod
    def _model_sd(model: torch.nn.Module | None):
        if model is None:
            return None
        if _is_fsdp_wrapped(model):
            return get_model_state_dict(model)
        return model.state_dict()

    def state_dict(self):
        sd: dict = {"step": self.step, "epoch": self.epoch, "batch_idx": self.batch_idx}
        if self.text_encoder is not None and self._included("text_encoder"):
            sd["text_encoder"] = self._model_sd(self.text_encoder)
        if self.transformer is not None and self._included("transformer"):
            sd["transformer"] = self._model_sd(self.transformer)
        if self.transformer_2 is not None and self._included("transformer_2"):
            sd["transformer_2"] = self._model_sd(self.transformer_2)
        sd["rng_cpu"] = torch.random.get_rng_state()
        sd["rng_cuda"] = torch.cuda.get_rng_state()
        return sd

    @staticmethod
    def _apply(model: torch.nn.Module | None, sd: dict, key: str) -> None:
        if model is None or key not in sd:
            return
        if _is_fsdp_wrapped(model):
            set_model_state_dict(model, model_state_dict=sd[key])
        else:
            model.load_state_dict(sd[key])

    def load_state_dict(self, state_dict):
        self._apply(self.text_encoder, state_dict, "text_encoder")
        self._apply(self.transformer, state_dict, "transformer")
        self._apply(self.transformer_2, state_dict, "transformer_2")
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]
        self.batch_idx = state_dict.get("batch_idx", 0)
        if "rng_cpu" in state_dict:
            torch.random.set_rng_state(state_dict["rng_cpu"])
        if "rng_cuda" in state_dict:
            torch.cuda.set_rng_state(state_dict["rng_cuda"])


# ---------------------------------------------------------------------------
# Optimizer shards — one .pt per shard rank, written outside DCP so the DCP
# save plan stays small (resume reads only the matching rank's shard).
# ---------------------------------------------------------------------------


def save_optimizer_shard(
    path: Path,
    model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
) -> bool:
    if model is None or optimizer is None:
        return False
    state = (
        get_optimizer_state_dict(model, optimizer)
        if _is_fsdp_wrapped(model)
        else optimizer.state_dict()
    )
    torch.save(state, path)
    return True


def load_optimizer_shard(
    path: Path,
    model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
) -> bool:
    if model is None or optimizer is None or not path.exists():
        return False
    state = torch.load(path, map_location="cpu", weights_only=False)
    # Tolerate the legacy wrapper dict {"format_version", "process_group_size",
    # "optimizer_state"} written by older trainer versions.
    if isinstance(state, dict) and "optimizer_state" in state and "param_groups" not in state:
        state = state["optimizer_state"]
    if _is_fsdp_wrapped(model):
        set_optimizer_state_dict(model, optimizer, state)
    else:
        optimizer.load_state_dict(state)
    return True


# ---------------------------------------------------------------------------
# LoRA export — gather DTensors → CPU → PEFT-format safetensors.
# ---------------------------------------------------------------------------


def gather_full_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Collective. Returns the unsharded state_dict on rank 0; ``{}`` on others.

    MUST be called on every rank in the FSDP group — internally runs an
    all-gather and offloads to CPU. Safe to call on a non-FSDP module too
    (just returns a plain CPU state_dict).
    """
    if _is_fsdp_wrapped(model):
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        return get_model_state_dict(model, options=opts)
    # Non-FSDP: still move to CPU for parity with the FSDP path.
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def write_peft_lora_adapter(
    out_dir: Path,
    model: torch.nn.Module,
    full_state_dict: dict[str, torch.Tensor],
    adapter_name: str = "default",
) -> None:
    """Write ``adapter_model.safetensors`` + ``adapter_config.json`` for one PEFT adapter.

    Caller's responsibility:
      * invoke on rank 0 only,
      * pass a state_dict that has been gathered to CPU plain tensors,
      * ensure ``model`` carries the matching ``peft_config[adapter_name]``.
    """
    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_sd = get_peft_model_state_dict(
        model, state_dict=full_state_dict, adapter_name=adapter_name
    )
    adapter_sd = {k: v.detach().contiguous().cpu() for k, v in adapter_sd.items()}
    save_file(adapter_sd, str(out_dir / "adapter_model.safetensors"))

    cfg = model.peft_config[adapter_name]
    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg.__dict__)
    with open(out_dir / "adapter_config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Inference: load DCP into a non-FSDP diffusers pipeline (single process).
# Only handles **full fine-tune** checkpoints — for LoRA, convert offline with
# scripts/convert_dcp_to_lora.py and use ``pipe.load_lora_weights(...)``.
# ---------------------------------------------------------------------------


def load_dcp_into_pipeline(pipe, checkpoint_path: str, use_ema: bool = False) -> None:
    needs_cleanup = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo")
        needs_cleanup = True
    try:
        for name, dcp_path in _detect_layout(checkpoint_path).items():
            model = getattr(pipe, name, None)
            if model is None:
                continue
            if use_ema:
                _load_ema_into_module(name, model, dcp_path)
            else:
                _load_weights_into_module(name, model, dcp_path)
    finally:
        if needs_cleanup:
            dist.destroy_process_group()


def _detect_layout(checkpoint_path: str) -> dict[str, str]:
    """Return ``{module_name: dcp_dir}`` for flat or expert-parallel layouts."""
    p = Path(checkpoint_path)
    high, low = p / "high", p / "low"
    if (high / ".metadata").exists() or (low / ".metadata").exists():
        layout: dict[str, str] = {}
        if (high / ".metadata").exists():
            layout["transformer"] = str(high)
        if (low / ".metadata").exists():
            layout["transformer_2"] = str(low)
        return layout
    return {"transformer": checkpoint_path, "transformer_2": checkpoint_path}


def _load_weights_into_module(name: str, model: torch.nn.Module, dcp_path: str) -> None:
    placeholder = {pname: torch.empty_like(t, device="cpu") for pname, t in model.state_dict().items()}
    state = {"train_state": {name: placeholder}}
    dcp.load(state, checkpoint_id=dcp_path)
    model.load_state_dict(state["train_state"][name])
    logger.info("Loaded {} weights from {}", name, dcp_path)


def _load_ema_into_module(name: str, model: torch.nn.Module, dcp_path: str) -> None:
    shadow: dict[str, torch.Tensor] = {
        f"{name}.{pname}": torch.empty_like(p, device="cpu")
        for pname, p in model.named_parameters()
    }
    state = {"ema": {"shadow": shadow, "decay": torch.tensor(0.0)}}
    dcp.load(state, checkpoint_id=dcp_path)
    prefix = f"{name}."
    sd = {k.removeprefix(prefix): v for k, v in shadow.items() if k.startswith(prefix)}
    model.load_state_dict(sd, strict=False)
    logger.info("Loaded {} EMA weights from {}", name, dcp_path)


# ---------------------------------------------------------------------------
# Init-from-checkpoint — read a DCP checkpoint as weight-only initialisation
# for a fresh run.  Handles plain → LoRA remap (full-FT → LoRA workflow).
# ---------------------------------------------------------------------------


def read_dcp_to_flat_dict(path: Path | str) -> dict[str, torch.Tensor]:
    """Single-process read of an entire DCP checkpoint into a CPU flat dict.

    Uses the same internal primitive as
    ``torch.distributed.checkpoint.format_utils.dcp_to_torch_save`` (which
    reads via ``_EmptyStateDictLoadPlanner`` + ``no_dist=True``), so this is
    safe to call inside an already-initialised distributed context — it does
    not participate in any collective.  Callers typically invoke it only on
    rank 0 to avoid redundant I/O.
    """
    # Private API, but the same one torchtitan and PyTorch's own format_utils
    # use.  Stable across 2.4+ (no public replacement yet).
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    sd: dict = {}
    _load_state_dict(
        sd,
        storage_reader=FileSystemReader(str(path)),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return _flatten_nested(sd)


def _flatten_nested(d, prefix: str = "") -> dict[str, torch.Tensor]:
    """``{"a": {"b": t}} → {"a.b": t}``. Skips non-tensor leaves (scalars, etc.)."""
    out: dict[str, torch.Tensor] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten_nested(v, key))
        elif isinstance(v, torch.Tensor):
            out[key] = v
    return out


def extract_init_weights(
    flat: dict[str, torch.Tensor],
    model_key: str,
) -> tuple[dict[str, torch.Tensor], str]:
    """Pull a single model's init weights out of a flat DCP dict.

    Prefers ``ema.shadow.<model_key>.*`` over ``train_state.<model_key>.*``
    (EMA is a higher-quality init when available).  Keys in the returned dict
    are local to the model (e.g. ``blocks.0.attn1.to_q.weight``).

    Returns ``(weights, source_tag)`` where ``source_tag`` is ``"EMA"`` or
    ``"raw"``.  Raises ``RuntimeError`` if neither source is present.
    """
    ema_prefix = f"ema.shadow.{model_key}."
    raw_prefix = f"train_state.{model_key}."

    ema_keys = [k for k in flat if k.startswith(ema_prefix)]
    if ema_keys:
        return {k[len(ema_prefix):]: flat[k] for k in ema_keys}, "EMA"

    raw_keys = [k for k in flat if k.startswith(raw_prefix)]
    if raw_keys:
        return {k[len(raw_prefix):]: flat[k] for k in raw_keys}, "raw"

    raise RuntimeError(
        f"Checkpoint has no data for {model_key} "
        f"(looked for prefixes {ema_prefix!r} and {raw_prefix!r})."
    )


def remap_for_current_model(
    weights: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Remap source keys into the current model's parameter-name namespace.

    Four cases, decided from source-vs-model PEFT wrapping:

    - **Same architecture** (both plain, or both LoRA-wrapped): identity.
    - **Plain source → LoRA model** (full-FT → LoRA workflow): insert
      ``.base_layer.`` before the trailing ``.weight`` / ``.bias`` for keys
      that have a LoRA base_layer counterpart in the model.  Non-target
      modules (e.g. ``patch_embedding.weight``) pass through unchanged.
      ``lora_A/B`` keys are absent from the result; caller loads with
      ``strict=False`` so they keep PEFT's default init.
    - **LoRA source → plain model**: raises — requires an offline merge.
    - **LoRA source → LoRA model, different rank/targets**: raises.

    Shape mismatches on same-name keys also raise (caller should then either
    change ``model_path`` or convert offline).
    """
    model_keys = {k: v.shape for k, v in model.state_dict().items()}
    source_has_lora = _has_peft_segments(weights)
    model_has_lora = _has_peft_segments(model_keys)

    if source_has_lora and not model_has_lora:
        raise _lora_to_plain_error()
    if source_has_lora and model_has_lora:
        return _remap_lora_to_lora(weights, model_keys)
    if not source_has_lora and model_has_lora:
        return _remap_plain_to_lora(weights, model_keys)
    # Both plain.
    return _remap_same_arch(weights, model_keys)


def _has_peft_segments(keys) -> bool:
    """Match PEFT path segments (``base_layer`` or ``lora_*``) anywhere in the key."""
    for k in keys:
        for part in k.split("."):
            if part == "base_layer" or part.startswith("lora_"):
                return True
    return False


def _remap_same_arch(
    weights: dict[str, torch.Tensor], model_keys: dict[str, torch.Size]
) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in weights.items():
        if k not in model_keys:
            logger.warning("Source key {!r} has no counterpart in model; dropped.", k)
            continue
        if tuple(v.shape) != tuple(model_keys[k]):
            raise ValueError(
                f"Shape mismatch on {k}: source {tuple(v.shape)} vs model {tuple(model_keys[k])}."
            )
        out[k] = v
    return out


def _remap_plain_to_lora(
    weights: dict[str, torch.Tensor], model_keys: dict[str, torch.Size]
) -> dict[str, torch.Tensor]:
    out = {}
    for src_key, tensor in weights.items():
        # Non-target module — key exists as-is in the LoRA-wrapped model.
        if src_key in model_keys:
            if tuple(tensor.shape) != tuple(model_keys[src_key]):
                raise ValueError(
                    f"Shape mismatch on {src_key}: source {tuple(tensor.shape)} "
                    f"vs model {tuple(model_keys[src_key])}."
                )
            out[src_key] = tensor
            continue
        # LoRA-wrapped target — insert `.base_layer` before the suffix.
        for suffix in (".weight", ".bias"):
            if src_key.endswith(suffix):
                base_key = src_key[: -len(suffix)] + ".base_layer" + suffix
                if base_key in model_keys:
                    if tuple(tensor.shape) != tuple(model_keys[base_key]):
                        raise ValueError(
                            f"Shape mismatch on {base_key}: source {tuple(tensor.shape)} "
                            f"vs model {tuple(model_keys[base_key])}."
                        )
                    out[base_key] = tensor
                    break
        else:
            logger.warning(
                "Source key {!r} has no plain or .base_layer. counterpart in model; dropped.",
                src_key,
            )
    return out


def _remap_lora_to_lora(
    weights: dict[str, torch.Tensor], model_keys: dict[str, torch.Size]
) -> dict[str, torch.Tensor]:
    out = {}
    mismatches: list[str] = []
    for k, v in weights.items():
        if k not in model_keys:
            mismatches.append(f"  - key missing in model: {k}")
            continue
        if tuple(v.shape) != tuple(model_keys[k]):
            mismatches.append(
                f"  - {k}: source {tuple(v.shape)} vs model {tuple(model_keys[k])}"
            )
            continue
        out[k] = v
    if mismatches:
        # Most common cause: LoRA rank / target_modules changed between runs.
        head = "\n".join(mismatches[:10])
        more = f"\n  ...and {len(mismatches) - 10} more" if len(mismatches) > 10 else ""
        raise ValueError(
            "LoRA checkpoint does not match current LoRA architecture.\n"
            "Likely cause: lora_rank or target_modules differ between runs.\n"
            f"First mismatches:\n{head}{more}\n"
            "Fix: merge the source LoRA into its base (see the LoRA→plain "
            "error message below for the recipe), then re-LoRA fresh from "
            "the merged model_path."
        )
    return out


def _lora_to_plain_error() -> ValueError:
    return ValueError(
        "Cannot init a full-FT model from a LoRA-trained checkpoint — the LoRA "
        "delta must be merged into base weights first. Recipe:\n"
        "  1. uv run python scripts/convert_dcp_to_lora.py \\\n"
        "         --config <original_train_config.yaml> \\\n"
        "         --checkpoint <ckpt_path> \\\n"
        "         --output <adapter_out>\n"
        "  2. Use PEFT merge_and_unload to fold the adapter into base weights:\n"
        "         from diffusers.models import WanTransformer3DModel\n"
        "         from peft import PeftModel\n"
        "         base = WanTransformer3DModel.from_pretrained('<orig>/transformer')\n"
        "         merged = PeftModel.from_pretrained(base, '<adapter_out>/transformer').merge_and_unload()\n"
        "         merged.save_pretrained('<merged_model>/transformer')\n"
        "  3. Point cfg.model_path at <merged_model> and start a fresh run "
        "     (no resume_from)."
    )
