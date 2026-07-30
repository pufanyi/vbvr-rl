"""Run a bounded single-rank GRPO config and prove trainable weights changed.

This is stricter than checking for a successful process exit: it snapshots all
trainable tensors before ``trainer.train()`` and fails if the optimizer step
leaves them byte-identical.

Example:
    .venv/bin/torchrun --standalone --nproc_per_node=1 \
        -m scripts.dev.validate_grpo_parameter_update \
        --config configs/train_dancegrpo_vbvr_pro_5b_512x512x81_official_base_smoke_1gpu.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from src.trainer import DanceGRPOTrainer, RLConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _named_trainable_parameters(trainer: DanceGRPOTrainer):
    modules = (
        ("text_encoder", trainer.model.text_encoder),
        ("transformer", trainer.model.transformer),
        ("transformer_2", trainer.model.transformer_2),
    )
    for module_name, module in modules:
        if module is None:
            continue
        for parameter_name, parameter in module.named_parameters():
            if parameter.requires_grad:
                yield f"{module_name}.{parameter_name}", parameter


def main() -> None:
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("validate_grpo_parameter_update.py is intentionally single-rank")

    cfg = RLConfig(**(yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}))
    if cfg.max_steps != 1:
        raise ValueError(f"parameter-update validation requires max_steps: 1, got {cfg.max_steps}")

    trainer = DanceGRPOTrainer(cfg)
    trainable = list(_named_trainable_parameters(trainer))
    if not trainable:
        raise RuntimeError("No trainable parameters found")
    before = {name: parameter.detach().cpu().clone() for name, parameter in trainable}

    try:
        trainer.train()
        changed_tensors = 0
        changed_elements = 0
        max_abs_delta = 0.0
        squared_delta = 0.0
        for name, parameter in trainable:
            delta = parameter.detach().cpu().float() - before[name].float()
            tensor_max = delta.abs().max().item()
            if tensor_max > 0:
                changed_tensors += 1
                changed_elements += int(torch.count_nonzero(delta).item())
            max_abs_delta = max(max_abs_delta, tensor_max)
            squared_delta += delta.double().square().sum().item()

        summary = {
            "trainable_tensors": len(trainable),
            "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
            "changed_tensors": changed_tensors,
            "changed_elements": changed_elements,
            "max_abs_delta": max_abs_delta,
            "delta_l2": math.sqrt(squared_delta),
        }
        print("PARAMETER_UPDATE " + json.dumps(summary, sort_keys=True))
        if changed_tensors == 0:
            raise RuntimeError("Optimizer step completed but no trainable parameter changed")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
