"""Optimizer factory.

When using Muon, parameters are split by dimensionality:
  - >= 2D parameters → Muon (Newton-Schulz orthogonalization)
  - < 2D parameters (biases, norms) → AdamW fallback

The first optimizer returned is the "primary" used for DCP checkpointing.
Any extra optimizers (the AdamW fallback for 1D params) are stepped during
training but not checkpointed — their state is cheap to recompute on resume.
"""

from collections.abc import Iterable

import torch
from loguru import logger
from torch.nn import Parameter

from src.trainer.config import TrainConfig


def build_optimizer(
    params: Iterable[Parameter], cfg: TrainConfig
) -> tuple[torch.optim.Optimizer, list[torch.optim.Optimizer]]:
    """Create optimizer(s) for the given parameters.

    Returns:
        (primary_optimizer, extra_optimizers) — primary is used for DCP
        checkpointing; extras are stepped but not checkpointed.
    """
    params = list(params)

    if cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=cfg.adamw_betas,
            fused=cfg.adamw_fused,
        )
        return opt, []

    elif cfg.optimizer == "muon":
        params_2d = [p for p in params if p.ndim >= 2]
        params_1d = [p for p in params if p.ndim < 2]

        if not params_2d:
            raise ValueError("Muon requires at least some >= 2D parameters, but none were found")

        muon = torch.optim.Muon(
            params_2d,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            momentum=cfg.muon_momentum,
            nesterov=cfg.muon_nesterov,
            ns_steps=cfg.muon_ns_steps,
            adjust_lr_fn=cfg.muon_adjust_lr_fn,
        )

        extras: list[torch.optim.Optimizer] = []
        if params_1d:
            adamw_fallback = torch.optim.AdamW(
                params_1d,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                betas=cfg.adamw_betas,
                fused=cfg.adamw_fused,
            )
            extras.append(adamw_fallback)
            logger.info(
                "Muon: {} params (>=2D) + AdamW fallback: {} params (<2D)",
                len(params_2d),
                len(params_1d),
            )
        else:
            logger.info("Muon: {} params (all >=2D, no fallback needed)", len(params_2d))

        return muon, extras

    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
