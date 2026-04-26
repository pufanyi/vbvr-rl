"""Wan2.2 I2V training with on-policy correction loss.

Usage:
    .venv/bin/torchrun --nproc_per_node=8 -m src.cli.train_i2v_correction \
        --config configs/train_i2v_correction.yaml
"""

import argparse
from pathlib import Path

import yaml

from src.trainer import CorrectionConfig, I2VCorrectionTrainer


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 I2V + on-policy correction training")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config file")
    for name, field_info in CorrectionConfig.model_fields.items():
        if field_info.annotation is bool:
            parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=None)
        elif field_info.annotation is int:
            parser.add_argument(f"--{name}", type=int, default=None)
        elif field_info.annotation is float:
            parser.add_argument(f"--{name}", type=float, default=None)
        else:
            parser.add_argument(f"--{name}", type=str, default=None)
    args = parser.parse_args()

    cfg_dict: dict = {}
    if args.config:
        cfg_dict = yaml.safe_load(Path(args.config).read_text()) or {}
    for name in CorrectionConfig.model_fields:
        v = getattr(args, name, None)
        if v is not None:
            cfg_dict[name] = v

    cfg = CorrectionConfig(**cfg_dict)
    trainer = I2VCorrectionTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
