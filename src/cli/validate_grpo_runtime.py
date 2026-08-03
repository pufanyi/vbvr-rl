"""Cheap, config-aware runtime preflight for GRPO launchers."""

from __future__ import annotations

import argparse
import socket
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from src.eval.vbvr_runtime import VBVRScorerRuntimeError, validate_vbvr_scorer_runtime


def _selected_runtime(argv: Sequence[str] | None = None) -> tuple[str | None, str | None]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    parser.add_argument("--grpo_reward_fn")
    parser.add_argument("--attention_backend")
    args, _ = parser.parse_known_args(argv)

    config: dict = {}
    if args.config:
        config = yaml.safe_load(Path(args.config).read_text()) or {}
        if not isinstance(config, dict):
            raise TypeError(f"GRPO config must contain a mapping: {args.config}")
    return (
        args.grpo_reward_fn or config.get("grpo_reward_fn"),
        args.attention_backend or config.get("attention_backend"),
    )


def _selected_reward(argv: Sequence[str] | None = None) -> str | None:
    """Backward-compatible reward-only view used by callers and tests."""
    return _selected_runtime(argv)[0]


def main(argv: Sequence[str] | None = None) -> int:
    try:
        reward, attention = _selected_runtime(argv)
    except Exception as exc:
        print(f"[error] Could not resolve the GRPO runtime contract: {exc}", file=sys.stderr)
        return 1
    if reward == "vbvr_rule":
        try:
            report = validate_vbvr_scorer_runtime()
        except VBVRScorerRuntimeError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1
        key_versions = ", ".join(
            f"{name}={report['distributions'][name]}"
            for name in ("opencv-python", "opencv-python-headless", "numpy", "scipy", "scikit-image")
        )
        print(
            "[preflight] VBVR scorer runtime passed on "
            f"{socket.gethostname()}: sha256={report['sha256']} "
            f"cv2={report['modules']['cv2']} HoughLinesP={report['probes']['hough_lines_p_layout']}"
        )
        print(f"[preflight] VBVR scorer key distributions: {key_versions}")
    else:
        print(f"[preflight] No VBVR scorer runtime required for grpo_reward_fn={reward!r}")

    if attention is not None:
        try:
            from src.trainer.utils import prepare_diffusers_attention_backend

            prepare_diffusers_attention_backend(attention)
        except Exception as exc:
            print(f"[error] Could not prepare attention backend {attention!r}: {exc}", file=sys.stderr)
            return 1
        print(f"[preflight] Attention backend ready on {socket.gethostname()}: {attention}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
