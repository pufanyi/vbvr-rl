"""Cheap, config-aware runtime preflight for GRPO launchers."""

from __future__ import annotations

import argparse
import socket
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from src.eval.vbvr_runtime import VBVRScorerRuntimeError, validate_vbvr_scorer_runtime


def _selected_reward(argv: Sequence[str] | None = None) -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    parser.add_argument("--grpo_reward_fn")
    args, _ = parser.parse_known_args(argv)

    config: dict = {}
    if args.config:
        config = yaml.safe_load(Path(args.config).read_text()) or {}
        if not isinstance(config, dict):
            raise TypeError(f"GRPO config must contain a mapping: {args.config}")
    return args.grpo_reward_fn or config.get("grpo_reward_fn")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        reward = _selected_reward(argv)
    except Exception as exc:
        print(f"[error] Could not resolve the GRPO runtime contract: {exc}", file=sys.stderr)
        return 1
    if reward != "vbvr_rule":
        print(f"[preflight] No VBVR scorer runtime required for grpo_reward_fn={reward!r}")
        return 0

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
