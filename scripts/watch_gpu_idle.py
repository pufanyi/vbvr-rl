#!/usr/bin/env python3
"""Watch GPU utilization and launch a command after a sustained idle window.

Default behavior:
  - poll every 2 minutes
  - require all GPUs to stay below 10% utilization
  - require 30 consecutive minutes of low utilization
  - then launch:
      torchrun --nproc_per_node=8 -m src.cli.train_i2v --config configs/train_sft_maze.yaml

Examples:
  python scripts/watch_gpu_idle.py
  python scripts/watch_gpu_idle.py --dry-run
  python scripts/watch_gpu_idle.py --command "echo triggered"
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

DEFAULT_COMMAND = "torchrun --nproc_per_node=8 -m src.cli.train_i2v --config configs/train_sft_maze.yaml"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a command after all GPUs remain below a utilization threshold for a sustained period."
    )
    parser.add_argument(
        "--command",
        default=DEFAULT_COMMAND,
        help="Command to execute once the idle condition is satisfied.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=120,
        help="Polling interval in seconds. Default: 120.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="All GPUs must stay strictly below this utilization percentage. Default: 10.",
    )
    parser.add_argument(
        "--idle-minutes",
        type=float,
        default=30.0,
        help="Required consecutive low-utilization duration in minutes. Default: 30.",
    )
    parser.add_argument(
        "--workdir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Working directory used when launching the command. Default: repository root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not execute the command; only report when it would have been launched.",
    )
    return parser.parse_args()


def query_gpu_utilizations() -> list[float]:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi not found in PATH")

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    utils: list[float] = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        index_str, util_str = [part.strip() for part in line.split(",", maxsplit=1)]
        try:
            _ = int(index_str)
            utils.append(float(util_str))
        except ValueError as exc:
            raise RuntimeError(f"Unexpected nvidia-smi output line: {line!r}") from exc

    if not utils:
        raise RuntimeError("No GPUs reported by nvidia-smi")
    return utils


def launch_command(command: str, workdir: str, dry_run: bool) -> int | None:
    if dry_run:
        log(f"[dry-run] Idle window satisfied. Would run: {command}")
        return None

    proc = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=workdir,
        start_new_session=True,
    )
    log(f"Launched command with PID {proc.pid}: {command}")
    return proc.pid


def main() -> int:
    args = parse_args()

    interval = args.interval_seconds
    if interval <= 0:
        raise SystemExit("--interval-seconds must be > 0")
    if args.idle_minutes <= 0:
        raise SystemExit("--idle-minutes must be > 0")

    idle_required_seconds = args.idle_minutes * 60.0
    workdir = str(Path(args.workdir).resolve())
    low_since: float | None = None

    log(
        "Starting GPU idle watchdog: "
        f"threshold<{args.threshold:.1f}%, "
        f"interval={interval}s, "
        f"idle_window={args.idle_minutes:.1f}min, "
        f"workdir={workdir}"
    )
    log(f"Target command: {args.command}")

    while True:
        try:
            utils = query_gpu_utilizations()
        except Exception as exc:
            log(f"GPU query failed: {exc}. Retrying in {interval}s.")
            time.sleep(interval)
            continue

        all_low = all(util < args.threshold for util in utils)
        util_str = ", ".join(f"{u:.0f}%" for u in utils)

        now = time.monotonic()
        if all_low:
            if low_since is None:
                low_since = now
                elapsed = 0.0
            else:
                elapsed = now - low_since

            remaining = max(idle_required_seconds - elapsed, 0.0)
            log(
                f"All GPUs below threshold [{util_str}]. "
                f"Idle streak={elapsed / 60.0:.1f}min, remaining={remaining / 60.0:.1f}min."
            )

            if elapsed >= idle_required_seconds:
                launch_command(args.command, workdir, args.dry_run)
                return 0
        else:
            if low_since is not None:
                elapsed = now - low_since
                log(f"Idle streak reset after {elapsed / 60.0:.1f}min because GPU utilizations are [{util_str}].")
            else:
                log(f"GPUs are active [{util_str}].")
            low_since = None

        time.sleep(interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted by user.")
        raise SystemExit(130) from None
