#!/usr/bin/env python3
"""Watch 5B VBVR SFT checkpoints and run 256x256x161 lmms-eval jobs.

The watcher scans checkpoint roots periodically, evaluates any complete DCP
checkpoint that has not produced a result yet, and keeps a JSON state file so it
can be restarted without duplicating finished runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = [
    "storage/checkpoints/sft_vbvr_5b_256x256x161_full_lr_1e-5",
    "storage/checkpoints/sft_vbvr_5b_256x256x161_full_lr_5e-6",
]
DEFAULT_STATE = "storage/eval_watch/vbvr_5b_256x256x161/state.json"
DEFAULT_LOG_DIR = "storage/eval_watch/vbvr_5b_256x256x161/logs"
DEFAULT_EVAL_ROOT = "storage/lmms_eval/vbvr_5b_256x256x161"
DEFAULT_CONVERTED_ROOT = "storage/models/dcp_converted_5b"
BASE_MODEL = "storage/models/Wan2.2-TI2V-5B-Diffusers"


def _now() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"created_at": _now(), "entries": {}}
    with path.open() as f:
        return json.load(f)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _complete_dcp_checkpoint(path: Path) -> bool:
    if (path / ".metadata").is_file():
        return True
    has_high_dir = (path / "high").is_dir()
    has_low_dir = (path / "low").is_dir()
    if has_high_dir and has_low_dir:
        return (path / "high/.metadata").is_file() and (path / "low/.metadata").is_file()
    return (path / "high/.metadata").is_file() or (path / "low/.metadata").is_file()


_STEP_RE = re.compile(r"^checkpoint-(\d+)$")
_EPOCH_RE = re.compile(r"^checkpoint-epoch(\d+)$")


def _checkpoint_sort_key(path: Path) -> tuple[str, int, int, str]:
    run = path.parent.name
    name = path.name
    step = _STEP_RE.match(name)
    if step is not None:
        return run, 0, int(step.group(1)), name
    epoch = _EPOCH_RE.match(name)
    if epoch is not None:
        return run, 1, int(epoch.group(1)), name
    return run, 2, 0, name


def _discover_checkpoints(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("checkpoint-") and _complete_dcp_checkpoint(child):
                found.append(child.resolve())
    return sorted(found, key=_checkpoint_sort_key)


def _safe_name_for_checkpoint(checkpoint: Path) -> str:
    try:
        rel = checkpoint.relative_to(REPO_ROOT / "storage/checkpoints")
        return str(rel).replace("/", "_")
    except ValueError:
        return f"{checkpoint.parent.name}_{checkpoint.name}"


def _result_path(eval_dir: Path) -> Path:
    return eval_dir / "submissions/vbvr_eval_results.json"


def _gpu_count() -> int:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        if cuda_visible == "NoDevFiles":
            return 0
        return len([item for item in cuda_visible.split(",") if item.strip()])
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return 1
    return len([line for line in out.splitlines() if line.strip().startswith("GPU ")]) or 1


def _free_gpu_count(max_used_mb: int) -> int:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return _gpu_count()
    free = 0
    for line in out.splitlines():
        try:
            used = int(line.strip())
        except ValueError:
            continue
        if used <= max_used_mb:
            free += 1
    return free


def _run_eval(
    checkpoint: Path,
    *,
    eval_dir: Path,
    converted_root: Path,
    log_dir: Path,
    data_parallel: int,
    num_frames: int,
    height: int,
    width: int,
    num_inference_steps: int,
    enable_torch_compile: str,
) -> tuple[int, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{_safe_name_for_checkpoint(checkpoint)}.log"
    env = os.environ.copy()
    env.update(
        {
            "CHECKPOINT_ROOT": str((REPO_ROOT / "storage/checkpoints").resolve()),
            "CONVERTED_ROOT": str(converted_root.resolve()),
            "BASE_MODEL": str((REPO_ROOT / BASE_MODEL).resolve()),
            "EVAL_OUTPUT_DIR": str(eval_dir.resolve()),
            "DATA_PARALLEL": str(data_parallel),
            "NUM_FRAMES": str(num_frames),
            "HEIGHT": str(height),
            "WIDTH": str(width),
            "NUM_INFERENCE_STEPS": str(num_inference_steps),
            "ENABLE_TORCH_COMPILE": enable_torch_compile,
            "TORCH_DTYPE": "bfloat16",
            "USE_EMA": "1",
            "MERGE_LORA": "1",
            "SAFE_SERIALIZATION": "1",
        }
    )
    cmd = ["fish", "scripts/eval/lmms_eval_checkpoint.fish", str(checkpoint)]
    with log_path.open("a") as log:
        log.write(f"\n===== {_now()} START {' '.join(cmd)} =====\n")
        log.write(
            "env: "
            f"DATA_PARALLEL={data_parallel} HEIGHT={height} WIDTH={width} "
            f"NUM_FRAMES={num_frames} NUM_INFERENCE_STEPS={num_inference_steps} "
            f"EVAL_OUTPUT_DIR={eval_dir} CONVERTED_ROOT={converted_root}\n"
        )
        log.flush()
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        return_code = proc.wait()
        log.write(f"===== {_now()} END return_code={return_code} =====\n")
    return return_code, log_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", default=[], help="Checkpoint run root to scan")
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--eval-root", default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--converted-root", default=DEFAULT_CONVERTED_ROOT)
    parser.add_argument("--poll-seconds", type=int, default=1800)
    parser.add_argument("--duration-hours", type=float, default=24.0)
    parser.add_argument("--data-parallel", type=int, default=0, help="0 = use visible GPU count")
    parser.add_argument("--min-free-gpus", type=int, default=0, help="0 = data_parallel")
    parser.add_argument("--max-used-mb-for-free", type=int, default=2048)
    parser.add_argument("--num-frames", type=int, default=161)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--enable-torch-compile", default="True")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    run_roots = [REPO_ROOT / item for item in (args.run_root or DEFAULT_RUNS)]
    state_path = REPO_ROOT / args.state
    log_dir = REPO_ROOT / args.log_dir
    eval_root = REPO_ROOT / args.eval_root
    converted_root = REPO_ROOT / args.converted_root
    data_parallel = args.data_parallel if args.data_parallel > 0 else _gpu_count()
    min_free_gpus = args.min_free_gpus if args.min_free_gpus > 0 else data_parallel
    deadline = time.monotonic() + args.duration_hours * 3600.0

    state = _load_json(state_path)
    state.update(
        {
            "updated_at": _now(),
            "watch_until": datetime.fromtimestamp(time.time() + args.duration_hours * 3600.0)
            .astimezone()
            .isoformat(timespec="seconds"),
            "poll_seconds": args.poll_seconds,
            "data_parallel": data_parallel,
            "num_frames": args.num_frames,
            "height": args.height,
            "width": args.width,
            "run_roots": [str(path) for path in run_roots],
        }
    )
    _save_json(state_path, state)

    print(f"[{_now()}] watcher started")
    print(f"state={state_path}")
    print(f"log_dir={log_dir}")
    print(f"eval_root={eval_root}")
    print(f"converted_root={converted_root}")
    print(f"data_parallel={data_parallel} shape={args.width}x{args.height}x{args.num_frames}")

    while True:
        state = _load_json(state_path)
        state["updated_at"] = _now()
        entries: dict[str, Any] = state.setdefault("entries", {})
        checkpoints = _discover_checkpoints(run_roots)
        pending: list[Path] = []
        for checkpoint in checkpoints:
            key = str(checkpoint)
            name = _safe_name_for_checkpoint(checkpoint)
            eval_dir = eval_root / name
            result = _result_path(eval_dir)
            entry = entries.get(key, {})
            if result.is_file():
                entry.update(
                    {
                        "status": "done",
                        "checkpoint": key,
                        "eval_dir": str(eval_dir),
                        "result_path": str(result),
                        "updated_at": _now(),
                    }
                )
                entries[key] = entry
                continue
            if entry.get("status") == "running":
                entry["status"] = "failed"
                entry["error"] = "watcher restarted or previous process exited before result appeared"
                entry["updated_at"] = _now()
                entries[key] = entry
            if entry.get("status") == "failed" and not args.retry_failed:
                continue
            if entry.get("status") != "done":
                pending.append(checkpoint)

        state["last_scan_at"] = _now()
        state["discovered_count"] = len(checkpoints)
        state["pending_count"] = len(pending)
        _save_json(state_path, state)

        if pending:
            checkpoint = pending[0]
            key = str(checkpoint)
            name = _safe_name_for_checkpoint(checkpoint)
            eval_dir = eval_root / name
            converted_dir = converted_root / name
            free_gpus = _free_gpu_count(args.max_used_mb_for_free)
            if free_gpus < min_free_gpus:
                print(f"[{_now()}] waiting for GPUs: free={free_gpus} required={min_free_gpus}; pending={len(pending)}")
                time.sleep(args.poll_seconds)
                continue

            print(f"[{_now()}] evaluating {checkpoint} -> {eval_dir}")
            entries[key] = {
                "status": "running",
                "checkpoint": key,
                "eval_dir": str(eval_dir),
                "converted_dir": str(converted_dir),
                "started_at": _now(),
            }
            state["updated_at"] = _now()
            _save_json(state_path, state)

            return_code, log_path = _run_eval(
                checkpoint,
                eval_dir=eval_dir,
                converted_root=converted_root,
                log_dir=log_dir,
                data_parallel=data_parallel,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                enable_torch_compile=args.enable_torch_compile,
            )

            state = _load_json(state_path)
            entries = state.setdefault("entries", {})
            result = _result_path(eval_dir)
            if return_code == 0 and result.is_file():
                status = "done"
                error = None
            else:
                status = "failed"
                error = f"return_code={return_code}; result_exists={result.is_file()}"
            entries[key] = {
                **entries.get(key, {}),
                "status": status,
                "checkpoint": key,
                "eval_dir": str(eval_dir),
                "converted_dir": str(converted_dir),
                "result_path": str(result),
                "log_path": str(log_path),
                "finished_at": _now(),
                "return_code": return_code,
            }
            if error is not None:
                entries[key]["error"] = error
            else:
                entries[key].pop("error", None)
            state["updated_at"] = _now()
            _save_json(state_path, state)
            print(f"[{_now()}] finished {checkpoint.name}: status={status} log={log_path}")
            continue

        if time.monotonic() >= deadline:
            print(f"[{_now()}] watch duration reached and no pending checkpoints remain")
            state["finished_at"] = _now()
            state["updated_at"] = _now()
            _save_json(state_path, state)
            return 0

        print(f"[{_now()}] no pending checkpoints; sleeping {args.poll_seconds}s")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
