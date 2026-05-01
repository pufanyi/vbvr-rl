#!/usr/bin/env python3
"""Monitor Wan-Trainer checkpoints and run VBVR evals.

The monitor is intentionally conservative:
- it only evaluates complete DCP checkpoints;
- it reuses existing complete eval outputs;
- it records per-run metadata next to the eval output;
- it writes separate vLLM-Omni and FastVideo result tables.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = REPO_ROOT / "storage/checkpoints"
CONVERTED_ROOT = REPO_ROOT / "storage/models/dcp_converted"
BASE_MODEL = REPO_ROOT / "storage/models/Wan2.2-I2V-A14B-Diffusers"
FASTVIDEO_REPO = Path("/mnt/umm/users/pufanyi/workspace/lmms-eval")
VLLM_REPO = Path("/mnt/umm/users/pufanyi/workspace/lmms-eval-vllm")
FASTVIDEO_OUTPUT_ROOT = REPO_ROOT / "storage/lmms_eval"
VLLM_OUTPUT_ROOT = REPO_ROOT / "storage/eval_out"
LOG_ROOT = REPO_ROOT / "storage/eval_logs/vbvr_monitor"
REPORT_ROOT = REPO_ROOT / "storage/eval_reports"
STATE_PATH = REPORT_ROOT / "vbvr_monitor_state.json"
LOCK_PATH = REPORT_ROOT / "vbvr_monitor.lock"
FASTVIDEO_TASK_PATH = REPO_ROOT / "storage/eval_tasks"
VLLM_TASK_PATH = REPO_ROOT / "storage/eval_tasks_vllm"
VBVR_ROOT = REPO_ROOT / "storage/datasets/VBVR-Bench"

SCORE_COLUMNS = [
    ("overall", "overall"),
    ("In_Domain", "in"),
    ("Out_of_Domain", "out"),
    ("abstraction", "abs"),
    ("knowledge", "know"),
    ("perception", "perc"),
    ("spatiality", "spat"),
    ("transformation", "trans"),
]


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    safe_name: str
    model_dir: Path
    complete: bool
    reason: str


@dataclass
class ResultRow:
    checkpoint: str
    backend: str
    source_checkpoint: str
    model_dir: str
    output_path: str
    result_path: str
    summary: dict[str, Any]
    status: str = "complete"
    note: str | None = None
    returncode: int | None = None
    duration_seconds: float | None = None
    started_at: str | None = None
    finished_at: str | None = None
    log_path: str | None = None


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def hms(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds_i = int(round(seconds))
    h, rem = divmod(seconds_i, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def score(summary: dict[str, Any], key: str) -> float | None:
    if key == "overall":
        return summary.get("overall")
    value = summary.get(key)
    if isinstance(value, dict):
        return value.get("score")
    return None


def fmt_score(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"runs": []}
    try:
        data = load_json(STATE_PATH)
    except Exception:
        return {"runs": []}
    if not isinstance(data, dict):
        return {"runs": []}
    data.setdefault("runs", [])
    return data


def save_state(state: dict[str, Any]) -> None:
    atomic_write_json(STATE_PATH, state)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class FileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                pid = int(self.path.read_text().strip().split()[0])
            except Exception:
                pid = -1
            if pid > 0 and pid_alive(pid):
                raise RuntimeError(f"monitor already running with pid {pid}: {self.path}")
            self.path.unlink(missing_ok=True)

        self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(self.fd, f"{os.getpid()} {datetime.now().isoformat()}\n".encode())
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.path.unlink(missing_ok=True)


def discover_checkpoints() -> list[Checkpoint]:
    candidates: set[Path] = set()
    if CHECKPOINT_ROOT.exists():
        candidates.update(path for path in CHECKPOINT_ROOT.glob("*/checkpoint-*") if path.is_dir())
        for metadata in CHECKPOINT_ROOT.rglob(".metadata"):
            parent = metadata.parent
            ckpt = parent.parent if parent.name in {"high", "low"} else parent
            candidates.add(ckpt)

    checkpoints: list[Checkpoint] = []
    for ckpt in sorted(candidates):
        try:
            rel = ckpt.relative_to(CHECKPOINT_ROOT)
        except ValueError:
            rel = Path(ckpt.parent.name) / ckpt.name
        safe_name = "_".join(rel.parts)
        high_meta = ckpt / "high/.metadata"
        low_meta = ckpt / "low/.metadata"
        top_meta = ckpt / ".metadata"
        if top_meta.exists():
            complete = True
            reason = "top-level DCP metadata"
        elif high_meta.exists() and low_meta.exists():
            complete = True
            reason = "high/low DCP metadata"
        elif high_meta.exists() or low_meta.exists():
            complete = False
            missing = "low" if high_meta.exists() else "high"
            reason = f"incomplete high/low DCP checkpoint; missing {missing}/.metadata"
        else:
            complete = False
            reason = "no DCP metadata"
        checkpoints.append(
            Checkpoint(
                path=ckpt,
                safe_name=safe_name,
                model_dir=CONVERTED_ROOT / safe_name,
                complete=complete,
                reason=reason,
            )
        )
    return checkpoints


def load_complete_summary(path: Path) -> dict[str, Any] | None:
    try:
        summary = load_json(path).get("summary")
    except Exception:
        return None
    if not isinstance(summary, dict):
        return None
    if summary.get("n") != 500:
        return None
    return summary


def metadata_for_output(output_dir: Path) -> dict[str, Any]:
    for name in ("monitor_metadata.json", "metadata.json"):
        path = output_dir / name
        if path.exists():
            try:
                data = load_json(path)
            except Exception:
                continue
            if isinstance(data, dict):
                return data
    return {}


def result_matches_backend(output_dir: Path, safe_name: str, backend: str) -> bool:
    name = output_dir.name
    if backend == "vllm":
        cache_prefixes = (
            f"vbvr_vllm_omni_{safe_name}_",
            f"vbvr_vllm_omni_cache_dit_{safe_name}_",
        )
        return name.startswith(cache_prefixes) and "nocache" not in name
    if backend == "fastvideo":
        return safe_name in name and output_dir.is_relative_to(FASTVIDEO_OUTPUT_ROOT)
    raise ValueError(f"unknown backend: {backend}")


def iter_result_paths(root: Path) -> Iterable[Path]:
    """Yield known lmms-eval result locations without recursing into videos."""
    if not root.exists():
        return
    for output_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        candidates = [
            output_dir / "submissions/vbvr_eval_results.json",
            output_dir / "logs/submissions/vbvr_eval_results.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                yield candidate


def discover_results_for(ckpt: Checkpoint, backend: str, state: dict[str, Any]) -> list[ResultRow]:
    root = VLLM_OUTPUT_ROOT if backend == "vllm" else FASTVIDEO_OUTPUT_ROOT
    state_by_output = {
        run.get("output_path"): run
        for run in state.get("runs", [])
        if isinstance(run, dict) and run.get("backend") == backend
    }
    rows: list[ResultRow] = []
    for result_path in iter_result_paths(root):
        output_dir = result_path.parents[1]
        if output_dir.name == "logs":
            output_dir = output_dir.parent
        if not result_matches_backend(output_dir, ckpt.safe_name, backend):
            continue
        summary = load_complete_summary(result_path)
        if summary is None:
            continue
        meta = metadata_for_output(output_dir)
        run_state = state_by_output.get(str(output_dir), {})
        rows.append(
            ResultRow(
                checkpoint=ckpt.safe_name,
                backend=backend,
                source_checkpoint=str(ckpt.path),
                model_dir=str(ckpt.model_dir),
                output_path=str(output_dir),
                result_path=str(result_path),
                summary=summary,
                duration_seconds=meta.get("duration_seconds") or run_state.get("duration_seconds"),
                started_at=meta.get("started_at") or run_state.get("started_at"),
                finished_at=meta.get("finished_at") or run_state.get("finished_at"),
                log_path=meta.get("log_path") or run_state.get("log_path"),
            )
        )
    rows.sort(key=lambda row: Path(row.result_path).stat().st_mtime, reverse=True)
    return rows


def best_result(ckpt: Checkpoint, backend: str, state: dict[str, Any]) -> ResultRow | None:
    rows = discover_results_for(ckpt, backend, state)
    return rows[0] if rows else None


def log_tail_text(path: str | Path | None, max_bytes: int = 200_000) -> str:
    if not path:
        return ""
    log_path = Path(path)
    if not log_path.exists():
        return ""
    with log_path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        return f.read().decode("utf-8", errors="replace")


def failure_note(record: dict[str, Any]) -> str:
    text = log_tail_text(record.get("log_path")).lower()
    rc = record.get("returncode")
    if "latent_model_input contains nan" in text:
        return "latent_model_input contains nan"
    if "out of memory" in text or "cuda oom" in text or "torch.outofmemoryerror" in text:
        return "out of memory"
    if "traceback" in text:
        return f"traceback, rc={rc}"
    if rc not in (None, 0):
        return f"rc={rc}"
    return "missing or incomplete result"


def failed_runs_for(ckpt: Checkpoint, backend: str, state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in state.get("runs", []):
        if not isinstance(run, dict):
            continue
        if run.get("checkpoint") != ckpt.safe_name or run.get("backend") != backend:
            continue
        result_path = Path(run.get("result_path", ""))
        has_summary = load_complete_summary(result_path) is not None
        if run.get("returncode") == 0 and has_summary:
            continue
        rows.append(run)
    rows.sort(key=lambda run: run.get("finished_at") or run.get("started_at") or "", reverse=True)
    return rows


def latest_failed_result(ckpt: Checkpoint, backend: str, state: dict[str, Any]) -> ResultRow | None:
    failed = failed_runs_for(ckpt, backend, state)
    if not failed:
        return None
    run = failed[0]
    return ResultRow(
        checkpoint=ckpt.safe_name,
        backend=backend,
        source_checkpoint=str(ckpt.path),
        model_dir=str(ckpt.model_dir),
        output_path=str(run.get("output_path", "")),
        result_path=str(run.get("result_path", "")),
        summary={},
        status="failed",
        note=failure_note(run),
        returncode=run.get("returncode"),
        duration_seconds=run.get("duration_seconds"),
        started_at=run.get("started_at"),
        finished_at=run.get("finished_at"),
        log_path=run.get("log_path"),
    )


def all_best_results(checkpoints: Iterable[Checkpoint], backend: str, state: dict[str, Any]) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for ckpt in checkpoints:
        if not ckpt.complete:
            continue
        row = best_result(ckpt, backend, state)
        if row is not None:
            rows.append(row)
            continue
        failed = latest_failed_result(ckpt, backend, state)
        if failed is not None:
            rows.append(failed)
    rows.sort(key=lambda row: row.checkpoint)
    return rows


def markdown_table(rows: list[ResultRow], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| checkpoint | status | overall | in | out | abs | know | perc | spat | trans | n | duration | finished_at | output | note |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        summary = row.summary
        values = [fmt_score(score(summary, key)) for key, _ in SCORE_COLUMNS]
        out = row.output_path
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.checkpoint}`",
                    row.status,
                    *values,
                    str(summary.get("n", "-")),
                    hms(row.duration_seconds),
                    row.finished_at or "-",
                    f"`{out}`",
                    row.note or "-",
                ]
            )
            + " |"
        )
    if not rows:
        lines.append("| _none_ | - | - | - | - | - | - | - | - | - | - | - | - | - | - |")
    lines.append("")
    return "\n".join(lines)


def pending_report(checkpoints: list[Checkpoint], state: dict[str, Any], backends: list[str]) -> str:
    lines = [
        "# VBVR Pending Checkpoints",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| checkpoint | status | converted | missing_eval | reason |",
        "|---|---|---|---|---|",
    ]
    for ckpt in checkpoints:
        missing: list[str] = []
        if ckpt.complete:
            for backend in backends:
                if best_result(ckpt, backend, state) is None:
                    failed = latest_failed_result(ckpt, backend, state)
                    missing.append(f"{backend} failed" if failed is not None else backend)
        else:
            missing = backends
        converted = "yes" if (ckpt.model_dir / "model_index.json").exists() else "no"
        status = "complete" if ckpt.complete else "incomplete"
        lines.append(
            f"| `{ckpt.safe_name}` | {status} | {converted} | {', '.join(missing) or '-'} | {ckpt.reason} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(checkpoints: list[Checkpoint], state: dict[str, Any], backends: list[str]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    if "vllm" in backends:
        atomic_write_text(
            REPORT_ROOT / "vbvr_vllm_results.md",
            markdown_table(all_best_results(checkpoints, "vllm", state), "VBVR vLLM-Omni Results"),
        )
    if "fastvideo" in backends:
        atomic_write_text(
            REPORT_ROOT / "vbvr_fastvideo_results.md",
            markdown_table(all_best_results(checkpoints, "fastvideo", state), "VBVR FastVideo Results"),
        )
    atomic_write_text(REPORT_ROOT / "vbvr_pending.md", pending_report(checkpoints, state, backends))


def gpu_memory_used() -> list[int]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except Exception:
        return []
    used: list[int] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            used.append(int(line))
        except ValueError:
            continue
    return used


def gpus_idle(threshold_mib: int) -> bool:
    used = gpu_memory_used()
    return bool(used) and all(value <= threshold_mib for value in used)


def command_to_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def terminate_process_group(proc: subprocess.Popen[Any], timeout_seconds: int = 60) -> int:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return proc.wait()
    except PermissionError:
        proc.terminate()

    try:
        return proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            proc.kill()
        return proc.wait(timeout=30)


def fatal_log_match(log_path: Path, patterns: tuple[str, ...]) -> str | None:
    if not patterns:
        return None
    text = log_tail_text(log_path, max_bytes=80_000).lower()
    for pattern in patterns:
        if pattern.lower() in text:
            return pattern
    return None


def run_command(
    *,
    name: str,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
    fatal_log_patterns: tuple[str, ...] = (),
) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"{name}: {command_to_text(cmd)}")
    if dry_run:
        return 0, 0.0

    started = time.monotonic()
    with log_path.open("a", buffering=1) as log_file:
        log_file.write(f"\n# started_at={datetime.now().isoformat(timespec='seconds')}\n")
        log_file.write(f"# cwd={cwd}\n")
        log_file.write(f"# command={command_to_text(cmd)}\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        rc: int | None = None
        while rc is None:
            rc = proc.poll()
            if rc is not None:
                break
            matched = fatal_log_match(log_path, fatal_log_patterns)
            if matched is not None:
                message = f"{name}: fatal log pattern matched ({matched}); terminating process group"
                log(message)
                log_file.write(f"\n# {message}\n")
                log_file.flush()
                rc = terminate_process_group(proc)
                break
            time.sleep(10)
        log_file.write(f"\n# finished_at={datetime.now().isoformat(timespec='seconds')} rc={rc}\n")
    return int(rc), time.monotonic() - started



def ensure_local_tasks() -> None:
    missing = []
    for path in (FASTVIDEO_TASK_PATH / "vbvr_local.yaml", VLLM_TASK_PATH / "vbvr_local.yaml"):
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise RuntimeError("missing local VBVR task files: " + ", ".join(missing))


def convert_if_needed(ckpt: Checkpoint, args: argparse.Namespace) -> bool:
    if (ckpt.model_dir / "model_index.json").exists():
        return True
    if not ckpt.complete:
        return False

    ts = stamp()
    log_path = LOG_ROOT / f"convert_{ckpt.safe_name}_{ts}.log"
    env = os.environ.copy()
    env.update(
        {
            "CHECKPOINTS": str(ckpt.path),
            "CHECKPOINT_ROOT": str(CHECKPOINT_ROOT),
            "OUTPUT_ROOT": str(CONVERTED_ROOT),
            "BASE_MODEL": str(BASE_MODEL),
            "DEVICE": args.convert_device,
        }
    )
    cmd = ["fish", "scripts/convert/dcp_to_diffusers.fish"]
    rc, duration = run_command(
        name=f"convert {ckpt.safe_name}",
        cmd=cmd,
        cwd=REPO_ROOT,
        env=env,
        log_path=log_path,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return False
    if rc != 0:
        log(f"conversion failed for {ckpt.safe_name}; log={log_path}; duration={hms(duration)}")
        return False
    if not (ckpt.model_dir / "model_index.json").exists():
        log(f"conversion did not produce model_index.json for {ckpt.safe_name}; log={log_path}")
        return False
    log(f"conversion complete for {ckpt.safe_name}; duration={hms(duration)}")
    return True


def append_run_record(record: dict[str, Any]) -> None:
    state = load_state()
    state.setdefault("runs", []).append(record)
    save_state(state)


def finish_run_record(
    *,
    ckpt: Checkpoint,
    backend: str,
    output_dir: Path,
    result_path: Path,
    log_path: Path,
    command: list[str],
    started_at: str,
    duration: float,
    rc: int,
) -> dict[str, Any]:
    finished_at = datetime.now().isoformat(timespec="seconds")
    record = {
        "backend": backend,
        "checkpoint": ckpt.safe_name,
        "source_checkpoint": str(ckpt.path),
        "model_dir": str(ckpt.model_dir),
        "output_path": str(output_dir),
        "result_path": str(result_path),
        "log_path": str(log_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration,
        "returncode": rc,
        "command": command,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "monitor_metadata.json", record)
    append_run_record(record)
    return record


def run_vllm(ckpt: Checkpoint, args: argparse.Namespace) -> bool:
    ts = stamp()
    output_dir = VLLM_OUTPUT_ROOT / f"vbvr_vllm_omni_cache_dit_{ckpt.safe_name}_{ts}"
    log_path = LOG_ROOT / f"vllm_{ckpt.safe_name}_{ts}.log"
    model_args = ",".join(
        [
            f"model={ckpt.model_dir}",
            "tensor_parallel_size=1",
            "data_parallel_size=1",
            "gpu_memory_utilization=0.9",
            f"output_dir={output_dir / 'videos'}",
            "output_modalities=video",
            "cache_backend=cache_dit",
            f"num_inference_steps={args.steps}",
            "guidance_scale=5.0",
            "num_frames=81",
            "height=384",
            "width=384",
            "fps=16",
            "seed=42",
            "boundary_ratio=0.9",
            "flow_shift=3.0",
            "tqdm_mode=rank",
        ]
    )
    cmd = [
        ".venv/bin/torchrun",
        "--standalone",
        "--nproc_per_node",
        str(args.vllm_dp),
        "-m",
        "lmms_eval",
        "eval",
        "--model",
        "vllm_omni",
        "--model_args",
        model_args,
        "--tasks",
        "vbvr_local",
        "--include_path",
        str(VLLM_TASK_PATH),
        "--batch_size",
        "1",
        "--log_samples",
        "--output_path",
        str(output_dir),
    ]
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": args.hf_home,
            "VBVR_GT_PATH": str(VBVR_ROOT),
            "CUDA_VISIBLE_DEVICES": args.gpus,
            "NCCL_BLOCKING_WAIT": "1",
            "NCCL_TIMEOUT": "18000000",
        }
    )
    started_at = datetime.now().isoformat(timespec="seconds")
    rc, duration = run_command(
        name=f"vllm {ckpt.safe_name}",
        cmd=cmd,
        cwd=VLLM_REPO,
        env=env,
        log_path=log_path,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return False
    result_path = output_dir / "submissions/vbvr_eval_results.json"
    finish_run_record(
        ckpt=ckpt,
        backend="vllm",
        output_dir=output_dir,
        result_path=result_path,
        log_path=log_path,
        command=cmd,
        started_at=started_at,
        duration=duration,
        rc=rc,
    )
    summary = load_complete_summary(result_path)
    if rc == 0 and summary is not None:
        log(f"vllm complete for {ckpt.safe_name}; overall={summary['overall']:.4f}; duration={hms(duration)}")
        return True
    log(f"vllm failed/incomplete for {ckpt.safe_name}; rc={rc}; log={log_path}; duration={hms(duration)}")
    return False


def run_fastvideo(ckpt: Checkpoint, args: argparse.Namespace) -> bool:
    ts = stamp()
    output_dir = FASTVIDEO_OUTPUT_ROOT / f"vbvr_fastvideo_{ckpt.safe_name}_{ts}"
    log_path = LOG_ROOT / f"fastvideo_{ckpt.safe_name}_{ts}.log"
    video_dir = output_dir / "generated_videos" / ckpt.safe_name
    model_args = ",".join(
        [
            f"model={ckpt.model_dir}",
            f"output_dir={video_dir}",
            f"data_parallel={args.fastvideo_dp}",
            "num_gpus=1",
            "sp_size=1",
            "tp_size=1",
            f"num_inference_steps={args.steps}",
            "num_frames=81",
            "height=384",
            "width=384",
            "fps=16",
            "dit_cpu_offload=False",
            "text_encoder_cpu_offload=True",
            "image_encoder_cpu_offload=False",
            "vae_cpu_offload=False",
            "enable_torch_compile=True",
        ]
    )
    cmd = [
        ".venv/bin/python",
        "-m",
        "lmms_eval",
        "eval",
        "--model",
        "fastvideo",
        "--model_args",
        model_args,
        "--tasks",
        "vbvr_local",
        "--include_path",
        str(FASTVIDEO_TASK_PATH),
        "--batch_size",
        "1",
        "--log_samples",
        f"--output_path={output_dir}",
    ]
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": args.hf_home,
            "VBVR_GT_PATH": str(VBVR_ROOT),
            "LMMS_EVAL_DATASETS_CACHE": f"/tmp/lmms_eval_hf_datasets_{os.environ.get('USER', 'user')}",
            "CUDA_VISIBLE_DEVICES": args.gpus,
        }
    )
    started_at = datetime.now().isoformat(timespec="seconds")
    rc, duration = run_command(
        name=f"fastvideo {ckpt.safe_name}",
        cmd=cmd,
        cwd=FASTVIDEO_REPO,
        env=env,
        log_path=log_path,
        dry_run=args.dry_run,
        fatal_log_patterns=("latent_model_input contains nan",),
    )
    if args.dry_run:
        return False
    result_path = output_dir / "submissions/vbvr_eval_results.json"
    finish_run_record(
        ckpt=ckpt,
        backend="fastvideo",
        output_dir=output_dir,
        result_path=result_path,
        log_path=log_path,
        command=cmd,
        started_at=started_at,
        duration=duration,
        rc=rc,
    )
    summary = load_complete_summary(result_path)
    if rc == 0 and summary is not None:
        log(f"fastvideo complete for {ckpt.safe_name}; overall={summary['overall']:.4f}; duration={hms(duration)}")
        return True
    log(f"fastvideo failed/incomplete for {ckpt.safe_name}; rc={rc}; log={log_path}; duration={hms(duration)}")
    return False


def process_once(args: argparse.Namespace) -> None:
    ensure_local_tasks()
    state = load_state()
    checkpoints = discover_checkpoints()
    write_reports(checkpoints, state, args.backends)

    complete = [ckpt for ckpt in checkpoints if ckpt.complete]
    incomplete = [ckpt for ckpt in checkpoints if not ckpt.complete]
    log(f"discovered {len(complete)} complete checkpoint(s), {len(incomplete)} incomplete checkpoint(s)")
    for ckpt in incomplete:
        log(f"skip incomplete {ckpt.safe_name}: {ckpt.reason}")

    if not args.skip_gpu_idle_check and not args.dry_run and not gpus_idle(args.gpu_idle_threshold_mib):
        log(f"GPU memory is above {args.gpu_idle_threshold_mib} MiB on at least one GPU; skip this interval")
        return

    for ckpt in complete:
        state = load_state()
        missing_backends: list[str] = []
        skipped_failed: list[str] = []
        for backend in args.backends:
            if best_result(ckpt, backend, state) is not None:
                continue
            failed_count = len(failed_runs_for(ckpt, backend, state))
            if failed_count >= args.max_failed_attempts:
                skipped_failed.append(f"{backend} failed_attempts={failed_count}")
                continue
            missing_backends.append(backend)
        if not missing_backends:
            if skipped_failed:
                log(f"skip {ckpt.safe_name}: {', '.join(skipped_failed)}")
            else:
                log(f"skip {ckpt.safe_name}: vllm/fastvideo results already complete")
            continue

        log(f"{ckpt.safe_name}: missing {', '.join(missing_backends)}")
        if not convert_if_needed(ckpt, args):
            if args.dry_run:
                continue
            log(f"skip eval for {ckpt.safe_name}: converted model not ready")
            continue

        for backend in missing_backends:
            state = load_state()
            if best_result(ckpt, backend, state) is not None:
                continue
            if backend == "vllm":
                run_vllm(ckpt, args)
            elif backend == "fastvideo":
                run_fastvideo(ckpt, args)
            else:
                raise ValueError(f"unknown backend: {backend}")
            write_reports(discover_checkpoints(), load_state(), args.backends)

    write_reports(discover_checkpoints(), load_state(), args.backends)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run one scan/eval pass and exit")
    parser.add_argument("--dry-run", action="store_true", help="print intended work without converting/evaluating")
    parser.add_argument("--interval-seconds", type=int, default=1800, help="monitor interval; default 1800")
    parser.add_argument(
        "--backends",
        default="vllm,fastvideo",
        help="comma-separated backends to run: vllm,fastvideo",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--vllm-dp", type=int, default=8)
    parser.add_argument("--fastvideo-dp", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--hf-home", default="/tmp/lmms_eval_hf_vbvr")
    parser.add_argument("--convert-device", default="cuda:0")
    parser.add_argument("--skip-gpu-idle-check", action="store_true")
    parser.add_argument("--gpu-idle-threshold-mib", type=int, default=1024)
    parser.add_argument("--max-failed-attempts", type=int, default=1)
    args = parser.parse_args()

    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    invalid = sorted(set(backends) - {"vllm", "fastvideo"})
    if invalid:
        parser.error(f"invalid backend(s): {', '.join(invalid)}")
    args.backends = backends
    return args


def main() -> int:
    args = parse_args()
    try:
        with FileLock(LOCK_PATH):
            while True:
                process_once(args)
                if args.once:
                    break
                log(f"sleep {args.interval_seconds}s")
                time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        log("interrupted")
        return 130
    except RuntimeError as exc:
        log(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
