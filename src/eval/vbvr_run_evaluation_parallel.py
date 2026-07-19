r"""Parallel drop-in replacement for VBVR-EvalKit's ``run_evaluation.py``.

Reuses EvalKit's functions verbatim (collect_videos, find_gt_info,
evaluate_single_video, aggregate_score, finalize_summary) — the only change
is the scoring loop uses multiprocessing.Pool so scoring scales with CPU
cores instead of being a single tqdm for-loop.

Output JSON is byte-compatible with run_evaluation.py.

EvalKit's EasyOCR-backed evaluators create GPU readers whenever CUDA is
visible, regardless of the ``--device`` value. Run scoring with CUDA hidden
and point EasyOCR at a writable or pre-populated model directory, for example::

    CUDA_VISIBLE_DEVICES="" EASYOCR_MODULE_PATH=/personal/easyocr_module_root \
        .venv/bin/python -m src.eval.vbvr_run_evaluation_parallel ...

Main-v2 also has evaluators that explicitly use ``./easyocr_models``. Workers
run from ``--evalkit_dir`` so relative annotations resolve correctly; that
checkout must therefore contain an ``easyocr_models`` directory or symlink.

Usage (called by scripts/eval/vbvr/vbvr_rule_score.fish):
    .venv/bin/python -m src.eval.vbvr_run_evaluation_parallel \
        --model_path /abs/path/model_out \
        --gt_base    /abs/path/VBVR-Bench \
        --output_dir /abs/path/model_out/score \
        --evalkit_dir /abs/path/VBVR-EvalKit-main-v2 \
        --expected_videos 500 \
        --num_workers 64
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from types import ModuleType

from tqdm import tqdm

_HERE = Path(__file__).resolve()
_DEFAULT_EVALKIT = _HERE.parents[2] / "third_party" / "VBVR-EvalKit"
_REQUIRED_EVALKIT_API = (
    "NumpyEncoder",
    "aggregate_score",
    "collect_videos",
    "finalize_summary",
    "find_gt_info",
    "init_results",
    "print_results",
    "evaluate_single_video",
)
_WORKER_REK: ModuleType | None = None


def _module_is_from(module: ModuleType, directory: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        Path(module_file).resolve().relative_to(directory)
    except ValueError:
        return False
    return True


def _purge_foreign_evalkit_modules(evalkit_dir: Path) -> None:
    """Remove an already-imported ``vbvr_bench`` from another EvalKit tree."""
    root = sys.modules.get("vbvr_bench")
    if root is None or _module_is_from(root, evalkit_dir):
        return
    for name in list(sys.modules):
        if name == "vbvr_bench" or name.startswith("vbvr_bench."):
            sys.modules.pop(name, None)


def _load_evalkit(evalkit_dir: Path | str) -> ModuleType:
    """Load one EvalKit checkout without relying on the process CWD."""
    directory = Path(evalkit_dir).expanduser().resolve()
    entrypoint = directory / "run_evaluation.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"EvalKit entrypoint not found: {entrypoint}")

    _purge_foreign_evalkit_modules(directory)
    directory_str = str(directory)
    if not sys.path or sys.path[0] != directory_str:
        sys.path.insert(0, directory_str)
    importlib.invalidate_caches()

    digest = hashlib.sha256(str(entrypoint).encode()).hexdigest()[:16]
    module_name = f"_wan_trainer_vbvr_run_evaluation_{digest}"
    cached = sys.modules.get(module_name)
    if cached is not None and _module_is_from(cached, directory):
        return cached

    spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {entrypoint}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    missing = [name for name in _REQUIRED_EVALKIT_API if not hasattr(module, name)]
    if missing:
        sys.modules.pop(module_name, None)
        raise ImportError(f"EvalKit {entrypoint} is missing required API: {', '.join(missing)}")
    return module


def _configure_worker_threads(threads: int) -> None:
    """Bound native library pools so scorer processes do not oversubscribe CPUs."""
    thread_value = str(threads)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = thread_value

    import cv2
    import torch

    cv2.setNumThreads(threads)
    torch.set_num_threads(threads)
    # A forked parent may already have initialized the inter-op pool.
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def _init_worker(evalkit_dir: str, threads_per_worker: int) -> None:
    global _WORKER_REK
    _configure_worker_threads(threads_per_worker)
    directory = Path(evalkit_dir).resolve()
    os.chdir(directory)
    _WORKER_REK = _load_evalkit(directory)


def _score_one(task: dict) -> dict:
    """Pool worker: score one video, return a sample_result dict."""
    if _WORKER_REK is None:
        raise RuntimeError("EvalKit was not initialized in this scoring worker")
    result = _WORKER_REK.evaluate_single_video(
        task["video_path"],
        task["task_name"],
        task["gt_info"],
        task["device"],
    )
    return {
        "video_path": task["video_path"],
        "video_file": task["video_file"],
        "task_name": task["task_name"],
        "split": task["split"],
        "category": task["category"],
        "folder": task["folder"],
        "score": float(result.get("score", 0.0)),
        "dimensions": result.get("dimensions", {}),
        "error": result.get("error", None),
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "EasyOCR scoring should run with CUDA hidden to prevent every worker from loading a model on GPU 0:\n"
            '  CUDA_VISIBLE_DEVICES="" EASYOCR_MODULE_PATH=/personal/easyocr_module_root <command>\n'
            "Workers chdir to --evalkit_dir; main_v2 also requires <evalkit_dir>/easyocr_models (a symlink is fine)."
        ),
    )
    ap.add_argument("--model_path", required=True, type=Path)
    ap.add_argument("--gt_base", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument(
        "--evalkit_dir",
        type=Path,
        default=_DEFAULT_EVALKIT,
        help="EvalKit checkout containing run_evaluation.py (default: repository third_party checkout)",
    )
    ap.add_argument(
        "--expected_videos",
        type=_positive_int,
        default=None,
        help="Fail before scoring unless EvalKit discovers exactly this many videos",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=_positive_int, default=max(1, (os.cpu_count() or 1) - 1))
    ap.add_argument(
        "--threads_per_worker",
        type=_positive_int,
        default=None,
        help="Native OpenCV/PyTorch threads per scorer worker (default: CPU count divided by workers)",
    )
    args = ap.parse_args()
    threads_per_worker = args.threads_per_worker or max(1, (os.cpu_count() or 1) // args.num_workers)

    evalkit_dir = args.evalkit_dir.expanduser().resolve()
    model_path_obj = args.model_path.expanduser().resolve()
    gt_base = args.gt_base.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    try:
        rek = _load_evalkit(evalkit_dir)
    except Exception as exc:
        print(f"[error] Failed to load EvalKit from {args.evalkit_dir}: {exc}", file=sys.stderr)
        return 1

    model_path = str(model_path_obj)
    model_name = os.path.basename(model_path.rstrip("/"))

    print(f"\n{'=' * 60}\nEvaluating: {model_name}\nPath: {model_path}\n{'=' * 60}")

    videos = rek.collect_videos(model_path)
    if not videos:
        print(f"No videos found in {model_path}")
        return 1
    if args.expected_videos is not None and len(videos) != args.expected_videos:
        print(
            f"[error] Expected exactly {args.expected_videos} videos, but EvalKit discovered {len(videos)}",
            file=sys.stderr,
        )
        return 1

    for v in videos:
        v["gt_info"] = rek.find_gt_info(v["task_name"], v["video_idx"], str(gt_base))
        v["device"] = args.device

    print(
        f"Found {len(videos)} videos; scoring with {args.num_workers} workers "
        f"and {threads_per_worker} native threads per worker"
    )

    results = rek.init_results(model_name, model_path)

    with Pool(
        processes=args.num_workers,
        initializer=_init_worker,
        initargs=(str(evalkit_dir), threads_per_worker),
    ) as pool:
        for sample in tqdm(
            pool.imap(_score_one, videos),
            total=len(videos),
            desc=f"Evaluating {model_name}",
        ):
            results["samples"].append(sample)
            rek.aggregate_score(results, sample)

    rek.finalize_summary(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{model_name}_vbvr_results.json"
    with out_file.open("w") as f:
        json.dump(results, f, indent=2, cls=rek.NumpyEncoder)

    # Match run_evaluation.py's stdout for easy eyeballing
    print(f"\nResults saved to {out_file}")
    rek.print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
