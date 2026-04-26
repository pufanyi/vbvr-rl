"""Parallel drop-in replacement for third_party/VBVR-EvalKit/run_evaluation.py.

Reuses EvalKit's functions verbatim (collect_videos, find_gt_info,
evaluate_single_video, aggregate_score, finalize_summary) — the only change
is the scoring loop uses multiprocessing.Pool so scoring scales with CPU
cores instead of being a single tqdm for-loop.

Output JSON is byte-compatible with run_evaluation.py.

Usage (called by scripts/eval/eval_vbvr_rule.fish):
    .venv/bin/python -m src.eval.vbvr_run_evaluation_parallel \
        --model_path /abs/path/model_out \
        --gt_base    /abs/path/VBVR-Bench \
        --output_dir /abs/path/model_out/score \
        --num_workers 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

_HERE = Path(__file__).resolve()
_EVALKIT = _HERE.parents[2] / "third_party" / "VBVR-EvalKit"
sys.path.insert(0, str(_EVALKIT))

# Reuse EvalKit's own functions so results stay byte-compatible.
import run_evaluation as rek  # noqa: E402


def _score_one(task: dict) -> dict:
    """Pool worker: score one video, return a sample_result dict."""
    result = rek.evaluate_single_video(
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True, type=Path)
    ap.add_argument("--gt_base", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    args = ap.parse_args()

    model_path = str(args.model_path)
    model_name = os.path.basename(model_path.rstrip("/"))

    print(f"\n{'=' * 60}\nEvaluating: {model_name}\nPath: {model_path}\n{'=' * 60}")

    videos = rek.collect_videos(model_path)
    if not videos:
        print(f"No videos found in {model_path}")
        return 1

    for v in videos:
        v["gt_info"] = rek.find_gt_info(v["task_name"], v["video_idx"], str(args.gt_base))
        v["device"] = args.device

    print(f"Found {len(videos)} videos; scoring with {args.num_workers} workers")

    results = rek.init_results(model_name, model_path)

    with Pool(processes=args.num_workers) as pool:
        for sample in tqdm(
            pool.imap_unordered(_score_one, videos),
            total=len(videos),
            desc=f"Evaluating {model_name}",
        ):
            results["samples"].append(sample)
            rek.aggregate_score(results, sample)

    rek.finalize_summary(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output_dir / f"{model_name}_vbvr_results.json"
    with out_file.open("w") as f:
        json.dump(results, f, indent=2, cls=rek.NumpyEncoder)

    # Match run_evaluation.py's stdout for easy eyeballing
    print(f"\nResults saved to {out_file}")
    rek.print_results(results)
    results["timestamp"] = datetime.now().isoformat()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
