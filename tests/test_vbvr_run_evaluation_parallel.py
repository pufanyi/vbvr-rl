from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_fake_evalkit(path: Path) -> None:
    path.mkdir()
    (path / "run_evaluation.py").write_text(
        textwrap.dedent(
            """
            import json
            import os
            import time


            class NumpyEncoder(json.JSONEncoder):
                pass


            def collect_videos(model_path):
                order = [2, 0, 1]
                return [
                    {
                        "video_path": f"{model_path}/task/{idx:05d}.mp4",
                        "video_file": f"{idx:05d}.mp4",
                        "task_name": "fake-task",
                        "video_idx": idx,
                        "split": "In_Domain",
                        "category": "Fake",
                        "folder": "In-Domain_50",
                    }
                    for idx in order
                ]


            def find_gt_info(task_name, video_idx, gt_base):
                return {"video_idx": video_idx, "gt_base": gt_base}


            def evaluate_single_video(video_path, task_name, gt_info, device):
                import cv2
                import torch

                idx = gt_info["video_idx"]
                time.sleep({2: 0.12, 0: 0.0, 1: 0.04}[idx])
                return {
                    "score": idx / 10,
                    "dimensions": {
                        "worker_loaded": True,
                        "cwd_is_evalkit": os.getcwd() == os.path.dirname(__file__),
                        "paths_are_absolute": os.path.isabs(video_path) and os.path.isabs(gt_info["gt_base"]),
                        "native_threads": {
                            "env": os.environ["OMP_NUM_THREADS"],
                            "opencv": cv2.getNumThreads(),
                            "torch": torch.get_num_threads(),
                        },
                    },
                }


            def init_results(model_name, model_path):
                return {
                    "model_name": model_name,
                    "model_path": model_path,
                    "samples": [],
                    "summary": {"overall": {"scores": []}},
                }


            def aggregate_score(results, sample):
                results["summary"]["overall"]["scores"].append(sample["score"])


            def finalize_summary(results):
                scores = results["summary"]["overall"]["scores"]
                results["summary"]["overall"]["mean_score"] = sum(scores) / len(scores)


            def print_results(results):
                print("fake scorer complete")
            """
        ).lstrip()
    )


def _run_cli(
    fake_evalkit: Path,
    model_path: Path,
    gt_base: Path,
    output_dir: Path,
    expected: int,
    *,
    relative_paths: bool = False,
):
    def cli_path(path: Path) -> str:
        return os.path.relpath(path, _REPO_ROOT) if relative_paths else str(path)

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "src.eval.vbvr_run_evaluation_parallel",
            "--model_path",
            cli_path(model_path),
            "--gt_base",
            cli_path(gt_base),
            "--output_dir",
            cli_path(output_dir),
            "--evalkit_dir",
            cli_path(fake_evalkit),
            "--expected_videos",
            str(expected),
            "--num_workers",
            "2",
            "--threads_per_worker",
            "3",
            "--device",
            "cpu",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_custom_evalkit_loads_in_workers_and_preserves_collected_order(tmp_path: Path):
    fake_evalkit = tmp_path / "fake_evalkit"
    model_path = tmp_path / "model_videos"
    gt_base = tmp_path / "gt"
    output_dir = tmp_path / "results"
    _write_fake_evalkit(fake_evalkit)
    model_path.mkdir()
    gt_base.mkdir()

    completed = _run_cli(fake_evalkit, model_path, gt_base, output_dir, expected=3, relative_paths=True)

    assert completed.returncode == 0, completed.stderr
    result_path = output_dir / "model_videos_vbvr_results.json"
    result = json.loads(result_path.read_text())
    assert [sample["video_file"] for sample in result["samples"]] == ["00002.mp4", "00000.mp4", "00001.mp4"]
    assert all(
        sample["dimensions"]
        == {
            "worker_loaded": True,
            "cwd_is_evalkit": True,
            "paths_are_absolute": True,
            "native_threads": {"env": "3", "opencv": 3, "torch": 3},
        }
        for sample in result["samples"]
    )
    assert abs(result["summary"]["overall"]["mean_score"] - 0.1) < 1e-12


def test_expected_video_count_fails_before_scoring(tmp_path: Path):
    fake_evalkit = tmp_path / "fake_evalkit"
    model_path = tmp_path / "model_videos"
    gt_base = tmp_path / "gt"
    output_dir = tmp_path / "results"
    _write_fake_evalkit(fake_evalkit)
    model_path.mkdir()
    gt_base.mkdir()

    completed = _run_cli(fake_evalkit, model_path, gt_base, output_dir, expected=4)

    assert completed.returncode == 1
    assert "Expected exactly 4 videos, but EvalKit discovered 3" in completed.stderr
    assert not output_dir.exists()
