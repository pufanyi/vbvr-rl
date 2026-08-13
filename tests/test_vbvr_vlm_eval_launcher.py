from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT
    / "scripts"
    / "eval"
    / "vbvr_pro"
    / "dancegrpo_vlm_qwen36_512x512x81"
    / "evaluate_incremental_multinode.fish"
)


def _write_fish(path: Path, text: str) -> None:
    path.write_text("#!/usr/bin/env fish\n" + text)


def test_formal_auto_judge_freezes_cells_and_crosses_strict_barrier(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    for step in (100, 200):
        metadata = checkpoint_root / f"checkpoint-{step}" / "high" / ".metadata"
        metadata.parent.mkdir(parents=True)
        metadata.write_text("complete")

    delegate_log = tmp_path / "delegate.log"
    judge_log = tmp_path / "judge.log"
    fake_delegate = tmp_path / "delegate.fish"
    fake_judge = tmp_path / "judge.fish"
    _write_fish(
        fake_delegate,
        "echo (string join ' ' -- $argv) >>$FAKE_DELEGATE_LOG\n"
        "if contains -- --assignment-only $argv\n"
        "    echo '[discover] formal new or incomplete: none'\n"
        "end\n",
    )
    _write_fish(
        fake_judge,
        "echo (string join ' ' -- $argv) >$FAKE_JUDGE_LOG\n"
        "echo tp=$WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE >>$FAKE_JUDGE_LOG\n"
        "echo dp=$WAN_TRAINER_VLM_DATA_PARALLEL_SIZE >>$FAKE_JUDGE_LOG\n",
    )

    env = {
        **os.environ,
        "WORLD_SIZE": "2",
        "RANK": "1",
        "OUTPUT_BASE": str(tmp_path / "formal-results"),
        "FAKE_DELEGATE_LOG": str(delegate_log),
        "FAKE_JUDGE_LOG": str(judge_log),
        "VLM_EVAL_SHARED_INCREMENTAL_LAUNCHER": str(fake_delegate),
        "VLM_EVAL_JUDGE_LAUNCHER": str(fake_judge),
    }
    for name in (
        "WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE",
        "WAN_TRAINER_VLM_DATA_PARALLEL_SIZE",
        "WAN_TRAINER_VLM_DATA_PARALLEL_SIZE_LOCAL",
    ):
        env.pop(name, None)
    result = subprocess.run(
        [
            "fish",
            str(LAUNCHER),
            "formal",
            "--checkpoint-root",
            str(checkpoint_root),
            "--nproc",
            "8",
            "--vlm-concurrency",
            "16",
            "--vlm-output-root",
            str(tmp_path / "judge-results"),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    delegate_calls = delegate_log.read_text().splitlines()
    assert delegate_calls == [
        "formal --nproc 8 --checkpoints 100,200",
        "formal --assignment-only --nproc 8 --checkpoints 100,200",
    ]
    judge_lines = judge_log.read_text().splitlines()
    judge_args = judge_lines[0].split()
    assert judge_args[:7] == [
        "score",
        "--input-root",
        str(tmp_path / "formal-results"),
        "--concurrency",
        "16",
        "--output-root",
        str(tmp_path / "judge-results"),
    ]
    expected_cells = {
        f"dancegrpo_vbvr_pro_5b_checkpoint-{step}-{label}"
        for step in (100, 200)
        for label in (
            "cps-noise-0.1",
            "cps-noise-0.3",
            "cps-noise-0.7",
            "cps-noise-0.9",
            "euler-ode-30steps-cfg1",
            "unipc-ode-30steps-cfg1",
        )
    }
    actual_cells = {judge_args[index + 1] for index, value in enumerate(judge_args) if value == "--cell"}
    assert actual_cells == expected_cells
    assert judge_lines[1:] == ["tp=2", "dp=4"]
    assert "strict formal barrier complete on node 1/2" in result.stdout


def test_assignment_only_never_invokes_auto_judge(tmp_path: Path) -> None:
    metadata = tmp_path / "checkpoints" / "checkpoint-100" / "high" / ".metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("complete")
    fake_delegate = tmp_path / "delegate.fish"
    fake_judge = tmp_path / "judge.fish"
    judge_marker = tmp_path / "judge-was-called"
    _write_fish(fake_delegate, "exit 0\n")
    _write_fish(fake_judge, "touch $FAKE_JUDGE_MARKER\n")
    env = {
        **os.environ,
        "WORLD_SIZE": "1",
        "RANK": "0",
        "FAKE_JUDGE_MARKER": str(judge_marker),
        "VLM_EVAL_SHARED_INCREMENTAL_LAUNCHER": str(fake_delegate),
        "VLM_EVAL_JUDGE_LAUNCHER": str(fake_judge),
    }
    result = subprocess.run(
        [
            "fish",
            str(LAUNCHER),
            "formal",
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--assignment-only",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not judge_marker.exists()
    assert "assignment-only mode; automatic judge was not launched" in result.stdout
