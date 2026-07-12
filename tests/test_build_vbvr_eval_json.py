import json
from pathlib import Path

import pytest

from src.eval.build_vbvr_eval_json import build_entries


def _write_sample(root: Path, domain: str, task: str, index: str, prompt: str) -> None:
    sample = root / domain / task / index
    sample.mkdir(parents=True)
    (sample / "first_frame.png").write_bytes(b"image")
    (sample / "prompt.txt").write_text(prompt, encoding="utf-8")


def test_build_entries_uses_evalkit_domain_layout(tmp_path: Path):
    _write_sample(tmp_path, "In-Domain_50", "task-in", "00000", "in prompt")
    _write_sample(tmp_path, "Out-of-Domain_50", "task-out", "00004", "out prompt")
    _write_sample(tmp_path, "Extra_50", "task-extra", "00000", "extra prompt")

    entries = build_entries(tmp_path)

    assert [entry["name"] for entry in entries] == [
        "In-Domain_50/task-in/00000",
        "Out-of-Domain_50/task-out/00004",
    ]
    assert [entry["prompt"] for entry in entries] == ["in prompt", "out prompt"]


def test_build_entries_can_keep_legacy_fixed_split_layout(tmp_path: Path):
    _write_sample(tmp_path, "In-Domain_50", "task", "00000", "prompt")

    entries = build_entries(tmp_path, layout="split", split="Hidden_40")

    assert entries[0]["name"] == "Hidden_40/task/00000"


def test_build_entries_validates_flattened_sample_against_manifest(tmp_path: Path):
    _write_sample(tmp_path, "In-Domain_50", "task", "00000", "prompt")
    sample = tmp_path / "In-Domain_50/task/00000"
    (sample / "metadata.json").write_text(json.dumps({"task_id": "sample-id"}), encoding="utf-8")
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text(
        json.dumps([{"task": "task", "split": "In-Domain_50", "bench": ["sample-id"]}]),
        encoding="utf-8",
    )

    entries = build_entries(tmp_path, split_manifest=manifest)

    assert len(entries) == 1

    (sample / "metadata.json").write_text(json.dumps({"task_id": "stale-id"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Manifest mismatch"):
        build_entries(tmp_path, split_manifest=manifest)
