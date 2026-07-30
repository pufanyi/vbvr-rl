import hashlib
import io
import json
import tarfile
from pathlib import Path

from scripts.data.vbvr_pro_unpack_hf import _FIELD_TARGETS, materialize


def _add_bytes(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(value)
    archive.addfile(info, io.BytesIO(value))


def test_materialize_published_vbvr_pro_snapshot(tmp_path: Path) -> None:
    dataset_root = tmp_path / "snapshot"
    data_dir = dataset_root / "data"
    data_dir.mkdir(parents=True)
    key = "00000007"
    values = {
        "first.png": b"first",
        "metadata.json.bin": b'{"meta": true}',
        "final.png": b"final",
        "gt.mp4": b"video",
        "video_prompt.txt": b"move the object",
    }
    shard = data_dir / "shard-00000.tar"
    with tarfile.open(shard, "w") as archive:
        _add_bytes(archive, f"{key}.json", b"{}")
        for field, value in values.items():
            _add_bytes(archive, f"{key}.{field}", value)
        _add_bytes(archive, f"{key}.extras.zip.bin", b"unused")

    record = {
        "key": key,
        "task_name": "G-1_example_data-generator",
        "sample_id": "example_00005001",
        "task_split": "In-Domain_50",
        "shard": "data/shard-00000.tar",
        "files": [
            {
                "field": field,
                "size": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
            for field, value in values.items()
        ],
    }
    (dataset_root / "samples.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    output_dir = tmp_path / "materialized"
    results = materialize(dataset_root, output_dir, expected_samples=1, workers=2)

    assert sum(result.written_files for result in results) == len(_FIELD_TARGETS)
    sample_dir = output_dir / "raw" / record["task_name"] / record["sample_id"]
    assert (sample_dir / "first_frame.png").read_bytes() == values["first.png"]
    assert (sample_dir / "metadata.json").read_bytes() == values["metadata.json.bin"]
    assert (sample_dir / "video/final_frame.png").read_bytes() == values["final.png"]
    assert (sample_dir / "video/ground_truth.mp4").read_bytes() == values["gt.mp4"]
    assert (sample_dir / "video/prompt.txt").read_bytes() == values["video_prompt.txt"]

    descriptor = json.loads((output_dir / "dataset.json").read_text(encoding="utf-8"))
    assert descriptor[0]["format"] == "vbvr_pro"
    assert descriptor[0]["data_roots"] == ["raw"]
    manifest = json.loads((output_dir / "split_manifest_rl.json").read_text(encoding="utf-8"))
    assert manifest == [
        {
            "task": record["task_name"],
            "source": record["task_name"],
            "split": record["task_split"],
            "rl": [record["sample_id"]],
        }
    ]

    resumed = materialize(dataset_root, output_dir, expected_samples=1, workers=1, verify_existing=True)
    assert sum(result.reused_files for result in resumed) == len(_FIELD_TARGETS)
