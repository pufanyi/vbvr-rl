"""Build an eval JSON for VBVR-Bench from the ground-truth directory.

Walks <gt_base>/{In-Domain_50,Out-of-Domain_50}/<task>/<idx>/ and emits a
JSON file compatible with `src.cli.eval_i2v`. Each entry's `name` is
"<domain>/<task>/<idx>" by default, so generated videos already use the layout
expected by EvalKit's `run_evaluation.py`. The legacy fixed split layout remains
available for `run_evaluation_video_icml.py` callers.

Usage:
    .venv/bin/python -m src.eval.build_vbvr_eval_json \
        --gt_base /mnt/umm/users/wangruisi/01-project/mllm/hokin_data/VBVR-Bench \
        --output data/vbvr_eval.json
"""

import argparse
import json
import re
from pathlib import Path

DOMAINS = {
    "In-Domain_50": "In_Domain",
    "Out-of-Domain_50": "Out_of_Domain",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build VBVR-Bench eval JSON")
    parser.add_argument(
        "--gt_base",
        type=str,
        default="/mnt/umm/users/wangruisi/01-project/mllm/hokin_data/VBVR-Bench",
        help="Base directory with In-Domain_50 / Out-of-Domain_50",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Where to write the eval JSON",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="Open_60",
        choices=["Open_60", "Hidden_40"],
        help="Directory label when --layout=split",
    )
    parser.add_argument(
        "--layout",
        choices=["domain", "split"],
        default="domain",
        help="Use EvalKit domain folders or one legacy Open_60/Hidden_40 folder",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Only include these task names (default: all)",
    )
    parser.add_argument(
        "--expected_samples",
        type=int,
        default=None,
        help="Fail unless exactly this many samples are found",
    )
    parser.add_argument(
        "--split_manifest",
        type=Path,
        default=None,
        help="Validate flattened sample metadata against this VBVR-Pro bench manifest",
    )
    return parser.parse_args()


def _load_manifest_bench(path: Path) -> dict[str, tuple[str, list[str]]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a list in split manifest {path}")
    expected: dict[str, tuple[str, list[str]]] = {}
    for record in records:
        domain = record.get("split")
        if domain not in DOMAINS:
            continue
        task_name = str(record["task"])
        if task_name in expected:
            raise ValueError(f"Duplicate task {task_name!r} in split manifest {path}")
        bench = record.get("bench")
        if not isinstance(bench, list):
            raise ValueError(f"Task {task_name!r} has no bench list in {path}")
        expected[task_name] = (str(domain), [str(sample_id) for sample_id in bench])
    return expected


def build_entries(
    gt_base: Path,
    *,
    layout: str = "domain",
    split: str = "Open_60",
    task_filter: set[str] | None = None,
    split_manifest: Path | None = None,
) -> list[dict[str, str]]:
    manifest_bench = _load_manifest_bench(split_manifest) if split_manifest is not None else None
    seen_manifest_samples: set[tuple[str, int]] = set()
    entries = []
    for domain_dir, domain_label in DOMAINS.items():
        domain_path = gt_base / domain_dir
        if not domain_path.is_dir():
            print(f"[warn] missing {domain_path}")
            continue
        for task_dir in sorted(domain_path.iterdir()):
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name
            if task_filter is not None and task_name not in task_filter:
                continue
            expected_task = manifest_bench.get(task_name) if manifest_bench is not None else None
            if manifest_bench is not None:
                if expected_task is None:
                    raise ValueError(f"Task {task_name!r} under {domain_dir} is absent from manifest bench")
                if expected_task[0] != domain_dir:
                    raise ValueError(
                        f"Task {task_name!r} is under {domain_dir}, but the manifest assigns it to {expected_task[0]}"
                    )
            for sample_dir in sorted(task_dir.iterdir()):
                if not sample_dir.is_dir() or re.fullmatch(r"\d{5}", sample_dir.name) is None:
                    continue
                first_frame = sample_dir / "first_frame.png"
                prompt_file = sample_dir / "prompt.txt"
                if not first_frame.exists() or not prompt_file.exists():
                    print(f"[skip] {sample_dir} missing first_frame.png or prompt.txt")
                    continue
                if expected_task is not None:
                    sample_index = int(sample_dir.name)
                    expected_ids = expected_task[1]
                    if sample_index >= len(expected_ids):
                        raise ValueError(f"Unexpected bench index {sample_dir.name} under task {task_name!r}")
                    metadata_path = sample_dir / "metadata.json"
                    if not metadata_path.is_file():
                        raise FileNotFoundError(f"Missing manifest validation metadata: {metadata_path}")
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    actual_id = str(metadata.get("task_id", ""))
                    if actual_id != expected_ids[sample_index]:
                        raise ValueError(
                            f"Manifest mismatch for {domain_dir}/{task_name}/{sample_dir.name}: "
                            f"expected task_id={expected_ids[sample_index]!r}, found {actual_id!r}"
                        )
                    seen_manifest_samples.add((task_name, sample_index))
                output_root = domain_dir if layout == "domain" else split
                entries.append(
                    {
                        "name": f"{output_root}/{task_name}/{sample_dir.name}",
                        "image": str(first_frame.resolve()),
                        "prompt": prompt_file.read_text(encoding="utf-8").strip(),
                        "task_name": task_name,
                        "video_idx": sample_dir.name,
                        "domain": domain_label,
                    }
                )
    if manifest_bench is not None:
        expected_samples = {
            (task_name, sample_index)
            for task_name, (_, sample_ids) in manifest_bench.items()
            for sample_index in range(len(sample_ids))
            if task_filter is None or task_name in task_filter
        }
        missing = expected_samples - seen_manifest_samples
        if missing:
            preview = ", ".join(f"{task}/{index:05d}" for task, index in sorted(missing)[:5])
            raise FileNotFoundError(f"Flattened GT is missing {len(missing)} manifest bench samples: {preview}")
    return entries


def main():
    args = parse_args()
    gt_base = Path(args.gt_base)
    task_filter = set(args.tasks) if args.tasks else None
    entries = build_entries(
        gt_base,
        layout=args.layout,
        split=args.split,
        task_filter=task_filter,
        split_manifest=args.split_manifest,
    )
    if args.expected_samples is not None and len(entries) != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} eval samples, found {len(entries)} under {gt_base}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(output)
    print(f"Wrote {len(entries)} entries to {output}")


if __name__ == "__main__":
    main()
