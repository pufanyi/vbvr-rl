"""Build an eval JSON for VBVR-Bench from the ground-truth directory.

Walks <gt_base>/{In-Domain_50,Out-of-Domain_50}/<task>/<idx>/ and emits a
JSON file compatible with `src.cli.eval_i2v`. Each entry's `name` is
"<split>/<task>/<idx>", so generated videos land at
<output_dir>/<split>/<task>/<idx>.mp4 — the layout expected by
`run_evaluation_video_icml.py`.

Usage:
    .venv/bin/python -m src.eval.build_vbvr_eval_json \
        --gt_base /mnt/umm/users/wangruisi/01-project/mllm/hokin_data/VBVR-Bench \
        --output data/vbvr_eval.json
"""

import argparse
import json
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
        help="Directory label for generated videos (scoring is split-agnostic)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Only include these task names (default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gt_base = Path(args.gt_base)
    task_filter = set(args.tasks) if args.tasks else None

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
            for sample_dir in sorted(task_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                first_frame = sample_dir / "first_frame.png"
                prompt_file = sample_dir / "prompt.txt"
                if not first_frame.exists() or not prompt_file.exists():
                    print(f"[skip] {sample_dir} missing first_frame.png or prompt.txt")
                    continue
                entries.append(
                    {
                        "name": f"{args.split}/{task_name}/{sample_dir.name}",
                        "image": str(first_frame.resolve()),
                        "prompt": prompt_file.read_text().strip(),
                        "task_name": task_name,
                        "video_idx": sample_dir.name,
                        "domain": domain_label,
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    print(f"Wrote {len(entries)} entries to {output}")


if __name__ == "__main__":
    main()
