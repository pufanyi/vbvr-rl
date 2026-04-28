"""Build a small held-out maze eval set with rendered references.

The generated JSON is consumable by ``src.cli.eval_maze`` for inference and by
``src.eval.maze_tracker_score`` for rule/tracker scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from diffusers.utils import export_to_video
from PIL import Image

from src.precompute.maze_generator import MazeSpec, build_maze_sample
from src.precompute.maze_webdataset import _sample_to_json_blob


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate held-out maze eval set")
    p.add_argument("--output_dir", default="data/maze/eval/maze_384x384x81_perfect_v2")
    p.add_argument("--per_difficulty", type=int, default=25)
    p.add_argument("--seed", type=int, default=20260428)
    p.add_argument("--cell_h", type=int, default=16)
    p.add_argument("--cell_w", type=int, default=16)
    p.add_argument("--cell_px", type=int, default=12)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--difficulties", default="easy,mid,hard,xhard")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "first_frames"
    videos_dir = output_dir / "reference_videos"
    meta_dir = output_dir / "metadata"
    for d in (frames_dir, videos_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    difficulties = [x.strip() for x in args.difficulties.split(",") if x.strip()]
    all_rows: list[dict] = []
    manifest = {
        "seed": args.seed,
        "per_difficulty": args.per_difficulty,
        "difficulties": difficulties,
        "cell_h": args.cell_h,
        "cell_w": args.cell_w,
        "cell_px": args.cell_px,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "rows": [],
    }

    for diff_idx, difficulty in enumerate(difficulties):
        diff_rows: list[dict] = []
        spec = MazeSpec(
            cell_h=args.cell_h,
            cell_w=args.cell_w,
            cell_px=args.cell_px,
            num_frames=args.num_frames,
            difficulty_names=(difficulty,),
            max_generation_attempts=1024,
        )
        for local_idx in range(args.per_difficulty):
            sample_id = f"{difficulty}_{local_idx:03d}"
            sample_seed = args.seed + diff_idx * 10_000 + local_idx
            rng = np.random.default_rng(sample_seed)
            video, sample = build_maze_sample(spec, rng, sample_seed=sample_seed)

            first_frame_path = frames_dir / f"{sample_id}.png"
            reference_video_path = videos_dir / f"{sample_id}.mp4"
            metadata_path = meta_dir / f"{sample_id}.json"

            Image.fromarray(video[0]).save(first_frame_path)
            export_to_video([Image.fromarray(frame) for frame in video], str(reference_video_path), fps=args.fps)

            maze_blob = _sample_to_json_blob(sample)
            metadata = {
                "id": sample_id,
                "difficulty": difficulty,
                "seed": sample_seed,
                "prompt": sample.prompt,
                "first_frame": str(first_frame_path.relative_to(output_dir)),
                "reference_video": str(reference_video_path.relative_to(output_dir)),
                "metadata": str(metadata_path.relative_to(output_dir)),
                "maze": maze_blob,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            diff_rows.append(metadata)
            all_rows.append(metadata)
            manifest["rows"].append(
                {
                    "id": sample_id,
                    "difficulty": difficulty,
                    "seed": sample_seed,
                    "path_len": sample.path_len,
                    "path_ratio": sample.path_ratio,
                }
            )

        diff_dir = output_dir / difficulty
        diff_dir.mkdir(parents=True, exist_ok=True)
        (diff_dir / "eval.json").write_text(json.dumps(diff_rows, indent=2) + "\n")

    (output_dir / "eval.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(all_rows)} eval samples to {output_dir}")


if __name__ == "__main__":
    main()
