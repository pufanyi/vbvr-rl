# Maze 100k Data Generation Plan

This document records the target spec for the next maze dataset generation run.
It is based on the approved preview in `tmp/maze_difficulty_preview_line_10s`.

## Target Dataset

- Total samples: `100000`
- Difficulty sampling: uniformly sample from `easy,mid,hard,xhard`
  - Exact `25000` samples per difficulty is not required.
  - With `100000` samples, the expected count is about `25000` per difficulty.
- Render mode: growing path line
  - Draw a red path line from start to goal over time.
  - Do not render a moving ball.
  - Keep the green goal marker.
- Video format:
  - Resolution: `384x384`
  - Frames: `161`
  - FPS: `16`
  - Duration: `10.06s`
  - Note: Wan2.2 first-frame conditioning requires `num_frames = 4k + 1`, so `160` frames is not valid for latent generation.
- Maze sizes:
  - `easy`: `8x8` cells, `cell_px=24`, logical grid `16x16`
  - `mid`: `12x12` cells, `cell_px=16`, logical grid `24x24`
  - `hard`: `16x16` cells, `cell_px=12`, logical grid `32x32`
  - `xhard`: `16x16` cells, `cell_px=12`, logical grid `32x32`
- `hard` and `xhard` should use the same maze size. Their difference should come from path length / path ratio.
- Suggested split:
  - SFT: `80000` samples
  - RL: `20000` samples

Each WebDataset JSON sidecar must contain enough `maze` metadata to reconstruct
the source RGB video frames without access to the original mp4. Required fields
include:

- `render_mode`
- `render_metadata`
- `grid`
- `path`
- `frame_positions_pix`
- `palette`
- `cell_px`
- `image_h`, `image_w`
- `num_frames`
  - `fps`

## Approved Preview

Preview directory:

```bash
tmp/maze_difficulty_preview_line_10s
```

Files:

```bash
tmp/maze_difficulty_preview_line_10s/reference_videos/easy.mp4
tmp/maze_difficulty_preview_line_10s/reference_videos/mid.mp4
tmp/maze_difficulty_preview_line_10s/reference_videos/hard.mp4
tmp/maze_difficulty_preview_line_10s/reference_videos/xhard.mp4
tmp/maze_difficulty_preview_line_10s/contact_sheet.png
tmp/maze_difficulty_preview_line_10s/manifest.json
```

The preview generation is CPU-only. It uses the pure maze renderer plus PIL/mp4 export.
It does not load Wan VAE, UMT5, or any GPU model.

## Metadata Support

`src/precompute/maze_webdataset.py` supports:

- `render_mode=growing_path_line`
- per-difficulty geometry through `difficulty_geometries`
- per-sample JSON metadata with `render_mode`, `render_metadata`, full maze layout, path, frame positions, and palette

The metadata can be used with:

```python
from src.precompute.maze_generator import render_video_from_metadata

video_uint8 = render_video_from_metadata(meta["maze"])
```

## Intended Output Layout

Use a new output root so this does not collide with the older ball-rendered dataset:

```bash
data/maze/latents/maze_384x384x161_line_v1/
```

Expected structure:

```bash
data/maze/latents/maze_384x384x161_line_v1/
  previews/
    ...
  webdataset/
    dataset_info.json
    sft/
      dataset_info.json
      shard-000000.tar
      ...
      shard-000079.tar
    rl/
      dataset_info.json
      shard-000000.tar
      ...
      shard-000019.tar
```

With `samples_per_shard=1000`, expected shard counts are:

- SFT: `80` shards
- RL: `20` shards
- Total: `100` shards

## Target Full-Generation Command

Run on an 8-GPU node:

```bash
GPUS=0,1,2,3,4,5,6,7 \
OUTPUT_ROOT=data/maze/latents/maze_384x384x161_line_v1 \
TAR_TAG=maze_384x384x161_line_v1 \
NUM_SAMPLES=100000 \
SFT_RATIO=0.8 \
SAMPLES_PER_SHARD=1000 \
SHARD_WRITE_BATCH_SIZE=64 \
SEED=4242 \
SPLIT_SEED=4242 \
NUM_FRAMES=161 \
NUM_PREVIEW_VIDEOS=100 \
DIFFICULTY_NAMES=easy,mid,hard,xhard \
DIFFICULTY_GEOMETRIES=easy:8x8x24,mid:12x12x16,hard:16x16x12,xhard:16x16x12 \
RENDER_MODE=growing_path_line \
bash scripts/precompute/maze_384_supervise_precompute_8gpu.bash
```

Notes:

- Full latent generation requires GPU. It loads Wan VAE for video/condition latents and UMT5 for prompt embeddings.
- The script should keep `--skip_existing` behavior so interrupted runs can resume by rerunning the same command.

## COS Line-To-Ball Variant

For Chain-of-Step training where the high-noise waypoint is a path-line plan
and the final target is the moving ball, generate a separate latent root:

```bash
GPUS=0,1,2,3,4,5,6,7 \
OUTPUT_ROOT=data/maze/latents/maze_384x384x161_line_to_ball_v1 \
TAR_TAG=maze_384x384x161_line_to_ball_v1 \
NUM_SAMPLES=100000 \
SFT_RATIO=0.8 \
SAMPLES_PER_SHARD=1000 \
SHARD_WRITE_BATCH_SIZE=64 \
SEED=4242 \
SPLIT_SEED=4242 \
NUM_FRAMES=161 \
NUM_PREVIEW_VIDEOS=100 \
DIFFICULTY_NAMES=easy,mid,hard,xhard \
DIFFICULTY_GEOMETRIES=easy:8x8x24,mid:12x12x16,hard:16x16x12,xhard:16x16x12 \
RENDER_MODE=moving_ball \
COS_CHAIN_MODE=line_to_moving_ball \
LINE_COMPLETION_FRACTION=0.5 \
bash scripts/precompute/maze_384_supervise_precompute_8gpu.bash
```

Each sample stores `latents_0` for the path-line waypoint and `latents_1` for
the moving-ball final target. `LINE_COMPLETION_FRACTION=0.5` makes the line
reach the goal halfway through the video frames. Train it with
`configs/train_cos_maze_line_to_ball_100k.yaml`, whose `cos_tau_sigma: 0.5`
allocates roughly half of the denoising path to the line-planning waypoint.

## Validation

After generation, verify shard counts:

```bash
find data/maze/latents/maze_384x384x161_line_v1/webdataset/sft -name 'shard-*.tar' | wc -l
find data/maze/latents/maze_384x384x161_line_v1/webdataset/rl -name 'shard-*.tar' | wc -l
```

Expected:

```text
80
20
```

Verify dataset metadata:

```bash
cat data/maze/latents/maze_384x384x161_line_v1/webdataset/dataset_info.json
cat data/maze/latents/maze_384x384x161_line_v1/webdataset/sft/dataset_info.json
cat data/maze/latents/maze_384x384x161_line_v1/webdataset/rl/dataset_info.json
```

Expected key values:

- `num_samples`: `100000`
- `num_frames`: `161`
- `image_h`: `384`
- `image_w`: `384`
- `difficulty_names`: `["easy", "mid", "hard", "xhard"]`
- `difficulty_geometries`:
  - `easy`: `8x8`, `cell_px=24`
  - `mid`: `12x12`, `cell_px=16`
  - `hard`: `16x16`, `cell_px=12`
  - `xhard`: `16x16`, `cell_px=12`
- render mode: `growing_path_line`

Count difficulty distribution from tar metadata:

```bash
.venv/bin/python - <<'PY'
from collections import Counter
from pathlib import Path
import json
import tarfile

root = Path("data/maze/latents/maze_384x384x161_line_v1/webdataset")
for split in ("sft", "rl"):
    counts = Counter()
    total = 0
    for tar_path in sorted((root / split).glob("shard-*.tar")):
        with tarfile.open(tar_path) as tar:
            for member in tar:
                if not member.name.endswith(".json"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                meta = json.load(f)
                counts[meta["maze"]["difficulty"]] += 1
                total += 1
    print(split, total, dict(sorted(counts.items())))
PY
```

Expected:

```text
sft 80000 { ... roughly balanced across easy/hard/mid/xhard ... }
rl 20000 { ... roughly balanced across easy/hard/mid/xhard ... }
```

Verify that metadata can reconstruct a video:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import json
import tarfile

from diffusers.utils import export_to_video
from PIL import Image

from src.precompute.maze_generator import render_video_from_metadata

tar_path = sorted(Path("data/maze/latents/maze_384x384x161_line_v1/webdataset/sft").glob("shard-*.tar"))[0]
out_path = Path("tmp/maze_metadata_reconstruct_check.mp4")
with tarfile.open(tar_path) as tar:
    member = next(m for m in tar if m.name.endswith(".json"))
    meta = json.load(tar.extractfile(member))

video = render_video_from_metadata(meta["maze"])
fps = meta["maze"].get("fps") or 16
export_to_video([Image.fromarray(frame) for frame in video], str(out_path), fps=fps)
print(out_path)
PY
```

Spot-check video previews:

```bash
find data/maze/latents/maze_384x384x161_line_v1/previews -maxdepth 1 -name '*.mp4' | sort | head
```

Each preview should show a red line being drawn through the maze, not a moving ball.
