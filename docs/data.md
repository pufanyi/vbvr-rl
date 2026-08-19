# Data and Precompute

VBVR-RL has two data paths: raw media through a Parquet-backed dataset, and precomputed latents through WebDataset tar shards.

## Raw I2V Dataset

The raw dataset is configured by a JSON file whose entries point to Parquet files:

```json
[
  {
    "data_path": "/path/to/train.parquet",
    "root": "/path/to/media/root",
    "num_frames": 81,
    "height": 256,
    "width": 256,
    "fps": 16
  }
]
```

The Parquet schema is intentionally small:

- `videos: list<string>`: ordered video chain, used by COS and multi-step training.
- `video: string`: single target video, used when `videos` is absent.
- `prompt: string`: text prompt.
- `image: string`: optional reference image; if absent, the first frame of the final video is used.[^i2v-dataset]

Paths are resolved against `root`, or against the Parquet parent if `root` is absent. Resolution is either fixed by config (`height` + `width`) or computed from `max_area` and the final video's aspect ratio. The result is rounded down to a multiple of 16, matching Wan2.2 VAE spatial scale and transformer patching assumptions.[^i2v-dataset]

## Raw Batch Contract

`I2VDataset.__getitem__` returns:

```python
{
    "index": idx,
    "videos": [video_0, ..., final_video],  # each (C, T, H, W), uint8
    "image": image,  # (C, H, W), uint8
    "prompt": prompt,
}
```

The common `collate` stacks tensor fields and keeps `videos` as a list of batched tensors, preserving chain order for COS.[^trainer-utils]

## Local Raw Smoke Fixture

Use the deterministic fixture generator when production media paths are not
mounted but the complete raw data path still needs validation:

```bash
.venv/bin/python scripts/dev/create_i2v_smoke_dataset.py \
  --output-dir storage/smoke/i2v_512x512x81 \
  --samples 4 --frames 81 --height 512 --width 512 --fps 16
```

It writes H.264 MP4s, first-frame PNGs, `samples.parquet`, and `dataset.json`
under the ignored `storage/` tree. The resulting descriptor is consumed by
`configs/train_dancegrpo_vbvr_pro_5b_512x512x81_official_base_smoke_1gpu.yaml`.[^smoke-data]

## Latent WebDataset

Latent training uses `VBVRLatentDataset`, an `IterableDataset` over `shard-*.tar` files.[^latent-dataset] Each tar sample contains:

```text
{key}.safetensors
{key}.json
```

The safetensors payload must contain:

- `prompt_embeds`: variable or fixed length text embeddings;
- `condition`: first-frame condition tensor;
- either `latents` for normal SFT/GRPO/correction, or `latents_0`, `latents_1`, ... for COS chains.

The loader pads/truncates prompt embeddings to 512 tokens and passes through non-reserved tensor keys. The pass-through path is used by MazeReward for tensors such as `maze_grid`, `maze_frame_positions_pix`, `maze_goal`, and `maze_ball_rgb`.[^latent-dataset][^maze-reward]

Because this is an iterable dataset, configs should set `dataset_size`. The trainer uses it to compute total optimizer steps and rank-local epoch lengths, preventing uneven-rank epoch endings that can deadlock FSDP/NCCL.[^base-trainer]

## Precompute From Existing I2V Data

`src.precompute.i2v_latent_webdataset` converts a raw Parquet-backed training config into the latent WebDataset contract.[^i2v-latent-precompute]

Example:

```bash
.venv/bin/torchrun --nproc_per_node=8 -m src.precompute.i2v_latent_webdataset \
  --config configs/train_sft_maze.yaml \
  --output_dir data/maze/latents/webdataset \
  --batch_size 4 \
  --samples_per_shard 1000
```

For COS, include every chain waypoint:

```bash
.venv/bin/torchrun --nproc_per_node=8 -m src.precompute.i2v_latent_webdataset \
  --config configs/train_cos_maze_cos_path_all_bfs_w_color.yaml \
  --output_dir data/maze_cos/latents/webdataset \
  --encode_all_videos
```

The precompute script writes `dataset_info.json` with recommended `latent_webdataset_dir` and `dataset_size` fields.[^i2v-latent-precompute]

## Synthetic Maze WebDataset

`src.precompute.maze_webdataset` generates synthetic mazes, renders ball-trajectory videos, encodes video latents and first-frame conditions with the Wan VAE, encodes prompts with UMT5, and writes WebDataset shards with extra `maze_*` reward tensors.[^maze-webdataset]

The fish launcher defaults to:

```fish
fish scripts/precompute/maze_webdataset.fish --num_samples 20000
```

Important generated tensors:

- `maze_grid`: wall/passage grid.
- `maze_frame_positions_pix`: expected ball positions per frame.
- `maze_goal`: goal cell.
- `maze_ball_rgb`: sample-specific ball color.
- `maze_cell_px`, `maze_image_hw`: geometry needed by MazeReward.[^maze-webdataset][^maze-reward]

## VBVR Precompute

The VBVR precompute code has separate VAE and text paths:

- `src.precompute.vbvr_vae_latents` writes one safetensors file per sample with `latents` and `condition`.[^vbvr-vae]
- `src.precompute.vbvr_prompt_embeds` writes one safetensors file per sample with `prompt_embeds`.[^vbvr-t5]
- packaging/shuffle helpers under `scripts/data/` and `src.precompute.build_webdataset` are used to assemble final tar shards.[^scripts-readme]

## Published VBVR-Pro RL Snapshot

The manifest-selected In-Domain RL view is published as the public Hugging Face
Dataset [`pufanyi/vbvr-pro-rl-indomain-50k`](https://huggingface.co/datasets/pufanyi/vbvr-pro-rl-indomain-50k).
It contains 50,000 samples from 50 tasks in 59 lossless WebDataset shards
(62,060,892,160 archive bytes). The source descriptor SHA-256 is
`1d397525869794cd3b608223f35bbb550b217e113be00bd1e913124c27507ac4`;
the selected split manifest SHA-256 is
`8eb86bf31b24dc5a21deb03a8294c15d299731db0f6f1ea0ddfcd3fd36619f32`.

Each sample exposes `json`, `first.png`, `image_prompt.txt`,
`metadata.json.bin`, `final.png`, `gt.mp4`, `video_prompt.txt`, and
`extras.zip.bin`. The ZIP retains the original image sequence and any
unrecognized source files, so the archive is a lossless packaging of the
selected source directories. Use
[`scripts/data/vbvr_pro_pack_hf.py`](../scripts/data/vbvr_pro_pack_hf.py) to
rebuild it; the generated `SHA256SUMS`, `samples.jsonl`,
`source_manifest.json`, `dataset_config.json`, and `audit.json` provide the
integrity and publication audit trail.

The published shards are raw backup assets rather than the latent tensors
accepted by `latent_webdataset_dir`. Restore the fields required by raw
training and `vbvr_rule` into an ignored standard VBVR-Pro tree before using
the raw-data configs:

```bash
hf download pufanyi/vbvr-pro-rl-indomain-50k \
  --repo-type dataset \
  --local-dir storage/datasets/vbvr-pro-rl-indomain-50k
```

```bash
.venv/bin/python -m scripts.data.vbvr_pro_unpack_hf \
  --dataset-root storage/datasets/vbvr-pro-rl-indomain-50k \
  --output-dir storage/datasets/vbvr-pro-rl-indomain-50k/materialized \
  --expected-samples 50000 --workers 8
```

The command verifies every newly written field against `samples.jsonl` and
writes `materialized/dataset.json`. It is resumable and restores only the five
training/reward-critical fields, requiring about 56.2 GiB in addition to the
downloaded tar snapshot.

## Dataset Design Tradeoffs

The current design makes GPU training fast by moving expensive VAE/T5 work offline. The cost is a stricter data contract:

- latent shards must be regenerated when video resolution, number of frames, prompt cleaning, VAE normalization, or condition construction changes;
- COS requires all chain latents, not just the final latent;
- rewards that need pixels still need the VAE loaded at training time;
- WebDataset exact epoch sizing must be managed explicitly with `dataset_size`.

[^i2v-dataset]: [`src/data/i2v_dataset.py`](../src/data/i2v_dataset.py)
[^trainer-utils]: [`src/trainer/utils.py`](../src/trainer/utils.py)
[^smoke-data]: [`scripts/dev/create_i2v_smoke_dataset.py`](../scripts/dev/create_i2v_smoke_dataset.py)
[^latent-dataset]: [`src/data/vbvr_latent_dataset.py`](../src/data/vbvr_latent_dataset.py)
[^maze-reward]: [`src/trainer/rewards/maze.py`](../src/trainer/rewards/maze.py)
[^base-trainer]: [`src/trainer/base_trainer.py`](../src/trainer/base_trainer.py)
[^i2v-latent-precompute]: [`src/precompute/i2v_latent_webdataset.py`](../src/precompute/i2v_latent_webdataset.py)
[^maze-webdataset]: [`src/precompute/maze_webdataset.py`](../src/precompute/maze_webdataset.py)
[^vbvr-vae]: [`src/precompute/vbvr_vae_latents.py`](../src/precompute/vbvr_vae_latents.py)
[^vbvr-t5]: [`src/precompute/vbvr_prompt_embeds.py`](../src/precompute/vbvr_prompt_embeds.py)
[^scripts-readme]: [`scripts/README.md`](../scripts/README.md)
