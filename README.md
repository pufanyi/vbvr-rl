# Wan-Trainer

Wan-Trainer is a research training stack for **Wan2.2 Image-to-Video** models. The current codebase supports supervised flow-matching fine-tuning, Chain-of-Step (COS) path training, on-policy correction, Flow-GRPO, DanceGRPO-style replay, latent WebDataset training, DCP checkpointing, LoRA extraction/loading, and VBVR-style evaluation.

The detailed English documentation lives in [`docs/`](docs/README.md). Start there if you need the full architecture and code-path analysis.

## Setup

```bash
# Python >= 3.12
uv sync
```

Expected local layout:

```text
storage/models/Wan2.2-I2V-A14B-Diffusers/
data/
storage/checkpoints/
storage/eval_out/
```

Most launchers source `scripts/lib/env.fish`, activate `.venv`, set `PYTHONPATH`, and run from the repository root.

## Main Workflows

Supervised I2V / COS:

```fish
fish scripts/train/i2v.fish --nproc 8 -- --config configs/train_sft_vbvr.yaml
fish scripts/train/i2v.fish --nproc 8 -- --config configs/train_cos_maze_cos_path_all_bfs_w_color_latent.yaml
fish scripts/train/i2v.fish --nproc 8 -- --config configs/train_sft_maze_lr_5e-6.yaml
```

Flow-GRPO / DanceGRPO:

```fish
fish scripts/train/grpo.fish --nproc 8 --config configs/train_grpo_maze.yaml
fish scripts/train/grpo.fish --nproc 8 --config configs/train_dancegrpo_maze.yaml
```

On-policy correction:

```fish
fish scripts/train/i2v_correction.fish --nproc 8 -- --config configs/train_correction_vbvr.yaml
```

Latent precompute:

```fish
fish scripts/precompute/maze_webdataset.fish --num_samples 20000

.venv/bin/torchrun --nproc_per_node=8 -m src.precompute.i2v_latent_webdataset \
  --config configs/train_sft_maze.yaml \
  --output_dir data/maze/latents/webdataset \
  --batch_size 4 \
  --samples_per_shard 1000
```

Inference and evaluation:

```bash
.venv/bin/python -m src.cli.infer_i2v \
  --image path/to/image.jpg \
  --prompt "A concise I2V prompt." \
  --output storage/outputs/sample.mp4

fish scripts/eval/vbvr_generate_score.fish
```

## Data Inputs

Raw training uses a JSON config that points to one or more Parquet files:

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

Each Parquet row should contain:

- `videos`: ordered `list<string>` for COS or multi-step chains, where the last item is the final target.
- `video`: single target video path, used when `videos` is absent.
- `prompt`: text prompt.
- `image`: optional reference image. If omitted, the first frame of the final video is used.

Latent training uses `latent_webdataset_dir` pointing at `shard-*.tar` files. Each sample stores `prompt_embeds`, `condition`, and either `latents` or `latents_0`, `latents_1`, ... for COS chains. Set `dataset_size` in latent configs so schedules and epoch lengths are well-defined.

## Repository Map

```text
src/cli/          entry points for training, inference, evaluation, conversion
src/models/       Wan2.2 training wrapper and COS path implementations
src/data/         raw Parquet and latent WebDataset loaders
src/trainer/      SFT, COS, correction, GRPO, checkpointing, EMA, optimizers
src/precompute/   VAE/T5 latent precompute and synthetic maze generation
src/eval/         VBVR generation/result tooling and VLM judge
configs/          runnable training configs
scripts/          fish launchers and operator utilities
tests/            focused unit/consistency checks
docs/             architecture, training, data, evaluation, and improvement docs
```

## Checkpoints

Training checkpoints use PyTorch Distributed Checkpoint (DCP). New checkpoints are written with a unified expert layout:

```text
checkpoint-N/
  high/
    .metadata
    *.distcp
    optimizer_transformer_rank*.pt
    dataloader_rank*.pt
    lora/transformer/
  low/
    .metadata
    *.distcp
    optimizer_transformer_2_rank*.pt
    dataloader_rank*.pt
    lora/transformer_2/
```

Use `--checkpoint <checkpoint-dir> --use_ema` with `src.cli.eval_i2v` to generate from a DCP checkpoint. Conversion helpers live under `src/cli/convert_dcp_to_diffusers.py` and `src/cli/convert_dcp_to_lora.py`.

## Documentation

- [`docs/architecture.md`](docs/architecture.md): system architecture and code-path analysis.
- [`docs/training.md`](docs/training.md): SFT, COS, correction, Flow-GRPO, and DanceGRPO behavior.
- [`docs/data.md`](docs/data.md): raw and latent dataset contracts.
- [`docs/evaluation.md`](docs/evaluation.md): generation, VBVR, VLM/rule scoring.
- [`docs/checkpoints.md`](docs/checkpoints.md): DCP, resume/init, LoRA, EMA.
- [`docs/improvements/`](docs/improvements/README.md): algorithm-to-engineering improvement plan.
