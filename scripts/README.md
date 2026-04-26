# Scripts

This directory only contains runnable launchers and small operator utilities.
Reusable Python logic should live under `src/`; generated outputs should live
under `storage/`, `data/`, or `docs/assets/`.

## Layout

- `lib/`: shared fish setup helpers.
- `train/`: single-node and multi-node training launchers.
- `inference/`: general inference and VAE smoke-test launchers.
- `precompute/`: latent, WebDataset, and benchmark precompute launchers.
- `data/`: dataset packaging, shuffling, and upload utilities.
- `eval/`: evaluation pipelines and compatibility wrappers for `src.eval`.
- `convert/`: checkpoint conversion launchers and compatibility wrappers.
- `download/`: model download helpers.
- `dev/`: local experiments and operator utilities.

## Common Entrypoints

```fish
fish scripts/train/i2v.fish --config configs/train_i2v.yaml
fish scripts/train/grpo.fish --config configs/train_grpo_maze.yaml
fish scripts/inference/i2v.fish --image path/to/image.jpg --prompt "..."
fish scripts/precompute/maze_webdataset.fish --num_samples 20000
fish scripts/eval/vbvr_generate_score.fish
fish scripts/convert/dcp_to_diffusers.fish
```

Most fish launchers source `scripts/lib/env.fish`, which changes to the repo
root, activates `.venv`, and exports `PYTHONPATH`.
