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
- `eval/`: evaluation pipelines grouped by runtime/benchmark; see
  [`eval/README.md`](eval/README.md).
- `convert/`: checkpoint conversion launchers and compatibility wrappers.
- `download/`: model download helpers.
- `dev/`: local experiments and operator utilities.

## Common Entrypoints

```fish
fish scripts/train/i2v.fish --config configs/train_i2v.yaml
fish scripts/train/grpo.fish --config configs/train_grpo_maze.yaml
fish scripts/inference/i2v.fish --image path/to/image.jpg --prompt "..."
fish scripts/precompute/vbvr_384_webdataset_single_node.fish
fish scripts/precompute/maze_webdataset.fish --num_samples 20000
fish scripts/eval/vbvr/vbvr_generate_score.fish
fish scripts/convert/dcp_to_diffusers.fish
```

Most fish launchers source `scripts/lib/env.fish`, which changes to the repo
root, activates `.venv`, exports `PYTHONPATH`, and makes matching Python
development headers available to Triton through `CPATH` when possible. The
multi-node GRPO launcher also preflights the Triton CUDA driver before loading
the model.

When a cluster image has Python 3.12 runtime files but no `Python.h`, provision
the ignored shared toolchain once before submitting the multi-node job:

```fish
fish scripts/dev/bootstrap_triton_python_headers.fish
```

The bootstrap uses `uv`, then forces a fresh-cache Triton driver compilation.
`scripts/lib/env.fish` discovers the resulting versioned include directory on
every node without downloading during launch. For a cheap scheduler-wide check,
set `WAN_TRAINER_TRITON_PREFLIGHT_ONLY=1` on all nodes; rerun without it after
all nodes report that the preflight passed.
