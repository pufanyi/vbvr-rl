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
- `serve/`: standalone model services used by training and evaluation.
- `dev/`: local experiments and operator utilities.

## Common Entrypoints

```fish
fish scripts/train/i2v.fish --config configs/train_i2v.yaml
fish scripts/train/grpo.fish --config configs/train_grpo_maze.yaml
fish scripts/train/grpo_vlm_eval_multinode.fish --config configs/train_dancegrpo_vbvr_pro_5b_384x384x81_vlm_qwen36_smoke_1node_3step.yaml
fish scripts/train/grpo_vlm_eval_cluster.fish --yaml=configs/train_dancegrpo_vbvr_pro_5b_512x512x81_vlm_qwen36_cps_from_nsft_bs_32_lr_5e-6_manifest_rl_multinode.yaml
fish scripts/inference/i2v.fish --image path/to/image.jpg --prompt "..."
fish scripts/precompute/vbvr_384_webdataset_single_node.fish
fish scripts/precompute/maze_webdataset.fish --num_samples 20000
fish scripts/eval/vbvr/vbvr_generate_score.fish
fish scripts/convert/dcp_to_diffusers.fish
```

`scripts/data/vbvr_pro_pack_hf.py` converts a manifest-selected raw VBVR-Pro
view into deterministic, lossless WebDataset shards suitable for a Hugging
Face Dataset repository. It also writes a sanitized source manifest, per-file
and per-shard checksums, a privacy/credential audit, and a dataset card:

```bash
.venv/bin/python scripts/data/vbvr_pro_pack_hf.py \
  --dataset-json data/vbvr_pro/vbvr_pro_rl_indomain_256x256x161_evalkit_6fedd9d9.json \
  --output-dir storage/hf/vbvr-pro-rl-indomain-50k \
  --repo-id pufanyi/vbvr-pro-rl-indomain-50k \
  --license-file /path/to/VBVR-Pro/LICENSE \
  --expected-samples 50000
```

Most fish launchers source `scripts/lib/env.fish`, which changes to the repo
root, activates `.venv`, exports `PYTHONPATH`, and makes matching Python
development headers available to Triton through `CPATH` when possible. The
multi-node GRPO launcher also reads the selected reward from the config and,
for `vbvr_rule`, validates the pinned scorer dependencies and OpenCV
`HoughLinesP` behavior on every node before loading the model. It then
preflights the Triton CUDA driver.

The generic VLM wrapper accepts the same scheduler environment and arguments
as `grpo_multinode.fish`, starts one node-local Qwen3.6 vLLM endpoint, probes
its generic vision/JSON path and exact task-prompt/regex path, and then
delegates to the standard launcher. The cluster wrapper detects 4/8/16 nodes,
defaults every endpoint to four local TP2 replicas with vLLM internal DP, and
isolates output names by world size. All topology controls remain
environment-variable overrides.
See [`docs/vlm_judge_reward.md`](../docs/vlm_judge_reward.md) for setup and GPU
memory controls.

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
