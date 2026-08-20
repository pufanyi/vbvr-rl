# Scripts

This directory contains launchers and bounded operator utilities. Reusable
training, data, checkpoint, and evaluation logic belongs under `src/`.
Generated artifacts belong under ignored directories such as `storage/`.

## Layout

| Directory | Purpose |
| --- | --- |
| `lib/` | Shared Fish environment setup |
| `train/` | Single- and multi-machine training launchers |
| `inference/` | General inference and sampler utilities |
| `precompute/` | Latent and WebDataset preparation |
| `data/` | Dataset packaging, materialization, shuffling, and upload tools |
| `eval/` | Benchmark-specific evaluation and reporting launchers |
| `convert/` | Checkpoint conversion wrappers |
| `download/` | External model download helpers |
| `serve/` | Standalone services used by rewards/evaluation |
| `dev/` | Bounded diagnostics and contract validators |

## Common Entrypoints

Create a deterministic raw-media fixture:

```bash
pixi run python scripts/dev/create_i2v_smoke_dataset.py \
  --output-dir storage/smoke/i2v_512x512x81 \
  --samples 4 --frames 81 --height 512 --width 512 --fps 16
```

Launch SFT:

```fish
pixi run fish scripts/train/i2v.fish --nproc 8 -- \
  --config configs/<reviewed-sft-config>.yaml
```

Launch DanceGRPO:

```fish
pixi run fish scripts/train/grpo.fish --nproc 8 \
  --config configs/<reviewed-rl-config>.yaml
```

Launch multi-machine DanceGRPO on every machine:

```bash
MASTER_ADDR=<rank-zero-host> MASTER_PORT=29500 \
WORLD_SIZE=<machine-count> RANK=<machine-rank> \
pixi run fish scripts/train/grpo_multinode.fish --nproc 8 -- \
  --config configs/<reviewed-rl-config>.yaml
```

Evaluate a VBVR-Pro checkpoint:

```fish
pixi run fish scripts/eval/vbvr_pro/run.fish \
  --checkpoint storage/checkpoints/<run>/checkpoint-100 \
  --converted-model storage/models/converted/<run>-checkpoint-100 \
  --output-root storage/eval_out/<run>/checkpoint-100/unipc \
  --sampler unipc \
  --dry-run
```

Convert a DCP checkpoint:

```bash
pixi run python -m src.cli.convert_dcp_to_diffusers \
  --checkpoint storage/checkpoints/<run>/checkpoint-100 \
  --base_model storage/models/Wan2.2-TI2V-5B-Diffusers \
  --output storage/models/converted/<run>-checkpoint-100
```

`scripts/convert/dcp_to_diffusers.fish` is the batch wrapper for discovering
and converting checkpoints beneath `CHECKPOINT_ROOT`.

## Launcher Conventions

Most Fish launchers source `scripts/lib/env.fish`. It enters the repository
root, activates the locked Pixi default environment, sets `PYTHONPATH`, and
exposes matching Python headers to Triton when available.

For multi-machine launchers:

- `WORLD_SIZE` is the machine count;
- `RANK` is the zero-based machine rank;
- `--nproc` is the number of local processes;
- the global process count is `WORLD_SIZE * --nproc`.

The GRPO launchers run cheap reward/attention checks before model loading. The
multi-machine launcher additionally exercises Triton's CUDA driver setup on
every machine. Use `WAN_TRAINER_TRITON_PREFLIGHT_ONLY=1` for a preflight-only
run.

## Public VBVR-Pro Dataset

[`data/vbvr_pro_unpack_hf.py`](data/vbvr_pro_unpack_hf.py) materializes the
official `Video-Reason/VBVR-Pro-RL` video archives into the `I2VDataset`
layout:

```bash
pixi run python -m scripts.data.vbvr_pro_unpack_hf \
  --dataset-root storage/datasets/VBVR-Pro-RL \
  --output-dir storage/datasets/VBVR-Pro-RL/materialized \
  --source-revision ca0aaffea93b07d269c6fe2fbfe533f1fdab9aa1 \
  --expected-tasks 50 --expected-samples 50000 --workers 8
```

Download only `VBVR-Pro-RL-Video/*.tar.gz`; those archives already include all
five fields required by raw I2V training and rule reward. The materializer
does not use unsafe whole-archive extraction and does not republish data.

## VLM Service

The optional Qwen judge uses an isolated runtime:

```fish
pixi run fish scripts/dev/setup_host_vllm.fish
pixi run fish scripts/download/qwen36_27b_hf_mirror.fish
pixi run fish scripts/serve/qwen36_27b_vllm.fish
```

Co-hosted training wrappers manage the service process group, probe its
multimodal and task-schema paths, delegate to the standard GRPO launcher, and
stop the service at exit. See
[`docs/vlm_judge_reward.md`](../docs/vlm_judge_reward.md).

## Development Rules

- Keep launchers thin and use Python modules for reusable behavior.
- Preserve explicit operator overrides; defaults should be repository-relative
  and safe for a fresh checkout.
- Validate required environment variables before starting expensive work.
- Use unique output namespaces for different model, data, sampler, or scorer
  contracts.
- Ensure background services and subprocess groups are stopped on success,
  failure, and signals.
- Run `fish -n <changed-launcher>` before committing.
