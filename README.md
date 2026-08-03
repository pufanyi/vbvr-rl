# Wan-Trainer

Wan-Trainer is a research training stack for **Wan2.2 Image-to-Video** models. The current codebase supports supervised flow-matching fine-tuning, Chain-of-Step (COS) path training, on-policy correction, DanceGRPO-style replay, TP+FSDP full fine-tuning, latent WebDataset training, DCP checkpointing, LoRA extraction/loading, and VBVR-style evaluation.

The detailed English documentation lives in [`docs/`](docs/README.md). Start there if you need the full architecture and code-path analysis.

## Setup

```bash
# Python >= 3.12
uv sync --frozen
uv sync --frozen --check
```

The lockfile selects the official PyTorch 2.11 CUDA 12.6 wheels. The Python
media stack uses `decord2`, headless OpenCV, and a bundled FFmpeg/ffprobe
fallback, so a system FFmpeg installation is optional.

Expected local layout:

```text
storage/models/Wan2.2-I2V-A14B-Diffusers/
data/
storage/checkpoints/
storage/eval_out/
```

Most launchers source `scripts/lib/env.fish`, activate `.venv`, set `PYTHONPATH`, and run from the repository root.

## Cluster Profiles and Operational Notes

This repository currently runs on two cluster profiles. The labels below are
the aliases used by existing logs and configs; select a profile from its
observed hardware, mounts, and data source rather than assuming paths are
portable between clusters.

| Property | ACP / H100 private-mount profile | Fujian / H800 materialized-snapshot profile |
| --- | --- | --- |
| Typical node | 8 x 80-GiB H100 | 8 x 80-GiB H800 |
| Repository path seen in jobs | `/mnt/umm/users/pufanyi/workspace/Wan-Trainer` | `/mnt/umm/users/pufanyi/projects/Wan-Trainer` |
| VBVR-Pro data | Read-only private manifest and raw trees under `/mnt/aigc/...` and `/mnt/umm/users/xujunxiang/...` | Public `pufanyi/vbvr-pro-rl-indomain-50k` snapshot restored under `storage/datasets/vbvr-pro-rl-indomain-50k/materialized` |
| Production config | `configs/train_dancegrpo_vbvr_pro_5b_384x384x81_rule_cps_from_nsft_bs_32_lr_1e-6_manifest_rl.yaml` | `configs/train_dancegrpo_vbvr_pro_5b_512x512x81_rule_cps_from_nsft_bs_32_lr_5e-6_manifest_rl_fujian.yaml` |
| Multi-node topology | Scheduler-driven HSDP; node count varies by job | Current production job is 16 nodes x 8 GPUs, world size 128 |
| Local validation | Eight-H100 production-shape runs recorded in `docs/training.md` | `configs/train_dancegrpo_vbvr_pro_5b_384x384x81_rule_cps_from_nsft_bs_4_lr_1e-6_manifest_rl_local_1node_10step.yaml` |
| Shared/local storage | `/mnt/umm` is shared; use node-local `/tmp` for disposable work | `/mnt/umm` is QuarkFS; `/tmp` is node-local container storage |

### Invariants Across Both Clusters

- Run the same Git commit on every node. Use `uv sync --frozen` followed by
  `uv sync --frozen --check`; do not let one node silently resolve a different
  package set.
- Keep the locked Python 3.12, PyTorch 2.11 CUDA 12.6, OpenCV, EasyOCR, NumPy,
  SciPy, and scikit-image versions identical even when the host driver reports
  a newer maximum CUDA version.
- Prefer repository-relative model, EvalKit, and EasyOCR paths. Dataset
  descriptors are the intentional cluster-specific exception.
- Run the VBVR runtime preflight on every node before loading the model. The
  imported `cv2` version, `HoughLinesP` behavior, scorer-source hash, and
  runtime fingerprint must match.
- Never put credentials or the workstation proxy URL in a config, log, or
  commit. AOSS and proxy credentials remain environment/user-config driven.

The multi-node launcher expects scheduler variables on every node:

```bash
# WORLD_SIZE is the number of nodes, not the number of GPUs.
# RANK is the zero-based node rank; all nodes share MASTER_ADDR/MASTER_PORT.
MASTER_ADDR=<rank-0-host> MASTER_PORT=29500 \
WORLD_SIZE=<nodes> RANK=<node-rank> \
fish scripts/train/grpo_multinode.fish --nproc 8 -- \
  --config <cluster-config.yaml>
```

Before a new image or a newly scaled topology runs training, submit an
all-node preflight with the same scheduler variables:

```bash
WAN_TRAINER_TRITON_PREFLIGHT_ONLY=1 \
fish scripts/train/grpo_multinode.fish --nproc 8 -- \
  --config <cluster-config.yaml>
```

Remove `WAN_TRAINER_TRITON_PREFLIGHT_ONLY` only after every node passes.
Runtime-only images often omit `Python.h`; provision the ignored shared
toolchain once with
`fish scripts/dev/bootstrap_triton_python_headers.fish`, or preferably install
the matching `python3.12-dev` package in the image. Triton cache defaults to
node-local `/tmp/wan-trainer-triton-cache`.

### Cluster-Specific Data Rules

On the ACP/H100 profile, source trees owned by other users are read-only.
Write checkpoints, conversions, caches, and scorer output only beneath this
repository's `storage/` tree or node-local `/tmp`. The production descriptor
contains absolute private paths, so it is not expected to run on Fujian.

On the Fujian/H800 profile, restore the public raw snapshot before training:

```bash
.venv/bin/python -m scripts.data.vbvr_pro_unpack_hf \
  --dataset-root storage/datasets/vbvr-pro-rl-indomain-50k \
  --output-dir storage/datasets/vbvr-pro-rl-indomain-50k/materialized \
  --expected-samples 50000 --workers 8
```

The 59 downloaded shards are publication assets, not latent WebDataset input.
Restoring the training/reward-critical fields creates a standard 50,000-sample
small-file tree and duplicates roughly 56.2 GiB. Its manifest is task-grouped,
so Fujian configs use `shuffle_raw_indices: true` with a fixed seed.

Do not use a cluster-specific absolute repository path for scorer inputs.
Spawned VBVR workers change their working directory to the pinned EvalKit
checkout. `VBVRRuleReward` therefore resolves GT video, first/final frame,
metadata, and source-directory paths before crossing the process boundary.
Without that normalization, repo-relative data works in the trainer but becomes
missing in the worker; EvalKit can then return a valid-looking reward of
exactly zero without raising an exception.

### Topology and Memory Rules

- Production multi-node jobs use `hsdp: true`; bounded one-node validation uses
  `hsdp: false` and plain FSDP.
- Keep `grpo_fsdp_sync_each_backward: true` for full fine-tuning. Suppressing
  FSDP gradient synchronization retains full unsharded gradients and is an OOM
  trap, especially for A14B's two experts.
- Keep `grpo_offload_inference_models: true` when memory headroom matters; T5
  and VAE are restored for raw encoding/reward and offloaded before replay.
- `batch_size` is the global prompt count in shared-prompt GRPO. At world size
  64, `batch_size=32` and `G=32` produce 16 rollouts per rank. The equivalent
  eight-GPU validation uses `batch_size=4`, still 16 rollouts per rank. Running
  batch 32 on one node creates 128 rollouts per rank and is not a
  production-equivalent test.
- `grpo_shared_prompt_microbatch_size` must divide both the global prompt batch
  and the data-parallel world. `G` must divide the ranks assigned to each
  prompt.
- The current replay path reuses rollout chunks; it does not independently
  enforce a smaller `grpo_train_sample_batch_size`. Size memory from
  `grpo_sample_batch_size` until replay rechunking is implemented.

For VBVR reward work, keep generated/prepared temporary videos and Triton
artifacts under `/tmp`, not QuarkFS. `vbvr_reward_cpu_workers` is per
reward-producing rank, so multiply it by eight to estimate the node-wide
process/thread budget. The 5B manifest configs use two workers x eight native
threads per rank. Point `WANDB_DIR` at a writable run-local directory when the
shared repository's `wandb/` ownership is unsuitable.

### Reward-Zero Triage and Validated Boundaries

An all-zero hard-rule reward is not normal. Before blaming the model:

1. Run `.venv/bin/python -m src.cli.validate_grpo_runtime --config <config>`.
2. Search every rank for `VBVR rule reward failed`, OpenCV/EasyOCR warnings,
   and dependency fingerprint changes.
3. Preserve one online scorer input with `vbvr_reward_keep_tmp: true` and one
   rollout video, then score that exact MP4 with
   `scripts/dev/validate_vbvr_reward_alignment.py`.
4. Check that every GT path passed to the scorer is absolute and exists from
   the worker process. `vbvr_reward_fail_on_error: false` cannot expose this
   bug because missing GT may produce a numeric zero rather than an exception.
5. If a run loaded a contaminated scorer environment, restart the complete job
   from the last clean checkpoint; repairing packages on disk does not replace
   modules already imported by scorer workers.

The path-boundary bug was isolated with a G-21 rollout: training initially
reported zero, while the preserved online `generated_raw.mp4` independently
scored `0.99115`. After the fix, the real eight-H800 production-scaled config
completed 10 full-FT optimizer steps at 384x384x81, `G=32`, `T=30`, and
Flow-CPS with nonzero reward at every step (`0.4437` to `0.6467`, mean
`0.5589`), gradient norms `0.0001` to `0.0002`, and a 25.7/28.4-GiB
allocated/reserved peak.

The same fix was also active in the earlier Fujian world-64 384x384 job. As of 2026-07-30, its
first 19 optimizer steps all had nonzero reward (`0.5102` to `0.6924`) with a
53.3/58.2-GiB allocated/reserved peak. This is strong early-run evidence, not a
claim about the current world-128 512x512 job. A new multi-node topology should
still be monitored through its first optimizer step before committing to a long
run.

Resume semantics also matter across clusters. `auto_resume: true` combined with
`reset_dataloader: true` loads the latest checkpoint as weight-only
initialization and restarts optimizer/step/epoch/dataloader state. It therefore
repeats the epoch-0 sample permutation. Use isolated output/W&B/tmp namespaces
for different clusters, resolutions, scorer revisions, learning rates, and
delayed-replay modes; never allow one profile to auto-resume the other's run.

## Main Workflows

Supervised I2V / COS:

```fish
fish scripts/train/i2v.fish --nproc 8 -- --config configs/train_sft_vbvr.yaml
fish scripts/train/i2v.fish --nproc 8 -- --config configs/train_cos_maze_cos_path_all_bfs_w_color_latent.yaml
fish scripts/train/i2v.fish --nproc 8 -- --config configs/train_sft_maze_lr_5e-6.yaml
```

DanceGRPO:

```fish
fish scripts/train/grpo.fish --nproc 8 --config configs/train_grpo_maze.yaml
fish scripts/train/grpo.fish --nproc 8 --config configs/train_dancegrpo_maze.yaml
fish scripts/train/dancegrpo_maze_split_multinode.fish --nproc 8

# Single-node A14B full fine-tuning: TP2 x FSDP4, global prompt batch 16.
fish scripts/train/grpo.fish --nproc 8 --config \
  configs/train_dancegrpo_vbvr_pro_a14b_256x256x161_rule_cps_from_sft_diffsynth_mix_260603_bs_16_lr_1e-5_full_tp2_fsdp4.yaml

# Four-node counterpart: TP2 x FSDP16, still global prompt batch 16.
# Run the same command on every scheduler node with WORLD_SIZE=4 and RANK=0..3.
fish scripts/train/dancegrpo_vbvr_pro_a14b_full_tp2_4node.fish
```

Single-GPU official Wan2.2-TI2V-5B end-to-end smoke:

```bash
.venv/bin/python scripts/dev/create_i2v_smoke_dataset.py \
  --output-dir storage/smoke/i2v_512x512x81 \
  --samples 4 --frames 81 --height 512 --width 512 --fps 16

.venv/bin/torchrun --standalone --nproc_per_node=1 \
  -m scripts.dev.validate_grpo_parameter_update \
  --config configs/train_dancegrpo_vbvr_pro_5b_512x512x81_official_base_smoke_1gpu.yaml
```

This smoke uses LoRA and the model-internal `neg_loss` reward. It verifies raw
T5/VAE encoding, Flow-CPS rollout, replay, backward, and a nonzero optimizer
update; it does not replace the full-FT, multi-node `vbvr_rule` production run.
To exercise the locally downloaded merged DiffSynth step-35500 with the same
bounded smoke, override only the model and output paths:

```bash
.venv/bin/torchrun --standalone --nproc_per_node=1 \
  -m src.cli.train_grpo \
  --config configs/train_dancegrpo_vbvr_pro_5b_512x512x81_official_base_smoke_1gpu.yaml \
  --model_path storage/models/diffsynth_converted_5b/wan2.2-TI2V-5B_260715_vbvr_pro_step-35500 \
  --output_dir storage/checkpoints/dancegrpo_vbvr_pro_5b_512x512x81_step35500_smoke_1gpu
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

fish scripts/eval/vbvr/vbvr_generate_score.fish
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
- [`docs/training.md`](docs/training.md): SFT, COS, correction, and DanceGRPO behavior.
- [`docs/data.md`](docs/data.md): raw and latent dataset contracts.
- [`docs/evaluation.md`](docs/evaluation.md): generation, VBVR, VLM/rule scoring.
- [`docs/vbvr_pro_eval.md`](docs/vbvr_pro_eval.md): VBVR-Pro main_v2 eight-GPU evaluation.
- [`docs/checkpoints.md`](docs/checkpoints.md): DCP, resume/init, LoRA, EMA.
- [`docs/improvements/`](docs/improvements/README.md): algorithm-to-engineering improvement plan.
