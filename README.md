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

Expected repository layout:

```text
storage/models/Wan2.2-I2V-A14B-Diffusers/
data/
storage/checkpoints/
storage/eval_out/
```

Most launchers source `scripts/lib/env.fish`, activate `.venv`, set `PYTHONPATH`, and run from the repository root.

## Runtime and Data Setup

The checked-in configs use repository-relative paths. Keep models, datasets,
checkpoints, evaluator checkouts, and generated outputs beneath the ignored
`storage/` tree; use `/tmp` only for disposable per-process artifacts. Supply
object-store credentials and proxy settings through the runtime environment,
never through committed files.

### Public VBVR-Pro Data

The runnable manifest-RL configs use the public
`pufanyi/vbvr-pro-rl-indomain-50k` snapshot. After downloading it beneath
`storage/datasets/vbvr-pro-rl-indomain-50k`, materialize the raw training tree:

```bash
.venv/bin/python -m scripts.data.vbvr_pro_unpack_hf \
  --dataset-root storage/datasets/vbvr-pro-rl-indomain-50k \
  --output-dir storage/datasets/vbvr-pro-rl-indomain-50k/materialized \
  --expected-samples 50000 --workers 8
```

The downloaded shards are publication assets, not latent WebDataset input.
Restoring the training/reward-critical fields creates the standard 50,000-sample
small-file tree consumed by the configs. Its manifest is task-grouped, so the
configs apply a deterministic raw-index shuffle.

Do not use host-specific absolute paths for scorer inputs.
Spawned VBVR workers change their working directory to the pinned EvalKit
checkout. `VBVRRuleReward` therefore resolves GT video, first/final frame,
metadata, and source-directory paths before crossing the process boundary.
Without that normalization, repo-relative data works in the trainer but becomes
missing in the worker; EvalKit can then return a valid-looking reward of
exactly zero without raising an exception.

### Optional Remote Media

The public configs use local files and require no object-store client. Custom
raw-data descriptors may still reference `s3://` media. To enable those paths,
set `WAN_TRAINER_REMOTE_DOWNLOADER` to an external command prefix; the loader
appends the source URI and local destination as two arguments and invokes the
command without a shell. For example:

```bash
export WAN_TRAINER_REMOTE_DOWNLOADER='/path/to/downloader --profile training'
export WAN_TRAINER_REMOTE_CACHE_DIR=storage/remote_cache
```

Install and authenticate that downloader outside the repository. Do not put
credentials in the command itself because process listings may expose its
arguments.

### Distributed Runtime

Run the same Git commit and locked environment on every machine. The launcher
expects `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE` (machine count), and `RANK`
(zero-based machine rank):

```bash
MASTER_ADDR=<rank-0-host> MASTER_PORT=29500 \
WORLD_SIZE=<machines> RANK=<machine-rank> \
fish scripts/train/grpo_multinode.fish --nproc 8 -- \
  --config configs/train_dancegrpo_vbvr_pro_5b_512x512x81_rule_cps_from_nsft_bs_32_lr_5e-6_manifest_rl.yaml
```

Before a new image or topology runs training, execute the all-machine compiler
and runtime preflight with the same scheduler variables:

```bash
WAN_TRAINER_TRITON_PREFLIGHT_ONLY=1 \
fish scripts/train/grpo_multinode.fish --nproc 8 -- \
  --config configs/train_dancegrpo_vbvr_pro_5b_512x512x81_rule_cps_from_nsft_bs_32_lr_5e-6_manifest_rl.yaml
```

Use `uv sync --frozen --check` to verify dependency parity. Run
`.venv/bin/python -m src.cli.validate_grpo_runtime --grpo_reward_fn vbvr_rule`
on every machine before loading model weights. Runtime-only images also need
matching Python development headers for fresh Triton builds; install the
matching `python3.12-dev` package or run
`fish scripts/dev/bootstrap_triton_python_headers.fish` once.

The Triton cache defaults to node-local `/tmp/wan-trainer-triton-cache`.
Downloaded attention kernels persist under `~/.cache/wan-trainer/kernels` or
the path supplied by `WAN_TRAINER_KERNELS_CACHE`. Prefetch the pinned FA3
artifact before submitting an offline job:

```bash
.venv/bin/python -m src.cli.prefetch_attention_kernel --backend _flash_3_hub
```

### Topology and Memory Rules

- Multi-node full fine-tuning uses `hsdp: true`; bounded one-node validation uses
  `hsdp: false` and plain FSDP.
- Keep `grpo_fsdp_sync_each_backward: true` for full fine-tuning. Suppressing
  FSDP gradient synchronization retains full unsharded gradients and is an OOM
  trap, especially for A14B's two experts.
- Keep `grpo_offload_inference_models: true` when memory headroom matters; T5
  and VAE are restored for raw encoding/reward and offloaded before replay.
- `batch_size` is the global prompt count in shared-prompt GRPO. The checked-in
  `batch_size=32`, `G=32`, and prompt-wave settings target world size 128.
  Adjust all three together when scaling to a different world size.
- `grpo_shared_prompt_microbatch_size` must divide both the global prompt batch
  and the data-parallel world. `G` must divide the ranks assigned to each
  prompt.
- The current replay path reuses rollout chunks; it does not independently
  enforce a smaller `grpo_train_sample_batch_size`. Size memory from
  `grpo_sample_batch_size` until replay rechunking is implemented.

For VBVR reward work, keep generated/prepared temporary videos under `/tmp`.
`vbvr_reward_cpu_workers` is per reward-producing rank, so size the aggregate
process/thread budget for the host and validate `reward_drain` before raising
worker or native-thread counts. Point `WANDB_DIR` at a writable run directory.

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

Resume semantics also matter across experiments. `auto_resume: true` combined with
`reset_dataloader: true` loads the latest checkpoint as weight-only
initialization and restarts optimizer/step/epoch/dataloader state. It therefore
repeats the epoch-0 sample permutation. Use isolated output/W&B/tmp namespaces
for different resolutions, scorer revisions, learning rates, and delayed-replay
modes; never auto-resume an incompatible run.

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

# Co-host a Qwen3.6 VLM judge per node, then run the standard scheduler
# contract (MASTER_ADDR/WORLD_SIZE/RANK) for a three-step smoke.
fish scripts/train/grpo_vlm_eval_multinode.fish --nproc 8 --config \
  configs/train_dancegrpo_vbvr_pro_5b_384x384x81_vlm_qwen36_smoke_1node_3step.yaml

# WORLD_SIZE=4/8/16: four TP2 judge replicas per machine.
fish scripts/train/grpo_vlm_eval_scaleout.fish \
  --yaml=configs/train_dancegrpo_vbvr_pro_5b_512x512x81_vlm_qwen36_cps_from_nsft_bs_32_lr_5e-6_manifest_rl_multinode.yaml \
  --max_steps 1 --save_steps 0 --no-save_final_checkpoint --no-auto_resume

# Lower-pressure native-384 variant.
fish scripts/train/grpo_vlm_eval_scaleout.fish \
  --yaml=configs/train_dancegrpo_vbvr_pro_5b_384x384x81_vlm_qwen36_cps_from_nsft_bs_32_lr_5e-6_manifest_rl_multinode.yaml \
  --max_steps 1 --save_steps 0 --no-save_final_checkpoint --no-auto_resume

# Single-node A14B full fine-tuning: TP2 x FSDP4, global prompt batch 16.
fish scripts/train/grpo.fish --nproc 8 --config \
  configs/train_dancegrpo_vbvr_pro_a14b_256x256x161_rule_cps_from_sft_diffsynth_mix_260603_bs_16_lr_1e-5_full_tp2_fsdp4.yaml

# Four-node counterpart: TP2 x FSDP16, still global prompt batch 16.
# Run the same command on every scheduler node with WORLD_SIZE=4 and RANK=0..3.
fish scripts/train/dancegrpo_vbvr_pro_a14b_full_tp2_4node.fish
```

The VLM co-hosting design, isolated vLLM environment, Qwen model download,
reward contract, and true multi-node vLLM alternatives are documented in
[`docs/vlm_judge_reward.md`](docs/vlm_judge_reward.md).

The VLM run's incremental formal evaluator now generates/EvalKit-scores missing
cells and then automatically fills only missing task-specific Qwen judgments:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_vlm_qwen36_512x512x81/evaluate_incremental_multinode.fish \
  formal --nproc 8
```

On one evaluation machine, omit scheduler variables; the adapter defaults to
`WORLD_SIZE=1` and `RANK=0`, while `--nproc` remains the local GPU count. For
multiple machines, run it on every node with both variables set. Completed
formal and VLM cells are audited and skipped; a node with no pending judge work
does not start Qwen. Use `--no-vlm-judge` for an EvalKit-only invocation.

To score another existing formal VBVR-Pro video tree with that same
task-specific Qwen contract, without running Wan inference again:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_vlm_qwen36_512x512x81/evaluate_vlm_judge_multinode.fish \
  score --input-root /path/to/formal-result-root --concurrency 16
```

The command is single-node by default and uses evaluation-machine
`WORLD_SIZE/RANK` for deterministic multi-node cell sharding when provided.
After a checkpoint-only judge matrix completes, render its audited standalone
curve with a contract-matched six-sampler baseline root:

```fish
.venv/bin/python -m src.cli.plot_vbvr_vlm_checkpoint_trends \
  --vlm-judge-root /path/to/checkpoint-vlm-results \
  --vlm-baseline-root /path/to/complete-vlm-results-with-baselines \
  --output-dir /path/to/trend-plots
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
