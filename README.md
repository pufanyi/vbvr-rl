# VBVR-RL

VBVR-RL is a research training and evaluation stack for reinforcement learning
of Wan2.2 image-to-video models on VBVR-Pro. It includes supervised
fine-tuning, DanceGRPO-style replay, Flow-CPS sampling, distributed full
fine-tuning, LoRA, DCP checkpointing, and provenance-checked VBVR-Pro
evaluation.

This repository is a source release. Model weights, datasets, generated media,
and the third-party VBVR evaluator are not bundled.

## Highlights

- Wan2.2 I2V A14B and TI2V-5B training through Diffusers.
- Raw Parquet and latent WebDataset input pipelines.
- Supervised flow-matching and grouped on-policy RL objectives.
- FSDP2, HSDP, expert parallelism, and RL-only tensor parallelism.
- Fixed and randomized Flow-CPS coefficients with replay-consistent sampling.
- Rule-based VBVR-Pro rewards and task-specific Qwen VLM rewards.
- Unified high/low DCP checkpoints, EMA, and PEFT-compatible LoRA sidecars.
- Resumable VBVR-Pro generation, media preparation, scoring, and provenance
  audits.

## Release Scope

The repository contains the training runtime, configs, launchers, tests, and
VBVR-Pro integration code. It intentionally does not contain:

- Wan or Qwen model weights;
- the VBVR-Pro evaluation set or the 50,000-sample RL dataset;
- VBVR-EvalKit source code or EasyOCR model weights;
- credentials, object-store SDKs, scheduler settings, or host-specific paths;
- generated checkpoints, videos, W&B runs, or evaluation outputs.

All local artifacts should live under the ignored `storage/` directory.

## Requirements

- Linux
- Python 3.12
- `uv`
- Fish shell for the provided launchers
- An NVIDIA GPU and driver compatible with the locked PyTorch 2.11 CUDA 12.6
  wheels for GPU workflows
- A host C/C++ compiler and Python development headers when Triton must compile
  a fresh CUDA driver helper

CPU-only documentation checks and most unit tests do not require model weights.
Training and video generation require CUDA.

## Quick Start

Clone and create the locked environment:

```bash
git clone https://github.com/pufanyi/vbvr-rl.git
cd vbvr-rl
uv sync --frozen
uv sync --frozen --check
```

Check the installed scorer/media runtime:

```bash
.venv/bin/python -m src.eval.vbvr_runtime
```

For a one-GPU end-to-end training smoke, first create a deterministic local
fixture:

```bash
.venv/bin/python scripts/dev/create_i2v_smoke_dataset.py \
  --output-dir storage/smoke/i2v_512x512x81 \
  --samples 4 \
  --frames 81 \
  --height 512 \
  --width 512 \
  --fps 16
```

Place the official TI2V-5B Diffusers model at
`storage/models/Wan2.2-TI2V-5B-Diffusers`, then run:

```bash
.venv/bin/torchrun --standalone --nproc_per_node=1 \
  -m scripts.dev.validate_grpo_parameter_update \
  --config configs/train_dancegrpo_vbvr_pro_5b_512x512x81_official_base_smoke_1gpu.yaml
```

This bounded smoke uses LoRA and the model-internal `neg_loss` reward. It
verifies data loading, T5/VAE encoding, Flow-CPS rollout, replay, backward, and
a nonzero optimizer update. It does not validate the external rule scorer or a
production distributed topology.

See [Getting Started](docs/getting_started.md) for model, dataset, evaluator,
and distributed setup.

## Artifact Layout

The checked-in configs use repository-relative paths. A typical installation
looks like this:

```text
storage/
  models/
    Wan2.2-TI2V-5B-Diffusers/
    Wan2.2-I2V-A14B-Diffusers/
  datasets/
    VBVR-Pro-RL/
    vbvr-pro-eval-500/
  evalkits/
    <external-compatible-evalkit-checkout>/
    easyocr-shared/
  checkpoints/
  eval_out/
  tmp/
```

The paths under `storage/` are examples of the runtime contract, not bundled
assets.

## Public VBVR-Pro RL Data

The manifest-RL configs use the official public Hugging Face dataset
[`Video-Reason/VBVR-Pro-RL`](https://huggingface.co/datasets/Video-Reason/VBVR-Pro-RL).
This release pins revision
`ca0aaffea93b07d269c6fe2fbfe533f1fdab9aa1`. The 50 video archives already
contain the first frame, metadata, target video, final frame, and prompt, so
the separate image archives are not required for I2V training.

Download only the video archives into the expected ignored directory:

```bash
.venv/bin/hf download Video-Reason/VBVR-Pro-RL \
  --repo-type dataset \
  --revision ca0aaffea93b07d269c6fe2fbfe533f1fdab9aa1 \
  --include 'VBVR-Pro-RL-Video/*.tar.gz' \
  --local-dir storage/datasets/VBVR-Pro-RL
```

Materialize the five fields required by raw training and rule scoring:

```bash
.venv/bin/python -m scripts.data.vbvr_pro_unpack_hf \
  --dataset-root storage/datasets/VBVR-Pro-RL \
  --output-dir storage/datasets/VBVR-Pro-RL/materialized \
  --source-revision ca0aaffea93b07d269c6fe2fbfe533f1fdab9aa1 \
  --expected-tasks 50 \
  --expected-samples 50000 \
  --workers 8
```

The command safely reads, validates, and flattens the task archives, is
resumable, and writes `materialized/dataset.json` plus source provenance.
These are raw assets, not trainer-ready latent WebDataset shards. Details are
in [Data](docs/data.md).

## External VBVR Evaluator

VBVR-EvalKit is deliberately not vendored. Every `vbvr_rule` training config
must set both:

```yaml
vbvr_reward_evalkit_dir: storage/evalkits/<checkout>
vbvr_reward_evalkit_source_sha256: <64-hex-scorer-contract-digest>
```

Offline rule scoring likewise requires `--evalkit_dir` and
`--expected_evalkit_source_sha256`. The fingerprint covers the entrypoint,
evaluator Python files, annotations, and requirements. A source mismatch is a
hard error because changing the evaluator changes the RL objective and the
reported metric.

The recorded `main_v2` experiments use a compatibility fork and must not be
silently substituted with the public upstream default branch. Obtain a
compatible checkout separately, or override the revision and digest only when
you intentionally define a new scorer contract. See
[External EvalKit](docs/external_evalkit.md).

## Training

Single-machine SFT uses `scripts/train/i2v.fish`:

```fish
fish scripts/train/i2v.fish --nproc 8 -- \
  --config configs/train_sft_vbvr_5b_256x256x161_lr_1e-5.yaml
```

Single-machine RL uses `scripts/train/grpo.fish`:

```fish
fish scripts/train/grpo.fish --nproc 8 \
  --config configs/train_dancegrpo_vbvr_pro_5b_512x512x81_official_base_smoke_1gpu.yaml
```

The one-GPU config above is a smoke config; pass `--nproc 1` when running it
through the Fish launcher. Production configs are topology-specific reference
configs and must be reviewed before launch.

For multiple machines, run the same command on every machine with the
scheduler-provided rendezvous values:

```bash
MASTER_ADDR=<rank-zero-host> \
MASTER_PORT=29500 \
WORLD_SIZE=<machine-count> \
RANK=<machine-rank> \
fish scripts/train/grpo_multinode.fish --nproc 8 -- \
  --config configs/train_dancegrpo_vbvr_pro_5b_512x512x81_rule_cps_from_nsft_bs_32_lr_5e-6_manifest_rl.yaml
```

`WORLD_SIZE` is the machine count; `--nproc` is the number of local training
processes. The selected YAML still determines global prompt, rollout, replay,
and sharding semantics. Read [Configuration](docs/configuration.md) and
[Training](docs/training.md) before adapting a reference config.

## VBVR-Pro Evaluation

The supported public evaluation path lives under `scripts/eval/vbvr_pro/`.
The shared launcher performs:

1. DCP-to-Diffusers conversion or validation of a preconverted model;
2. exact-manifest video generation;
3. frame-preserving resize/pad/retime preparation;
4. CPU rule scoring through an external pinned EvalKit checkout;
5. provenance validation and aggregate export.

Inspect a configured run without loading weights:

```bash
DRY_RUN=1 \
CHECKPOINT=storage/checkpoints/<run>/checkpoint-100 \
GT_BASE=storage/datasets/vbvr-pro-eval-500 \
EVALKIT_DIR=storage/evalkits/<checkout> \
fish scripts/eval/vbvr_pro/vbvr_pro_5b_main_v2.fish
```

For a real run, also provide a compatible evaluator source through either an
existing `EVALKIT_DIR` or `EVALKIT_REPO`, and use an output directory dedicated
to that model, sampler, media, manifest, and scorer contract.

The full procedure and completion criteria are documented in
[Evaluation](docs/evaluation.md) and
[VBVR-Pro Evaluation](docs/vbvr_pro_eval.md).

## Checkpoints

Training uses PyTorch Distributed Checkpoint. New checkpoints have stable
`high/` and `low/` expert directories regardless of whether the producing job
used expert parallelism. Model/EMA state is stored with DCP; optimizer and
dataloader state are rank-local sidecars; LoRA runs additionally emit
PEFT-compatible adapter folders.

Use the conversion CLI for portable Diffusers inference:

```bash
.venv/bin/python -m src.cli.convert_dcp_to_diffusers \
  --checkpoint storage/checkpoints/<run>/checkpoint-100 \
  --base_model storage/models/Wan2.2-TI2V-5B-Diffusers \
  --output storage/models/converted/<run>-checkpoint-100 \
  --merge_lora
```

See [Checkpoints](docs/checkpoints.md) for resume versus weight-only
initialization semantics.

## Validation

Run project tests from the explicit test directory:

```bash
.venv/bin/python -m pytest tests
```

Run the same lint and formatting checks as CI:

```bash
.venv/bin/ruff check --output-format=github .
.venv/bin/ruff format --check .
```

For rule-reward configs, validate the complete GRPO runtime before allocating
model memory:

```bash
.venv/bin/python -m src.cli.validate_grpo_runtime \
  --config configs/<rule-reward-config>.yaml
```

## Repository Map

```text
src/cli/          training, inference, conversion, and evaluation entrypoints
src/models/       Wan2.2 model wrapper and LoRA integration
src/data/         raw-media, WebDataset, and remote-I/O loaders
src/trainer/      SFT/RL trainers, rewards, distributed runtime, checkpoints
src/precompute/   latent and synthetic-data builders
src/eval/         VBVR-Pro scoring, provenance, and reporting helpers
configs/          reference experiment and smoke configs
scripts/          Fish launchers and operator utilities
tests/            focused unit and contract tests
docs/             public guides and technical references
```

## Documentation

- [Documentation index](docs/README.md)
- [Getting started](docs/getting_started.md)
- [Configuration](docs/configuration.md)
- [Architecture](docs/architecture.md)
- [Data and precompute](docs/data.md)
- [Training](docs/training.md)
- [Checkpoints](docs/checkpoints.md)
- [Evaluation](docs/evaluation.md)
- [External EvalKit](docs/external_evalkit.md)
- [VBVR-Pro evaluation reference](docs/vbvr_pro_eval.md)
- [Qwen VLM reward and judge](docs/vlm_judge_reward.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Known Boundaries

- Checked-in large-scale configs preserve specific research semantics; they are
  examples, not automatic hardware discovery profiles.
- Rule-based reward and reporting require a separately obtained compatible
  evaluator checkout and EasyOCR weights.
- The public 50,000-sample RL snapshot contains raw publication assets and must
  be materialized before raw training.
- The one-GPU smoke proves plumbing and an optimizer update, not model quality
  or production-scale memory capacity.
- VLM rewards require a separately hosted OpenAI-compatible multimodal service.
- Existing `WAN_TRAINER_*` environment-variable names are retained for
  compatibility even though the released project is named VBVR-RL.

## Citation

GitHub-compatible citation metadata is provided in
[`CITATION.cff`](CITATION.cff). Update its version and release date together
with future tagged releases.

## License

The repository source is released under the [MIT License](LICENSE). Models,
datasets, evaluator code, and other external artifacts retain their own
licenses and terms. Review those terms before downloading or redistributing
them.
