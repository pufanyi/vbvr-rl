# Getting Started

This guide takes a fresh checkout to a bounded one-GPU training smoke. Model
weights, datasets, evaluator source, and generated artifacts are external to
the repository and should be placed under the ignored `storage/` directory.

## 1. Install the Environment

Requirements:

- Linux and Python 3.12;
- [`uv`](https://docs.astral.sh/uv/);
- Fish for the provided launchers;
- FFmpeg and ffprobe on `PATH` for media workflows;
- a CUDA-capable NVIDIA GPU for training and generation;
- a C/C++ compiler and matching Python headers when Triton compiles locally.

Clone the repository and reproduce the lockfile exactly:

```bash
git clone https://github.com/pufanyi/vbvr-rl.git
cd vbvr-rl
uv sync --frozen
uv sync --frozen --check
```

Run a cheap import and media-runtime check:

```bash
.venv/bin/python -m src.eval.vbvr_runtime
```

The project intentionally uses direct `.venv/bin/python` and
`.venv/bin/torchrun` commands. This makes the interpreter used by launchers,
workers, and tests unambiguous.

## 2. Download a Base Model

The smallest public smoke config expects the official TI2V-5B Diffusers model
at `storage/models/Wan2.2-TI2V-5B-Diffusers`:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --local-dir storage/models/Wan2.2-TI2V-5B-Diffusers
```

The A14B reference configs instead expect
`storage/models/Wan2.2-I2V-A14B-Diffusers`. Review the model license and access
requirements before downloading either artifact. A config may point at a
compatible converted or fine-tuned Diffusers directory through `model_path`.

## 3. Create a Local Smoke Dataset

Generate four deterministic H.264 samples with matching first frames and a
trainer descriptor:

```bash
.venv/bin/python scripts/dev/create_i2v_smoke_dataset.py \
  --output-dir storage/smoke/i2v_512x512x81 \
  --samples 4 \
  --frames 81 \
  --height 512 \
  --width 512 \
  --fps 16
```

The output is ignored by Git. Its `dataset.json` follows the same raw-data
contract as the released VBVR-Pro training configs.

## 4. Run the One-GPU Update Smoke

The validator launches a bounded DanceGRPO step and checks that trainable
tensors actually change:

```bash
.venv/bin/torchrun --standalone --nproc_per_node=1 \
  -m scripts.dev.validate_grpo_parameter_update \
  --config configs/train_dancegrpo_vbvr_pro_5b_512x512x81_official_base_smoke_1gpu.yaml
```

This smoke uses LoRA, Flow-CPS, and the model-internal `neg_loss` reward. It
covers raw loading, prompt and video encoding, rollout, replay, backward, and
optimizer update. It deliberately does not require VBVR-Pro data or the
external rule evaluator.

Start with this smoke before increasing resolution, frame count, group size,
sampling steps, batch size, or distributed world size. Those dimensions
multiply memory and runtime.

## 5. Prepare the Public RL Dataset

Download the published raw snapshot:

```bash
hf download pufanyi/vbvr-pro-rl-indomain-50k \
  --repo-type dataset \
  --local-dir storage/datasets/vbvr-pro-rl-indomain-50k
```

Materialize the fields required by `I2VDataset` and `vbvr_rule`:

```bash
.venv/bin/python -m scripts.data.vbvr_pro_unpack_hf \
  --dataset-root storage/datasets/vbvr-pro-rl-indomain-50k \
  --output-dir storage/datasets/vbvr-pro-rl-indomain-50k/materialized \
  --expected-samples 50000 \
  --workers 8
```

The unpacker is resumable, validates file hashes from `samples.jsonl`, and
writes `materialized/dataset.json`. The published tar shards contain raw
publication assets; they are not compatible with `latent_webdataset_dir`.
See [Data and Precompute](data.md) for the complete schemas.

## 6. Install the Rule Evaluator When Needed

`vbvr_rule` is optional and its evaluator is not bundled. A rule-reward config
must point to a separately obtained compatible checkout and pin its source
fingerprint:

```yaml
grpo_reward_fn: vbvr_rule
vbvr_reward_evalkit_dir: storage/evalkits/<checkout>
vbvr_reward_evalkit_source_sha256: <64-hex-digest>
```

Follow [External EvalKit](external_evalkit.md) to validate the checkout,
EasyOCR assets, and scorer runtime. Do not replace an evaluator revision in an
existing result namespace: evaluator source is part of the metric definition.

## 7. Review a Config Before Launch

At minimum, verify:

- `model_path` exists and matches the model family used to create any latents;
- exactly one intended data path is selected;
- raw dimensions and frame count match the experiment;
- `dataset_size` is correct for latent WebDataset input;
- `output_dir` is new or has the intended resume checkpoint;
- batch, group, prompt-wave, and topology constraints are satisfied;
- reward-specific paths and service endpoints are available from every rank;
- `max_steps`, save cadence, and W&B settings are intentional.

Configuration precedence is defaults, then YAML, then explicit CLI overrides.
See [Configuration](configuration.md) for field semantics.

## 8. Launch Training

Single-machine SFT or COS:

```fish
fish scripts/train/i2v.fish --nproc 8 -- \
  --config configs/train_sft_vbvr_5b_256x256x161_lr_1e-5.yaml
```

Single-machine DanceGRPO:

```fish
fish scripts/train/grpo.fish --nproc 8 \
  --config configs/<reviewed-rl-config>.yaml
```

Multi-machine DanceGRPO uses the same command on every machine:

```bash
MASTER_ADDR=<rank-zero-host> \
MASTER_PORT=29500 \
WORLD_SIZE=<machine-count> \
RANK=<machine-rank> \
fish scripts/train/grpo_multinode.fish --nproc 8 -- \
  --config configs/<reviewed-rl-config>.yaml
```

Here `WORLD_SIZE` is the machine count and `--nproc` is the local process
count. The global process count is their product. The launcher performs cheap
runtime checks before loading model weights.

## 9. Validate the Checkout

Run tests from the explicit project test directory:

```bash
.venv/bin/python -m pytest tests
.venv/bin/ruff check --output-format=github .
.venv/bin/ruff format --check .
```

For a selected RL config, run the same preflight used by the launcher:

```bash
.venv/bin/python -m src.cli.validate_grpo_runtime \
  --config configs/<reviewed-rl-config>.yaml
```

## Troubleshooting

### `cv2` or EasyOCR cannot load `libGL.so.1`

Both pinned OpenCV distributions expose the same `cv2` package. On a headless
host, reinstall the headless wheel last, then repeat the runtime check:

```bash
uv pip install --python .venv/bin/python --reinstall --no-deps \
  opencv-python-headless==4.13.0.92
.venv/bin/python -m src.eval.vbvr_runtime
```

### Triton reports a missing `Python.h`

Install the development package matching Python 3.12. If the system image
cannot be changed, the helper can provision an ignored project-local header
toolchain:

```fish
fish scripts/dev/bootstrap_triton_python_headers.fish
```

The distributed launcher reports this failure before `torchrun` starts.

### A rule config fails before model loading

Confirm both evaluator fields are present, the checkout exists on every
machine, and its computed fingerprint matches the YAML. Then run:

```bash
.venv/bin/python -m src.cli.validate_grpo_runtime \
  --config configs/<rule-reward-config>.yaml
```

### Rewards are all zero

Do not assume this is a model-quality result. Check scorer warnings, metadata
paths, unsupported-task counts, prepared videos, and per-sample errors. Input
paths passed to scorer workers must resolve before those workers change their
working directory. For stochastic smoke rewards, score multiple members from
the same group together so group advantages are not accidentally flat.

### A restart repeats data unexpectedly

`auto_resume: true` resumes the latest checkpoint under `output_dir`.
Explicit `resume_from` with the default `reset_dataloader` behavior is
weight-only initialization and resets counters. Set the mode deliberately;
details are in [Checkpoints](checkpoints.md).
