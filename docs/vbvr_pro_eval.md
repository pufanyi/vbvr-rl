# VBVR-Pro `main_v2` Evaluation

This is the detailed operator reference for the manifest-locked, rule-based
VBVR-Pro evaluation pipeline. The shared Fish launcher is:

```text
scripts/eval/vbvr_pro/vbvr_pro_5b_main_v2.fish
```

It is designed to fail closed when model, data, media, evaluator, or runtime
provenance changes. Use [Evaluation](evaluation.md) for the conceptual overview
and [External EvalKit](external_evalkit.md) for evaluator installation.

## Release Scope

The repository includes:

- DCP conversion and Diffusers validation;
- exact-manifest eval JSON construction;
- ODE and Flow-CPS generation entrypoints;
- frame-preserving video preparation;
- a parallel adapter for external EvalKit;
- stage provenance, completion checks, sweep wrappers, and result summaries.

It does not include model weights, VBVR-Pro evaluation data, evaluator source,
EasyOCR weights, or generated results.

## Required Artifact Layout

A typical local setup is:

```text
storage/
  models/
    Wan2.2-TI2V-5B-Diffusers/
  checkpoints/
    <training-run>/checkpoint-100/
  datasets/
    vbvr-pro-eval-500/
      split_manifest.json
      In-Domain_50/<task>/<00000>/
      Out-of-Domain_50/<task>/<00000>/
  evalkits/
    <compatible-checkout>/
    easyocr-shared/model/
      craft_mlt_25k.pth
      english_g2.pth
  eval_out/
```

Each flattened sample directory must contain at least:

```text
first_frame.png
prompt.txt
ground_truth.mp4
metadata.json
```

The metadata task ID must match the selected entry in `split_manifest.json`.

## Evaluator Contract

The launcher defaults encode the recorded compatible `main_v2` contract:

```text
revision: e140038f2aee76ca518f464755fa8bc19b783ba5
source SHA-256: 4cc7d028d4106a28190a63bc179562d5ac9add9263cb71926dd6385c5714bcf8
```

These values do not imply that the corresponding revision is on the public
upstream default branch. Supply a checkout that actually matches both values,
or deliberately override both and create a new output namespace. Never keep an
old result label while changing evaluator source.

The Python environment is also part of the scorer contract. Before a run:

```bash
.venv/bin/python -m src.eval.vbvr_runtime
```

The launcher repeats this check and records the full runtime report.

## Dry Run

Inspect the resolved top-level selections without cloning evaluator source,
loading weights, or touching outputs:

```bash
DRY_RUN=1 \
CHECKPOINT=storage/checkpoints/<run>/checkpoint-100 \
BASE_MODEL=storage/models/Wan2.2-TI2V-5B-Diffusers \
CONVERTED_MODEL=storage/models/converted/<run>-checkpoint-100 \
GT_BASE=storage/datasets/vbvr-pro-eval-500 \
EVALKIT_DIR=storage/evalkits/<compatible-checkout> \
OUTPUT_ROOT=storage/eval_out/<run>/checkpoint-100-unipc \
fish scripts/eval/vbvr_pro/vbvr_pro_5b_main_v2.fish
```

Always dry-run a wrapper after changing environment variables. A dry run does
not prove that paths exist or fingerprints match; those checks occur in the
real pipeline.

## Core Environment Variables

### Model and data

| Variable | Meaning |
| --- | --- |
| `CHECKPOINT` | DCP checkpoint directory when conversion is required |
| `BASE_MODEL` | Base Diffusers model used to interpret the checkpoint |
| `CONVERTED_MODEL` | Stable Diffusers output or preconverted input |
| `PRECONVERTED_MODEL` | `1` to skip DCP conversion after validating the model and its provenance |
| `CONVERSION_PROVENANCE` | Conversion/import manifest; default depends on preconverted mode |
| `GT_BASE` | Flattened VBVR-Pro evaluation tree |
| `SPLIT_MANIFEST` | Exact bench selection, default `<GT_BASE>/split_manifest.json` |
| `EXPECTED_VIDEOS` | Required number of samples and outputs |

### Evaluator and OCR

| Variable | Meaning |
| --- | --- |
| `EVALKIT_DIR` | External checkout path |
| `EVALKIT_REPO` | Optional Git source used only when `EVALKIT_DIR` is missing |
| `EVALKIT_REV` | Exact Git revision required by the run |
| `EVALKIT_SOURCE_SHA256` | Complete source-contract fingerprint |
| `EASYOCR_SOURCE_MODELS` | Directory containing pre-populated OCR weight files |
| `EASYOCR_ROOT` | Runtime EasyOCR root exposed to scorer workers |
| `SCORE_WORKERS` | CPU scorer process count |
| `SCORE_THREADS_PER_WORKER` | Native threads allowed inside each scorer process |

The launcher creates `EVALKIT_DIR/easyocr_models` as a symlink to the staged
OCR model directory. It refuses to replace a real directory at that path.

### Generation

| Variable | Meaning |
| --- | --- |
| `NUM_GPUS` | Local data-parallel generation process count |
| `CUDA_DEVICES` | Visible CUDA device list for generation/conversion |
| `HEIGHT`, `WIDTH` | Generated spatial size |
| `NUM_FRAMES` | Fixed generated frame count |
| `INFER_FPS` | Encoded generation FPS |
| `NUM_INFERENCE_STEPS` | Denoising step count |
| `GUIDANCE_SCALE` | Classifier-free guidance scale |
| `SEED` | Deterministic base seed |
| `GENERATION_MODE` | `ode` or `cps` |
| `ODE_SOLVER` | `unipc` or `euler` when `GENERATION_MODE=ode` |
| `CPS_NOISE_LEVEL` | Flow-CPS coefficient in `[0, 1]` |
| `USE_ITEM_NUM_FRAMES` | `1` to derive a GT-duration-matched length per sample |
| `TEMPORAL_ALIGNMENT` | Temporal alignment used for per-item frame lengths |

### Preparation and outputs

| Variable | Meaning |
| --- | --- |
| `OUTPUT_ROOT` | Unique root for the complete evaluation cell |
| `EVAL_JSON` | Generated manifest-validated item list |
| `GENERATED_DIR` | Raw generated video tree |
| `PREPARED_DIR` | Rule-ready video tree |
| `SCORE_DIR` | Result JSON directory |
| `PREPARED_HEIGHT`, `PREPARED_WIDTH` | Rule scorer canvas |
| `MAX_DURATION` | Maximum prepared-video duration |
| `PREP_CRF` | H.264 preparation quality setting |
| `PREP_WORKERS` | Parallel FFmpeg preparation workers |

Paths default beneath `storage/`, but wrappers may override them. An output
root must identify the checkpoint, sampler, generation shape, manifest, and
scorer contract well enough to avoid accidental reuse.

## Run a Complete Cell

Example ODE evaluation:

```bash
CHECKPOINT=storage/checkpoints/<run>/checkpoint-100 \
BASE_MODEL=storage/models/Wan2.2-TI2V-5B-Diffusers \
CONVERTED_MODEL=storage/models/converted/<run>-checkpoint-100 \
GT_BASE=storage/datasets/vbvr-pro-eval-500 \
EVALKIT_DIR=storage/evalkits/<compatible-checkout> \
EVALKIT_REV=<revision> \
EVALKIT_SOURCE_SHA256=<64-hex-digest> \
EASYOCR_SOURCE_MODELS=storage/evalkits/easyocr-shared/model \
OUTPUT_ROOT=storage/eval_out/<run>/checkpoint-100-unipc \
GENERATION_MODE=ode \
ODE_SOLVER=unipc \
NUM_GPUS=8 \
fish scripts/eval/vbvr_pro/vbvr_pro_5b_main_v2.fish
```

Flow-CPS changes the sampler and therefore requires a different output root:

```bash
OUTPUT_ROOT=storage/eval_out/<run>/checkpoint-100-cps-0.7 \
GENERATION_MODE=cps \
CPS_NOISE_LEVEL=0.7 \
fish scripts/eval/vbvr_pro/vbvr_pro_5b_main_v2.fish
```

Do not compare an ODE cell and a CPS cell unless inference steps, sigma grid,
CFG, seed, resolution, frame policy, preparation, and scoring contracts are
also controlled and reported.

## Stage 1: Evaluator and Runtime Preflight

Before touching model outputs, the launcher:

1. obtains a missing evaluator checkout only when `EVALKIT_REPO` is set;
2. serializes concurrent installation with a directory lock;
3. checks out the exact revision in detached state;
4. computes and compares the complete source fingerprint;
5. stages the two required EasyOCR weights and its compatibility symlink;
6. validates the pinned Python/scientific-media runtime.

If an evaluator directory exists but is incomplete, the launcher stops instead
of overwriting it. If the directory is a Git checkout, its `HEAD` must match
`EVALKIT_REV` as well as the content fingerprint.

## Stage 2: Checkpoint Conversion

For DCP input, one process acquires a conversion lock and runs:

```text
src.cli.convert_dcp_to_diffusers
```

The public launcher converts raw, non-EMA weights, merges LoRA when present,
uses bf16 safe serialization, and validates the resulting Diffusers tree. It
fingerprints the checkpoint, base model, converter source, conversion options,
and complete output tree.

Concurrent jobs targeting the same converted model wait for the lock and then
reuse a complete validated result. An incomplete directory or incompatible
provenance fails instead of being silently repaired from different inputs.

Set `CONVERSION_ONLY=1` to stop after successful conversion and provenance
promotion.

### Preconverted models

Set `PRECONVERTED_MODEL=1` only for a complete Diffusers directory. It must
carry its own immutable `conversion_metadata.json` or an explicitly selected
`CONVERSION_PROVENANCE`. The launcher validates structure, fingerprints the
tree, and checks that it remains stable during the validation interval.

## Stage 3: Manifest Construction

`src.eval.build_vbvr_eval_json` verifies each flattened sample against the
split manifest and writes names in the evaluator's domain layout:

```text
In-Domain_50/<task>/<index>
Out-of-Domain_50/<task>/<index>
```

The item count must equal `EXPECTED_VIDEOS`. The generation JSON contains
absolute first-frame paths and prompts, avoiding dependence on a later working
directory.

When `USE_ITEM_NUM_FRAMES=1`, the builder probes each ground-truth video and
selects the closest legal `alignment * k + 1` generation length at
`INFER_FPS`. This mode currently requires ODE generation.

## Stage 4: Video Generation

The launcher first validates any existing generated tree without loading
weights. If both media and provenance are complete, it skips generation.
Otherwise it resumes missing/invalid videos through the selected module:

- `src.cli.eval_i2v` for UniPC ODE;
- `src.cli.eval_i2v_euler` for Euler ODE;
- `src.cli.eval_i2v_cps` for Flow-CPS.

Generation is local multi-GPU data parallel. Every expected path is derived
from the eval JSON; extra, missing, duplicate, corrupt, wrong-size,
wrong-frame-count, or wrong-FPS outputs fail exact-set validation.

Set `FORCE_REGENERATE=1` only when you intentionally want to replace a tree
whose provenance differs. Prefer a fresh `OUTPUT_ROOT` for a changed contract.

## Stage 5: Frame-Preserving Preparation

`src.cli.prepare_vbvr_eval_videos` mirrors the generated tree. For each MP4 it:

1. probes exact dimensions, frames, FPS, and duration;
2. scales to fit the configured canvas without cropping;
3. pads the remaining area;
4. retains every frame;
5. raises FPS when needed to satisfy `MAX_DURATION`;
6. writes H.264 through a temporary file and validates it before promotion.

The prepared output must contain exactly `EXPECTED_VIDEOS`. Preparation input,
parameters, implementation source, and full output tree are fingerprinted.

Set `FORCE_REPREPARE=1` only for an intentional rewrite. Regenerating videos
automatically invalidates downstream preparation.

## Stage 6: Rule Scoring

The launcher hides CUDA and sets native thread limits before starting CPU
scorer workers. The adapter:

- validates the evaluator source and dependency runtime again;
- changes worker directories only after all relevant paths are resolved;
- requires EvalKit to discover exactly the expected number of videos;
- captures one result per sample;
- writes domain and overall aggregates;
- preserves sample errors instead of dropping them.

The result file name is derived from the prepared directory. The launcher
deletes stale score/provenance files before the scoring stage because partial
score reuse is not safe. It promotes score provenance only when sample count is
exact and no record contains an error.

## Output and Provenance Layout

```text
<output-root>/
  eval_samples.json
  generation-provenance.json
  preparation-provenance.json
  score-provenance.json
  generated_<shape>/
    In-Domain_50/<task>/<index>.mp4
    Out-of-Domain_50/<task>/<index>.mp4
  eval_<shape-and-duration>/
    ... mirrored MP4 tree ...
  scores/
    <prepared-name>_vbvr_results.json
```

Conversion provenance normally sits beside `CONVERTED_MODEL`, allowing
multiple evaluation cells to reuse one immutable conversion safely.

Each stage manifest records a state such as `in_progress_resume`,
`in_progress_rewrite`, or `complete`; scalar parameters; input files/trees;
implementation files; and output-tree fingerprints. A `complete` manifest is
valid only while all fingerprints still match.

## Resume Rules

Safe automatic resume applies to:

- a complete matching converted model;
- individual generated videos that pass exact media validation under matching
  generation inputs;
- individual prepared videos under matching preparation inputs.

Scoring starts from a clean result because merging arbitrary partial evaluator
outputs can hide contract changes or missing errors.

If an output tree exists without compatible provenance, the launcher stops and
asks for an explicit force variable or a fresh namespace. A fresh namespace is
the recommended response to any semantic change.

## Checkpoint and Sampler Sweeps

Wrapper scripts under `scripts/eval/vbvr_pro/` set checkpoint- and
sampler-specific environment variables, then delegate to the shared launcher.
They should contain no independent scoring logic.

When adding a sweep:

1. keep the split manifest and scoring contract fixed;
2. assign a unique converted-model and output path to every checkpoint;
3. encode ODE/CPS solver and coefficient in the cell name;
4. dry-run every cell;
5. run cells independently so one failure is visible;
6. aggregate only complete score JSON files with matching provenance.

Use
[`scripts/eval/vbvr_pro/summarize_vbvr_pro_results.fish`](../scripts/eval/vbvr_pro/summarize_vbvr_pro_results.fish)
to summarize completed rule-result trees. Presentation builders and static
result explorers are optional reporting tools; they do not define the scoring
contract.

## Training Reward Parity

`vbvr_rule` training reward uses the same external evaluator source pin,
dependency runtime, frame-preserving preparation dimensions, duration limit,
and task-specific scorer interface. The relevant YAML fields are:

```yaml
vbvr_reward_evalkit_dir: storage/evalkits/<compatible-checkout>
vbvr_reward_evalkit_source_sha256: <64-hex-digest>
vbvr_reward_fps: 16
vbvr_reward_prepared_width: 1024
vbvr_reward_prepared_height: 1024
vbvr_reward_max_duration_seconds: 5.0
vbvr_reward_prepare_crf: 12
```

Parity still depends on generated frame count, FPS, task metadata, and
evaluator revision. Keep those aligned when using offline ground-truth checks
to validate the online reward.

## Failure Recovery

### Evaluator mismatch

Restore the intended checkout. If the evaluator change is deliberate, compute
a new digest, change both config and output namespace, and report it as a new
metric contract.

### Incomplete converted model

Do not write into it concurrently by hand. Remove or relocate only the exact
incomplete artifact after confirming no job owns the conversion lock, then
rerun conversion. Never replace a complete model behind matching provenance.

### Generated set validation failure

Read the reported paths and media mismatch. Rerunning normally repairs missing
or invalid files under matching provenance. Use `FORCE_REGENERATE=1` only when
the entire prior generation contract is intentionally superseded.

### Preparation failure

Verify FFmpeg/ffprobe availability and inspect the source MP4. The preparer
reports exact dimension, frame, FPS, or duration violations and does not
promote an invalid temporary output.

### Partial or errored score JSON

Inspect per-sample `error` fields and evaluator logs. Fix evaluator assets,
metadata paths, or task-specific dependencies, then rerun the score stage
through the launcher. Do not manually remove errored records and recompute the
mean.

### All scores are zero

Confirm EvalKit found the intended prepared paths and ground truth. Check task
support, metadata identity, OCR assets, worker exceptions, and the source/runtime
fingerprints before interpreting the result as model behavior.

## Completion Audit

Before publishing a number, verify:

- [ ] the repository revision is recorded;
- [ ] conversion provenance is complete and the model tree is stable;
- [ ] the eval JSON matches the intended split manifest and exact sample count;
- [ ] generated and prepared trees have the exact expected path sets;
- [ ] every generated/prepared MP4 passes media validation;
- [ ] evaluator revision and source digest match the declared contract;
- [ ] scorer runtime digest and OCR assets are recorded;
- [ ] result JSON contains all samples and zero error records;
- [ ] In-Domain, Out-of-Domain, and overall aggregates come from that file;
- [ ] generation, preparation, and score provenance are complete;
- [ ] all launcher processes exited successfully.

Retain the config, launcher environment, score JSON, and provenance manifests
with any reported aggregate. Generated videos may be stored separately, but
their complete tree fingerprint must remain auditable.
