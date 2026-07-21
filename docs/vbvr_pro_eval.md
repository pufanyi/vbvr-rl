# VBVR-Pro main_v2 Evaluation

This workflow evaluates Wan2.2 TI2V 5B checkpoints on the current VBVR-Pro
bench with the `main_v2` interleave rule scorer. It does not use FastVideo.

## Data And Scorer

The authoritative raw inputs are read-only:

- `/mnt/umm/users/xujunxiang/VBVR-Pro_revise`
- `/mnt/umm/users/xujunxiang/VBVR-Pro`
- `/mnt/aigc/xujunxiang/Code/VBVR-Pro/scripts/split_manifest.json`

The manifest's evaluation subset is `bench` (five samples per task). The
`main_v2` scorer supports the 50 `In-Domain_50` and 50 `Out-of-Domain_50`
tasks, for 500 scored videos. It does not register the 50 `Extra_50` tasks.
These domain names follow the official benchmark definition: In-Domain tasks
were present in the original VBVR-Dataset, while Out-of-Domain tasks were held
out there for generalization testing. They are not defined relative to the
current DanceGRPO run. Its
`data/vbvr_pro/vbvr_pro_train_supported_256x256x161.json` configuration loads
manifest `train` samples for all 100 EvalKit-supported tasks: 50 official
In-Domain tasks and 50 official Out-of-Domain tasks, 250k samples from each.
Consequently, the reported Out-of-Domain score is not a held-out-task score for
this RL-trained model; only the unsupported `Extra_50` tasks are absent from
this RL dataset and they are not included in `main_v2` scoring. The manifest
updated on 2026-07-20 removes every bench ID from every task's `train` list, so
the current 500-video scored set is sample-disjoint from manifest training IDs.
Treat its domain names as benchmark split labels, not held-out-task labels for
this particular RL run.

For a strict In-Domain-only RL run, use
`configs/train_dancegrpo_vbvr_pro_5b_256x256x161_rule_cps_from_nsft_bs_32_lr_1e-6.yaml`.
It points to `data/vbvr_pro/vbvr_pro_train_indomain_strict_256x256x161.json`,
which selects only manifest records labeled `In-Domain_50` and subtracts every
record's `bench` IDs from its `train` IDs. The verified current result is 50
tasks and 250,000 samples with zero benchmark-ID overlap. Its checkpoint output path is
separate and ends in `_indomain_strict`, so it cannot auto-resume a prior
100-task run.
The flattened, manifest-matched GT view can be read directly from:

```text
/mnt/aigc/xujunxiang/VR_Data/VBVR-Bench_Pro-video
```

Generation must use each sample's `video/prompt.txt`; the flattened view's
`prompt.txt` already contains that prompt. Do not edit any of these shared
paths. Personal scorer clones and all generated artifacts belong under the
repository's ignored `storage/` tree.

The current verified scorer revision (2026-07-24) is
`6fedd9d9edb8daafa56aca8e53885aa8ad6f6037` from the `main_v2` branch. Its
scorer-contract SHA-256 is
`eb977da60e95456734063ba018b14d805680179fdf0e3e3b2ba6f603f27a935c`; this
covers the entrypoint, evaluator Python, bundled annotations, and
`requirements.txt`.

- GitLab: `https://gitlab.bj.sensetime.com/zeotrope/multimodal/vbvr-evalkit-interleave`
- GitHub browser: `https://github.com/xujunxiangwork/VBVR-Evalkit-Interleave`
- GitHub SSH: `git@github.com:xujunxiangwork/VBVR-Evalkit-Interleave.git`

Relative to the historical `42a1593d` revision, this is 14 commits and changes
13 evaluator files (`+1756/-359`). The public runner and the 100 registered task
names are unchanged, but scoring semantics changed substantially: the updates
include per-task geometry/segmentation fixes, blank-output handling, removal of
several hard score cliffs, and a consistency-penalty reformulation. Treat this
as a new reward/evaluation objective, not a drop-in relabeling of old scores.

## One-Command Run

The launcher converts the DCP checkpoint when needed, installs the exact
EvalKit revision when its revision-specific checkout is absent, builds and validates the
500-sample JSON, runs eight independent Diffusers pipelines with `torchrun`,
prepares scorer videos, then runs the rule scorer. It verifies the full
scorer-contract digest before scoring and again during provenance promotion:

```fish
fish scripts/eval/vbvr_pro/vbvr_pro_5b_main_v2.fish
```

Important defaults are explicit in the launcher:

- generation: `256x256`, 161 frames, 16 FPS, 50 inference steps, CFG 5.0,
  seed 0;
- distributed execution: 8 GPUs, one full pipeline per rank, round-robin data
  parallelism;
- scorer preparation: retain all 161 frames, resize without cropping to the
  1024x1024 scorer canvas, and raise FPS to at least 33 so playback is at most
  five seconds;
- scoring: latest `main_v2`, 500 expected videos, CPU multiprocessing with
  CUDA hidden from EasyOCR workers and bounded native thread pools (8 workers
  x 16 threads on the current 128-core host).

Latest-scorer outputs default to
`storage/eval_out/vbvr_pro_main_v2_evalkit_eb977da6/`. Historical results and
the published Space remain tied to revision `42a1593d`; their scores must not
be relabeled or mixed with `6fedd9d9` scores.

Keep the 1024x1024 scorer resize for future runs. On checkpoint-1200, the
overall score was 0.447580 for 1024x1024x161, 0.445539 for unmodified
256x256x161, and 0.437559 for uniformly sampled 256x256x81. The small gain over
raw resolution and the larger loss from temporal sampling support retaining all
161 frames and consistently resizing them to the 1024x1024 scorer canvas.

## Training Reward Parity

The `vbvr_rule` training reward uses this same `main_v2` source fingerprint,
Diffusers source-video encoder, 1024x1024/all-frame/maximum-five-second
preparation function, GT metadata contract, and isolated CPU scoring entrypoint.
Training configs pin the full scorer-source SHA-256, not only the branch or Git
revision, and fail on scorer errors by default. See
[`docs/training.md`](training.md#vbvr-pro-main_v2-reward-contract) for the
configuration contract.

After any change to generation video I/O, preparation, raw-data metadata, or
EvalKit sources, run `scripts/dev/validate_vbvr_reward_alignment.py`. It
constructs independent training and final-eval videos from the same RGB frames
and requires raw-frame equality, prepared-frame equality, and exact score
equality. Include at least one ordinary geometry/trajectory task and one
EasyOCR-backed task.

Common overrides use environment variables:

```fish
set -lx NUM_GPUS 8
set -lx SCORE_WORKERS 8
set -lx OUTPUT_ROOT storage/eval_out/vbvr_pro_main_v2_evalkit_eb977da6/my_run
fish scripts/eval/vbvr_pro/vbvr_pro_5b_main_v2.fish
```

Set `DRY_RUN=1` to resolve and print the generation mode, checkpoint,
converted-model path, output root, sampling steps, CFG, and CPS noise (when
applicable) without loading a model or writing evaluation artifacts.

For an already-converted Diffusers model, set `PRECONVERTED_MODEL=1` and point
`CONVERSION_PROVENANCE` at its immutable conversion/import metadata file. The
shared launcher validates the Diffusers structure, verifies that the model
tree is stable, and fingerprints both the metadata and full model tree in
generation provenance; it does not manufacture a DCP-conversion record. The
fixed DiffSynth step-35500 wrapper uses this path and keeps the RL reward's
exact 32-FPS/5.03125-second timing:

```fish
fish scripts/eval/vbvr_pro/vbvr_pro_5b_diffsynth_step35500_cps0p7_main_v2.fish
```

Its 81-frame standard-ODE comparison uses 50-step UniPC, CFG 5.0, seed 0,
and exact 16 FPS (5.0625 seconds):

```fish
fish scripts/eval/vbvr_pro/vbvr_pro_5b_diffsynth_step35500_unipc_50steps_81f_16fps_main_v2.fish
```

The completed 500-sample run scored 0.406543 Overall, 0.475460 In-Domain, and
0.337626 Out-of-Domain with no scorer errors. It generates 81 frames directly
rather than subsampling an existing 161-frame video. All generated and
1024x1024 prepared files were independently checked as exactly 81 frames,
16 FPS, and 5.0625 seconds. This result is not a frame-count-only ablation
against the 161-frame CPS run because the sampler, sampling-step count, CFG,
frame count, and FPS differ.

Checkpoint-specific wrappers may add report generation after the shared
pipeline. For example, the SFT epoch-1 wrapper runs the full evaluation and
then exports all 100 per-task averages plus the domain/category summary to an
Excel workbook:

```fish
fish scripts/eval/vbvr_pro/vbvr_pro_5b_sft_full_lr1e5_epoch1_main_v2.fish
```

The reusable exporter is `python -m src.cli.export_vbvr_task_scores`; it
validates the expected sample/task counts before writing the workbook. With
`--summary-output`, it also writes a compact text file whose first three lines
are Overall, In-Domain, and Out-of-Domain.

The DanceGRPO checkpoint series has fixed-step wrappers named
`scripts/eval/vbvr_pro/dancegrpo_bs32/vbvr_pro_5b_dancegrpo_checkpoint_<STEP>_main_v2.fish` for steps
300, 600, 900, 1200, 1500, 1800, 2100, 2400, and 2700. Each wrapper calls the
same `vbvr_pro_5b_dancegrpo_checkpoint_main_v2.fish` implementation and keeps
its converted model, videos, score JSON, Excel workbook, and text summary in a
checkpoint-specific path. For example:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_bs32/vbvr_pro_5b_dancegrpo_checkpoint_1500_main_v2.fish
```

## Checkpoint-300 CPS Noise Sweep

The checkpoint-300 CPS wrappers generate the same 500-sample evaluation set
with the training-time `flowcps` sampler. They use 30 sampling steps, CFG 1.0,
the same base seed for corresponding samples, and one isolated output root per
noise level:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_bs32/vbvr_pro_5b_dancegrpo_checkpoint_300_cps_noise_0p1_main_v2.fish
fish scripts/eval/vbvr_pro/dancegrpo_bs32/vbvr_pro_5b_dancegrpo_checkpoint_300_cps_noise_0p3_main_v2.fish
fish scripts/eval/vbvr_pro/dancegrpo_bs32/vbvr_pro_5b_dancegrpo_checkpoint_300_cps_noise_0p7_main_v2.fish
fish scripts/eval/vbvr_pro/dancegrpo_bs32/vbvr_pro_5b_dancegrpo_checkpoint_300_cps_noise_0p9_main_v2.fish
```

Each command independently performs CPS generation, 1024x1024x161 video
preparation, `main_v2` scoring, per-task Excel export, and concise
`final_scores.txt` export. Run them sequentially on the same eight GPUs.

For fixed CPS noise levels of 0.3 and 0.7, every DanceGRPO checkpoint from 300
through 2700 has a wrapper named
`scripts/eval/vbvr_pro/dancegrpo_bs32/vbvr_pro_5b_dancegrpo_checkpoint_<STEP>_cps_noise_<LEVEL>_main_v2.fish`.
These wrappers delegate to `vbvr_pro_5b_dancegrpo_checkpoint_cps_main_v2.fish`
and isolate all outputs by checkpoint step and noise level. Run the jobs
sequentially.

## bs32 lr=1e-6 Checkpoint Sweep

The checkpoint root
`storage/checkpoints/dancegrpo_vbvr_pro_5b_256x256x161_rule_cps_from_nsft_bs32_lr_1e-6`
contains checkpoints 100 through 800 in increments of 100. Its 24 fixed
entrypoints plus 8 Euler entrypoints live under
`scripts/eval/vbvr_pro/dancegrpo_bs32_lr_1e-6/`: each checkpoint has the
original UniPC ODE (`..._checkpoint_<STEP>_main_v2.fish`), a first-order
rectified-flow Euler ODE (`..._checkpoint_<STEP>_euler_main_v2.fish`), and two
Flow-CPS wrappers (`..._cps_noise_0p3_...` and `..._cps_noise_0p7_...`). For
example:

```fish
set series scripts/eval/vbvr_pro/dancegrpo_bs32_lr_1e-6
fish $series/vbvr_pro_5b_dancegrpo_checkpoint_100_main_v2.fish
fish $series/vbvr_pro_5b_dancegrpo_checkpoint_100_euler_main_v2.fish
fish $series/vbvr_pro_5b_dancegrpo_checkpoint_100_cps_noise_0p3_main_v2.fish
fish $series/vbvr_pro_5b_dancegrpo_checkpoint_100_cps_noise_0p7_main_v2.fish
```

The configured scheduler for the original ODE scripts is
`UniPCMultistepScheduler`: second-order `bh2`, flow prediction, flow shift 5.0,
and 50 inference steps. The Euler scripts instead install the deterministic
`FlowMatchEulerDiscreteScheduler` with the same flow shift, 50 steps, CFG 5.0,
and per-sample seeds. Its update is the first-order step
`x_next = x + (sigma_next - sigma) * v_theta(x, sigma)`; stochastic sampling
is disabled. This is not Euler Ancestral.

The wrappers pin distinct converted-model and evaluation roots for this run,
so they cannot accidentally reuse artifacts from the original bs32 or strict
In-Domain series. The four modes for one checkpoint intentionally share its
converted Diffusers model. Exact-revision EvalKit installation and conversion
are guarded by separate atomic locks, so overlapping ODE/CPS jobs elect one
writer and the followers reuse the same complete checkout and conversion.
Before publishing provenance, the launcher validates all
Diffusers components, safetensors headers, index keys, and referenced shards,
then requires the output tree to remain stable. A stale output fingerprint
from the older concurrent-writer race is refreshed only when all recorded
conversion inputs still match and those validation checks pass. Genuine input
mismatches still fail. Run the 32 eight-GPU jobs sequentially when they use the
same GPU pool.

All jobs use one stable EasyOCR cache under `storage/evalkits/easyocr-shared`.
Model weights and the EvalKit symlink are installed atomically, so concurrent
scorers cannot redirect or truncate each other's per-run OCR cache.

## Strict In-Domain Checkpoint Sweep

The strict lr=1e-6 checkpoint root contains steps 300, 600, 900, 1200, and
1500. Run its complete four-mode matrix sequentially on one eight-GPU node:

```fish
set -lx STRICT_REEVALUATE_COMPLETE 1  # optional: refresh even complete runs
fish scripts/eval/vbvr_pro/dancegrpo_indomain_strict/vbvr_pro_5b_dancegrpo_indomain_strict_sweep_main_v2.fish
```

For every complete checkpoint, the sweep runs second-order UniPC ODE,
first-order deterministic FlowMatch Euler ODE, Flow-CPS 0.3, and Flow-CPS 0.7.
The modes share one converted Diffusers model but use isolated output roots.
The default result root is derived from the first eight characters of both the
current `SPLIT_MANIFEST` SHA-256 and scorer-contract SHA-256. For the current
pins it is
`storage/eval_out/vbvr_pro_main_v2_indomain_strict_manifest_326f7bda_evalkit_eb977da6/`.
The completion check compares the full manifest hash, EvalKit revision, and
scorer-contract hash and recomputes every recorded input/output artifact
fingerprint and path binding before skipping a run.
Jobs are strictly sequential, write one log per checkpoint/mode under
`storage/eval_logs/vbvr_pro_main_v2_indomain_strict/`, and count a run complete
only after 500 generated and prepared videos, its 500-sample error-free JSON,
all three complete provenance stages, a 100-task workbook, and
`final_scores.txt` all exist.

### Strict-sweep result explorer

Build the static task- and video-level explorer directly from the completed
strict sweep and SFT epoch-1 baseline:

```bash
.venv/bin/python scripts/eval/vbvr_pro/build_vbvrpro_space.py
```

The default local build is `storage/hf_spaces/vbvrpro_output/` and hard-links
all scored videos where the filesystem permits it. The deployed frontend is
[`pufanyi/vbvrpro_output`](https://huggingface.co/spaces/pufanyi/vbvrpro_output).
It stores the 10,500 scored videos in the companion
[`pufanyi/vbvrpro_output-data`](https://huggingface.co/datasets/pufanyi/vbvrpro_output-data)
Dataset and streams them into the static Space. To reproduce that lightweight
Space tree, run:

```bash
.venv/bin/python scripts/eval/vbvr_pro/build_vbvrpro_space.py \
  --output-root storage/hf_spaces/vbvrpro_output_site \
  --skip-videos \
  --video-url-prefix https://huggingface.co/datasets/pufanyi/vbvrpro_output-data/resolve/main/videos
```

The generator validates 500 error-free samples per run, stable sample identity
across the baseline and all 20 strict-sweep runs, five samples per task, and
the exact task means before writing score indexes or materializing media.

Use `FORCE_REGENERATE=1` only when intentionally replacing personal generated
videos after a generation-provenance change; the launcher will also rebuild
the prepared videos. Use `FORCE_REPREPARE=1` for a preparation-only change such
as CRF. A converted-model provenance mismatch requires a fresh conversion path
or an explicit reconversion rather than silently overwriting a model tree.

## Outputs And Resume

The default run root contains:

```text
eval_samples.json
generation-provenance.json
preparation-provenance.json
score-provenance.json
generated_256x256x161/
eval_1024x1024_161f_5s/
scores/
```

The SFT epoch-1 wrapper additionally writes
`scores/eval_1024x1024_161f_5s_task_scores.xlsx` and `final_scores.txt`. The
DanceGRPO fixed-step wrappers write the same two report names beneath their
checkpoint-specific output roots.

To refresh cross-run Excel summaries after more evaluations finish, run:

```fish
fish scripts/eval/vbvr_pro/summarize_vbvr_pro_results.fish
```

It scans complete standard 1024x1024x161 results and writes five workbooks to
`storage/eval_out/vbvr_pro_main_v2_evalkit_eb977da6/reports/`: all-run aggregate/domain/category
scores, per-task CPS deltas against the same checkpoint's ODE result, and ODE
aggregate/domain scores by DanceGRPO training step. It also writes a CPS 0.3
training curve that uses the SFT epoch-1 ODE result as step-0 baseline and the
DanceGRPO CPS results for steps 300 through 2700. Separate task-level training
workbooks are written for CPS noise 0.3 and 0.7: each of the 100 tasks is one
row, with SFT ODE baseline, per-step CPS scores, deltas versus baseline and the
previous step, best step, and a secondary aggregate sheet. Partial or
scorer-error runs are excluded and listed separately. Complete score provenance
is required, its result path and fingerprint must match the JSON being read,
and the summarizer refuses to combine different scorer-contract hashes.

Generation and video preparation write temporary MP4s and atomically rename
them only after validation. Re-running the launcher checks the exact manifest
path set plus every generated video's width, height, frame count, and FPS
before skipping generation. Prepared videos are likewise probed before reuse.
Conversion, generation, preparation, and scoring also write provenance JSONs
that fingerprint their inputs and settings. Score provenance records the
declared and actual EvalKit revision, full scorer-contract digest, requirements,
EasyOCR weight files, and key installed dependency versions. A changed
checkpoint, model, manifest, seed, CFG, inference-step count, CRF, runtime
version, or scorer tree cannot silently reuse older outputs. Interrupted stages remain marked
`in_progress`; ordinary runs resume validated outputs, while an interrupted
configuration-changing rewrite stays a full rewrite. Scoring always runs again
with the pinned, clean EvalKit revision, so a stale result JSON is never
accepted. The pipeline is complete only when all three media stages contain 500
samples and the score JSON reports 500 samples without scorer errors.

Some `main_v2` evaluators require Norfair and EasyOCR beyond the scorer's
declared requirements. The launcher checks those imports and reads the
pre-populated EasyOCR weights from:

```text
/mnt/aigc/xujunxiang/Code/VBVR-Bench/VBVR-EvalKit/easyocr_models
```

The launcher copies those two files into the stable personal
`storage/evalkits/easyocr-shared/model`, creates a writable `user_network/`,
and points the revision-specific EvalKit checkout's `easyocr_models` link at
that personal copy. `EASYOCR_MODULE_PATH` must be the personal parent
directory, not the shared source directory: EasyOCR appends
`model/` and may write there. Workers run from the personal checkout so its
relative annotations resolve correctly. Keep CUDA hidden during scoring;
otherwise every worker may instantiate EasyOCR on GPU 0.
