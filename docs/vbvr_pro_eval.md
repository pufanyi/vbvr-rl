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

The access-controlled Hugging Face snapshot `pufanyi/vbvr-pro-eval-500`
revision `181e201076063eb8abbbd9d803f83258472d60a2` is a self-contained
alternative under `storage/datasets/vbvr-pro-eval-500`. It contains the same
500 supported samples, a sanitized path-free split manifest, and a complete
`SHA256SUMS` file. The sanitized manifest SHA-256 is
`afab352e08c590c9f4b480ef314b37f6896eef6430f42ea6c0ce0494f2aa8c4e`;
its `dataset_config.json` also records the authoritative source-manifest hash
`326f7bda3743e9c66dc0c29445661a5dda4ad0cee4cb8838c3fcfd0c4a149deb`.
Download the pinned revision and verify it before use:

```fish
hf download pufanyi/vbvr-pro-eval-500 \
    --repo-type dataset \
    --revision 181e201076063eb8abbbd9d803f83258472d60a2 \
    --local-dir storage/datasets/vbvr-pro-eval-500
cd storage/datasets/vbvr-pro-eval-500
sha256sum -c SHA256SUMS --quiet
```

On this workstation, the 2026-08-02 mirror metadata probe was faster, but
`huggingface_hub.snapshot_download` against `hf-mirror.com` did not receive the
Hub metadata headers it requires, and anonymous direct mirror downloads later
hit HTTP 429. Authenticated direct downloads from `huggingface.co` completed
the pinned snapshot reliably. Prefer the official authenticated endpoint for
the full snapshot; use the mirror only as a connectivity probe or fallback.

Move Hugging Face's generated local-dir `.cache` outside `GT_BASE` before a
run so mutable download metadata is not included in the evaluation-source
provenance fingerprint. The checkpoint sweep below enforces this clean tree.

Generation must use each sample's `video/prompt.txt`; the flattened view's
`prompt.txt` already contains that prompt. Do not edit any of these shared
paths. Personal scorer clones and all generated artifacts belong under the
repository's ignored `storage/` tree.

The current verified scorer revision (2026-08-03) is
`e140038f2aee76ca518f464755fa8bc19b783ba5` from the `main_v2` branch. Its
scorer-contract SHA-256 is
`4cc7d028d4106a28190a63bc179562d5ac9add9263cb71926dd6385c5714bcf8`; this
covers the entrypoint, evaluator Python, bundled annotations, and
`requirements.txt`.

- GitLab: `https://gitlab.bj.sensetime.com/zeotrope/multimodal/vbvr-evalkit-interleave`
- GitHub browser: `https://github.com/xujunxiangwork/VBVR-Evalkit-Interleave`
- GitHub SSH: `git@github.com:xujunxiangwork/VBVR-Evalkit-Interleave.git`

The immediately preceding verified series used revision
`6fedd9d9edb8daafa56aca8e53885aa8ad6f6037` and scorer-contract SHA-256
`eb977da60e95456734063ba018b14d805680179fdf0e3e3b2ba6f603f27a935c`.
Revision `e140038f` changes thousands of lines across the task evaluators, so
its scores are a new reward/evaluation objective rather than a drop-in
relabeling of the `6fedd9d9` results. Keep both revision-specific checkouts and
output namespaces for reproducibility.

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
`storage/eval_out/vbvr_pro_main_v2_evalkit_4cc7d028/`. Historical results tied
to `42a1593d` or `6fedd9d9` keep their original paths; their scores must not be
relabeled or mixed with `e140038f` scores.

The formal evaluation for the native-512 Fujian run trained against the e140
reward uses its matching rollout policy and an isolated converted-model
namespace:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_512x512x81/vbvr_pro_5b_dancegrpo_manifest_rl_fujian_new_e140_cps0p7_sweep_main_v2.fish
```

This targets
`dancegrpo_vbvr_pro_5b_512x512x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl_fujian_new_evalkit_e140038f`,
discovers every complete DCP checkpoint, and generates native 512x512x81 with
30-step Flow-CPS 0.7 / CFG 1.0 / seed 0. It then resizes/pads every frame to
the 1024x1024 scorer canvas while preserving all 81 frames at exact 16 FPS.

The 2026-08-06 run completed checkpoints 100--800. All 4,000 generated videos,
4,000 prepared videos, and 4,000 finite scores completed without errors; every
checkpoint contained 500 samples over 100 tasks with the expected 250/250
domain split. All 24 generation/preparation/score provenance manifests passed
both the launcher's strict audit and a separate full recomputation. One native
and prepared pair per checkpoint also passed a physical media probe.

The sampler-matched step-0 baseline is the merged DiffSynth step-35500
initialization, evaluated with the same Flow-CPS-30/0.7, CFG-1, seed-0,
512x512x81, exact-16-FPS, and e140 contract:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_512x512x81/vbvr_pro_5b_diffsynth_step35500_baseline_cps0p7_main_v2.fish
```

Its additional 500 generated videos, 500 prepared videos, and 500 finite scores
completed without errors. All three provenance manifests passed the launcher
audit and a separate full recomputation, and the native/prepared media probe
confirmed 512/1024 square, 81 frames, exact 16 FPS, and 5.0625 seconds:

| Step | Overall | Delta vs baseline | In-Domain | Out-of-Domain |
| ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.472177 | -- | 0.628099 | 0.316256 |
| 100 | 0.481617 | +0.009440 | 0.634598 | 0.328637 |
| 200 | 0.490144 | +0.017967 | 0.645314 | 0.334975 |
| 300 | 0.500188 | +0.028011 | 0.666405 | 0.333972 |
| 400 | 0.509672 | +0.037495 | 0.673760 | 0.345585 |
| 500 | 0.512673 | +0.040495 | 0.673658 | 0.351688 |
| 600 | 0.518889 | +0.046711 | 0.679263 | 0.358515 |
| **700** | **0.523973** | **+0.051795** | **0.682113** | **0.365832** |
| 800 | 0.519987 | +0.047810 | 0.676687 | 0.363287 |

Checkpoint 100 already improves Overall over the sampler-matched baseline by
`+0.009440`; a paired 100-task, 100,000-resample bootstrap gives a 95% interval
of `[+0.002448, +0.017438]`. Checkpoint 700 is the best Overall, In-Domain,
and Out-of-Domain point estimate. Its Overall gain over baseline is `+0.051795`
with interval `[+0.032053, +0.072562]`; its In-Domain and Out-of-Domain gains
are `+0.054014` and `+0.049576`, with intervals
`[+0.031450, +0.079340]` and `[+0.017611, +0.082936]`. Its five category point
estimates all improve over baseline, led by Spatiality `+0.084247` and
Perception `+0.075632`.

Checkpoint 700's `+0.005084` lead over checkpoint 600 has interval
`[-0.001573, +0.012038]`, and its `+0.003986` lead over checkpoint 800 has
interval `[-0.003360, +0.011956]`. Select checkpoint 700 when one point
estimate is required, but treat checkpoints 600--800 as a statistically tied
late plateau. The complete results, per-category tables, task workbooks, and
provenance live under
`storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_fujian_new_e140_lr5e6_eval500_181e2010_manifest_afab352e_evalkit_4cc7d028/`.

For the earlier 384-trained Fujian manifest-RL checkpoint series, the formal
evaluation also generates native 512x512x81 video before the same scorer
preparation:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_384x384x81_fujian/vbvr_pro_5b_dancegrpo_manifest_rl_fujian_cps0p7_512x512_sweep_main_v2.fish
```

It discovers every complete `checkpoint-N/high/.metadata`, generates with
30-step Flow-CPS 0.7 / CFG 1.0 / seed 0, and fills each generation wave across
all eight local GPUs. The unsuffixed
`...fujian_cps0p7_sweep_main_v2.fish` entry point instead generates native
384x384x81 and is retained only as a controlled resolution ablation. A
checkpoint is complete only after 500 generated and prepared videos, 500
error-free scores across 100 tasks, three recomputed provenance manifests, the
task workbook, and `final_scores.txt` all pass.

When the audited native-512 videos and their 1024x1024 prepared copies already
exist, migrate only the scorer with:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_384x384x81_fujian/vbvr_pro_5b_dancegrpo_manifest_rl_fujian_rescore_512x512_main_v2.fish
```

The wrapper requires generation provenance for native 512x512x81 media and
preparation provenance for 1024x1024x81 scorer media before EvalKit starts.
The unsuffixed `...rescore_main_v2.fish` requires native 384 provenance and is
the matching ablation-only scorer path. Neither path loads a checkpoint,
generates video, nor runs video preparation. Both write scores, workbooks,
complete score provenance, `checkpoint_scores.tsv`, and
`scorer_migration.tsv` under resolution-specific
`...rescore_from_evalkit_eb977da6_to_evalkit_4cc7d028` namespaces.

The 2026-08-04 formal native-512 scorer-only e140 migration completed all 14
checkpoints. It reused 7,000 audited native-512 videos and their 7,000
1024x1024/81-frame/exact-16-FPS prepared copies, ran two scorer workers with
eight native threads per worker and up to eight checkpoints in parallel, and
did not generate or prepare any video. All 7,000 scores were finite and
error-free, every run contained 500 samples across 100 tasks, all three
provenance stages passed independent recomputation, one native/prepared pair
per checkpoint passed a physical media probe, and the score-only output tree
contains zero MP4s:

| Step | Overall | In-Domain | Out-of-Domain |
| ---: | ---: | ---: | ---: |
| 100 | 0.489767 | 0.647479 | 0.332054 |
| 200 | 0.504265 | 0.664863 | 0.343667 |
| 300 | 0.507089 | 0.659857 | 0.354321 |
| 400 | 0.514215 | 0.663105 | 0.365325 |
| 500 | 0.517292 | 0.672174 | 0.362409 |
| **600** | 0.518330 | 0.669176 | **0.367484** |
| 700 | 0.519915 | 0.672993 | 0.366837 |
| 800 | 0.520253 | 0.675117 | 0.365389 |
| 900 | 0.523158 | 0.683787 | 0.362530 |
| 1000 | 0.523431 | 0.687914 | 0.358948 |
| 1100 | 0.528449 | 0.692038 | 0.364861 |
| 1200 | 0.525869 | 0.693249 | 0.358488 |
| **1300** | **0.528663** | **0.697320** | 0.360007 |
| 1400 | 0.524030 | 0.695880 | 0.352180 |

Checkpoint 1300 is the best Overall and In-Domain point estimate, while
checkpoint 600 is best Out-of-Domain. Step 1300 improves Overall by
`+0.038897` over step 100; a paired 100-task, 100,000-resample bootstrap gives
a 95% interval of `[+0.018933, +0.059656]`. Its `+0.000214` Overall advantage
over the second-place checkpoint 1100 has interval
`[-0.006819, +0.007810]`, so the late leaders remain statistically tied.
Relative to `6fedd9d9` on the identical native-512/prepared videos, e140 shifts
Overall by an average `-0.064370` (range `-0.066971` to `-0.062819`), with
average In-Domain and Out-of-Domain shifts of `-0.029423` and `-0.099317`.
This is a scorer-objective migration, not a model regression. The complete
formal result is under
`storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_fujian_eval500_181e2010_manifest_afab352e_rescore_from_evalkit_eb977da6_to_evalkit_4cc7d028/`.

The earlier native-384 e140 migration is a resolution ablation, not the formal
checkpoint curve. It completed the same 14 checkpoints with 7,000 error-free
scores and zero output MP4s under
`storage/eval_out/vbvr_pro_main_v2_384x384x81_manifest_rl_fujian_eval500_181e2010_manifest_afab352e_rescore_from_evalkit_eb977da6_to_evalkit_4cc7d028/`:

| Step | Overall | In-Domain | Out-of-Domain |
| ---: | ---: | ---: | ---: |
| 100 | 0.429949 | 0.574266 | 0.285632 |
| 200 | 0.460630 | 0.610357 | 0.310904 |
| 300 | 0.460650 | 0.615194 | 0.306106 |
| 400 | 0.467069 | 0.624927 | 0.309210 |
| 500 | 0.469105 | 0.634865 | 0.303344 |
| 600 | 0.470054 | 0.625885 | 0.314223 |
| 700 | 0.478236 | 0.637653 | 0.318819 |
| 800 | 0.479912 | 0.641542 | 0.318282 |
| 900 | 0.480224 | 0.644147 | 0.316301 |
| 1000 | 0.485576 | 0.648811 | 0.322342 |
| 1100 | 0.488447 | 0.651377 | 0.325518 |
| **1200** | **0.489997** | **0.658287** | 0.321708 |
| 1300 | 0.488286 | 0.653709 | 0.322862 |
| 1400 | 0.488462 | 0.648898 | **0.328026** |

Under the same e140 scorer, native 512 improves Overall at every checkpoint by
`+0.035568` to `+0.059817`, averaging `+0.043438`; the mean In-Domain and
Out-of-Domain gains are `+0.043217` and `+0.043659`. At step 100 the
512-minus-384 gain is `+0.059817` with paired-task 95% interval
`[+0.030753, +0.089972]`; at step 1300 it is `+0.040378` with interval
`[+0.017841, +0.064201]`. This explains the initially low `0.429949`: the
correct e140/native-512 value is `0.489767`; the remaining `-0.066971` versus
the historical native-512 `6fedd9d9` value `0.556738` is the scorer revision,
not a resolution or model change.

Evaluate the DiffSynth step-35500 initialization on that exact snapshot with
its requested 50-step UniPC ODE recipe using:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_384x384x81_fujian/vbvr_pro_5b_dancegrpo_manifest_rl_fujian_initial_unipc_main_v2.fish
```

This baseline keeps native 384x384x81 generation, CFG 5.0, seed 0, exact 16
FPS, and the same 1024x1024 all-frame scorer preparation. Its audited
2026-08-03 rerun on HF revision `181e2010...` scored `0.514934` Overall,
`0.608369` In-Domain, and `0.421500` Out-of-Domain. All 500 generated and 500
prepared videos, 500 error-free scores over 100 tasks, workbook, and three
recomputed provenance stages passed. Keep it separate from the CPS checkpoint
curve: it is a complete UniPC-50/CFG-5 serving-recipe baseline, not a
sampler-matched checkpoint-0 ablation.

The historical 2026-08-03 `6fedd9d9` sweep completed checkpoints 100 through
1400 against the pinned HF snapshot and scorer contract. All 14 runs passed the
strict audit. Aggregate scores were:

| Step | Overall | In-Domain | Out-of-Domain |
| ---: | ---: | ---: | ---: |
| 100 | 0.505991 | 0.617560 | 0.394423 |
| 200 | 0.532319 | 0.646827 | 0.417810 |
| 300 | 0.532665 | 0.649466 | 0.415864 |
| 400 | 0.537259 | 0.657328 | 0.417190 |
| 500 | 0.539255 | 0.667002 | 0.411509 |
| 600 | 0.541091 | 0.659629 | 0.422553 |
| 700 | 0.547363 | 0.669679 | 0.425047 |
| 800 | 0.548401 | 0.674610 | 0.422192 |
| 900 | 0.548897 | 0.676404 | 0.421390 |
| 1000 | 0.553764 | 0.680293 | 0.427235 |
| 1100 | 0.556831 | 0.682909 | 0.430754 |
| **1200** | **0.557430** | **0.687768** | 0.427093 |
| 1300 | 0.556404 | 0.684874 | 0.427933 |
| 1400 | 0.557272 | 0.682538 | **0.432006** |

Checkpoint 1200 is the best Overall/ID point estimate, while checkpoint 1400
is best OOD. Overall improves by `+0.051439` from step 100 to 1200 and then
plateaus. A paired 100-task bootstrap with 100,000 resamples puts the
step-1200-minus-step-1400 Overall difference (`+0.000158`) at a 95% interval
of `[-0.010245, +0.011591]`, so do not claim that the top two are meaningfully
separated. Use step 1200 as the default Overall/ID selection and step 1400 only
when the OOD point estimate is the primary target.

Against the UniPC initialization recipe, checkpoint 100 CPS is `-0.008943`
Overall with a paired 100-task 95% bootstrap interval of
`[-0.037543, +0.019823]`. Checkpoint 1200 CPS is `+0.042496` Overall,
`+0.079399` In-Domain, and `+0.005592` Out-of-Domain; its Overall interval is
`[+0.010640, +0.074736]` with 63 task means improved, 2 tied, and 35
regressed. These comparisons include the decoding-policy change and therefore
do not isolate the effect of RL weights.

Run the same model series with direct native 512x512x81 generation using the
fixed-resolution entry points:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_384x384x81_fujian/vbvr_pro_5b_dancegrpo_manifest_rl_fujian_initial_unipc_512x512_main_v2.fish
fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_384x384x81_fujian/vbvr_pro_5b_dancegrpo_manifest_rl_fujian_cps0p7_512x512_sweep_main_v2.fish
```

The audited 2026-08-03 native-512 rerun completed the initialization and all
14 checkpoints with 7,500 generated videos, 7,500 prepared videos, 7,500
error-free scores, and 45 independently recomputed provenance manifests.
Checkpoint 1100 is the best Overall point estimate at `0.592322`; checkpoint
1300 is best In-Domain at `0.725404`, and checkpoint 600 is best Out-of-Domain
at `0.466747`. The UniPC initialization scores `0.548310` Overall, `0.664182`
In-Domain, and `0.432439` Out-of-Domain. Checkpoint 1100 improves over that
serving-recipe baseline by `+0.044012` Overall, with a paired 100-task,
100,000-resample 95% interval of `[+0.019310, +0.069940]` and 64/5/31
improved/tied/regressed task means. Checkpoints 1100 and 1300 remain
statistically indistinguishable: their Overall difference is `+0.000403`,
with interval `[-0.006153, +0.006701]`.

Native 512 improves Overall over native 384 for the initialization and every
RL checkpoint. Across the 14 same-checkpoint comparisons, the gain ranges
from `+0.031715` to `+0.050747` and averages `+0.037926`. For checkpoint 1100,
the 512-minus-384 delta is `+0.035491`, with paired-task 95% interval
`[+0.016061, +0.055661]`. The complete curve, resolution table, bootstrap
comparisons, and audit are under
`storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_fujian_eval500_181e2010_manifest_afab352e_evalkit_eb977da6/`.

This native-512 comparison intentionally keeps each model's requested serving
recipe: UniPC-50/CFG-5 for the initialization and CPS-30/CFG-1 for the RL
checkpoints. It supports an end-to-end model-selection conclusion, not a
sampler-matched causal claim about weights alone.

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
set -lx OUTPUT_ROOT storage/eval_out/vbvr_pro_main_v2_evalkit_4cc7d028/my_run
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

Generating at the checkpoint's 512x512 training resolution while keeping the
same 81-frame UniPC settings scored 0.548651 Overall, 0.665233 In-Domain, and
0.432068 Out-of-Domain. This controlled `+0.142108` Overall gain over direct
256x256 generation shows that native generation resolution, rather than LoRA
conversion scale, was a major source of the earlier low score.

The controlled intermediate-resolution counterpart is:

```fish
fish scripts/eval/vbvr_pro/vbvr_pro_5b_diffsynth_step35500_unipc_50steps_384x384_81f_16fps_main_v2.fish
```

It changes only native generation to 384x384 and scored 0.514761 Overall,
0.608461 In-Domain, and 0.421061 Out-of-Domain. This is `+0.108218` over
256x256 and `-0.033890` below 512x512 Overall: 384 retains 76.15% of the
controlled 256-to-512 gain while using 56.25% as many latent spatial positions
as 512. Against 512, 32 paired task means improved, 6 tied, and 62 regressed;
a 100,000-resample paired-task bootstrap gave a 95% Overall-delta interval of
`[-0.058508, -0.009946]`. The loss is therefore measurable rather than exact
quality parity, with most of the domain-level gap in In-Domain (`-0.056772`)
and a smaller Out-of-Domain delta (`-0.011007`). All 500 generated and 500
prepared videos passed exact size/81-frame/16-FPS/5.0625-second checks, all
scores were finite and error-free, and all three provenance manifests passed
artifact-fingerprint recomputation.

The GT-duration-matched counterpart is:

```fish
fish scripts/eval/vbvr_pro/vbvr_pro_5b_diffsynth_step35500_unipc_50steps_512x512_gt_duration_16fps_main_v2.fish
```

It probes every GT video, derives the requested frame count at 16 FPS, applies
Wan's `4k+1` temporal contract exactly as Diffusers does, generates that many
frames directly at 512x512, and resizes every frame to the GT canvas without
cropping. All 500 current GT videos are 1024x1024 at 16 FPS. Their raw lengths
are 10--282 frames; aligned generation lengths are 9--281 frames. The completed
run scored 0.547386 Overall, 0.657960 In-Domain, and 0.436812 Out-of-Domain.
Against the otherwise identical fixed-81-frame native-512 run, the Overall
delta was `-0.001265`; a 100,000-resample paired-task bootstrap gave a 95%
interval of `[-0.024533, +0.022437]`. Thus GT-duration matching is not a stable
aggregate improvement for this checkpoint. See
[`reports/vbvr_diffsynth_step35500_gt_duration_eval_2026-07-26.md`](reports/vbvr_diffsynth_step35500_gt_duration_eval_2026-07-26.md)
for the paired analysis and audit.

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
`storage/eval_out/vbvr_pro_main_v2_indomain_strict_manifest_326f7bda_evalkit_4cc7d028/`.
The completion check compares the full manifest hash, EvalKit revision, and
scorer-contract hash and recomputes every recorded input/output artifact
fingerprint and path binding before skipping a run.
Jobs are strictly sequential, write one log per checkpoint/mode under
`storage/eval_logs/vbvr_pro_main_v2_indomain_strict_evalkit_4cc7d028/`, and count a run complete
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

### Per-step clean-trajectory visualization

The unified `python -m src.inference --save_steps` renderer supports ODE, SDE,
and Flow-CPS with one definition: cells 1 through `T-1` decode the post-CFG
predicted-clean endpoint `x0 = x_sigma - sigma * velocity` from the sampler's
actual current state, while cell `T` decodes the actual final latent at sigma
zero. These are clean endpoint predictions, not decoded noisy sampler states or
pixel-space interpolation. For 5B `expand_timesteps` I2V, the renderer re-pins
latent frame zero to the encoded input condition in every preview, matching the
real rollout's frozen-first-frame contract. Grid/contact-sheet labels are
one-based and include the source sigma; compatibility MP4 filenames remain
zero-based (`step_00.mp4` through `step_{T-1}.mp4`). `manifest.json` records the
kind, source sigma, output sigma, and file for every displayed cell.

For the native-512 Fujian comparison, run the complete matched matrix with:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_512x512x81/vbvr_pro_5b_sampler_matrix_30steps_main_v2.fish
```

It evaluates the DiffSynth step-35500 baseline plus every complete checkpoint
with Flow-CPS coefficients 0.1/0.3/0.7/0.9, deterministic FlowMatch Euler,
and UniPC. Every cell fixes 30 steps, CFG 1.0, seed 0, 512x512x81, exact 16
FPS, the 500-sample manifest, and EvalKit e140. The launcher resumes by a
recorded-contract audit and uses four two-GPU jobs per eight-GPU wave.
The 2026-08-06 quantitative process captured the immutable model snapshot
`baseline,100,200,...,900` when it started. Checkpoints 1000 and later appeared
while that process was running and are not members of this 60-cell comparison;
they need all six formal evaluations before being added to its trajectory set.

After every quantitative cell is complete, render the fixed sample's full
30-step gallery with:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_512x512x81/render_vbvr_pro_sampler_matrix_30steps.fish
```

The renderer defaults to one cell per local GPU. To fill only otherwise-idle
devices while a quantitative wave is running, set the Fish list
`TRAJECTORY_CUDA_DEVICES` before launch; `MODEL_FILTER` and `SAMPLER_FILTER`
select one model/sampler cell. A filtered render must finish before the next
quantitative wave reclaims that GPU.

The trajectory renderer intercepts Euler/UniPC only after the real scheduler
step and computes `x0 = x_t - sigma * velocity` on CPU. The displayed cell 30
and `final_00.mp4` are copied byte-for-byte from the quantitative MP4 scored by
EvalKit; this avoids claiming that a separately observed fused-CUDA run is
bit-exact. Build score tables and the 60-cell HTML video gallery with:

```bash
.venv/bin/python -m src.cli.summarize_vbvr_sampler_matrix \
  --eval-output-base storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_fujian_new_e140_lr5e6_eval500_181e2010_manifest_afab352e_evalkit_4cc7d028 \
  --trajectory-root storage/eval_out/vbvr_pro_sampler_matrix_30step_trajectories
```

The fixed-sample gallery above is only a compact diagnostic. To render the
requested trajectory for all 500 outputs in every one of the 60 formal cells,
run:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_512x512x81/render_vbvr_pro_sampler_matrix_all_outputs_30steps.fish
```

The full launcher first audits every quantitative cell, then keeps each model
loaded while it renders its assigned samples. It is sample-level resumable and
pins its default checkpoint snapshot to 100 through 900 so later training
checkpoints cannot enter only part of the matrix. `MODEL_FILTER`,
`SAMPLER_FILTER`, and the Fish-list `TRAJECTORY_CUDA_DEVICES` narrow a run;
`TRAJECTORY_LIMIT` is intended only for small end-to-end smoke tests.
`TRAJECTORY_SAMPLE_SHARDS_PER_CELL=N` splits every cell into `N` deterministic
sample shards, while `TRAJECTORY_WORKERS_PER_GPU=N` caps concurrent independent
batch-one workers per selected GPU. The shard count defaults to the worker cap
for backward compatibility. This is process-level concurrency, not a tensor
batch: it preserves each sample's seed, CPS random stream, and numerical path
while allowing one worker's media-writing gaps to overlap another worker's GPU
work. A custom shard count must fit within and evenly divide the node's
`GPU count * workers/GPU` slots so a cell never straddles launch waves. The
renderer deliberately creates no tar archive. The default unpacked layout is:

```text
storage/eval_out/vbvr_pro_sampler_matrix_all_500_30step_trajectories/
  baseline-cps0p1/
    cell_manifest.json
    In-Domain_50/<task>/<sample>/
      step_00.mp4 ... step_29.mp4
      final_00.mp4
      steps_grid.mp4
      step_contact_sheet.jpg
      manifest.json
```

This is 30,000 sample trajectories and 900,000 individual step MP4s, plus
final/grid/contact/manifest files. A two-sample native-512 CPS smoke measured
63.23 and 55.06 seconds per sample after one model load; use roughly three days
as an eight-GPU order-of-magnitude runtime, with sampler and filesystem
variation. Measured representative media projected about 45–52 GiB allocated;
retain 60–80 GiB free for margin. `src.cli.audit_vbvr_i2v_trajectories` is the
lightweight strict completion check used by the launcher: it verifies the full
contract, CPS coefficient, exact artifact set, and SHA-256 equality between
`step_29.mp4`, `final_00.mp4`, and the formal quantitative MP4.

### Baseline versus checkpoint-2200 trajectory Space

Build the interactive left/right comparison site from the completed all-output
archive with:

```bash
.venv/bin/python scripts/eval/vbvr_pro/build_vbvr_trajectory_space.py \
  --output-root storage/hf_spaces/vbvrpro_sampler_trajectories \
  --skip-videos \
  --media-url-prefix https://huggingface.co/datasets/pufanyi/vbvrpro_sampler_trajectories-data/resolve/main/videos \
  --step-media-url-prefix baseline=https://huggingface.co/datasets/pufanyi/vbvrpro_sampler_trajectories-baseline-steps/resolve/main/videos \
  --step-media-url-prefix checkpoint-2200=https://huggingface.co/datasets/pufanyi/vbvrpro_sampler_trajectories-2200-steps/resolve/main/videos
```

The builder selects exactly the DiffSynth step-35500 baseline and DanceGRPO
checkpoint 2200 across CPS 0.1/0.3/0.7/0.9, Euler ODE, and UniPC ODE. It
requires the same 500 sample identities in all 12 cells and validates every
cell manifest, prompt, 30-step schedule, all native `step_00.mp4` through
`step_29.mp4` files, `steps_grid.mp4`, and `final_00.mp4`. It also resolves the
exact formal result for each cell, verifies the result fingerprint and complete
score provenance, requires one numeric `[0, 1]` score for every matched sample,
and verifies that each public final/step-30 video is bound to the corresponding
formally scored output. The compact index therefore contains 6,000 aligned
scores plus Overall/In-Domain/Out-of-Domain cell means. The complete deployment
is 192,000 MP4s/about 6.51 GiB: 180,000 native step videos/about 5.65 GiB plus
the existing 6,000 overview grids and 6,000 native finals/about 0.86 GiB.

The public frontend is
[`pufanyi/vbvrpro_sampler_trajectories`](https://huggingface.co/spaces/pufanyi/vbvrpro_sampler_trajectories),
backed by
[`pufanyi/vbvrpro_sampler_trajectories-data`](https://huggingface.co/datasets/pufanyi/vbvrpro_sampler_trajectories-data),
[`pufanyi/vbvrpro_sampler_trajectories-baseline-steps`](https://huggingface.co/datasets/pufanyi/vbvrpro_sampler_trajectories-baseline-steps),
and
[`pufanyi/vbvrpro_sampler_trajectories-2200-steps`](https://huggingface.co/datasets/pufanyi/vbvrpro_sampler_trajectories-2200-steps).
The two native-step archives each contain 90,000 MP4s, keeping both below the
[Hub's recommended 100,000 files per Git-backed repository](https://huggingface.co/docs/hub/storage-limits#repository-limitations-and-recommendations).
The static UI independently selects the model and sampler on each side,
synchronizes paired playback, selects any of the 30 original step videos,
auto-advances through a complete path, retains the compressed grid only as an
optional overview, preserves comparisons and the selected step in shareable
URLs, and lazily opens the full 2x6 matrix at the current step/view. Both sides
and every matrix card show the selected test case's final EvalKit score to four
decimal places and the cell's 500-sample mean. The score is explicitly
final-only (public step 30 / `final_00.mp4`); it remains visible while stepping
through the path but does not claim that intermediate previews were rescored.

Publish or resume the native files directly from the strict trajectory source
tree with:

```bash
ulimit -n 65536
HF_XET_HIGH_PERFORMANCE=1 \
  .venv/bin/python scripts/eval/vbvr_pro/upload_vbvr_trajectory_steps.py \
    --batch-size 1000 \
    --num-threads 16
```

The uploader is additive: it enumerates the exact expected paths, skips files
already present on each Dataset, commits bounded batches, retries uncertain
requests after checking the remote paths, and performs a strict 90,000-video
completion audit per model. The descriptor limit is process-local; do not
change the system-wide limit.

For a scheduler allocation with multiple eight-GPU machines, launch this on
every node instead:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_512x512x81/render_vbvr_pro_sampler_matrix_all_outputs_30steps_multinode.fish
```

It consumes the same scheduler contract as `scripts/train/grpo_multinode.fish`:
`WORLD_SIZE` is the number of machines and `RANK` is the zero-based machine
rank. `--nproc 8` and `--workers-per-gpu 2` are the defaults;
`--sample-shards-per-cell N` optionally decouples sample partitioning from the
per-GPU concurrency cap. The wrapper does
not create a `torchrun` process group; it round-robin shards the fixed 60-cell
list by node, splits each local cell into even/odd sample shards, and runs 14–16
independent batch-one workers per node when `WORLD_SIZE=8`. Every completed
sample is discovered and skipped before generation, including samples made by
the older unsharded launcher. Only a strict audit of all 500 samples publishes
the canonical complete-cell manifest.

Two workers were pressure-tested on one 80-GiB H800 at a 53,963-MiB combined
peak. Ten one-second samples during the main phase reported 100% SM utilization,
and all 136 non-manifest artifacts from four trajectories were byte-identical
to their single-worker counterparts. This concurrency is primarily a safe
gap-hiding knob: a deliberately synchronized same-cell pair took about 119
seconds for two samples versus about 115 seconds sequentially, so it does not
speed up the already saturated transformer work. Use `--workers-per-gpu 1` on
cards with less memory or when duplicate model residency is not worthwhile;
do not raise it above two without a new production-shape memory test.

The launcher prints trajectory counts, percent complete, run throughput, ETA,
and per-cell counts every 30 seconds (`--progress-interval N` changes the
period). Each node reports its own assigned cells. On any node that can see the
shared output tree, run this for one global 60-cell view:

```bash
.venv/bin/python -m src.cli.watch_vbvr_trajectory_progress \
  --trajectory-root storage/eval_out/vbvr_pro_sampler_matrix_all_500_30step_trajectories \
  --shard-count 2 --watch --interval 30
```

All nodes must see the same repository, formal evaluation roots, converted
models, trajectory root, and log directory. Set
`TRAJECTORY_ASSIGNMENT_ONLY=1` to print the node/GPU/sample-shard assignment
without loading a model or writing outputs. The wrapper captures then removes
the ambient distributed rank variables before entering Python, because these
jobs are independent single-GPU inference processes rather than distributed
model workers.

The immutable 2026-08-07 extension for checkpoints 1000, 1100, 1200, 1300,
and 1400 uses one two-stage wrapper. Run the formal stage on all eight nodes
first:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_512x512x81/evaluate_new_checkpoints_1000_1400_multinode.fish \
  formal --nproc 8
```

After every node exits successfully, submit the same eight nodes for all-output
trajectories:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_512x512x81/evaluate_new_checkpoints_1000_1400_multinode.fish \
  trajectories --nproc 8 --workers-per-gpu 2 --progress-interval 30
```

The formal stage contains exactly 30 cells (five checkpoints times six
samplers), assigns four disjoint two-GPU cells to ranks 0–5 and three to ranks
6–7, and strictly skips complete results. The trajectory stage contains the
same 30 cells split into 60 deterministic sample shards; ranks 0–5 use all
eight GPUs and ranks 6–7 use six. It reuses the shared output roots, skips every
already-valid sample, and refuses to start generation until all corresponding
formal cells pass their provenance audit. Baseline and checkpoints 100–900 are
excluded from both stages. Add `--assignment-only` to either command to inspect
the complete node/GPU plan without conversion, inference, or scoring. There is
deliberately no cross-submission barrier: do not start `trajectories` until the
eight `formal` commands have all succeeded.

The immutable 2026-08-08 extension covers the next four complete checkpoints,
1500, 1600, 1700, and 1800. Run its two stages on all eight nodes in the same
order:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_512x512x81/evaluate_new_checkpoints_1500_1800_multinode.fish \
  formal --nproc 8
```

After all formal processes exit successfully:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_512x512x81/evaluate_new_checkpoints_1500_1800_multinode.fish \
  trajectories --nproc 8 --workers-per-gpu 2 --progress-interval 30
```

This snapshot adds 24 formal cells, exactly three two-GPU cells per node, and
48 trajectory shards, exactly six single-GPU workers per node. GPUs 6–7 are
therefore unused by this balanced four-checkpoint extension. Both stages
exclude baseline and checkpoints 100–1400, retain strict completion/provenance
checks, and resume at cell/sample granularity. Add `--assignment-only` to
either command for a side-effect-free assignment audit.

For all later checkpoints in this same rule-reward series, use the reusable
incremental wrapper instead of creating another fixed step-range script. On
every evaluation node run:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_512x512x81/evaluate_incremental_multinode.fish \
  formal --nproc 8
```

It discovers only complete numeric DCP checkpoints, strictly audits all six
formal sampler cells, and delegates only checkpoints with at least one missing
or invalid result. After every node exits successfully, the corresponding
incremental all-output trajectory command is:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_512x512x81/evaluate_incremental_multinode.fish \
  trajectories --nproc 8 --workers-per-gpu 2 --progress-interval 30
```

The trajectory stage accepts only checkpoints whose six formal cells already
pass, then relies on the existing strict cell/sample audit to skip complete
trajectory cells and resume partial ones. `formal` is the default stage, and
`--assignment-only` remains available for both stages. Evaluation node count is
not fixed; scheduler `WORLD_SIZE=6`, `RANK=0..5` is valid. Do not overlap two
invocations, and do not start `trajectories` until every formal node exits.

To generate only checkpoint 2200, pin the same snapshot on every node. First
audit/resume its six formal 500-sample cells:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_512x512x81/evaluate_incremental_multinode.fish \
  formal --checkpoints 2200 --nproc 8
```

After every formal node exits successfully, render all 500 outputs for CPS
0.1/0.3/0.7/0.9, Euler ODE, and UniPC ODE, retaining every one of the 30 clean
endpoint previews:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_manifest_rl_512x512x81/evaluate_incremental_multinode.fish \
  trajectories --checkpoints 2200 --nproc 8 \
  --workers-per-gpu 1 --sample-shards-per-cell 8 --progress-interval 30
```

`--checkpoints` also accepts a comma-separated list such as `1900,2000,2100,2200`.
The selected checkpoints must exist as complete numeric DCP directories; an
unknown or incomplete requested step fails immediately instead of silently
falling back to the automatic discovery set. For exactly one checkpoint, a
six-node allocation maps its six sampler cells one per node; eight shards then
use all eight local GPUs with one model process per GPU. With more than six
nodes, the extra ranks correctly exit with no assigned cell.

After all cells finish, build the lazy-loading browser without copying or
archiving media (30,000 outputs for the original snapshot, 45,000 after the
five-checkpoint extension, or 57,000 after both extensions):

```bash
.venv/bin/python -m src.cli.build_vbvr_trajectory_gallery \
  --eval-output-base storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_fujian_new_e140_lr5e6_eval500_181e2010_manifest_afab352e_evalkit_4cc7d028 \
  --trajectory-root storage/eval_out/vbvr_pro_sampler_matrix_all_500_30step_trajectories
```

Its matrix index is `gallery/index.html`; each cell page (60 originally, 90
after the first extension, or 114 after both) provides a searchable list of 500
lazy-loaded grid videos, contact sheets, exact scored finals, manifests, and
direct links to all 30 individual step MP4s.
The score-table summarizer also understands this layout:

```bash
.venv/bin/python -m src.cli.summarize_vbvr_sampler_matrix \
  --eval-output-base storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_fujian_new_e140_lr5e6_eval500_181e2010_manifest_afab352e_evalkit_4cc7d028 \
  --trajectory-root storage/eval_out/vbvr_pro_sampler_matrix_all_500_30step_trajectories \
  --trajectory-layout all-samples \
  --output-dir storage/eval_out/vbvr_pro_sampler_matrix_all_500_30step_trajectories/summary
```

### Incremental Qwen3.6-VLM checkpoint evaluation

The native-512 Qwen3.6-VLM DanceGRPO run has a thin multi-node adapter around
the shared native-512 incremental evaluator. To evaluate exactly one DCP save,
run the same command on every evaluation node and pass the complete
`checkpoint-N` directory directly:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_vlm_qwen36_512x512x81/evaluate_incremental_multinode.fish \
  formal \
  --checkpoint-dir storage/checkpoints/dancegrpo_vbvr_pro_5b_512x512x81_vlm_qwen36_task_prompts_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl_new_2_nodes16_world128/checkpoint-600 \
  --nproc 8
```

It uses scheduler `WORLD_SIZE` as the current evaluation-machine count and
`RANK` as the zero-based machine rank; this is independent of the source
training topology, so five or six eight-GPU evaluation nodes are valid even
though this checkpoint was trained with world128. `--checkpoint-dir` derives
the parent checkpoint root and numeric step, then isolates the converted model,
formal results, logs, and optional trajectory outputs using the parent run
name. It cannot be combined with `--checkpoints` because it already selects one
step.

Omit `--checkpoint-dir` to scan the default output of
`configs/train_dancegrpo_vbvr_pro_5b_512x512x81_vlm_qwen36_cps_from_nsft_bs_32_lr_5e-6_manifest_rl_multinode.yaml`,
currently the topology-suffixed `...manifest_rl_new_2_nodes16_world128` root.
Use `--checkpoint-root PATH` for another run, optionally with
`--checkpoints N[,N...]`; the older `VLM_EVAL_CHECKPOINT_ROOT` environment
override remains supported.

Every invocation discovers directories named exactly `checkpoint-<integer>`
with `high/.metadata`, ignores incomplete saves and aliases such as
`checkpoint-epoch0`, and strictly audits CPS 0.1/0.3/0.7/0.9 plus matched
30-step Euler and UniPC results. A checkpoint is skipped only when all six
500-sample results, workbooks, media counts, and provenance contracts pass.
New or partial checkpoints reuse conversion artifacts and resume the existing
generation/preparation/scoring trees.

For this VLM-trained-run adapter, `formal` now continues automatically into the
training-time Qwen3.6 judge. Checkpoint discovery is frozen at invocation so
formal inference and judge evaluation use the same exact six-cell-per-step
snapshot. After every node finishes its local formal shard, each node polls the
shared evaluator's strict, side-effect-free audit until all selected formal
cells are complete; only then are those exact cell names deterministically
sharded across judge nodes. The judge's source/contract audit skips completed
cells and resumes partial JSONL, and a node with no pending assigned cell does
not start Qwen. Thus rerunning the same `formal --nproc 8` command is the normal
way to discover and fill missing judge results without regenerating videos.
The default judge root is the formal output root plus
`_vlm_qwen36_27b_task_judge_4d315923`.

All nodes should come from the same scheduler launch. If training may finish a
new DCP save while evaluation nodes are still starting, pass an explicit
`--checkpoints N[,N...]` snapshot so different ranks cannot discover different
checkpoint sets.

Use `--no-vlm-judge` (or `VLM_EVAL_AUTO_JUDGE=0`) for the previous EvalKit-only
behavior. `--vlm-concurrency N` defaults to twice `--nproc`, and
`--vlm-output-root PATH` overrides the independent judge destination. The
validated eight-GPU default is Qwen TP2 x DP4; for another even `--nproc`, the
adapter defaults to TP2 and one DP replica per GPU pair. Add
`--assignment-only` to inspect formal allocation without conversion,
evaluation, Qwen startup, or judge writes. Wait for every node from one
invocation to exit before starting the next one; overlapping invocations can
observe different completion snapshots while a cell is being promoted.

After all six formal cells for an exact checkpoint are complete, the same
adapter also delegates the optional all-output path without duplicating the
evaluation logic:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_vlm_qwen36_512x512x81/evaluate_incremental_multinode.fish \
  trajectories --checkpoint-dir /path/to/checkpoint-600 \
  --nproc 8 --workers-per-gpu 1 --sample-shards-per-cell 8
```

Use `FORCE_REGENERATE=1` only when intentionally replacing personal generated
videos after a generation-provenance change; the launcher will also rebuild
the prepared videos. Use `FORCE_REPREPARE=1` for a preparation-only change such
as CRF. A converted-model provenance mismatch requires a fresh conversion path
or an explicit reconversion rather than silently overwriting a model tree.

The automatic judge stage uses the existing formal videos directly and never
runs a second Wan inference stage. The standalone command remains useful for
backfilling any compatible native-512 formal result root or for operating a
separately managed judge service:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_vlm_qwen36_512x512x81/evaluate_vlm_judge_multinode.fish \
  score --input-root /path/to/formal-result-root --concurrency 16
```

The input may be the VLM-trained checkpoint result root or any compatible
native-512 formal sampler matrix. `WORLD_SIZE/RANK` shard complete sampler
cells across evaluation machines; one machine needs neither variable. The
judge reads each cell's `eval_samples.json` plus complete generation
provenance, sends its existing `generated_512x512x81` MP4 directly with the
matching first frame, and writes an independent resumable result root. See
`docs/vlm_judge_reward.md` for the exact request, retry, fingerprint, startup,
and aggregation contracts.

The 2026-08-13 one-node run over the complete baseline-plus-checkpoint-100--2300
matrix finished all 144 sampler cells and 72,000 videos with no request,
semantic, or fallback errors. Its global Qwen mean was `0.587300`; checkpoint
2200 had the best six-sampler mean (`0.601000`), and checkpoint-2200 Euler was
the best individual cell (`0.615940`). The detailed sampler/domain breakdown
and judge-versus-EvalKit comparison are recorded in
`docs/vlm_judge_reward.md`.

### Audited sampler checkpoint trend plots

Use `src.cli.plot_vbvr_checkpoint_trends` to render the matched six-sampler
checkpoint curves for an offline task-judge root and a formal EvalKit root:

```fish
.venv/bin/python -m src.cli.plot_vbvr_checkpoint_trends \
  --vlm-judge-root $vlm_judge_root \
  --evalkit-root $evalkit_checkpoint_root \
  --evalkit-baseline-root $matched_formal_baseline_root \
  --output-dir $trend_output_root
```

The command audits every plotted cell as a 500-sample, zero-error result. It
cross-checks the offline judge CSV against each cell summary and judge
contract; formal results additionally require complete score provenance plus
the recorded result size and SHA-256. The six formal baselines may come from a
different result root only when its EvalKit source and runtime contract exactly
match the checkpoint results. Incremental cells whose score provenance is not
yet complete remain explicit gaps instead of being treated as zero or stale
scores.

The output directory receives two individual PNG/SVG plots, one 2x3 comparison
plot, a combined CSV, and a JSON audit/best-checkpoint summary. The two rows use
different evaluators, so the plot labels this explicitly: compare trends within
each row and do not interpret absolute Qwen-judge and EvalKit scores as directly
comparable. Keeping all plots in the independent output root also supports
read-only evaluation result directories.

To plot a single checkpoint-only offline VLM-judge run, use the dedicated
entry point and supply any already-complete VLM result root that contains the
six matched DiffSynth baselines:

```fish
.venv/bin/python -m src.cli.plot_vbvr_vlm_checkpoint_trends \
  --vlm-judge-root $checkpoint_vlm_root \
  --vlm-baseline-root $matched_vlm_baseline_root \
  --output-dir $trend_output_root
```

The loader imports only the six baseline rows and requires the complete judge
contract to match exactly, including Qwen revision, prompt source, media
settings, and evaluator protocol. It refuses partial checkpoint matrices,
nonzero-error cells, duplicate rows, and a mismatched external baseline. The
output directory receives the standalone PNG/SVG curve, audited score CSV, and
JSON summary.

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
`storage/eval_out/vbvr_pro_main_v2_evalkit_4cc7d028/reports/`: all-run aggregate/domain/category
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
