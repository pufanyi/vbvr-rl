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

The formal current evaluation for the Fujian manifest-RL checkpoint series
generates native 512x512x81 video, then resizes/pads every frame to the
1024x1024 scorer canvas while retaining all 81 frames at exact 16 FPS:

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
