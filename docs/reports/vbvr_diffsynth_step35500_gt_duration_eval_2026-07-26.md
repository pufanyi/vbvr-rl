# DiffSynth Step-35500 GT-Duration VBVR-Pro Evaluation

Date: 2026-07-26

## Question

Evaluate
`storage/models/diffsynth_converted_5b/wan2.2-TI2V-5B_260715_vbvr_pro_step-35500`
at exact 16 FPS while deriving each generated video's length from its GT
duration, then resize every generated frame to the GT spatial size.

## Protocol

- Eval set: 100 EvalKit-supported tasks x 5 bench samples = 500 videos.
- Scorer: `main_v2` revision `6fedd9d9edb8daafa56aca8e53885aa8ad6f6037`;
  contract SHA-256
  `eb977da60e95456734063ba018b14d805680179fdf0e3e3b2ba6f603f27a935c`.
- Generation: native 512x512, 50-step UniPC ODE, CFG 5.0, seed 0, 16 FPS.
- Per-item requested length:
  `round(GT decoded frames * 16 / GT FPS)`.
- Wan-compatible realized length:
  `floor(requested length / 4) * 4 + 1`, matching the Diffusers pipeline.
- Preparation: preserve every generated frame and encode at exact 16 FPS;
  direct 512x512 to 1024x1024 resize without crop or padding.

All 500 GT files are 1024x1024 at exact 16 FPS. GT lengths range from 10 to
282 frames (mean 57.688); aligned generated lengths range from 9 to 281 frames
(mean 57.808). Alignment changes a same-FPS GT length by at most two frames.

The runnable wrapper is
`scripts/eval/vbvr_pro/vbvr_pro_5b_diffsynth_step35500_unipc_50steps_512x512_gt_duration_16fps_main_v2.fish`.

## Results

| Evaluation | Overall | In-Domain | Out-of-Domain |
| --- | ---: | ---: | ---: |
| 256x256, fixed 81 frames | 0.406543 | 0.475460 | 0.337626 |
| 384x384, fixed 81 frames | 0.514761 | 0.608461 | 0.421061 |
| 512x512, fixed 81 frames | 0.548651 | 0.665233 | 0.432068 |
| 512x512, GT-duration matched | 0.547386 | 0.657960 | 0.436812 |
| Dynamic minus fixed-81 native-512 | -0.001265 | -0.007274 | +0.004744 |

The dynamic and fixed-81 native-512 evaluations hold model, prompt, sampler,
steps, CFG, seed, FPS, native resolution, scorer preparation, and scorer
revision fixed. Only the generated length changes.

Across 100 paired task means, dynamic length improved 50 tasks, tied 12, and
regressed 38. A 100,000-resample paired-task bootstrap with seed 0 gave an
Overall delta 95% interval of `[-0.024533, +0.022437]`.

Length-stratified sample results explain the near-zero aggregate change:

| Dynamic length | Samples | Dynamic score | Fixed-81 score | Delta |
| --- | ---: | ---: | ---: | ---: |
| `<81` | 388 | 0.566032 | 0.558161 | +0.007871 |
| `=81` | 39 | 0.543205 | 0.543205 | 0.000000 |
| `>81` | 73 | 0.450518 | 0.501014 | -0.050497 |

For the 39 samples whose aligned length is exactly 81, both generated and
prepared MP4 files were byte-identical between the two runs, and all scores
matched exactly. This confirms that the comparison is a controlled temporal
ablation.

## Intermediate-Resolution Decision

A separate fixed-81 sweep held model, prompt, sampler, 50 steps, CFG 5.0,
seed 0, 16 FPS, scorer preparation, and EvalKit revision fixed while changing
only native generation resolution. Native 384x384 improved over 256x256 by
`+0.108218` Overall, `+0.133001` In-Domain, and `+0.083435` Out-of-Domain.
It retained 76.15%, 70.08%, and 88.35% of the corresponding 256-to-512 gains.
The 384 latent spatial grid has 56.25% as many positions as 512.

Against 512x512, the 384x384 deltas were `-0.033890` Overall, `-0.056772`
In-Domain, and `-0.011007` Out-of-Domain. Across 100 paired task means, 384
improved 32 tasks, tied 6, and regressed 62. The 100,000-resample Overall
task-bootstrap interval was `[-0.058508, -0.009946]`; the In-Domain interval
was `[-0.092168, -0.024703]`, while the Out-of-Domain interval
`[-0.043842, +0.023400]` crossed zero. Thus 384 is not score-equivalent to 512,
but is a practical compute/quality compromise for RL.

The 384 evaluation's 500 generated files were exactly 384x384x81 and its 500
prepared files exactly 1024x1024x81; all were exact 16 FPS/5.0625 seconds.
Path sets, 500 finite error-free scores across 100 tasks, and all three complete
provenance manifests passed independent audit. Its runnable wrapper is
`scripts/eval/vbvr_pro/vbvr_pro_5b_diffsynth_step35500_unipc_50steps_384x384_81f_16fps_main_v2.fish`.

The resulting 384x384x81 RL config is
`configs/train_dancegrpo_vbvr_pro_5b_384x384x81_rule_cps_from_nsft_bs_32_lr_1e-6_manifest_rl.yaml`.
An eight-H100, one-step full-FT smoke with local batch/prompt-wave 4 completed
the real Flow-CPS rollout, latest EvalKit reward, delayed-replay boundary
flush, 17-timestep replay, gradient clipping, and AdamW update in 220.09
seconds. It reported reward `0.5524 +/- 0.4138`, grad norm `0.0001`, and
48.7/53.3 GiB allocated/reserved peak memory. This validates single-node FSDP
memory but not production multi-node HSDP.

## Diagnosis

GT-duration matching does not explain or repair the earlier low aggregate
score. The largest confirmed protocol mismatch was spatial: changing only
native generation from 256x256 to the checkpoint's 512x512 training resolution
improved Overall by `+0.142108`.

The checkpoint training metadata records 512x512x201 at exact 16 FPS, LoRA
rank/alpha 32/32, and learning rate `1e-4`. The conversion metadata records
scale 1.0 and 300 merged target modules, and prior direct merged-weight checks
matched `B @ A` at approximately 1.0, so there is no current evidence for a
missing or duplicated LoRA multiplier.

The SFT dataset descriptor contains 250 tasks. It overlaps the current
EvalKit-supported set on all 50 In-Domain tasks and zero of the 50
Out-of-Domain tasks. The native-512 dynamic result's 0.657960 versus 0.436812
domain split is therefore consistent with task-distribution shift. For this
checkpoint, fixed 81 frames remains marginally better Overall, while dynamic
GT length is useful as a diagnostic rather than a new default.

## Audit

- 500 generated paths and 500 prepared paths exactly matched the eval manifest.
- Independent `ffprobe` checks covered all 1,000 MP4s.
- Generated files were exactly 512x512; prepared files were exactly 1024x1024.
- Every generated/prepared pair retained the manifest's 9--281 aligned frame
  count, exact 16 FPS, and duration `frames / 16`.
- All 500 scores were finite and error-free across exactly 100 tasks.
- Generation, preparation, and score provenance manifests were complete and
  passed independent artifact-fingerprint recomputation.

Artifacts are under
`storage/eval_out/vbvr_pro_main_v2_evalkit_eb977da6/diffsynth_step35500-unipc-50steps-cfg5-512x512-gt-duration-fps16/`.
