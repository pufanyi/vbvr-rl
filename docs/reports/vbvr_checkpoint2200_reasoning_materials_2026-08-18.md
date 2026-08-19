# Checkpoint-2200 reasoning-chain materials (2026-08-18)

## Outcome

The folder package is under
`storage/presentations/vbvr_checkpoint2200_vs_baseline_reasoning_chains_20260818`.
It contains the previously selected 50 paired checkpoint-2200-versus-DiffSynth examples without a PPT or GIF.
Every case includes:

- the question first frame, prompt, complete metadata, semantic GT, GT video, and GT final frame;
- exact native and prepared/scored final MP4s for DiffSynth+UniPC and checkpoint-2200+CPS 0.7;
- all 30 exact sampler-trajectory MP4s per side, seven fixed milestones, the complete overview MP4, and contact sheet;
- the formal score, per-step visual-convergence metrics, exact trajectory manifest, and final-video SHA binding;
- a human-readable scorer audit plus its complete JSON evidence and evaluator internals.

The package has 5,181 files and 232,594,437 bytes. It contains 3,000 complete reasoning-step MP4s, 700 milestone
MP4 hardlinks/copies, 100 native finals, 100 formal prepared scorer inputs, 50 question images, and 50 GT videos.
`SHA256SUMS` covers every other package file.

## Scorer audit

`scripts/eval/vbvr_pro/audit_checkpoint2200_reasoning_cases.py` used the exact pinned e140 EvalKit checkout and
runtime that produced the formal result. For every selected sample it independently evaluated the baseline prepared
video, checkpoint prepared video, and GT video, retained `_last_task_details`, inspected the concrete registry class
and source line, and validated the metadata's evaluator key and required semantic fields.

All 150 scoring calls completed without errors. All 100 generated-video scores reproduced the published values with
absolute tolerance `1e-12`, all checkpoint scores remained above the paired baseline scores, and all 50 evaluator
mappings/metadata contracts passed. GT self-scores had minimum 0.988333, mean 0.999003, and maximum 1.0. The exact
EvalKit source fingerprint is `4cc7d028d4106a28190a63bc179562d5ac9add9263cb71926dd6385c5714bcf8`; the runtime
fingerprint is `49ea34669caaef54d82d86f93a21acf3c012fbb07994a4b0ab60f2c2d820cc2e`.

## Trajectory audit and meaning

The exact matched trajectory cells are `baseline-unipc` and `2200-cps0p7` under
`storage/eval_out/vbvr_pro_sampler_matrix_all_500_30step_trajectories`. Both passed
`src.cli.audit_vbvr_i2v_trajectories` read-only over all 500 samples. Every selected trajectory manifest binds
`step_29.mp4` (human-facing step 30), `final_00.mp4`, and the formal native final by SHA-256.

Packaging decoded every one of the 3,000 selected step videos and required 512x512 RGB, 81 frames, and 16 FPS. All
100 chains have 30 distinct step hashes and nonzero visual evolution before converging exactly to the formal final.
Across the 100 chains, sampled-video MAE to formal final fell from mean 0.007359 at step 1 to mean 0.002330 at step
15 and mean 0.001330 at step 29; step 30 is exactly zero by construction/binding. The per-transition distance is not
required to be monotonic because stochastic/multistep sampling can revise the predicted-clean endpoint.

Steps 1–29 are full-video post-CFG predicted-clean estimates at their source sigmas; step 30 is the actual final
latent at sigma zero. EvalKit scores only the formal final. These are useful sampler-state visualizations, not textual
chain-of-thought and not separately rewarded intermediate reasoning steps.

## Evidence boundary

The 50 examples are selected for the largest exact checkpoint-minus-baseline score gaps, so they are qualitative
presentation material rather than an unbiased performance estimate. The comparison also changes both weights and
sampler: checkpoint 2200 uses CPS 0.7 while the DiffSynth baseline uses UniPC. It therefore demonstrates the combined
generation configuration, not an RL-only causal effect; use the sampler-matched CPS checkpoint curve for that claim.
