---
title: VBVR-Pro Sampler Trajectories
emoji: 🔬
colorFrom: green
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
short_description: Baseline vs DanceGRPO-2200 diffusion trajectories
tags:
  - video-generation
  - evaluation
  - wan
  - vbvr
---

# VBVR-Pro sampler trajectory compare

Interactive left/right comparison of native 512×512×81 VBVR-Pro generation
trajectories for Wan2.2 TI2V-5B.

- DiffSynth step-35500 baseline and DanceGRPO checkpoint 2200
- Flow-CPS noise 0.1, 0.3, 0.7, and 0.9
- deterministic FlowMatch Euler ODE and UniPC ODE
- 500 matched samples, 100 tasks, and 30 inference steps per trajectory
- every native `step_00.mp4` through `step_29.mp4`, selected one step at a time
- a complete 2×6 matrix view for every sample
- per-test-case final EvalKit scores on a 0–1 scale, plus each cell's 500-sample mean

The default view streams the original 512×512 MP4 for the selected inference
step. Steps 1–29 show the post-CFG predicted-clean endpoint at the displayed
source sigma; step 30 is the actual final latent decoded at sigma zero. A
compressed synchronized 6×5 grid remains available only as an optional overview,
and the dedicated final tab shows the sigma-zero result at native resolution.
The score shown while browsing a trajectory applies only to that case's formal
final output (public step 30 / `final_00.mp4`); intermediate clean-endpoint
previews are not scored separately. The compact index binds all 6,000 values to
the formal result JSONs and the pinned EvalKit `e140038f` / source hash
`4cc7d028` scorer contract.

The static frontend and compact sample index live in this Space. Overview and
final media are hosted by
[`pufanyi/vbvrpro_sampler_trajectories-data`](https://huggingface.co/datasets/pufanyi/vbvrpro_sampler_trajectories-data)
while 180,000 native step videos are split between the public
[`baseline`](https://huggingface.co/datasets/pufanyi/vbvrpro_sampler_trajectories-baseline-steps)
and [`checkpoint-2200`](https://huggingface.co/datasets/pufanyi/vbvrpro_sampler_trajectories-2200-steps)
step archives. All media is streamed on demand through the Hub CDN.

This Space is generated from the strict local trajectory archive by
`scripts/eval/vbvr_pro/build_vbvr_trajectory_space.py` in Wan-Trainer.
