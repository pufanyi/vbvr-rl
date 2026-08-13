---
pretty_name: VBVR-Pro Baseline and DanceGRPO-2200 Sampler Trajectories
language:
  - en
tags:
  - video-generation
  - evaluation
  - wan
  - vbvr
size_categories:
  - 1K<n<10K
---

# VBVR-Pro sampler trajectory media

This media archive backs the interactive
[`pufanyi/vbvrpro_sampler_trajectories`](https://huggingface.co/spaces/pufanyi/vbvrpro_sampler_trajectories)
Space.

It contains 12 matched evaluation cells:

- DiffSynth step-35500 baseline and DanceGRPO checkpoint 2200
- Flow-CPS noise 0.1, 0.3, 0.7, and 0.9
- deterministic FlowMatch Euler ODE and UniPC ODE
- 500 samples per cell across 100 VBVR-Pro tasks

The deployment is split across three public media repositories so each Git-backed
Dataset remains below Hugging Face's recommended 100,000-file threshold:

- [`pufanyi/vbvrpro_sampler_trajectories-data`](https://huggingface.co/datasets/pufanyi/vbvrpro_sampler_trajectories-data):
  optional 6×5 overview grids and native final outputs
- [`pufanyi/vbvrpro_sampler_trajectories-baseline-steps`](https://huggingface.co/datasets/pufanyi/vbvrpro_sampler_trajectories-baseline-steps):
  90,000 original baseline step videos
- [`pufanyi/vbvrpro_sampler_trajectories-2200-steps`](https://huggingface.co/datasets/pufanyi/vbvrpro_sampler_trajectories-2200-steps):
  90,000 original checkpoint-2200 step videos

For each of the 6,000 model/sampler/sample combinations, the complete deployment
stores all 30 native `step_00.mp4` through `step_29.mp4` files without spatial
repacking, plus:

- `steps_grid.mp4`: a compressed synchronized 6×5 overview
- `final_00.mp4`: the native 512×512, 81-frame final output at sigma zero

Paths follow:

```text
videos/{cell_id}/{domain_folder}/{task_name}/{sample_id}/{filename.mp4}
```

The companion `data/index.json` records sample identity, exact prompts, seeds,
sampler schedules, trajectory semantics, and 6,000 aligned final-only EvalKit
scores on a 0–1 scale. A displayed score evaluates public step 30 /
`final_00.mp4`, not the intermediate clean-endpoint previews. No training data
or model weights are included.
