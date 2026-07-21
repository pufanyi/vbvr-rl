---
pretty_name: VBVR-Pro Strict In-Domain Sweep Scored Videos
language:
  - en
tags:
  - video-generation
  - evaluation
  - wan
  - vbvr
size_categories:
  - 10K<n<100K
---

# VBVR-Pro strict-sweep scored videos

This media archive backs the interactive
[`pufanyi/vbvrpro_output`](https://huggingface.co/spaces/pufanyi/vbvrpro_output)
Space.

It contains 10,500 scorer-input MP4 files:

- 20 DanceGRPO results: 5 checkpoints × 4 sampling modes × 500 samples
- 1 SFT epoch-1 baseline: 500 samples
- 100 EvalKit-supported tasks, with 5 samples per task and run
- 1024×1024, 161-frame H.264 videos encoded at 33 FPS

Paths follow:

```text
videos/{run_id}/{domain_folder}/{task_name}/{video_idx}.mp4
```

The Space contains the task catalog, exact per-video scores, baseline deltas,
sample-specific prompts, run summaries, and copies of the authoritative scorer
JSON files.

Evaluation provenance:

- VBVR split-manifest SHA-256:
  `326f7bda3743e9c66dc0c29445661a5dda4ad0cee4cb8838c3fcfd0c4a149deb`
- VBVR EvalKit `main_v2` revision:
  `42a1593d8e493370c768be8e43646f0e0a9d8525`

“Out-of-Domain” follows the original VBVR benchmark naming. Those task
families were present in this RL training dataset, so the label must not be
interpreted as zero-shot task generalization. The 500 scored benchmark samples
are sample-disjoint from the manifest training IDs.
