---
title: VBVR-Pro Output Explorer
emoji: 🧭
colorFrom: indigo
colorTo: green
sdk: static
app_file: index.html
pinned: false
short_description: Task-level and video-level VBVR-Pro training result explorer
tags:
  - video-generation
  - evaluation
  - wan
  - vbvr
---

# VBVR-Pro Output Explorer

Interactive task- and sample-level visualization of the strict In-Domain DanceGRPO
sweep for Wan2.2 TI2V-5B.

- 5 checkpoints × 4 sampling modes
- 100 EvalKit-supported tasks and 500 scored samples per run
- SFT epoch-1 baseline comparison at both task and sample level
- Every scored 1024×1024, 161-frame video, paired with its exact `main_v2` score

The static frontend and score indexes live in this Space. The 3.67 GiB media
archive is hosted in the companion public dataset
[`pufanyi/vbvrpro_output-data`](https://huggingface.co/datasets/pufanyi/vbvrpro_output-data)
and streamed through the Hub CDN.

The strict-sweep results use split-manifest SHA-256
`326f7bda3743e9c66dc0c29445661a5dda4ad0cee4cb8838c3fcfd0c4a149deb`
and VBVR EvalKit `main_v2` revision
`42a1593d8e493370c768be8e43646f0e0a9d8525`.

“Out-of-Domain” follows the original VBVR benchmark naming. Those task families were
present in this RL training dataset, so the label must not be interpreted as zero-shot
task generalization. The 500 scored benchmark samples are sample-disjoint from the
manifest training IDs.

This Space is generated from the authoritative score JSON files by
`scripts/eval/vbvr_pro/build_vbvrpro_space.py` in VBVR-RL.
