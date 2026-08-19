# Changelog

All notable public changes to VBVR-RL are documented here. The project uses
[Semantic Versioning](https://semver.org/) for source releases.

## [0.1.0] - 2026-08-19

Initial public source release.

### Included

- Wan2.2 TI2V-5B and I2V-A14B training with SFT, COS, correction, and
  DanceGRPO-style reinforcement learning.
- Flow-CPS rollout and replay, LoRA/full fine-tuning, FSDP2/HSDP, expert
  parallelism, and RL tensor parallelism.
- Raw Parquet and latent WebDataset pipelines, plus a resumable materializer
  for the public 50,000-sample VBVR-Pro RL snapshot.
- Manifest-locked VBVR-Pro generation, media preparation, external rule
  scoring, VLM judging, and stage provenance.
- DCP checkpoint save/resume, Diffusers conversion, LoRA extraction, tests,
  launchers, and public operator documentation.

### Release boundaries

- VBVR-EvalKit is not vendored. Rule reward and evaluation require an explicit
  external checkout and exact source fingerprint.
- Model weights, datasets, OCR weights, generated media, checkpoints, and
  machine-specific infrastructure are not included.
- The old non-VBVR-Pro evaluation implementation and compatibility wrappers
  have been removed; `scripts/eval/vbvr_pro/` is the supported evaluation
  surface.

[0.1.0]: https://github.com/pufanyi/vbvr-rl/releases/tag/v0.1.0
