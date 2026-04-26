# Training Systems Improvements

## 1. Consolidate SFT And RL Base Infrastructure

`BaseTrainer` and `BaseRLTrainer` duplicate distributed init, model build, FSDP mesh creation, dataset construction, optimizer construction, EMA setup, compile, checkpoint integration, and resume behavior.[^base-trainer][^base-rl]

Recommended work:

- Extract shared mixins for distributed setup, model loading, FSDP mesh creation, dataset construction, optimizer setup, and checkpoint state.
- Keep RL-specific reference policy and reward logic in `BaseGRPOTrainer`.
- Add parity tests that instantiate both base stacks with minimal fake modules and compare key derived fields.

Why it matters: any future fix to HSDP, expert parallel, dataloader epoch sizing, or checkpoint semantics currently has to be made twice.

## 2. Harden Expert Parallel Communication

Expert parallel uses direct `dist.send` / `dist.recv` between peer ranks for metrics, GRPO boundary latents, rewards, and advantages.[^cos-trainer][^grpo-trainer]

Recommended work:

- Wrap peer sends/receives in named helper functions with tensor shape checks.
- Add timeouts and clear rank/group diagnostics.
- Log the expert group, peer rank, and local schedule once at startup.
- Add a small distributed integration test with fake tensors to validate the high-to-low GRPO handshake.
- Prefer collective operations where the communication pattern is symmetric and can be expressed as group all-gather or broadcast.

## 3. Make FSDP Routing Invariants Explicit

FSDP requires ranks to execute compatible wrapped-module forward sequences. The code already uses synchronized timestep sampling in the model wrapper and broadcasts DanceGRPO replay timesteps.[^wan-wrapper][^dancegrpo-trainer]

Recommended work:

- Add runtime assertions in debug mode that all ranks sampled the same expert route histogram.
- Record high/low selected counts per step in W&B.
- Add tests for empty expert selections and dummy forward behavior in rewards.
- Document which functions are allowed to use rank-local randomness.

## 4. Improve Checkpoint Observability

The unified high/low checkpoint layout is strong, but observability can improve.[^checkpoint-runtime]

Recommended work:

- Write `checkpoint_manifest.json` at the checkpoint root with layout, step, epoch, model path, config, world size, expert-parallel mode, HSDP mesh, LoRA rank, EMA status, and source commit.
- Validate a checkpoint immediately after save by checking required files and DCP metadata existence.
- Add a CLI: `python -m src.cli.inspect_checkpoint <path>`.
- Add a smoke test for high-only, low-only, and high+low checkpoints with fake modules.

## 5. Reduce GRPO Memory Pressure

GRPO stores sampled trajectories on CPU and moves them back to GPU during replay.[^grpo-trainer] This is practical but can still become a bottleneck for long videos and large group sizes.

Recommended work:

- Store only replay-selected timesteps for DanceGRPO mode.
- Add activation/memory logging around sampling, reward, and replay phases.
- Stream replay chunks instead of collecting every chunk before policy update where algorithmically acceptable.
- Add optional latent compression/offload for CPU trajectory storage.
- Separate sampling and training devices only after the current single-pool path is well instrumented.

## 6. Optimize MazeReward Throughput

MazeReward decodes generated latents to pixels and computes RGB-distance ball localization.[^maze-reward] This can dominate GRPO step time.

Recommended work:

- Decode only selected frames if the VAE API can support temporal slicing safely.
- Cache or precompute condition-independent geometry tensors on device.
- Replace full-frame RGB argmin with a downsampled search followed by local refinement.
- Add per-reward timing and memory metrics.
- Consider a latent-space auxiliary reward trained to approximate the pixel reward.

## 7. Clarify Optimizer Checkpoint Semantics

Muon uses AdamW fallback for non-2D parameters, but the optimizer factory notes extra optimizers are stepped and not checkpointed.[^optimizer]

Recommended work:

- Expose this as a config warning when `optimizer: muon`.
- Optionally checkpoint fallback optimizer state.
- Add optimizer state coverage to checkpoint manifests.
- Add a resume test that confirms LR bases and fallback optimizers are rebuilt as expected.

## 8. Add Performance Regression Tracking

MFU monitoring exists for I2V, COS, correction, and GRPO, but it is mostly log-time instrumentation.[^flops]

Recommended work:

- Write structured performance logs with step time, MFU, data wait, forward/backward time, optimizer time, checkpoint time, and reward time.
- Track compile settings, FSDP/HSDP/EP mode, GPU type, sequence length, and batch size.
- Add a benchmark command that runs N warmup + M measured steps on fake or tiny data.

[^base-trainer]: [`src/trainer/base_trainer.py`](../../src/trainer/base_trainer.py)
[^base-rl]: [`src/trainer/base_rl_trainer.py`](../../src/trainer/base_rl_trainer.py)
[^cos-trainer]: [`src/trainer/cos_trainer.py`](../../src/trainer/cos_trainer.py)
[^grpo-trainer]: [`src/trainer/grpo_trainer.py`](../../src/trainer/grpo_trainer.py)
[^wan-wrapper]: [`src/models/wan_i2v.py`](../../src/models/wan_i2v.py)
[^dancegrpo-trainer]: [`src/trainer/dancegrpo_trainer.py`](../../src/trainer/dancegrpo_trainer.py)
[^checkpoint-runtime]: [`src/trainer/checkpoint_runtime.py`](../../src/trainer/checkpoint_runtime.py)
[^maze-reward]: [`src/trainer/rewards/maze.py`](../../src/trainer/rewards/maze.py)
[^optimizer]: [`src/trainer/optimizer.py`](../../src/trainer/optimizer.py)
[^flops]: [`src/trainer/flops.py`](../../src/trainer/flops.py)
