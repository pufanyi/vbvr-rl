# Training

Wan-Trainer implements four training modes on top of the same Wan2.2 I2V wrapper: SFT, COS, correction, and DanceGRPO-style replay.

## Shared Flow-Matching Base

The standard supervised objective samples a shifted sigma, constructs

```text
x_sigma = sigma * noise + (1 - sigma) * x_0
target  = noise - x_0
```

and minimizes weighted MSE between the model velocity prediction and `target`.[^wan-compute-loss] This follows the flow-matching idea of learning a vector field that transports noise to data.[^flow-matching]

Wan-Trainer uses a scheduler-matched shifted sigma schedule:

```text
sigma = shift * s / (1 + (shift - 1) * s)
```

where `shift` is read from the Diffusers scheduler config.[^wan-wrapper] Loss weights use a BSMNTW-style Gaussian curve centered around timestep 500, normalized across the 1000 training timesteps.[^wan-wrapper]

## SFT / I2V

`I2VTrainer` is the simplest path:

1. unpack raw media or precomputed latent tensors;
2. encode prompt/video/condition when raw data is used;
3. call `WanI2VForTraining.compute_loss`;
4. accumulate gradients;
5. clip, cosine-decay LR, step all optimizers, update EMA, log, checkpoint.[^i2v-trainer]

The same trainer supports full fine-tuning and LoRA because LoRA is installed inside the model wrapper before FSDP sharding.[^wan-wrapper] `prompt_dropout` zeroes entire prompt embeddings for unconditional CFG-branch training.[^wan-wrapper]

Typical config:

```yaml
trainer: i2v
model_path: storage/models/Wan2.2-I2V-A14B-Diffusers
latent_webdataset_dir: data/vbvr/latents/sft
dataset_size: 800000
output_dir: storage/checkpoints/sft_vbvr
batch_size: 12
learning_rate: 1.0e-5
ema_decay: 0.999
prompt_dropout: 0.1
expert_parallel: true
train_experts: both
```

## COS: Chain-of-Step

COS changes the flow path. Instead of learning a direct noise-to-final path, it learns a chain:

```text
noise -> waypoint_0 -> waypoint_1 -> ... -> final_video
```

The chain boundaries are `cos_tau_sigma`, and the number of tau values must equal `len(video_latents) - 1`.[^cos-path][^cos-trainer] Raw COS datasets provide `videos` as an ordered list; latent COS datasets must store `latents_0`, `latents_1`, ... and are produced with `--encode_all_videos`.[^i2v-latent-precompute]

Supported path families:

- `linear`: N-step piecewise linear passthrough. It passes through each waypoint at its tau boundary but is only C0 at boundaries.
- `target_cosine`: N-step smooth target blend. It does not pass exactly through intermediate waypoints, but it makes the effective target C1 across the chain.
- Legacy 2-step-only paths: `cosine`, `cubic_hermite`, `smooth_blend`, `quadratic_bezier`, and `target_linear`.[^cos-path]

`compute_cos_loss` runs one dedicated pass per available expert. That guarantees each expert receives a training signal every step when both experts are loaded.[^wan-cos-loss] In expert-parallel mode, each half of the world loads only its own expert while rank-0 logging receives the low-expert metrics by point-to-point communication.[^cos-trainer]

Typical config:

```yaml
trainer: cos
latent_webdataset_dir: data/maze_cos/latents/webdataset
dataset_size: 300000
cos_tau_sigma: 0.8
cos_boundary_noise_std: 0.02
cos_path_type: target_cosine
expert_parallel: true
train_experts: both
```

## On-Policy Correction

`I2VCorrectionTrainer` is still supervised MSE, not policy gradient. It adds a correction branch:

```text
noise  = epsilon
x_hat  = short EMA-teacher rollout(epsilon, condition)
x_sigma = sigma * epsilon + (1 - sigma) * x_hat
target  = (x_sigma - x_GT) / sigma
```

When `x_hat == x_GT`, this reduces to the normal flow-matching target. When the rollout drifts, it teaches the student a velocity that redirects the path toward the real ground truth.[^wan-correction-loss][^correction-trainer]

Correction is expensive because each correction micro-step adds `K` no-grad rollout forwards plus a student forward/backward. The trainer supports `correction_every_n_steps` to amortize that cost by firing the correction branch only every N micro-steps and scaling the correction weight accordingly.[^correction-trainer]

Current correction code explicitly forbids expert parallel because the teacher rollout advances the entire batch through one sigma at a time, leaving the remote expert group idle in an EP split.[^correction-trainer]

Operational warning: if `ema_decay <= 0`, the code logs that the teacher will use live student weights. That defeats much of the intended teacher-stability effect.[^correction-trainer]

## RL Objective Background

The RL stack adapts policy-gradient optimization to flow-matching video generation. The code keeps the Flow-GRPO idea of converting deterministic flow ODE sampling into a stochastic SDE transition so rollout steps have tractable log probabilities, but the supported trainer path is DanceGRPO.[^flowgrpo][^wan-sde]

DanceGRPO does:

1. encode one batch;
2. sample `G` videos per prompt in chunks of `grpo_sample_batch_size`;
3. compute reward for final generated latents;
4. compute group-relative z-score advantages;
5. replay selected denoising steps, recompute log probabilities, apply PPO-style clipped policy loss, and optionally add a KL penalty to a reference policy.[^dancegrpo-trainer][^ppo][^grpo]

For LoRA runs, the reference policy is the base model with adapters disabled. For full fine-tuning, frozen reference transformer copies are created before FSDP sharding.[^base-grpo]

## DanceGRPO-Style Replay

`DanceGRPOTrainer` keeps the standard single-group execution path for normal launches and can also split
rollout/reward actors from the smaller training FSDP group for multi-node launches. It adopts two ideas from DanceGRPO:

- all samples in a prompt group can share the same initial noise;
- policy replay can use only a selected subset of denoising timesteps.[^dancegrpo][^dancegrpo-trainer]

The timestep subset is generated consistently across ranks so all FSDP ranks call the same expert modules in the same order. This is necessary because per-rank divergence in expert routing can deadlock FSDP collectives.[^dancegrpo-trainer]

DanceGRPO currently rejects expert parallel. For split RL, `rl_train_node_count: 1` means node 0 trains and the remaining nodes run rollout/reward actors; rollout actors partition `grpo_group_size` across cards.[^dancegrpo-trainer]

## Reward Functions

Rewards are registered through `src.trainer.rewards.registry` and built by name from `RLConfig.grpo_reward_fn`.[^reward-registry]

Current rewards:

- `neg_loss`: a model-internal negative flow-matching loss against GT. It supports expert filtering and runs dummy FSDP forwards when no sample routes to an expert.[^neg-loss]
- `maze`: VAE-decodes generated latents, detects the ball by RGB distance, and combines trajectory, on-path, and goal rewards. It requires `maze_*` metadata from the latent dataset and forces the VAE to load even in precomputed-latent training.[^maze-reward]

## Practical Launch Notes

Use `src.cli.train_i2v` for both SFT and COS when your YAML has `trainer: i2v` or `trainer: cos`:

```fish
fish scripts/train/i2v.fish --nproc 8 -- --config configs/train_sft_vbvr.yaml
fish scripts/train/i2v.fish --nproc 8 -- --config configs/train_cos_maze_cos_path_all_bfs_w_color_latent.yaml
```

Use the GRPO launcher for DanceGRPO:

```fish
fish scripts/train/grpo.fish --nproc 8 --config configs/train_grpo_maze.yaml
fish scripts/train/grpo.fish --nproc 8 --config configs/train_dancegrpo_maze.yaml
fish scripts/train/dancegrpo_maze_split_multinode.fish --nproc 8
```

Use the correction launcher for `CorrectionConfig`:

```fish
fish scripts/train/i2v_correction.fish --nproc 8 -- --config configs/train_correction_vbvr.yaml
```

## Key Failure Modes

- `latent_webdataset_dir` without `dataset_size` makes total steps ambiguous and can break scheduling assumptions.
- COS latent training without `latents_0`, `latents_1`, ... raises an explicit error; regenerate with `--encode_all_videos`.
- Expert-parallel GRPO needs a sampling schedule that crosses the high/low boundary exactly once.
- FSDP trainers require all ranks to call the same wrapped module sequence; any per-rank routing divergence can hang.
- MazeReward is VAE-bound and can become the throughput bottleneck in GRPO.

[^wan-compute-loss]: [`src/models/wan_i2v.py`](../src/models/wan_i2v.py)
[^flow-matching]: Lipman et al., "Flow Matching for Generative Modeling", arXiv:2210.02747, https://arxiv.org/abs/2210.02747
[^wan-wrapper]: [`src/models/wan_i2v.py`](../src/models/wan_i2v.py)
[^i2v-trainer]: [`src/trainer/i2v_trainer.py`](../src/trainer/i2v_trainer.py)
[^cos-path]: [`src/models/cos_path.py`](../src/models/cos_path.py)
[^cos-trainer]: [`src/trainer/cos_trainer.py`](../src/trainer/cos_trainer.py)
[^i2v-latent-precompute]: [`src/precompute/i2v_latent_webdataset.py`](../src/precompute/i2v_latent_webdataset.py)
[^wan-cos-loss]: [`src/models/wan_i2v.py`](../src/models/wan_i2v.py)
[^wan-correction-loss]: [`src/models/wan_i2v.py`](../src/models/wan_i2v.py)
[^correction-trainer]: [`src/trainer/i2v_correction_trainer.py`](../src/trainer/i2v_correction_trainer.py)
[^flowgrpo]: Liu et al., "Flow-GRPO: Training Flow Matching Models via Online RL", arXiv:2505.05470, https://arxiv.org/abs/2505.05470
[^wan-sde]: [`src/models/wan_i2v.py`](../src/models/wan_i2v.py)
[^ppo]: Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347, https://arxiv.org/abs/1707.06347
[^grpo]: Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", arXiv:2402.03300, https://arxiv.org/abs/2402.03300
[^base-grpo]: [`src/trainer/base_grpo_trainer.py`](../src/trainer/base_grpo_trainer.py)
[^dancegrpo]: DanceGRPO authors, "DanceGRPO: Unleashing GRPO on Visual Generation", arXiv:2505.07818, https://arxiv.org/abs/2505.07818
[^dancegrpo-trainer]: [`src/trainer/dancegrpo_trainer.py`](../src/trainer/dancegrpo_trainer.py)
[^reward-registry]: [`src/trainer/rewards/registry.py`](../src/trainer/rewards/registry.py)
[^neg-loss]: [`src/trainer/rewards/neg_loss.py`](../src/trainer/rewards/neg_loss.py)
[^maze-reward]: [`src/trainer/rewards/maze.py`](../src/trainer/rewards/maze.py)
