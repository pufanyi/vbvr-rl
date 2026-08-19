# Algorithmic Improvements

## 1. Validate The Sigma Schedule And Expert Boundary

The code correctly reads `boundary_ratio` and `flow_shift`, then computes `boundary_idx` from shifted timesteps.[^wan-wrapper] This should become a tested invariant because every trainer relies on the same high/low split.

Recommended work:

- Add unit tests that assert the shifted schedule is monotonic, begins near 1, ends near 0, and yields a single high-prefix/low-suffix expert routing.
- Add tests for configs where `boundary_ratio` is not 0.9 or 0.875.
- Log both `boundary_timestep` and `boundary_idx` to W&B for every run.
- Store boundary metadata in checkpoints and latent dataset metadata.

Why it matters: Wan2.2's model card describes expert specialization by denoising stage.[^wan22] If schedule or boundary drift occurs, the wrong expert receives gradients and RL replay can deadlock under FSDP.

## 2. Turn COS Path Choice Into An Ablation Matrix

COS currently supports N-step `linear` and `target_cosine`, plus legacy 2-step paths.[^cos-path] The code is flexible, but there is no documented experiment matrix that connects path type, tau placement, target norm, and downstream video quality.

Recommended work:

- Build a small standard COS benchmark with fixed seeds and equal compute budgets.
- Compare `linear` vs `target_cosine` for 1-step, 2-step, and 3-step chains.
- Track target velocity norm by expert and by COS stage, not only averaged debug values.
- Add boundary continuity tests with finite differences for each path type.
- Explore learned or curriculum tau schedules instead of static `cos_tau_sigma`.

Why it matters: Flow Matching depends on fitting a vector field along a chosen probability path.[^fm] COS changes that path; the path should be treated as an algorithmic object with measurable continuity, norm, and sample-quality consequences.

## 3. Make Correction Training Actually Teacher-Stable

Correction is designed around an EMA teacher rollout, but current correction configs set `ema_decay: 0`, and the trainer warns that it will use live student weights.[^correction-trainer][^correction-configs]

Recommended work:

- Enable EMA by default for correction configs, e.g. `ema_decay: 0.999` or higher.
- Add a config validator that warns or fails when `correction_weight > 0` and `ema_decay <= 0`.
- Log teacher rollout quality proxies: endpoint MSE to GT, endpoint norm, and correction loss firing ratio.
- Compare deterministic ODE teacher vs SDE teacher under the same seed and batch.
- Cache short teacher rollouts inside a micro-step only when it does not change gradient semantics.

Why it matters: the correction target is useful only if `x_hat` is a meaningful off-policy or slowly moving teacher endpoint. Live-student targets make the target distribution shift every optimizer step.

## 4. Improve RL Reward Robustness

`neg_loss` is easy to run but may reward generated latents for being easy to denoise rather than for task success.[^neg-loss] `maze` is closer to task semantics but relies on VAE decoding and RGB ball detection.[^maze-reward]

Recommended work:

- Split rewards into latent, pixel, and semantic categories with explicit `requires_vae` and expected metadata contracts.
- Add reward normalization diagnostics before GRPO z-scoring: raw reward mean/std, rank correlations between components, and per-task histograms.
- For MazeReward, add color robustness checks and use connected-component localization instead of raw RGB argmin.
- Add a reward-hacking monitor: generated-video diversity, VAE reconstruction sanity, and negative prompt artifacts.
- Create a reward registry test that instantiates every reward and checks shape, dtype, and device behavior on synthetic tensors.

Why it matters: Flow-GRPO explicitly relies on online reward signals for flow-matching models.[^flowgrpo] Poor reward design can improve scalar reward while degrading visual quality or task validity.

## 5. Make GRPO Step Selection More Principled

DanceGRPO-style replay currently shares group initial noise and subsamples replay timesteps.[^dancegrpo-trainer] This is useful, but the selected timesteps are random and not linked to reward sensitivity or expert boundary coverage.

Recommended work:

- Compare uniform random timestep replay with stratified replay across high/low experts.
- Force at least one replay step near the high/low boundary.
- Prioritize timesteps with high policy/reference mean divergence or high reward sensitivity.
- Log selected timestep indices and high/low counts.
- Add a KL budget scheduler rather than a constant `grpo_kl_coeff`.

Why it matters: PPO-style clipped updates depend on reliable importance ratios.[^ppo] In flow models, timestep selection determines which parts of the generation trajectory receive policy gradients.

## 6. Add Reference-Policy Controls For Full Fine-Tuning

Full fine-tuning creates deep-copied reference transformers before FSDP sharding.[^base-grpo] That is correct but memory-heavy.

Recommended work:

- Add an option to keep reference policy on CPU or offload it between replay steps.
- Add a lower-memory reference mode using checkpoint snapshots or frozen LoRA-disabled base modules where possible.
- Track policy/reference KL by expert and by timestep.
- Add tests that LoRA reference mode truly disables and re-enables adapters after exceptions.

## 7. Upgrade Evaluation-Driven Training Signals

The VLM evaluator can provide useful qualitative scoring, but it is currently a post-hoc evaluator, not a calibrated reward model.[^vlm-judge]

Recommended work:

- Build a small human-labeled validation split and report VLM/rule/human agreement.
- Use pairwise preferences in addition to absolute 0-10 scores.
- Keep evaluation prompts versioned and immutable across model comparisons.
- Add confidence/uncertainty by judging the same sample with frame order checks or multiple sampled frame subsets.

[^wan-wrapper]: [`src/models/wan_i2v.py`](../../src/models/wan_i2v.py)
[^wan22]: Wan-AI, "Wan2.2-I2V-A14B-Diffusers", Hugging Face model card, https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers
[^cos-path]: [`src/models/cos_path.py`](../../src/models/cos_path.py)
[^fm]: Lipman et al., "Flow Matching for Generative Modeling", arXiv:2210.02747, https://arxiv.org/abs/2210.02747
[^correction-trainer]: [`src/trainer/i2v_correction_trainer.py`](../../src/trainer/i2v_correction_trainer.py)
[^correction-configs]: [`configs/train_correction_vbvr.yaml`](../../configs/train_correction_vbvr.yaml), [`configs/train_correction_maze.yaml`](../../configs/train_correction_maze.yaml)
[^neg-loss]: [`src/trainer/rewards/neg_loss.py`](../../src/trainer/rewards/neg_loss.py)
[^maze-reward]: [`src/trainer/rewards/maze.py`](../../src/trainer/rewards/maze.py)
[^flowgrpo]: Liu et al., "Flow-GRPO: Training Flow Matching Models via Online RL", arXiv:2505.05470, https://arxiv.org/abs/2505.05470
[^dancegrpo-trainer]: [`src/trainer/dancegrpo_trainer.py`](../../src/trainer/dancegrpo_trainer.py)
[^ppo]: Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347, https://arxiv.org/abs/1707.06347
[^base-grpo]: [`src/trainer/base_grpo_trainer.py`](../../src/trainer/base_grpo_trainer.py)
[^vlm-judge]: [`src/trainer/rewards/vbvr_vlm.py`](../../src/trainer/rewards/vbvr_vlm.py)
