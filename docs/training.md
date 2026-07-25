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
- `target_sigmoid`: N-step smooth target blend using normalized sigmoid easing. Like `target_cosine`, it does not pass exactly through intermediate waypoints, but it keeps exact 0/0.5/1 blend anchors around each tau and is C1 across the chain.
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
# For sigmoid easing:
# cos_path_type: target_sigmoid
# cos_sigmoid_steepness: 10.0
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

### VBVR-Pro `main_v2` Reward Contract

`vbvr_rule` uses the same observable contract as final VBVR-Pro reporting,
rather than calling a task evaluator directly on the native 256x256 training
video. Pin both the EvalKit path and its full source-tree fingerprint, and keep
the final-eval preparation parameters explicit:

```yaml
grpo_reward_fn: vbvr_rule
vbvr_reward_evalkit_dir: storage/evalkits/vbvr-evalkit-interleave-main_v2-6fedd9d9
vbvr_reward_evalkit_source_sha256: eb977da60e95456734063ba018b14d805680179fdf0e3e3b2ba6f603f27a935c
vbvr_reward_device: cpu
vbvr_reward_decode_batch_size: 8
vbvr_reward_max_pending_jobs: 8
vbvr_reward_cpu_workers: 2
vbvr_reward_cpu_threads_per_worker: 8
vbvr_reward_prepared_width: 1024
vbvr_reward_prepared_height: 1024
vbvr_reward_max_duration_seconds: 5.0
vbvr_reward_prepare_crf: 12
vbvr_reward_use_process_pool: true
vbvr_reward_fail_on_error: true
```

The pinned digest covers `run_evaluation.py`, all Python under `vbvr_bench`,
bundled annotation files, and `requirements.txt`. Both the path and digest are
mandatory for `vbvr_rule`; an implicit fallback scorer is rejected.

The reward shares inference's canonical decoded-tensor-to-uint8 conversion,
writes those frames through Diffusers `export_to_video`, matching the CPS/ODE
generation entrypoints, then calls the shared evaluator preparation function.
All 161 frames are retained; only the canvas and playback rate change, so a
161-frame sample is scored at 1024x1024 and 33 FPS. It forwards the sample
prompt, GT directory, and metadata file and invokes the exact
`main_v2 run_evaluation.evaluate_single_video` entrypoint.[^vbvr-rule][^vbvr-prepare][^vbvr-eval-wrapper]

Scoring happens in spawned CPU workers with CUDA hidden. This is required for
EasyOCR tasks and also prevents evaluator imports from inheriting training-GPU
state. `vbvr_reward_cpu_workers` counts processes per reward-producing rank;
keep the node-wide native-thread budget bounded when increasing it. The 5B
manifest-RL configs use two processes with eight threads each per rank after
that layout outperformed one process with 16 threads on a 50-task benchmark;
other standard configs may retain one x 16 per rank (or per TP pair). EvalKit
errors, non-finite scores, and scores outside `[0, 1]` stop training by default.

`vbvr_rule` is a bounded producer-consumer pipeline. The training thread
decodes each `vbvr_reward_decode_batch_size` chunk and immediately queues its
samples for background video preparation and CPU scoring. DanceGRPO then
continues the next GPU rollout chunk and resolves queued rewards, in submission
order, only after all local rollout chunks have been produced. On TP runs only
TP rank 0 produces rewards and the deferred resolve retains the existing TP
broadcast order. `vbvr_reward_max_pending_jobs` bounds decoded samples retained
per reward-producing rank; `0` selects
`max(vbvr_reward_decode_batch_size, 2 * vbvr_reward_cpu_workers)`. The optimized
DiffSynth manifest-RL configs use `16`, two decoded size-8 batches, so both
prompt waves can remain in flight. At 256x256x161 this bound retains about
483 MiB of decoded RGB per rank. Reward videos are disposable and
metadata-heavy; put `vbvr_reward_tmp_dir` under node-local `/tmp`, not the
shared QuarkFS project mount.

On 2026-07-25, an eight-H100 production-shape one-step smoke of the 161-frame
DiffSynth step-35500 config used shared prompts with local `batch_size: 4`,
`G=32`, `T=30`, two size-8 rollout/replay chunks, latest EvalKit, nonzero
learning rate, gradient clipping, and AdamW. The two-worker x eight-thread
pipeline completed in `204.18 s/step`, versus `251.32 s/step` for the prior
one-worker x 16-thread serial-reward smoke: `47.14` seconds (`18.8%`) less wall
time. Reward (`0.3141 +/- 0.2649`), grad norm (`0.0002`), and peak memory
(`46.8/50.8 GiB` allocated/reserved) matched exactly. The corresponding
81-frame config also completed a real step in `137.13 s`, with reward
`0.3771 +/- 0.3251`, grad norm `0.0002`, and `40.6/45.0 GiB` peak memory.

After changing video encoding, preparation, metadata, or scorer code, run
`scripts/dev/validate_vbvr_reward_alignment.py` on both a normal geometric
task and an EasyOCR task. The validator independently builds the training and
final-generation video paths from the same RGB frames, requires raw and prepared
decoded frames to be identical, and requires the isolated scorer results to
match exactly.[^vbvr-alignment-validator]

To keep the same 5B lr=1e-6 DanceGRPO setting while training on the dedicated
VBVR-Pro RL split with the latest scorer, use
`configs/train_dancegrpo_vbvr_pro_5b_256x256x161_rule_cps_from_nsft_bs_32_lr_1e-6_manifest_rl_evalkit_6fedd9d9.yaml`.
Its data descriptor selects `split: rl` from `split_manifest_rl.json` and uses
the explicit `/mnt/umm/users/xujunxiang/VBVR-Pro_10k` root. The verified
manifest contains 50 In-Domain tasks and 50,000 samples, so the YAML sets
`dataset_size: 50000`. It fixes the Flow-CPS coefficient at `0.7` (with no
`grpo_cps_noise_scale_range`) and isolates output, W&B, and reward-temp paths
with `cps0p7` plus the `_manifest_rl_evalkit_6fedd9d9` suffix. Because the original fp32 DCP
epoch-1 checkpoint is absent, it initializes model weights from the completed
`storage/models/dcp_converted_5b/sft_vbvr_5b_256x256x161_full_lr_1e-5_checkpoint-epoch1-main-v2`
Diffusers conversion and sets `resume_from: null`. This starts fresh optimizer
and dataloader state, and the fp32 training load cannot recover precision already
discarded by the BF16 conversion.

The unsuffixed config remains pinned to historical revision `42a1593d` for
reproducibility. Do not resume one of its checkpoints with the
`evalkit_6fedd9d9` config: the scorer update changes the RL objective.

The following smoke result belongs to the historical `42a1593d` scorer, not
the new objective. An eight-H100 one-step executable smoke completed with this initialization and
dataset contract, fixed CPS 0.7, Liger, fresh-cache Inductor compilation, the
aligned `main_v2` reward, full-FT backward/AdamW, and a complete DCP save. The
reduced batch-8/G=2/T=2 validation reported reward `0.7310 +/- 0.2571`, grad norm
`0.0024`, and 35.5/39.3 GiB peak allocated/reserved memory per rank. This proves
the pipeline and compiler-header path; it is not a production G=32/T=30 or
four-node throughput result.

## DanceGRPO-Style Replay

`DanceGRPOTrainer` keeps the standard single-group execution path for normal launches, supports a shared-prompt
all-rank mode where `batch_size` is the global prompt batch and each prompt's `grpo_group_size` samples are sharded
across all ranks, and can also split rollout/reward actors from the smaller training FSDP group for multi-node launches.
It adopts two ideas from DanceGRPO:

- all samples in a prompt group can share the same initial noise;
- policy replay can use only a selected subset of denoising timesteps.[^dancegrpo][^dancegrpo-trainer]

The timestep subset is generated consistently across ranks so all FSDP ranks call the same expert modules in the same order. This is necessary because per-rank divergence in expert routing can deadlock FSDP collectives.[^dancegrpo-trainer]

Flow-CPS training supports either a fixed coefficient or a uniformly sampled
per-prompt coefficient. The existing fixed behavior is:

```yaml
grpo_sde_formula: flowcps
grpo_sde_noise_scale: 0.7
```

To sample independently between prompt groups while keeping all `G` rollouts
inside one prompt group on the same coefficient, set a range:

```yaml
grpo_sde_formula: flowcps
grpo_cps_noise_scale_range: [0.0, 1.0]
```

The coefficient is sampled once per prompt and optimizer step. Shared-prompt
ranks and split rollout actors derive it from the same deterministic seed, and
the rollout payload stores it so policy replay uses the exact original value.
`grpo_sde_noise_scale` remains the fixed-mode value when the range is omitted.

Shared-prompt runs can divide the global prompt batch into reward/replay waves:

```yaml
batch_size: 32
grpo_shared_prompt_batch: true
grpo_shared_prompt_microbatch_size: 16
```

`batch_size` remains the number of prompts in one optimizer step. The wave size
must divide it, fit within the data-parallel world, and evenly divide that
world. All waves are generated before any replay backward, while the policy is
unchanged; replay of the first wave can therefore overlap CPU scoring for the
second wave without introducing stale-policy trajectories. Gradients from all
waves accumulate into one optimizer update, with synchronization enabled on the
last backward. At world size 32, a size-16 wave assigns two ranks and 16 of the
32 rollouts to each prompt per rank; at world size 64, it assigns four ranks and
eight rollouts per rank. This distributes OCR-heavy reward tails more broadly
than the single-wave world-32 layout, where one rank must score all 32 videos
for its prompt.

DanceGRPO currently rejects expert parallel. For split RL, the multinode launcher defaults to half the nodes training and half running rollout/reward actors. A manual `rl_train_node_count: 1` means node 0 trains and the remaining nodes run rollout/reward actors; rollout actors partition `grpo_group_size` across cards.[^dancegrpo-trainer]

### Single-Node Tensor Parallel + FSDP

Full A14B DanceGRPO can compose two-way tensor parallelism with four FSDP
data replicas on one eight-GPU node:

```yaml
fsdp: true
hsdp: false
tensor_parallel_size: 2
expert_parallel: false
lora_rank: 0
use_liger_kernel: true
torch_compile: true
torch_compile_backend: inductor
grpo_shared_prompt_batch: false
batch_size: 4  # per DP replica: 4 x DP4 = 16 global prompts
grpo_fsdp_sync_each_backward: true
```

The standard data path shards prompts over the four DP replicas and duplicates
each replica's batch only within its TP pair. Rollout inference and replay
training therefore use the same TP-sharded policy; FSDP adds parameter,
gradient, and optimizer-state sharding across the four DP replicas. The
runnable A14B configuration is
`configs/train_dancegrpo_vbvr_pro_a14b_256x256x161_rule_cps_from_sft_diffsynth_mix_260603_bs_16_lr_1e-5_full_tp2_fsdp4.yaml`.[^wan-tp-config]

Full-finetune DanceGRPO must not defer FSDP synchronization across hundreds of
replay backwards: doing so retains unsharded gradients. The
`grpo_fsdp_sync_each_backward` switch performs reduce-scatter after each replay
backward, bounding gradient memory while still accumulating the sharded
gradient for the optimizer step. The TP path also uses a topology-aware global
gradient norm because stock `clip_grad_norm_` cannot combine the 1D-DP and
2D-DP/TP DTensor meshes in the same Wan model. For raw full-A14B runs,
`grpo_offload_inference_models: true` moves frozen T5 to CPU after encoding and
the VAE to CPU after rule rewards, recovering replay/Adam headroom; both are
restored automatically before the next raw batch. The GRPO launcher defaults
the CUDA allocator to `expandable_segments:True` unless the operator already
set an override.[^base-rl][^wan-tp]

With TP, `use_liger_kernel: true` is compatibility-safe but is not currently a
Liger RMSNorm speedup: Wan A14B's RMSNorm modules are precisely the Q/K norms
whose statistic spans all TP-sharded heads, so they are converted to the
collective-aware TP implementation. `torch_compile` still compiles the
surrounding Transformer, T5, and VAE graphs. The TP RMSNorm is an intentional
eager graph boundary; tracing its `distributed.nn` autograd collective gives a
numerically correct forward but an incorrect backward in current PyTorch.
The fp32-load/bf16-FSDP activation-checkpoint wrapper puts autocast around
`checkpoint()` and relies on non-reentrant checkpointing to restore that
ambient state during recompute. Do not pass an autocast `context_fn`: PyTorch
2.11 Dynamo only supports `TorchDispatchMode` checkpoint contexts. In-place
`Module.compile()` preserves DCP/state-dict names. Inductor/Triton also needs a
working host C compiler and the matching Python development headers (for this
Python 3.12 environment, `Python.h` from `python3.12-dev`). The shared launcher
environment first checks an explicit `WAN_TRAINER_PYTHON_INCLUDE`, the system
include directory, and `CPATH`; if those are missing, it checks the ignored
shared `storage/toolchains/uv-python` tree and then an already-installed
user-level uv Python with the same major/minor ABI. It exports the selected
include through `CPATH` without downloading during launch. On runtime-only
cluster images, provision and fresh-cache validate the shared tree once with:

```fish
fish scripts/dev/bootstrap_triton_python_headers.fish
```

Multi-node GRPO uses a node-local Triton cache and runs a small driver preflight
on every node before `torchrun`, so a missing compiler/header fails before model
loading. Set `WAN_TRAINER_TRITON_PREFLIGHT_ONLY=1` for a cheap all-node
preflight job, then remove it for training. Installing the exact development
package in the production image remains preferred.

The production-shape Liger+Inductor validation completed one optimizer step on
eight 80-GiB H100s with TP2 x FSDP4, 16 global prompts, `G=16`, `T=30`, 17
replay timesteps, both A14B experts, raw T5/VAE, `vbvr_rule`, gradient clipping,
and AdamW at lr=1e-5. It took 4,096.42 seconds and reached 59.4 GiB allocated /
64.5 GiB reserved per process (about 69.1 GiB per card in `nvidia-smi`), with no
OOM. This establishes end-to-end memory feasibility; it is not a controlled
compile-speed benchmark.

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

For the four-node full-FT A14B TP2 run, use the fixed wrapper on every node.
It requires `WORLD_SIZE=4`, maps 32 ranks to DP16 x TP2, overrides the local
batch to 1 to preserve 16 global prompts, and uses a separate FSDP16 output/run
name:

```fish
fish scripts/train/dancegrpo_vbvr_pro_a14b_full_tp2_4node.fish
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
[^base-rl]: [`src/trainer/base_rl_trainer.py`](../src/trainer/base_rl_trainer.py)
[^wan-tp]: [`src/trainer/tensor_parallel.py`](../src/trainer/tensor_parallel.py)
[^wan-tp-config]: [`configs/train_dancegrpo_vbvr_pro_a14b_256x256x161_rule_cps_from_sft_diffsynth_mix_260603_bs_16_lr_1e-5_full_tp2_fsdp4.yaml`](../configs/train_dancegrpo_vbvr_pro_a14b_256x256x161_rule_cps_from_sft_diffsynth_mix_260603_bs_16_lr_1e-5_full_tp2_fsdp4.yaml)
[^reward-registry]: [`src/trainer/rewards/registry.py`](../src/trainer/rewards/registry.py)
[^neg-loss]: [`src/trainer/rewards/neg_loss.py`](../src/trainer/rewards/neg_loss.py)
[^maze-reward]: [`src/trainer/rewards/maze.py`](../src/trainer/rewards/maze.py)
[^vbvr-rule]: [`src/trainer/rewards/vbvr_rule.py`](../src/trainer/rewards/vbvr_rule.py)
[^vbvr-prepare]: [`src/cli/prepare_vbvr_eval_videos.py`](../src/cli/prepare_vbvr_eval_videos.py)
[^vbvr-eval-wrapper]: [`src/eval/vbvr_run_evaluation_parallel.py`](../src/eval/vbvr_run_evaluation_parallel.py)
[^vbvr-alignment-validator]: [`scripts/dev/validate_vbvr_reward_alignment.py`](../scripts/dev/validate_vbvr_reward_alignment.py)
