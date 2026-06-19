# Project Memory

## Maintenance

- Maintain this `AGENTS.md` as a living project memory file. When durable project-specific workflow preferences, evaluation procedures, environment lessons, or recurring pitfalls come up, update this file proactively in the same change.
- Prefer Jujutsu (`jj`) for version-control workflows in this repository. Use `git` only when a requested operation or external tool specifically requires Git semantics.
- Prefer project facts that will still matter weeks later. Do not add one-off command transcripts, temporary paths, or speculative notes unless they encode a repeatable workflow or pitfall.
- Keep `README.md`, `docs/`, runnable `configs/`, launcher scripts, and this file in sync when data contracts, checkpoint layouts, training modes, or evaluation flows change.
- For long-running tasks, keep monitoring the process instead of leaving it hanging. Poll logs or command output, verify the exit status or completion signal, and make sure no required background session is still running before reporting the task as done.

## Repository Shape

- Wan-Trainer is a research training stack for Wan2.2 I2V/TI2V Diffusers models. It supports SFT, COS path training, on-policy correction, DanceGRPO-style RL, latent WebDataset training, DCP checkpointing, LoRA extraction/loading, and VBVR/maze evaluation.
- Start with `docs/README.md` for architecture-level context. The main source map is: `src/cli/` entry points, `src/models/` Wan model wrapper and COS paths, `src/data/` raw/latent datasets and remote I/O, `src/trainer/` SFT/RL trainers, rewards, FSDP/DCP runtime, `src/precompute/` latent and synthetic data builders, `src/eval/` VBVR/maze scoring helpers, `scripts/` fish launchers/operator utilities, `configs/` runnable experiment configs, `tests/` focused unit/consistency checks.
- Most fish launchers source `scripts/lib/env.fish`, change to the repo root, activate `.venv`, and set `PYTHONPATH`. When running Python inside this project, prefer direct `.venv/bin/python` or `.venv/bin/torchrun` commands; do not default to `uv run`.
- Generated/heavy local artifacts belong under ignored paths such as `storage/`, `wandb/`, `logs/`, `tmp/`, and `data/diffsynth_mix/`. Do not commit generated latent shards or local dataset caches.

## Data Contracts

- Raw I2V data is configured by a JSON file pointing to one or more Parquet files, not by a flat list of samples. Rows use `videos` for ordered COS/multi-step chains or `video` for a single target, plus `prompt` and optional `image`; absent `image` falls back to the first frame of the final video.
- Raw media paths can be local or `s3://`. S3 localization goes through `src/data/remote_io.py` and the vendored `aoss/` client. Credentials/config are environment-driven: `WAN_TRAINER_AOSS_CONF_RULES`, `WAN_TRAINER_AOSS_CONF_PATH`, and `WAN_TRAINER_REMOTE_CACHE_DIR`.
- For remote raw training, `raw_remote_prefetch_lookahead`, `raw_remote_prefetch_workers`, `WAN_TRAINER_DECORD_NUM_THREADS=1`, and `dataloader_in_order=false` are practical knobs. The DiffSynth mix launcher `scripts/train/i2v_diffsynth_mix_260603_multinode.fish` records the current AOSS/cache defaults.
- Latent training uses WebDataset `shard-*.tar` files with `{key}.safetensors` and `{key}.json`. Required tensors are `prompt_embeds`, `condition`, and either `latents` or COS-style `latents_0`, `latents_1`, ...; extra tensors such as `maze_*` are passed through for rewards.
- Always set `dataset_size` for latent/WebDataset configs. Trainers derive per-rank epoch lengths from it; missing or wrong values can break LR scheduling or cause uneven-rank epoch endings.
- If shard count is smaller than rank count, `VBVRLatentDataset` falls back to sample-level splitting inside shard groups. This works, but resharding to at least the training world size gives better sequential I/O.
- 5B TI2V uses `expand_timesteps` and a different condition format from A14B I2V. Recompute latents/conditions with the same base model family used for training or evaluation.

## Model And Training

- `WanI2VForTraining` is the central model wrapper. It reads `boundary_ratio` from `model_index.json`, `flow_shift` from the scheduler config, builds the shifted sigma/timestep schedule, and routes A14B timesteps to high (`transformer`) or low (`transformer_2`) experts. 5B TI2V is a dense single-transformer model with expanded per-token timesteps.
- `src.cli.train_i2v` dispatches `SFTConfig.trainer: i2v` to `I2VTrainer` and `trainer: cos` to `COSTrainer`; `src.cli.train_i2v_correction` uses `CorrectionConfig`; `src.cli.train_grpo` uses `RLConfig` and `DanceGRPOTrainer`.
- SFT and RL use separate base stacks (`BaseTrainer` and `BaseRLTrainer`) with duplicated distributed/model/dataset/FSDP/checkpoint logic. Fixes to FSDP, HSDP, dataloading, checkpointing, or resume semantics often need to be mirrored in both stacks.
- Full fine-tuning loads trainable transformers in fp32 by default when `transformer_load_dtype: auto`; LoRA base weights stay bf16 unless overridden. If `param_dtype: bfloat16` with fp32 transformer load, gradient checkpoint recompute uses bf16 autocast.
- `disable_cudnn_sdp` defaults true because cuDNN SDPA backward has produced NaNs for Wan low-noise training on H100/PyTorch 2.11. Be explicit before changing attention backend behavior.
- Expert parallel splits the world into high/low expert groups and requires `fsdp=true`, `train_experts: both`, even world size, and synchronized control flow. `expert_parallel_data_mode: duplicate` makes expert groups see the same data stream; `split` shards data across all ranks for full global throughput.
- FSDP2 requires all ranks to execute compatible wrapped-module forward sequences. Keep timestep sampling and DanceGRPO replay timestep selection rank-synchronized; per-rank divergence in high/low routing can hang NCCL/FSDP collectives.
- COS supports N-step paths `linear`, `target_cosine`, and `target_sigmoid`; legacy paths are 2-step only. `len(cos_tau_sigma)` must equal `len(video_latents) - 1`, and latent COS datasets must be generated with all chain latents.
- Correction training is supervised correction, not policy gradient. It currently forbids expert parallel; use EMA for the teacher rollout unless intentionally testing live-student correction.

## DanceGRPO And Rewards

- DanceGRPO requires `grpo_group_size >= 2`, `grpo_num_sampling_steps >= 2`, and positive `grpo_sde_noise_scale`. It does not support expert parallel.
- Non-split `grpo_shared_prompt_batch=true` means every rank reads the same global prompt batch and shards each prompt's `grpo_group_size` samples across ranks. `batch_size` is the global prompt count and must divide `world_size`; `grpo_group_size` must divide the ranks-per-prompt count.
- Split RL is controlled by `rl_train_node_count` or `rl_train_rank_count`; train ranks come first, rollout actors are the remaining ranks. `rl_actor_weight_sync='lora'` requires LoRA, `full` currently requires `rl_async_rollout=true`, and `none` is only useful for plumbing/stale-actor smoke tests.
- Async split rollout keeps each optimizer step semantically unchanged but pre-generates future steps through a bounded queue. `rl_async_rollout_prefetch_steps: 0` auto-picks enough queued steps to keep rollout actors busy when `rollout_world_size > grpo_group_size`.
- Rewards are registered in `src/trainer/rewards/__init__.py`. Current names include `neg_loss`, `maze`, `maze_line`, `maze_tracker`, and `vbvr_rule`. Rewards with `requires_vae = True` force VAE loading even with precomputed latents.
- `vbvr_rule` decodes generated and GT latents to temporary MP4s and calls the vendored VBVR EvalKit. It requires WebDataset JSON metadata exposing `sample_tar` so the task name can be inferred, and it can filter RL data to EvalKit-supported tasks.

## Checkpoints And Conversion

- New training checkpoints use the unified DCP high/low layout under `checkpoint-N/high/` and `checkpoint-N/low/`, plus rank-local optimizer/dataloader files and optional PEFT LoRA sidecars. Treat `src/trainer/checkpoint_runtime.py` as authoritative for writes; `src/trainer/checkpoint.py` still contains some historical layout wording.
- Flat and expert-parallel checkpoints are intended to cross-load through the unified high/low layout. A checkpoint with only one expert subdirectory can be loaded for matching expert-only work but will warn/skip the missing expert in flat mode.
- Auto-resume from `output_dir` preserves optimizer/dataloader/counters by default; explicit `resume_from` defaults to weight/init behavior unless `reset_dataloader` says otherwise.
- For evaluation or FastVideo/lmms-eval, convert DCP checkpoints to regular Diffusers directories first. Use `scripts/convert/dcp_to_diffusers.fish` or `scripts/eval/lmms_eval_checkpoint.fish`; conversion completion is signaled by `model_index.json`.
- Converted full A14B Diffusers outputs are large but much smaller than full DCP training checkpoints because optimizer shards are excluded. Make sure interrupted conversions are removed or rerun with `OVERWRITE=1`.

## Evaluation Workflows

- For VBVR/lmms-eval checkpoint evaluation, read `docs/vbvr_lmms_eval.md` first. It documents the DCP-to-Diffusers conversion step, the lmms-eval command, hardware overrides, output paths, and lessons from the 2026-04-29 partial run.
- `scripts/eval/lmms_eval.fish` runs in `/mnt/umm/users/pufanyi/workspace/lmms-eval` with the FastVideo backend. Keep `DATA_PARALLEL * NUM_GPUS` within visible GPU count; default `DATA_PARALLEL=8` assumes an 8x H100 setup, so override on smaller machines.
- Do not report final VBVR metrics from partial generated videos. A valid full run should exit normally and write aggregate/submission JSON, including `submissions/vbvr_eval_results.json`.
- For 5B VBVR 256x256x161 checkpoint sweeps, `scripts/eval/watch_vbvr_5b_checkpoints.py` watches complete DCP checkpoints, records state under `storage/eval_watch/...`, converts with base `storage/models/Wan2.2-TI2V-5B-Diffusers`, and skips checkpoints with existing result JSON.
- For Maze checkpoint evaluation with aggregate scores, use the separate `../lmms-eval-maze` fork, not only the in-repo `src.cli.eval_maze` renderer. Read `../lmms-eval-maze/README_MAZE.md` and `../lmms-eval-maze/docs/maze_topology_eval.md` first. The task is `maze_line`, the runner is `../lmms-eval-maze/tools/run_maze_fastvideo.sh`, generated videos go under `storage/lmms_eval_maze/<run>/generated_videos/<model>/maze_line/`, and detailed scores are written to `storage/lmms_eval_maze/<run>/submissions/maze_line_eval_results.json`. Prefer the current topology score fields (`overall`/`topology_score`, pass rate, goal success, invalid rate, and per-difficulty scores); older `mask_f1` results may exist for historical runs.
- If evaluation or conversion is interrupted, check for remaining `lmms_eval`, `fastvideo`, `torchrun`, or conversion processes and verify GPU memory with `nvidia-smi` before reporting the system idle.

## Development And Verification

- Use targeted tests for the touched surface. Useful CPU tests include `tests/test_cos_path.py`, `tests/test_dancegrpo_split.py`, `tests/test_i2v_trainer.py`, `tests/test_vbvr_latent_dataset.py`, and `tests/test_trainer_utils.py`.
- This repo's `.venv` may not have dev tools installed until needed. Install with `uv pip install --python .venv/bin/python <package>` when a missing local test tool blocks verification.
- `pyproject.toml` uses Python >=3.12, `uv.lock`, and Ruff line length 120. Keep reusable Python logic under `src/`; scripts should stay thin launchers or operator utilities.
- When adding a reward, create a `BaseReward` subclass, decorate it with `@register_reward("name")`, import the module in `src/trainer/rewards/__init__.py`, and set `grpo_reward_fn` in the YAML.
- When changing data schema, latent precompute, condition construction, checkpoint remapping, or distributed routing, add or update focused tests. These areas silently break expensive runs if only checked manually.
