# Identified Issues and Potential Bugs

## 1. Shared Parameter Divergence in Expert Parallel (EP)
**File**: `src/trainer/base_trainer.py`
**Issue**: When `expert_parallel: true` is enabled, the world is split into two independent FSDP meshes. If `train_text_encoder: true` is also set, both meshes will update the text encoder based on their local data shards.
**Problem**: There is no cross-group gradient synchronization. The text encoder in the "high-noise" group will diverge from the "low-noise" group.
**Risk**: Significant performance degradation as the model loses a unified prompt embedding space.

## 2. VAE Decoding Bottleneck in RL (GRPO)
**File**: `src/trainer/rewards/maze.py`
**Issue**: The reward function decodes generated latents into pixel space using the VAE.
**Problem**: `pixel_video = self.trainer.model.decode_latents(generated_latents)`. For a group size of $G=8$ and batch size $B=1$, we decode 8 videos per step. If videos are 81 frames long, this is 648 frames per training step.
**Risk**: Massive VRAM consumption and slow training steps.

## 3. EMA Synchronization in EP Mode
**File**: `src/trainer/base_trainer.py`
**Issue**: Each expert group maintains its own EMA (Exponential Moving Average) of the model.
**Problem**: Similar to Issue #1, the EMA for shared parameters (non-expert layers) will diverge between groups. If a checkpoint is saved, which group's EMA is "correct"?
**Risk**: Inconsistent checkpoints.

## 4. Hard-coded MoE Boundaries
**File**: `src/models/wan_i2v.py`
**Issue**: The routing boundary is hard-coded to timestep 900 (sigma ~0.978).
**Problem**: While this matches the original Wan2.2 config, the trainer does not dynamically read this from the model config in all paths, potentially leading to mismatches if a user provides a custom-tuned MoE model.
