# Wan-Trainer Documentation

This directory documents the current Wan-Trainer codebase from source, not from older generated notes. All code citations point to repository files; external citations point to papers or official documentation that motivate the design.

## Reading Order

1. [Architecture](architecture.md) explains the execution model, trainer hierarchy, model wrapper, distributed setup, and checkpointing.
2. [Training](training.md) explains SFT, COS, correction, and DanceGRPO-style training.
3. [Data](data.md) explains raw Parquet inputs, latent WebDataset shards, and precompute pipelines.
4. [Evaluation](evaluation.md) explains inference, VBVR generation, VLM scoring, and rule scoring.
5. [Checkpoints](checkpoints.md) explains the unified high/low DCP layout, EMA, LoRA sidecars, and resume semantics.
6. [Improvements](improvements/README.md) lists algorithm, data, training-systems, and engineering improvements.

## Current System Summary

Wan-Trainer is centered on a Wan2.2 image-to-video training wrapper that loads the Diffusers tokenizer, UMT5 text encoder, Wan VAE, and one or both denoising transformers (`transformer` for high-noise timesteps and `transformer_2` for low-noise timesteps). The wrapper reads `boundary_ratio` from `model_index.json` and `flow_shift` from the scheduler config, then builds a shifted sigma schedule used by all trainers.[^model-wrapper]

The training stack has two parallel base hierarchies:

- `BaseTrainer` serves SFT-like objectives: I2V, COS, and correction.[^base-trainer]
- `BaseRLTrainer` / `BaseGRPOTrainer` serve the DanceGRPO RL objective.[^base-rl][^base-grpo]

Both stacks share the same model wrapper, optimizer factory, EMA implementation, FSDP2 sharding style, WebDataset latent loader, and DCP checkpoint runtime.

## Citation Convention

Local code citations are written as footnotes that point to source files. External references appear in each document's reference section. The most important external anchors are Wan2.2's model card for two-expert denoising, PyTorch FSDP2/DCP documentation for distributed training and checkpointing, Flow Matching for the base objective, and Flow-GRPO/DanceGRPO/PPO/GRPO papers for the RL layer.[^wan22][^fsdp2][^dcp][^fm][^flowgrpo][^dancegrpo][^ppo][^grpo]

[^model-wrapper]: [`src/models/wan_i2v.py`](../src/models/wan_i2v.py)
[^base-trainer]: [`src/trainer/base_trainer.py`](../src/trainer/base_trainer.py)
[^base-rl]: [`src/trainer/base_rl_trainer.py`](../src/trainer/base_rl_trainer.py)
[^base-grpo]: [`src/trainer/base_grpo_trainer.py`](../src/trainer/base_grpo_trainer.py)
[^wan22]: Wan-AI, "Wan2.2-I2V-A14B-Diffusers", Hugging Face model card, https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers
[^fsdp2]: PyTorch, "`torch.distributed.fsdp.fully_shard`", https://docs.pytorch.org/docs/2.8/distributed.fsdp.fully_shard.html
[^dcp]: PyTorch, "Distributed Checkpoint", https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
[^fm]: Lipman et al., "Flow Matching for Generative Modeling", arXiv:2210.02747, https://arxiv.org/abs/2210.02747
[^flowgrpo]: Liu et al., "Flow-GRPO: Training Flow Matching Models via Online RL", arXiv:2505.05470, https://arxiv.org/abs/2505.05470
[^dancegrpo]: DanceGRPO authors, "DanceGRPO: Unleashing GRPO on Visual Generation", arXiv:2505.07818, https://arxiv.org/abs/2505.07818
[^ppo]: Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347, https://arxiv.org/abs/1707.06347
[^grpo]: Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", arXiv:2402.03300, https://arxiv.org/abs/2402.03300
