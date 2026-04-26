# Improvement Plan

This folder lists improvements from algorithm design through engineering operations. The priorities are based on current source behavior, not speculative product goals.

## Files

- [Algorithmic Improvements](algorithmic.md): flow paths, reward design, RL stability, expert routing, and evaluation science.
- [Data And Evaluation Improvements](data_and_evaluation.md): data contracts, latent provenance, VBVR scoring, and dataset quality.
- [Training Systems Improvements](training_systems.md): FSDP/HSDP/EP reliability, checkpointing, performance, and memory.
- [Engineering Quality Improvements](engineering_quality.md): config hygiene, tests, docs, CLI consistency, and maintainability.

## Highest Priority Items

1. Add invariant tests around shifted sigma schedules, MoE boundary routing, COS tau/video-chain shape checks, and expert-parallel GRPO schedules.
2. Fix correction configs so EMA is either enabled or the correction teacher is explicitly documented as live-student.
3. Consolidate duplicated SFT/RL trainer infrastructure or introduce shared mixins for dataset/model/FSDP/checkpoint behavior.
4. Record latent provenance (`model_path`, VAE config hash, prompt-cleaning version, resolution, num frames, commit hash) in every WebDataset shard.
5. Add a lightweight smoke-test suite that does not require loading Wan2.2 A14B, plus a separate GPU/model integration suite.
6. Calibrate VLM and rule-based evaluation against a small human-labeled set before using scores as training-selection signals.

## Reference Anchors

The proposed changes are grounded in the repository sources and in these external references:

- Flow Matching for the base supervised objective.[^fm]
- Wan2.2's two-expert denoising design.[^wan22]
- Flow-GRPO, PPO, GRPO, and DanceGRPO for the RL layer.[^flowgrpo][^ppo][^grpo][^dancegrpo]
- PyTorch FSDP2 and DCP documentation for distributed training/checkpointing constraints.[^fsdp2][^dcp]
- WebDataset and StatefulDataLoader docs for data pipeline constraints.[^webdataset][^torchdata]

[^fm]: Lipman et al., "Flow Matching for Generative Modeling", arXiv:2210.02747, https://arxiv.org/abs/2210.02747
[^wan22]: Wan-AI, "Wan2.2-I2V-A14B-Diffusers", Hugging Face model card, https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers
[^flowgrpo]: Liu et al., "Flow-GRPO: Training Flow Matching Models via Online RL", arXiv:2505.05470, https://arxiv.org/abs/2505.05470
[^ppo]: Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347, https://arxiv.org/abs/1707.06347
[^grpo]: Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", arXiv:2402.03300, https://arxiv.org/abs/2402.03300
[^dancegrpo]: DanceGRPO authors, "DanceGRPO: Unleashing GRPO on Visual Generation", arXiv:2505.07818, https://arxiv.org/abs/2505.07818
[^fsdp2]: PyTorch, "`torch.distributed.fsdp.fully_shard`", https://docs.pytorch.org/docs/2.8/distributed.fsdp.fully_shard.html
[^dcp]: PyTorch, "Distributed Checkpoint", https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
[^webdataset]: Hugging Face Hub docs, "WebDataset", https://huggingface.co/docs/hub/datasets-webdataset
[^torchdata]: PyTorch/TorchData, "Stateful DataLoader", https://docs.pytorch.org/data/beta/torchdata.stateful_dataloader.html
