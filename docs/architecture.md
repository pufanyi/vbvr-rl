# System Architecture

Wan-Trainer is built for high-throughput training of the Wan2.2 MoE model (14B parameters). It leverages the latest features in PyTorch Distributed to manage the computational load.

## 1. Distributed Training with FSDP2
The project uses **PyTorch FSDP2** (Fully Sharded Data Parallel version 2) for model sharding.
- **Mixed Precision**: Parameters are sharded in `bfloat16`, while gradient reduction happens in `float32` by default for stability.
- **Device Mesh**: Supports 1D (all-GPU sharding) and 2D (HSDP - sharding within node, replication across nodes) meshes.
- **Liger Kernels**: Integrated Liger kernels for fused Triton-based RMSNorm to save VRAM and increase speed.

## 2. Mixture-of-Experts (MoE) Optimizations
Wan2.2 uses a noise-based routing strategy:
- `Transformer`: High-noise expert (Timestep 900-1000).
- `Transformer_2`: Low-noise expert (Timestep 0-900).

### Expert Parallel (EP) Mode
In EP mode, the GPUs are partitioned into two sub-meshes.
- **Group 0** handles only the `Transformer` shards.
- **Group 1** handles only the `Transformer_2` shards.
This reduces the VRAM requirement for each GPU by nearly 50% since only one expert is "trainable" and sharded per group.

## 3. Distributed Checkpoint (DCP)
Instead of traditional `.bin` or `.safetensors` files, this project uses **DCP**.
- **Parallel I/O**: Every GPU rank writes its own shard of the weights and optimizer states simultaneously.
- **Resharding on the Fly**: You can train on 8 GPUs and resume on 16 GPUs seamlessly; DCP handles the re-sharding automatically.
- **Stateful DataLoader**: We use `torchdata.stateful_dataloader` to save and restore the exact position in the dataset, including buffer states for WebDataset.

## 4. Latent Space Training
To bypass the expensive VAE and T5 encoding during training, the project supports **WebDataset latents**.
- T5 prompt embeds and VAE video latents are pre-computed and stored as `.tar` shards.
- The `VBVRLatentDataset` loads these directly, allowing for extremely high GPU utilization (reaching 90%+ MFU).
