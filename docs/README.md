# Wan-Trainer Documentation

This project is a high-performance training suite for the **Wan2.2** video generation model. It supports advanced training techniques including MoE-specific optimizations, Reinforcement Learning (GRPO), and Chain-of-Step flow matching.

## Table of Contents
- [Quick Start](README.md#setup)
- [Trainer Deep Dive](trainers.md)
- [Architecture & Expert Parallelism](architecture.md)
- [Bugs & Technical Debt](bugs/technical_debt.md)

## Core Algorithms
- **SFT (I2V)**: Standard supervised fine-tuning.
- **COS (Chain-of-Step)**: Two-stage piecewise flow matching for "Reasoning-to-Video".
- **Flow-GRPO**: Rule-based reinforcement learning for video alignment.
- **Correction**: On-policy drift correction using teacher rollouts.

## Key Features
- **FSDP2 & DCP**: Full sharding and distributed checkpoints for massive models.
- **Latent Precomputation**: Support for training directly on pre-encoded T5/VAE latents via WebDataset.
- **Expert Parallel (EP)**: Efficiently training Mixture-of-Experts by sharding different experts to different GPU groups.
