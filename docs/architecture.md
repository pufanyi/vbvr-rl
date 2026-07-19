# Architecture

Wan-Trainer is a distributed fine-tuning and evaluation stack around Wan2.2 I2V Diffusers checkpoints. Its core shape is:

```text
CLI or fish launcher
  -> Pydantic config
  -> Trainer hierarchy
  -> WanI2VForTraining
  -> raw media loader or latent WebDataset loader
  -> FSDP2/HSDP/expert-parallel execution
  -> DCP checkpoint runtime
```

## Entry Points

| Entry point | Config class | Trainer path | Notes |
| --- | --- | --- | --- |
| `src.cli.train_i2v` | `SFTConfig` | `I2VTrainer` or `COSTrainer` | Dispatches on `trainer: i2v` vs `trainer: cos`.[^train-i2v] |
| `src.cli.train_cos` | `SFTConfig` | `COSTrainer` | Dedicated COS entry point, equivalent to `trainer: cos`.[^train-cos] |
| `src.cli.train_i2v_correction` | `CorrectionConfig` | `I2VCorrectionTrainer` | SFT plus teacher-rollout correction; forbids expert parallel.[^train-correction] |
| `src.cli.train_grpo` | `RLConfig` | `DanceGRPOTrainer` | RL entry point; split rollout/train execution is controlled by `rl_train_node_count`.[^train-grpo] |
| `src.cli.eval_i2v` | argparse | Diffusers pipeline | Batch generation; can load DCP checkpoint weights.[^eval-i2v] |
| `src.cli.eval_vbvr` | argparse | VLM judge + runner | Scores already-generated VBVR videos.[^eval-vbvr] |

The fish wrappers in `scripts/` are thin operational launchers. They activate `.venv`, set `PYTHONPATH`, and call the Python module with `torchrun` where needed.[^scripts-env][^scripts-readme]

## Model Wrapper

`WanI2VForTraining` owns the training-facing view of the Wan2.2 pipeline. It can load:

- tokenizer and UMT5 text encoder, unless prompt embeddings are precomputed;
- Wan VAE, unless VAE latents and condition tensors are precomputed;
- `transformer` for high-noise denoising;
- `transformer_2` for low-noise denoising;
- optional LoRA adapters on attention projection modules.[^wan-wrapper]

The wrapper reads `boundary_ratio` from `model_index.json` and converts it into `boundary_timestep`. It also reads scheduler `flow_shift`, constructs shifted sigma values, and computes `boundary_idx` from shifted timesteps rather than assuming a linear index.[^wan-wrapper] This matters because the same shifted schedule is used in SFT, COS, correction, GRPO sampling, and reward evaluation.

The Wan2.2 model card describes the A14B family as a two-expert MoE design: a high-noise expert handles early layout-oriented denoising, and a low-noise expert handles later detail refinement.[^wan22] The code mirrors that design by routing `timestep >= boundary_timestep` to `transformer` and lower timesteps to `transformer_2`.[^wan-wrapper]

## Trainer Hierarchy

The SFT side uses `BaseTrainer` for shared infrastructure and subclasses for algorithm-specific loops:

```text
BaseTrainer
  I2VTrainer
  COSTrainer
  I2VCorrectionTrainer
```

`BaseTrainer` initializes distributed state, expert-parallel rank groups, the model, optional FSDP2/HSDP sharding, EMA, dataset, StatefulDataLoader, optimizer(s), DCP state, W&B, and resume handling.[^base-trainer]

The RL side is deliberately separate:

```text
BaseRLTrainer
  BaseGRPOTrainer
    DanceGRPOTrainer
```

`BaseRLTrainer` duplicates much of the SFT infrastructure because the RL pipeline supports different sampling/training GPU layouts. `BaseGRPOTrainer` adds reference policy handling, reward construction, group-relative advantage computation, SDE schedule helpers, and the outer DanceGRPO training loop.[^base-rl][^base-grpo]

## Distributed Execution

The main sharding mode is PyTorch FSDP2 through `fully_shard`, using `MixedPrecisionPolicy` and `DeviceMesh`.[^fsdp2] The code supports:

- plain FSDP over all ranks;
- HSDP with a 2D mesh: shard within node and replicate across nodes;
- expert parallel, where half the ranks train the high-noise expert and half train the low-noise expert;
- expert-parallel plus HSDP, building per-expert sharded/replicated meshes;
- DanceGRPO tensor parallel composed with FSDP, for example a single-node
  `DP=4 x TP=2` mesh over eight GPUs;
- non-FSDP mode with manual gradient all-reduce, mainly useful for small/LoRA runs.[^base-trainer][^base-rl]

The DanceGRPO TP path applies tensor parallelism before FSDP2. Q/K/V and the
first FFN projection are column-sharded; attention output and the second FFN
projection are row-sharded. Wan normalizes Q/K across all heads, so the TP
implementation uses an autograd-aware TP all-reduce for the RMS statistic
instead of changing the model to a local/per-head norm. Parameters outside
those projections remain TP-replicated and are FSDP-sharded over the DP
dimension.[^wan-tp][^pytorch-tp]

TP ranks form one logical data replica: they use the same sampler shard,
rollout seeds, reward, and replay inputs. Expensive VAE/CPU rewards run on TP
rank 0 and are broadcast to its partner; rewards that call the Wan policy are
marked `requires_policy_forward` and execute collectively on every TP rank.
The current TP implementation is RL-only and deliberately rejects LoRA,
HSDP, expert parallel, split RL, trainable text encoders, Liger RMSNorm, and
`torch.compile` rather than silently running a partially sharded topology.

Expert parallel changes the effective model loaded on each rank. Ranks in group 0 load/train only `transformer`; ranks in group 1 load/train only `transformer_2`. The checkpoint runtime still writes the same `high/` and `low/` layout regardless of whether a flat or expert-parallel trainer produced the checkpoint.[^checkpoint-runtime]

## Data Flow

Raw media path:

```text
JSON config -> Parquet rows -> videos/image/prompt
  -> UMT5 prompt embeddings
  -> VAE video latents
  -> VAE first-frame condition tensor
  -> trainer loss
```

Latent path:

```text
latent_webdataset_dir/shard-*.tar
  -> prompt_embeds + condition + latents
  -> optional maze_* metadata
  -> trainer loss or reward
```

The raw dataset is Parquet-backed and supports either one final video (`video`) or an ordered chain (`videos`) for COS.[^i2v-dataset] The latent dataset is an `IterableDataset` built on WebDataset tar shards; it pads/truncates prompt embeddings to 512 tokens and passes through extra safetensors keys such as `maze_*` for rewards.[^latent-dataset] WebDataset's tar-shard design matches the repository's large-scale latent training use case.[^webdataset]

## Checkpoint Runtime

Checkpointing uses PyTorch Distributed Checkpoint (DCP). DCP calls `state_dict` / `load_state_dict` on Stateful objects, which is why the repository wraps model weights and scalar training state in `TrainState` and EMA in `EMA`.[^dcp][^checkpoint][^ema]

Current checkpoints are written with a unified layout:

```text
checkpoint-N/
  high/
    .metadata + *.distcp
    optimizer_transformer_rank{R}.pt
    optimizer_text_encoder_rank{R}.pt
    dataloader_rank{R}.pt
    lora/transformer/
  low/
    .metadata + *.distcp
    optimizer_transformer_2_rank{R}.pt
    optimizer_text_encoder_rank{R}.pt
    dataloader_rank{R}.pt
    lora/transformer_2/
```

The design makes flat and expert-parallel checkpoints cross-loadable because both always use the same high/low directories. Legacy top-level DCP checkpoints are still loadable but are no longer written by the current runtime.[^checkpoint-runtime]

## Configuration Model

`TrainConfig` defines shared fields for model paths, raw/latent data, optimizer choice, trainable components, FSDP/HSDP/expert-parallel flags, LoRA, checkpointing, and logging. `SFTConfig`, `CorrectionConfig`, and `RLConfig` extend it with algorithm-specific fields.[^config]

Important config consequences:

- `latent_webdataset_dir` switches the model loader into precomputed mode, skipping VAE and text encoder unless required.
- `dataset_size` is required for correct scheduling and epoch length with iterable latent datasets.
- `train_experts` selects high, low, or both experts; `expert_parallel` requires `train_experts: both`.
- `transformer_load_dtype: auto` loads full fine-tuning transformers in fp32 but LoRA bases in bf16.
- `prompt_dropout` zeroes whole prompt embeddings to train an unconditional CFG branch.

## Architectural Tensions

The source shows several useful but risky tensions:

- SFT and RL base trainers duplicate infrastructure. This keeps RL independent but creates drift risk for FSDP, dataset, checkpoint, EMA, and optimizer behavior.
- `src/trainer/checkpoint.py` has a historical docstring that still describes a top-level flat checkpoint, while `checkpoint_runtime.py` now writes high/low subdirectories for all new checkpoints.
- Correction training recommends EMA in comments, but current correction configs set `ema_decay: 0`, causing teacher rollout to use live student weights and triggering a runtime warning.
- Expert parallel is effective for memory, but it relies on send/recv handshakes and identical control flow between peer ranks. Logging, reward computation, and sampling schedules must stay perfectly synchronized.

[^train-i2v]: [`src/cli/train_i2v.py`](../src/cli/train_i2v.py)
[^train-cos]: [`src/cli/train_cos.py`](../src/cli/train_cos.py)
[^train-correction]: [`src/cli/train_i2v_correction.py`](../src/cli/train_i2v_correction.py)
[^train-grpo]: [`src/cli/train_grpo.py`](../src/cli/train_grpo.py)
[^eval-i2v]: [`src/cli/eval_i2v.py`](../src/cli/eval_i2v.py)
[^eval-vbvr]: [`src/cli/eval_vbvr.py`](../src/cli/eval_vbvr.py)
[^scripts-env]: [`scripts/lib/env.fish`](../scripts/lib/env.fish)
[^scripts-readme]: [`scripts/README.md`](../scripts/README.md)
[^wan-wrapper]: [`src/models/wan_i2v.py`](../src/models/wan_i2v.py)
[^wan22]: Wan-AI, "Wan2.2-I2V-A14B-Diffusers", Hugging Face model card, https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers
[^base-trainer]: [`src/trainer/base_trainer.py`](../src/trainer/base_trainer.py)
[^base-rl]: [`src/trainer/base_rl_trainer.py`](../src/trainer/base_rl_trainer.py)
[^base-grpo]: [`src/trainer/base_grpo_trainer.py`](../src/trainer/base_grpo_trainer.py)
[^wan-tp]: [`src/trainer/tensor_parallel.py`](../src/trainer/tensor_parallel.py)
[^pytorch-tp]: PyTorch, "Tensor Parallelism - torch.distributed.tensor.parallel", https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html
[^fsdp2]: PyTorch, "`torch.distributed.fsdp.fully_shard`", https://docs.pytorch.org/docs/2.8/distributed.fsdp.fully_shard.html
[^checkpoint-runtime]: [`src/trainer/checkpoint_runtime.py`](../src/trainer/checkpoint_runtime.py)
[^i2v-dataset]: [`src/data/i2v_dataset.py`](../src/data/i2v_dataset.py)
[^latent-dataset]: [`src/data/vbvr_latent_dataset.py`](../src/data/vbvr_latent_dataset.py)
[^webdataset]: Hugging Face Hub docs, "WebDataset", https://huggingface.co/docs/hub/datasets-webdataset
[^dcp]: PyTorch, "Distributed Checkpoint", https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
[^checkpoint]: [`src/trainer/checkpoint.py`](../src/trainer/checkpoint.py)
[^ema]: [`src/trainer/ema.py`](../src/trainer/ema.py)
[^config]: [`src/trainer/config.py`](../src/trainer/config.py)
