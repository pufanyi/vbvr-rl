# Engineering Quality Improvements

## 1. Fix Documentation Drift At The Source

The previous root README described a JSON array of direct video records, but current raw training expects JSON configs pointing to Parquet files.[^i2v-dataset] This kind of drift is harmful because users will build the wrong data.

Recommended work:

- Treat `docs/` and `README.md` as part of the code review surface.
- Add a documentation checklist for every data/config/CLI change.
- Keep examples tied to real files under `configs/` and `scripts/`.
- Add `docs/current_limitations.md` or keep the limitations embedded in relevant docs.

## 2. Add Config Validation Beyond Pydantic Types

Pydantic currently validates some scalar ranges, such as `prompt_dropout` and `dancegrpo_timestep_selection_ratio`.[^config] Many cross-field constraints remain implicit.

Recommended validators:

- `expert_parallel=True` requires `fsdp=True`, `train_experts=both`, and even world size.
- correction with `correction_weight > 0` should require or warn on `ema_decay > 0`.
- latent configs require `dataset_size`.
- COS requires `len(cos_tau_sigma) == number_of_videos - 1`, validated early when dataset metadata is available.
- `height` and `width` must be divisible by VAE scale factor times patch size.
- GRPO expert-parallel configs must have a sampling schedule with both experts active.

## 3. Normalize CLI Override Parsing

The training CLIs auto-generate flags from Pydantic fields, but list fields such as `cos_tau_sigma` are parsed as strings when passed through CLI overrides.[^train-i2v][^train-grpo]

Recommended work:

- Use a shared config loader that handles YAML, JSON, and CLI overrides with type-aware parsing.
- Support dotted overrides such as `--set cos_tau_sigma=[0.9,0.8]`.
- Print the final resolved config on rank 0 and write it to checkpoint manifests.
- Make fish wrappers consistently handle `--nproc` and forwarded args.

## 4. Build A Two-Tier Test Suite

Current tests cover COS path shape/validation and a GPU/model consistency script.[^test-cos][^test-consistency] That is not enough for the complexity of this trainer.

Recommended tiers:

- **CPU unit tests**: config validation, COS paths, data schema scanners, checkpoint key remapping, reward registry, CLI parsing.
- **Small CUDA tests**: tiny fake transformer under FSDP2, expert-routing synchronization, DCP save/load, EMA swap.
- **Full integration tests**: Wan2.2 model parity, one training step per trainer, one eval generation sample.
- **Distributed tests**: two-rank fake expert-parallel communication and GRPO schedule checks.

## 5. Replace Duplicated Logic With Shared Utilities

Repeated logic appears in:

- `BaseTrainer` and `BaseRLTrainer`;
- MFU setup in I2V, COS, correction, and GRPO;
- raw dataset logic in `src/data/i2v_dataset.py` and precompute's `ParquetI2VDataset`;
- flow SDE transition formulas in `wan_i2v.py`, `grpo_trainer.py`, and `dancegrpo_trainer.py`.

Recommended work:

- Create shared `src/trainer/runtime.py` or focused mixins for duplicated trainer infrastructure.
- Create a single SDE transition helper that returns mean, log-probability, and noise scale.
- Create a shared raw dataset implementation used by both training and precompute.
- Create a shared MFU estimator that receives a phase plan instead of duplicating formulas.

## 6. Improve Error Messages And Recovery

Many runtime errors are already explicit, but distributed training failures are still hard to diagnose.

Recommended work:

- Prefix every distributed log with rank, local rank, expert group, and dp rank in debug mode.
- On expected config incompatibilities, fail before model load.
- On WebDataset empty shard discovery, report exact directory and glob pattern.
- On DCP load mismatch, write a short key/shape mismatch report to disk.
- On VLM evaluation errors, include sample path, task, split, and judge model in the JSONL error record.

## 7. Separate Research Defaults From Production Defaults

Some configs are experiment-specific and may be surprising as defaults: correction uses high LR LoRA with `ema_decay: 0`; SFT configs often enable expert parallel; GRPO configs use specific maze paths.[^configs]

Recommended work:

- Add `configs/templates/` with minimal safe templates.
- Add `configs/experiments/` for dated experiment configs.
- Keep machine-specific absolute paths out of templates.
- Add comments for experimental assumptions that are not generally safe.

## 8. Add Lightweight Developer Commands

Recommended commands:

```bash
.venv/bin/python -m unittest tests.test_cos_path
.venv/bin/ruff check src tests
.venv/bin/python -m src.cli.validate_dataset --config configs/train_sft_maze.yaml --limit 100
.venv/bin/python -m src.cli.inspect_checkpoint storage/checkpoints/.../checkpoint-...
```

The last two CLIs do not exist yet; they should be added because they directly reduce operator mistakes.

[^i2v-dataset]: [`src/data/i2v_dataset.py`](../../src/data/i2v_dataset.py)
[^config]: [`src/trainer/config.py`](../../src/trainer/config.py)
[^train-i2v]: [`src/cli/train_i2v.py`](../../src/cli/train_i2v.py)
[^train-grpo]: [`src/cli/train_grpo.py`](../../src/cli/train_grpo.py)
[^test-cos]: [`tests/test_cos_path.py`](../../tests/test_cos_path.py)
[^test-consistency]: [`tests/test_consistency.py`](../../tests/test_consistency.py)
[^configs]: [`configs/`](../../configs/)
