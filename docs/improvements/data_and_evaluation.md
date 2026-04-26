# Data And Evaluation Improvements

## 1. Add Latent Provenance Metadata

Latent shards currently store tensors and lightweight JSON metadata, but the training loader does not enforce provenance compatibility.[^latent-dataset][^i2v-latent-precompute]

Recommended metadata per shard:

- source repository commit hash;
- `model_path` and model config hash;
- VAE config hash and normalization constants;
- text encoder/tokenizer config hash;
- prompt-cleaning function version;
- `num_frames`, height, width, FPS, VAE scale factors;
- COS chain length and `num_latents`;
- generation/precompute command line.

Why it matters: latent datasets become stale silently when VAE normalization, prompt cleaning, resolution, or chain semantics change.

## 2. Validate Parquet And WebDataset Schemas Before Training

`I2VDataset` fails during item load if media paths or columns are bad.[^i2v-dataset] For long distributed runs, that is too late.

Recommended work:

- Add `python -m src.cli.validate_dataset --config ...`.
- Validate required columns, path existence, video decodability, frame count, prompt type, and image fallback.
- For COS, validate that all rows have the same chain length expected by `cos_tau_sigma`.
- For latent WebDataset, scan shard headers for required tensor keys and shape consistency.
- Produce a machine-readable report with sample counts per source dataset.

## 3. Make Dataset Size Discoverable

Latent configs require `dataset_size`, but this number can drift from the actual tar-shard contents.

Recommended work:

- Prefer `dataset_info.json` when present.
- Add a fast shard-count scanner that reads JSON members without loading safetensors tensors.
- Fail if configured `dataset_size` differs from shard metadata unless an explicit override is provided.
- Store per-rank epoch length in logs.

## 4. Improve VBVR Evaluation Reproducibility

`eval_i2v` uses a rank-local generator seeded once, then skips existing files. This makes resumed generation sensitive to which files already exist.[^eval-i2v]

Recommended work:

- Derive a deterministic seed per sample from `(global_seed, sample_id)`.
- Write a sidecar JSON next to every generated video with prompt, image path, checkpoint, seed, inference steps, guidance scale, and resolution.
- Add a `--force` option and a `--dry_run` option.
- Add a manifest at the output root so scoring can verify generation completeness.

## 5. Calibrate VLM Scoring

The current VLM judge uses one prompt, one model, and one sampled frame set.[^vlm-judge] It is useful for quick iteration but should not be treated as ground truth without calibration.

Recommended work:

- Build a 100-300 sample human-labeled calibration set.
- Measure VLM agreement with human labels by domain and task.
- Test score sensitivity to sampled frame count, frame order, inclusion of first frame, and judge model.
- Add pairwise A/B judging for checkpoint comparisons.
- Version `_JUDGE_SYSTEM` and write the prompt version into every result.

## 6. Unify Rule And VLM Outputs

The evaluation script supports both rule scoring and VLM scoring, but the paths are operationally separate.[^vbvr-script]

Recommended work:

- Define one internal `SampleScore` schema for both rule and VLM results.
- Write all scorers into the same `scores.rank*.jsonl` format.
- Include scorer type, scorer version, and task split in every score record.
- Add an aggregation CLI that can merge multiple scorers and report disagreements.

## 7. Make Synthetic Maze Data More Diagnostic

The maze generator exports rich metadata, but training and evaluation do not yet expose enough dataset diagnostics.[^maze-webdataset]

Recommended work:

- Log path length distributions, maze sizes, color palettes, and start/goal distances.
- Add held-out maze families by geometry and palette.
- Add adversarial palettes that make RGB ball detection harder.
- Generate negative examples for reward debugging.
- Add visual sample sheets for every generated shard.

[^latent-dataset]: [`src/data/vbvr_latent_dataset.py`](../../src/data/vbvr_latent_dataset.py)
[^i2v-latent-precompute]: [`src/precompute/i2v_latent_webdataset.py`](../../src/precompute/i2v_latent_webdataset.py)
[^i2v-dataset]: [`src/data/i2v_dataset.py`](../../src/data/i2v_dataset.py)
[^eval-i2v]: [`src/cli/eval_i2v.py`](../../src/cli/eval_i2v.py)
[^vlm-judge]: [`src/eval/vbvr/judges/vlm.py`](../../src/eval/vbvr/judges/vlm.py)
[^vbvr-script]: [`scripts/eval/vbvr_generate_score.fish`](../../scripts/eval/vbvr_generate_score.fish)
[^maze-webdataset]: [`src/precompute/maze_webdataset.py`](../../src/precompute/maze_webdataset.py)
