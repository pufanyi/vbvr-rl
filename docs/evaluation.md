# Evaluation

Evaluation has two phases: generate videos from prompts/images, then score generated videos with either the in-repo VLM evaluator or the bundled VBVR-EvalKit compatibility path.

## Batch I2V Generation

`src.cli.eval_i2v` loads a Diffusers `WanImageToVideoPipeline`, optionally loads a DCP checkpoint into the pipeline, and generates one `.mp4` per input record.[^eval-i2v]

Input JSON format:

```json
[
  {
    "name": "sample_0001",
    "image": "path/to/reference.png",
    "prompt": "A short task prompt."
  },
  {
    "name": "sample_0002",
    "video": "path/to/source.mp4",
    "prompt": "If image is absent, first video frame is used."
  }
]
```

The generator computes output resolution from image aspect ratio and `max_area`, rounded to the pipeline's VAE/patch multiple, or accepts an explicit `--height` and `--width`. Multi-GPU mode partitions records round-robin by rank. Seeds are derived from the global sample index, and videos are validated for resolution, frame count, and FPS before an atomic rename, so changing rank count or resuming around existing outputs does not change other samples. Fixed-resolution runs can use `--validate_only` to verify the exact expected path set without initializing CUDA or loading a model. Model loading can be parallel or rank-serialized to reduce host RAM spikes.[^eval-i2v]

Example:

```bash
.venv/bin/torchrun --nproc_per_node=8 -m src.cli.eval_i2v \
  --eval_json storage/eval_out/vbvr/vbvr_eval.json \
  --output_dir storage/eval_out/vbvr/checkpoint-4000 \
  --checkpoint storage/checkpoints/sft_vbvr_fixed/checkpoint-4000 \
  --use_ema
```

## DCP Loading For Evaluation

When `--checkpoint` is provided, `eval_i2v` calls `load_dcp_into_pipeline`. That function:

1. detects LoRA sidecars and wraps pipeline transformers if needed;
2. detects flat vs high/low checkpoint layout;
3. reads DCP into CPU tensors;
4. prefers EMA shadows when `--use_ema` is set;
5. remaps plain/LoRA key layouts into the current pipeline module;
6. loads the result with `strict=False` to tolerate adapter keys.[^checkpoint]

## VBVR VLM Scoring

`src.cli.eval_vbvr` expects generated videos under:

```text
<model_output>/
  Open_60/
    <task>/<idx>.mp4
  Hidden_40/
    <task>/<idx>.mp4
```

It discovers matching ground-truth records from `data/vbvr/VBVR-Bench`, builds `EvalSample` objects, loads a VLM judge, and writes resumable per-rank `scores.rank*.jsonl` shards plus rank-0 summaries.[^eval-vbvr][^vbvr-dataset][^vbvr-runner]

The VLM judge shows the task prompt, the expected final frame, optionally the starting frame, and uniformly sampled generated frames. It asks the model for strict JSON, then parses a 0-10 score into `[0, 1]`.[^vlm-judge]

Example:

```bash
.venv/bin/torchrun --nproc_per_node=8 -m src.cli.eval_vbvr \
  --model_output storage/eval_out/vbvr/checkpoint-4000 \
  --gt_base data/vbvr/VBVR-Bench \
  --output_dir storage/eval_out/vbvr_vlm \
  --judge_model google/gemma-4-26B-A4B-it \
  --num_frames 6
```

## Rule-Based VBVR-EvalKit Path

`scripts/eval/vbvr_generate_score.fish` automates a common checkpoint loop:

1. build an eval JSON if missing;
2. generate videos with `src.cli.eval_i2v`;
3. either run `src.cli.eval_vbvr` (`JUDGE=vlm`) or restructure outputs and call the rule scorer (`JUDGE=rule`).[^vbvr-script]

The rule path uses:

- `src.eval.vbvr_restructure_to_evalkit` to convert generation output into the layout expected by the third-party kit;
- `src.eval.vbvr_run_evaluation_parallel` for parallel rule scoring, with `--evalkit_dir` selecting the exact scorer checkout and `--expected_videos` enforcing completeness.[^vbvr-restructure][^vbvr-rule]

The current VBVR-Pro 5B workflow has a dedicated [main_v2 evaluation guide](vbvr_pro_eval.md). It covers the 500-sample manifest contract, eight-GPU native Diffusers generation, frame-preserving resize/retime preparation, and latest rule scorer.

## Output Files

The VLM runner writes:

```text
<output_dir>/<model_name>/
  scores.rank0.jsonl
  scores.rank1.jsonl
  ...
  eval_results.json
  summary.json
```

`eval_results.json` contains every `SampleScore`, plus `In_Domain`, `Out_of_Domain`, and overall summaries. `summary.json` contains only headline aggregate fields.[^vbvr-runner]

## Current Evaluation Limitations

- The VLM judge is not calibrated against a human-labeled validation set.
- The prompt in `_JUDGE_SYSTEM` asks for a short reasoning string, but no consistency or self-checking pass is performed.
- The rule path and VLM path are separate output formats until the script normalizes them.
- VLM scoring caches JSONL shards, but malformed/torn lines are only skipped, not repaired.

[^eval-i2v]: [`src/cli/eval_i2v.py`](../src/cli/eval_i2v.py)
[^checkpoint]: [`src/trainer/checkpoint.py`](../src/trainer/checkpoint.py)
[^eval-vbvr]: [`src/cli/eval_vbvr.py`](../src/cli/eval_vbvr.py)
[^vbvr-dataset]: [`src/eval/vbvr/dataset.py`](../src/eval/vbvr/dataset.py)
[^vbvr-runner]: [`src/eval/vbvr/runner.py`](../src/eval/vbvr/runner.py)
[^vlm-judge]: [`src/eval/vbvr/judges/vlm.py`](../src/eval/vbvr/judges/vlm.py)
[^vbvr-script]: [`scripts/eval/vbvr_generate_score.fish`](../scripts/eval/vbvr_generate_score.fish)
[^vbvr-restructure]: [`src/eval/vbvr_restructure_to_evalkit.py`](../src/eval/vbvr_restructure_to_evalkit.py)
[^vbvr-rule]: [`src/eval/vbvr_run_evaluation_parallel.py`](../src/eval/vbvr_run_evaluation_parallel.py)
