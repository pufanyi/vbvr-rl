# VBVR lmms-eval Notes

This note documents the checkpoint conversion and VBVR-Bench evaluation path used by
`scripts/convert/dcp_to_diffusers.fish` and `scripts/eval/lmms_eval.fish`.

## Pipeline

Evaluation is a two-step process:

1. Convert a DCP training checkpoint into a regular Diffusers model directory.
2. Run `lmms-eval` with the `fastvideo` model backend on the `vbvr` task.

The VBVR task is rule-scored by lmms-eval. The model generates one video per
VBVR sample, lmms-eval parses the generated video path, compares it against the
VBVR ground-truth assets, and aggregates metrics such as `vbvr_overall`,
`vbvr_in_domain`, `vbvr_out_of_domain`, and category scores.

## DCP to Diffusers Conversion

`scripts/convert/dcp_to_diffusers.fish` should be run from the Wan-Trainer repo
root. It sources `scripts/lib/env.fish`, activates the repo `.venv`, and sets
`PYTHONPATH`.

Important defaults:

- `CHECKPOINT_ROOT=storage/checkpoints`
- `OUTPUT_ROOT=storage/models/dcp_converted`
- `BASE_MODEL=storage/models/Wan2.2-I2V-A14B-Diffusers`
- `DEVICE=cuda`
- `TORCH_DTYPE=bfloat16`
- `USE_EMA=1`
- `MERGE_LORA=1`
- `SAFE_SERIALIZATION=1`

The script treats a checkpoint as DCP if it has either a top-level `.metadata`
or both `high/.metadata` and `low/.metadata`. For high/low MoE checkpoints it
requires both halves before conversion. The output directory is considered
complete only when it contains `model_index.json`.

To convert one checkpoint:

```bash
CHECKPOINTS=storage/checkpoints/sft_vbvr_fixed_1e-6/checkpoint-4000 \
DEVICE=cuda:0 \
fish scripts/convert/dcp_to_diffusers.fish
```

The Python converter loads the base Diffusers pipeline, loads DCP weights with
`load_dcp_into_pipeline`, prefers EMA weights when present, optionally merges
LoRA adapters, writes safetensors shards, and applies FastVideo compatibility
cleanup.

Observed on 2026-04-29:

- Input checkpoint: `storage/checkpoints/sft_vbvr_fixed_1e-6/checkpoint-4000`
- Converted model: `storage/models/dcp_converted/sft_vbvr_fixed_1e-6_checkpoint-4000`
- Converted size: about 65 GB
- Conversion finished successfully and wrote `model_index.json`
- The original DCP checkpoint was about 426 GB because it also includes
  optimizer shards; the converter only needs DCP model/EMA state, not the
  separate optimizer shard files.

The first sandboxed CUDA run failed with `cudaGetDeviceCount()` error 304. Run
conversion with normal CUDA access.

## lmms-eval VBVR Run

`scripts/eval/lmms_eval.fish` changes directory to
`/mnt/umm/users/pufanyi/workspace/lmms-eval` and executes:

```bash
.venv/bin/python -m lmms_eval eval \
  --model fastvideo \
  --tasks vbvr \
  --batch_size 1 \
  --log_samples \
  --output_path=$OUTPUT_DIR
```

The script requires `MODEL_DIR`. Generated videos are written under:

```text
$OUTPUT_DIR/generated_videos/<basename MODEL_DIR>/
```

For local VBVR assets, it sets:

```text
VBVR_GT_PATH=/mnt/umm/users/pufanyi/workspace/Wan-Trainer/storage/datasets/VBVR-Bench
```

Default generation settings:

- `DATA_PARALLEL=8`
- `NUM_GPUS=1`
- `SP_SIZE=1`
- `TP_SIZE=1`
- `NUM_INFERENCE_STEPS=50`
- `NUM_FRAMES=81`
- `HEIGHT=384`
- `WIDTH=384`
- `FPS=16`
- `ENABLE_TORCH_COMPILE=True`

On a 4 GPU machine, override `DATA_PARALLEL`:

```bash
MODEL_DIR=/mnt/umm/users/pufanyi/workspace/Wan-Trainer/storage/models/dcp_converted/sft_vbvr_fixed_1e-6_checkpoint-4000 \
DATA_PARALLEL=4 \
OUTPUT_DIR=/mnt/umm/users/pufanyi/workspace/Wan-Trainer/storage/lmms_eval \
fish scripts/eval/lmms_eval.fish
```

Keep `DATA_PARALLEL * NUM_GPUS` less than or equal to the number of visible GPUs.
The script defaults to 8-way data parallelism because it was written for 8x H100.

## Output Layout

Generated videos are stored in VBVR split/task folders, for example:

```text
storage/lmms_eval/generated_videos/sft_vbvr_fixed_1e-6_checkpoint-4000/
  In-Domain_50/
    G-131_select_next_figure_increasing_size_sequence_data-generator/
      00000.mp4
      00001.mp4
```

When a full run finishes, lmms-eval writes aggregate metrics under
`$OUTPUT_DIR` and detailed VBVR submission/evaluation JSON under
`$OUTPUT_DIR/submissions/`, including `vbvr_eval_results.json`.

The full `vbvr` task has 500 samples:

- 250 In-Domain samples
- 250 Out-of-Domain samples
- 100 tasks with 5 instances each

## 2026-04-29 Partial Run

The run was started with:

```bash
MODEL_DIR=/mnt/umm/users/pufanyi/workspace/Wan-Trainer/storage/models/dcp_converted/sft_vbvr_fixed_1e-6_checkpoint-4000 \
DATA_PARALLEL=4 \
OUTPUT_DIR=/mnt/umm/users/pufanyi/workspace/Wan-Trainer/storage/lmms_eval \
fish scripts/eval/lmms_eval.fish
```

It was stopped by request before completion. No final VBVR metrics were
produced.

Observed behavior:

- FastVideo spawned 4 workers on GPU groups `0`, `1`, `2`, and `3`.
- The converted model loaded successfully in all workers.
- lmms-eval linked the local cached `Video-Reason/VBVR-Bench-Data` snapshot and
  built 500 VBVR requests.
- Hugging Face network HEAD checks emitted warnings when the network was
  unreachable, but the local cached dataset still loaded.
- Actual sample settings in FastVideo logs were 384x384, 81 frames, 50 inference
  steps, fps 16, seed 42.
- The first batch of 4 videos took about 121 to 122 seconds per worker, including
  first-use overhead.
- 8 partial videos were generated before termination.
- GPU memory reached about 67 GB per H100 during generation.

The generated partial videos remain under:

```text
storage/lmms_eval/generated_videos/sft_vbvr_fixed_1e-6_checkpoint-4000/
```

The lmms-eval and FastVideo processes were terminated with `SIGTERM`, and GPUs
were confirmed idle afterward.

## Practical Tips

- Always convert DCP checkpoints to Diffusers before using `lmms_eval.fish`;
  FastVideo expects a regular model directory with `model_index.json`.
- Use `DATA_PARALLEL=4` on the current 4 GPU setup. Leave the default only on an
  8 GPU setup.
- Do not assume a final score exists unless `lmms-eval` exits normally. Partial
  generated videos are not enough for aggregate metrics.
- If a run is interrupted, check for remaining `lmms_eval` or `fastvideo`
  processes and verify GPU memory with `nvidia-smi`.
- Network warnings from Hugging Face are acceptable only if the local dataset
  cache is available and the `test` split is generated successfully.
