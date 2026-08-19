# VBVR-Pro Evaluation Launchers

This directory exposes a small, release-supported interface for VBVR-Pro
evaluation. Model paths, checkpoints, samplers, and output namespaces are
arguments; they are not encoded in experiment-specific script names.

## Files

| File | Purpose |
| --- | --- |
| `run.fish` | Evaluate one model with UniPC, Euler, or Flow-CPS |
| `sweep.fish` | Expand one model into several sampler cells, optionally sharded across machines |
| `summarize.fish` | Verify and summarize complete rule-score cells |
| `vlm_judge.fish` | Optionally judge generated-video cells with Qwen |
| `lib/rule_pipeline.fish` | Internal conversion, generation, preparation, scoring, and provenance implementation |

Do not invoke or copy `lib/rule_pipeline.fish` directly. Add stable options to
`run.fish` when the public contract needs to grow. A new checkpoint, sampler
coefficient, resolution, or output location does not need a new wrapper.

## One Rule-Evaluation Cell

Evaluate a DCP checkpoint with UniPC:

```fish
fish scripts/eval/vbvr_pro/run.fish \
  --checkpoint storage/checkpoints/<run>/checkpoint-100 \
  --converted-model storage/models/converted/<run>-checkpoint-100 \
  --output-root storage/eval_out/<run>/checkpoint-100/unipc \
  --sampler unipc
```

Evaluate an existing Diffusers model with Euler:

```fish
fish scripts/eval/vbvr_pro/run.fish \
  --model storage/models/<model> \
  --output-root storage/eval_out/<model>/euler \
  --sampler euler
```

Flow-CPS requires an explicit coefficient:

```fish
fish scripts/eval/vbvr_pro/run.fish \
  --model storage/models/<model> \
  --output-root storage/eval_out/<model>/cps-noise-0.7 \
  --sampler cps \
  --cps-noise 0.7
```

Add `--dry-run` to resolve the model, sampler, evaluator, media contract, and
output namespace without reading model or dataset artifacts. Run
`fish scripts/eval/vbvr_pro/run.fish --help` for every supported option.

## Sampler Sweep

`sweep.fish` owns the sampler and output-root arguments. Place all options for
`run.fish` after `--`:

```fish
fish scripts/eval/vbvr_pro/sweep.fish \
  --output-base storage/eval_out/<model> \
  --samplers unipc,euler,cps:0.3,cps:0.7 \
  -- \
  --model storage/models/<model> \
  --steps 30 \
  --guidance-scale 1.0
```

Cells run sequentially on each machine. To distribute them, pass identical
arguments everywhere and set `--world-size` and `--rank`, or provide the
equivalent `WORLD_SIZE` and `RANK` environment variables. Assignment is a
deterministic round robin over the declared sampler order. Inspect it first
with `--assignment-only`.

For a checkpoint sweep, invoke the same generic command once per checkpoint
with a distinct conversion path and output base. Conversion output can be
reused safely by all sampler cells because the pipeline locks and fingerprints
it.

## Summaries and Optional VLM Judging

Summarize one run or a parent containing sampler cells:

```fish
fish scripts/eval/vbvr_pro/summarize.fish \
  --root storage/eval_out/<model> \
  --expected-samples 500
```

The command verifies score provenance and refuses to mix evaluator source
fingerprints. It writes an Excel workbook, JSON summary, and `final_scores.txt`
under `<root>/reports` unless `--output-dir` is supplied.

The optional Qwen judge consumes generated-video cells without modifying rule
scores:

```fish
fish scripts/eval/vbvr_pro/vlm_judge.fish score \
  --input-root storage/eval_out/<model> \
  --output-root storage/eval_out/<model>-vlm-judge
```

See [`docs/vbvr_pro_eval.md`](../../../docs/vbvr_pro_eval.md) for artifact
requirements, provenance semantics, resume rules, and publication checks.
