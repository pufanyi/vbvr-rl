# Evaluation Scripts

The release-supported VBVR path is `vbvr_pro/`. It evaluates exact VBVR-Pro
manifests and keeps conversion, generation, media preparation, scoring, and
provenance in one auditable workflow.

## Directory Layout

| Directory | Status and purpose |
| --- | --- |
| `vbvr_pro/` | Supported VBVR-Pro `main_v2` pipeline, checkpoint/sampler wrappers, summaries, and optional result viewers |
| `maze/` | Synthetic maze evaluation utilities |
| `lmms/` | Optional lmms-eval/FastVideo integration |

## VBVR-Pro Entry Point

The shared launcher is:

```fish
fish scripts/eval/vbvr_pro/vbvr_pro_5b_main_v2.fish
```

Inspect a run first:

```bash
DRY_RUN=1 \
CHECKPOINT=storage/checkpoints/<run>/checkpoint-100 \
BASE_MODEL=storage/models/Wan2.2-TI2V-5B-Diffusers \
GT_BASE=storage/datasets/vbvr-pro-eval-500 \
EVALKIT_DIR=storage/evalkits/<compatible-checkout> \
OUTPUT_ROOT=storage/eval_out/<run>/checkpoint-100 \
fish scripts/eval/vbvr_pro/vbvr_pro_5b_main_v2.fish
```

The launcher requires a separately obtained evaluator. It verifies both an
exact Git revision, when available, and a complete source-contract digest.
There is no vendored evaluator fallback.

See [`docs/vbvr_pro_eval.md`](../../docs/vbvr_pro_eval.md) for environment
variables, stage contracts, resume behavior, and completion criteria.

## Wrappers and Sweeps

Subdirectories under `vbvr_pro/` contain experiment-specific wrappers. A
wrapper should only select checkpoint, model, sampler, media, manifest,
evaluator, and output variables before delegating to the shared launcher.

When adding a wrapper:

- use a unique `CONVERTED_MODEL` and `OUTPUT_ROOT` for each evaluation cell;
- encode the checkpoint and sampler in its name;
- keep the split manifest, preparation, and scorer contract explicit;
- support `DRY_RUN=1` through the shared launcher;
- do not duplicate generation or scorer implementation.

Summarize completed result trees with:

```fish
fish scripts/eval/vbvr_pro/summarize_vbvr_pro_results.fish \
  --root storage/eval_out/<result-root>
```

## VLM Judge

The optional offline Qwen judge reads completed generated-video cells and
writes a separate resumable result root. Its convenience launcher is under:

```text
vbvr_pro/dancegrpo_vlm_qwen36_512x512x81/
```

It does not replace the rule evaluator and must not share output files with
rule scoring. See
[`docs/vlm_judge_reward.md`](../../docs/vlm_judge_reward.md).

## Runtime Check

Before rule-based evaluation:

```bash
.venv/bin/python -m src.eval.vbvr_runtime
```

Training reward and offline scoring intentionally use the same pinned
scientific-media runtime. If the contract changes, update both paths and write
results to a new provenance namespace.
