# VBVR-Pro RL additional 30 four-rollout demos (2026-08-18)

## Outcome

This package adds 30 same-input groups to the original strict 20-group presentation set. The 30 canonical samples are
disjoint from the first set, and every group contains four stochastic Flow-CPS rollouts from one intermediate
DanceGRPO checkpoint. All 120 rollouts use CPS noise 0.7, 30 inference steps, CFG 1.0, 512x512x81 native generation,
16 FPS, and an explicit recorded seed.

The presentation package is under
`storage/eval_out/vbvr_pro_rl_rollout_demo_additional30_strict_20260818/`:

- `index.html`: self-contained gallery with the aggregate checkpoint curve, inputs, ground truth, four native
  rollouts, scores, and expandable temporal audit sheets.
- `manifest.json` / `manifest.csv`: checkpoint, task, seed, score, hash, provenance, and case-specific manual-review
  records.
- `verification.json`: independent second-pass verification of all 120 scores.
- `cases/demo_01` through `cases/demo_30`: native and scored videos plus their input, ground truth, and audit sheet.
- `evidence/`: the audited 500-sample sampler/checkpoint trend used as the quantitative RL evidence.

Serve the gallery over HTTP so browsers can seek all MP4 files reliably:

```bash
.venv/bin/python -m http.server 8000 \
    --directory storage/eval_out/vbvr_pro_rl_rollout_demo_additional30_strict_20260818
```

## Quantitative RL evidence

These examples were deliberately selected for clear within-group contrast, so they are qualitative explanations and
not an unbiased estimate of model quality. The quantitative claim remains the complete fixed-contract 500-sample
checkpoint curve bundled at the top of the gallery:

- Matched DiffSynth step-35500 baseline at CPS 0.7: **0.472177** overall.
- Best CPS 0.7 checkpoint: **step 2200, 0.547886** overall.
- Absolute gain: **+0.075709**, about **+16.0%** relative to the matched baseline.
- Latest recorded checkpoint: **step 2300, 0.543413**, still **+0.071235** above baseline.
- Curve audit: 500 samples per cell, 144 complete sampler/checkpoint cells, and zero scorer errors.

## Selection summary

The added set spans checkpoints 300, 600, 900, 1200, 1500, 1800, 2100, and 2200 and covers 15 task types. Ten of
the 30 groups are maze or ordered-grid planning tasks. The remaining groups include sliding puzzles, object merging
and counting, ball-eating dynamics, light sequences, color mixing, communicating vessels, deletion, and visual
selection.

The within-group score-range minimum is **0.261660**, median **0.403147**, mean **0.433035**, and maximum **1.0**.
Fifteen groups have a range of at least 0.4, and 22 have a range of at least 0.3. Every group has four distinct native
video SHA-256 hashes.

| Demo | Checkpoint | Task | Four scores | Range |
| --- | ---: | --- | --- | ---: |
| 01 | 1800 | G-161 mark second-largest shape | 0.000, 1.000, 0.000, 0.000 | 1.000 |
| 02 | 1200 | O-37 light sequence | 0.649, 0.290, 0.288, 1.000 | 0.712 |
| 03 | 300 | O-5 symbol deletion | 0.356, 1.000, 0.400, 0.356 | 0.644 |
| 04 | 2100 | O-37 light sequence | 0.200, 0.835, 0.200, 0.200 | 0.635 |
| 05 | 1200 | G-16 visit specified block | 0.308, 0.680, 0.589, 0.907 | 0.599 |
| 06 | 900 | O-29 merge and count balls | 0.419, 0.781, 0.975, 0.695 | 0.556 |
| 07 | 2200 | O-29 merge and count balls | 0.340, 0.694, 0.172, 0.472 | 0.522 |
| 08 | 600 | G-15 obstacle grid | 0.511, 0.317, 0.776, 0.819 | 0.502 |
| 09 | 600 | O-31 ball eating | 0.975, 0.792, 0.482, 0.842 | 0.493 |
| 10 | 2200 | O-37 light sequence | 0.200, 0.667, 0.200, 0.200 | 0.467 |
| 11 | 2200 | O-2 subtractive color mixing | 0.400, 0.765, 0.555, 0.861 | 0.461 |
| 12 | 600 | O-75 communicating vessels | 0.503, 0.891, 0.572, 0.440 | 0.451 |
| 13 | 1200 | O-47 sliding puzzle | 0.573, 0.272, 0.133, 0.441 | 0.440 |
| 14 | 600 | G-15 obstacle grid | 0.473, 0.167, 0.189, 0.039 | 0.434 |
| 15 | 900 | O-31 ball eating | 0.537, 0.544, 0.379, 0.800 | 0.421 |
| 16 | 300 | G-45 key-door maze | 0.688, 0.612, 0.304, 0.690 | 0.385 |
| 17 | 1200 | O-29 merge and count balls | 0.773, 0.460, 0.399, 0.625 | 0.374 |
| 18 | 2100 | O-31 ball eating | 0.887, 0.519, 0.537, 0.706 | 0.369 |
| 19 | 1800 | O-47 sliding puzzle | 0.333, 0.267, 0.133, 0.476 | 0.343 |
| 20 | 1800 | G-16 visit specified block | 0.087, 0.420, 0.394, 0.368 | 0.333 |
| 21 | 900 | O-47 sliding puzzle | 0.267, 0.227, 0.267, 0.552 | 0.326 |
| 22 | 1500 | G-47 multiple-key maze | 0.425, 0.120, 0.162, 0.103 | 0.323 |
| 23 | 900 | O-31 ball eating | 0.569, 0.769, 0.523, 0.474 | 0.294 |
| 24 | 1800 | G-13 numbered-waypoint grid | 0.517, 0.513, 0.529, 0.803 | 0.290 |
| 25 | 2200 | G-45 key-door maze | 0.663, 0.942, 0.696, 0.778 | 0.279 |
| 26 | 1200 | O-29 merge and count balls | 0.592, 0.532, 0.317, 0.381 | 0.274 |
| 27 | 900 | O-2 subtractive color mixing | 0.651, 0.803, 0.856, 0.924 | 0.273 |
| 28 | 2200 | O-39 maze | 0.980, 0.714, 0.957, 0.946 | 0.266 |
| 29 | 900 | O-39 maze | 0.328, 0.593, 0.429, 0.481 | 0.265 |
| 30 | 600 | O-16 additive color mixing | 0.539, 0.564, 0.581, 0.800 | 0.262 |

## Score and media correctness

The additional search generated and scored 736 fresh candidate videos across two broad 80-group rounds and one
targeted 24-group round. The final set also reuses strong, previously scored candidates that did not overlap the
original 20 examples. Every selected group received a five-timepoint visual comparison against its ground-truth
video; the package stores both the audit sheet and a case-specific explanation of why the four outcomes and scores
are credible.

The final 120 prepared videos were then staged into fresh canonical EvalKit trees and scored independently. The
packaged `verification.json` reports:

- **120/120 exact score matches**;
- maximum absolute score delta **0**;
- zero scorer errors;
- EvalKit source fingerprint
  `4cc7d028d4106a28190a63bc179562d5ac9add9263cb71926dd6385c5714bcf8`;
- scorer runtime fingerprint
  `49ea34669caaef54d82d86f93a21acf3c012fbb07994a4b0ab60f2c2d820cc2e`.

The gallery displays native 512x512 generations, while EvalKit scores the corresponding 1024x1024 prepared videos.
The manifest binds both versions by SHA-256. A final package audit found no missing files or hash mismatches, resolved
all gallery links, and decoded/probed all 240 native/scored assets as 81-frame, exact-16-FPS videos at their expected
resolutions.

## Reproducibility

The final reviewed selection is
`scripts/eval/vbvr_pro/dancegrpo_manifest_rl_512x512x81/rl_demo_additional30_final_selection.json`. Candidate building,
staging, aggregation, audit-sheet rendering, packaging, and exact independent verification use
`src.cli.vbvr_rl_demo`.

Some later converted Diffusers checkpoint trees were not reliably readable through `safe_open`. For those cases,
inference loaded the official local TI2V-5B base and the original DCP checkpoint; every successful DCP load reported
zero missing and zero unexpected transformer keys. The generated videos, seeds, source cases, load modes, scorer
contract, and manual review records are preserved in the candidate and final manifests.
