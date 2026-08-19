# VBVR-Pro RL four-rollout demo set (2026-08-18)

## Outcome

The presentation package contains 20 same-input groups and four stochastic Flow-CPS rollouts per group. It spans
DanceGRPO checkpoints 300, 600, 900, 1200, 1500, 1800, 2100, and 2200. Every rollout uses CPS noise 0.7, 30
inference steps, CFG 1.0, 512x512x81 native generation, and a recorded explicit seed.

The strict package is under
`storage/eval_out/vbvr_pro_rl_rollout_demo_final_strict_20260818/`:

- `index.html`: presentation gallery with the aggregate checkpoint curve, input, ground truth, four rollouts, scores,
  and expandable temporal audit sheets.
- `manifest.json` / `manifest.csv`: complete generation, scoring, hash, checkpoint, seed, and manual-review records.
- `verification.json`: independent second-pass score verification.
- `cases/demo_01` through `cases/demo_20`: self-contained presentation assets.
- `evidence/`: the audited 500-sample sampler/checkpoint plot, source CSV, SVG, and trend summary.

Serve the gallery over HTTP so every browser can seek the MP4 files reliably:

```bash
.venv/bin/python -m http.server 8000 \
    --directory storage/eval_out/vbvr_pro_rl_rollout_demo_final_strict_20260818
```

## Quantitative RL evidence

The selected examples are intentionally high-contrast and therefore must not be presented as an unbiased estimate of
training gain. The quantitative evidence is the complete fixed-contract 500-sample checkpoint curve bundled at the top
of the gallery:

- CPS 0.7 matched DiffSynth step-35500 baseline: **0.472177** overall.
- Best CPS 0.7 checkpoint: **step 2200, 0.547886** overall.
- Absolute gain: **+0.075709** (about **+16.0%** relative to the matched baseline).
- Latest recorded CPS 0.7 checkpoint: **step 2300, 0.543413**, still **+0.071235** above baseline.
- Trend audit: 500 samples per cell, 144 complete sampler/model cells, and zero scorer errors.

This curve supports the claim that RL improves the measured task score. The 20 selected groups explain what the reward
sees inside stochastic rollouts at intermediate model states.

## Final examples

The group score-range median is 0.519239 and the mean is 0.541948. Eleven of the 20 groups have a range of at least
0.5; the minimum retained range is 0.249396. All four native videos in every group have distinct SHA-256 hashes.

| Demo | Checkpoint | Task | Four scores | Range |
| --- | ---: | --- | --- | ---: |
| 01 | 300 | G-131 next figure by increasing size | 0.000, 0.000, 1.000, 0.000 | 1.000 |
| 02 | 300 | G-15 obstacle grid | 0.380, 0.120, 0.706, 0.997 | 0.877 |
| 03 | 2100 | G-54 connect matching colors | 1.000, 0.667, 0.133, 0.333 | 0.867 |
| 04 | 600 | G-54 connect matching colors | 0.133, 0.400, 0.400, 1.000 | 0.867 |
| 05 | 300 | O-39 maze | 0.231, 0.841, 0.688, 0.917 | 0.686 |
| 06 | 2100 | G-45 key-door maze | 0.244, 0.339, 0.324, 0.912 | 0.668 |
| 07 | 2200 | O-47 sliding puzzle | 0.309, 0.933, 0.667, 0.378 | 0.624 |
| 08 | 1200 | G-16 visit colored blocks in order | 0.350, 0.880, 0.347, 0.293 | 0.587 |
| 09 | 300 | G-41 highest-cost grid path | 0.777, 0.386, 0.945, 0.787 | 0.559 |
| 10 | 2100 | G-13 numbered waypoint path | 0.559, 0.025, 0.557, 0.561 | 0.537 |
| 11 | 900 | G-13 numbered waypoint path | 0.046, 0.547, 0.091, 0.151 | 0.502 |
| 12 | 1800 | G-202 mark wave peaks | 0.000, 0.467, 0.000, 0.290 | 0.467 |
| 13 | 1200 | O-37 light sequence | 0.595, 0.322, 0.200, 0.200 | 0.395 |
| 14 | 1800 | O-39 maze | 0.902, 0.608, 0.566, 0.525 | 0.377 |
| 15 | 2100 | G-13 numbered waypoint path | 0.047, 0.378, 0.140, 0.254 | 0.331 |
| 16 | 2200 | O-47 sliding puzzle | 0.387, 0.444, 0.578, 0.253 | 0.324 |
| 17 | 900 | G-15 obstacle grid | 0.301, 0.033, 0.343, 0.229 | 0.310 |
| 18 | 1500 | O-39 maze | 0.603, 0.910, 0.611, 0.825 | 0.307 |
| 19 | 600 | G-16 visit colored blocks in order | 0.137, 0.319, 0.247, 0.442 | 0.305 |
| 20 | 600 | G-47 multiple-key maze | 0.356, 0.119, 0.106, 0.181 | 0.249 |

The final set emphasizes difficult temporal/spatial behavior: three ordinary mazes, two key/door mazes, eight ordered
or optimization grid-navigation cases, and two sliding puzzles. Selection, connection, wave, and light tasks remain
because their reward differences are especially easy to explain on a slide.

## Score correctness checks

The score claim has three independent layers:

1. The 480 original candidate videos plus eight targeted G-15 replacement candidates were scored with the pinned e140
   EvalKit source contract `4cc7d028d4106a28190a63bc179562d5ac9add9263cb71926dd6385c5714bcf8`; all runs completed
   with zero scorer errors.
2. High-ranking candidate groups and the final composite replacement received manual five-timepoint review against
   their ground-truth videos. The final selection records a case-specific visual justification and includes the review
   sheet beside each demo.
3. The final 80 prepared videos were staged into fresh canonical trees and scored again. `verification.json` reports
   **80/80 exact score matches**, maximum absolute delta **0**, zero scorer errors, and the same EvalKit fingerprint.

The three strict replacements received additional task-level checks. For G-15, rollout 0 was refreshed at the user's
request with seed 2026086101: its left-side ascent and cross-grid route scores 0.380, while the other three remain at
0.120/0.706/0.997; rollout 2 covers about 78.6% of the optimal path and rollout 3 covers 100%. For G-41, the best
rollout legally visits 15 cells and collects cost 360 of the optimal 380, while the lowest visits seven cells,
collects 160, and has six illegal transitions; their scores are 0.945 and 0.386. The four O-47 videos have visibly
different move sequences and final boards, with the nearly solved board scoring 0.933.

The scorer inputs are the packaged 1024x1024 videos, while the gallery shows their corresponding native 512x512
generations. The manifest binds both versions by SHA-256. A final media audit decoded all 160 native/scored files as
81-frame, 16-FPS videos at the expected resolutions and resolved all 141 local gallery links.

## Inference and reproducibility notes

Checkpoints 300, 600, and 900 used their readable converted Diffusers trees. Several later converted trees were
structurally complete but had root-owned, mode-600 component shards. Those checkpoints were loaded from the readable
official TI2V-5B base plus the original DCP checkpoint instead; every load reported zero missing and zero unexpected
keys. `manifest.json` records `converted` or `base_plus_dcp` for every case.

The reusable workflow is `src.cli.vbvr_rl_demo` (`build`, `stage`, `aggregate`, `audit`, `package`, and `verify`). The
final selection, including all manual notes, is
`scripts/eval/vbvr_pro/dancegrpo_manifest_rl_512x512x81/rl_demo_final_selection.json`.
