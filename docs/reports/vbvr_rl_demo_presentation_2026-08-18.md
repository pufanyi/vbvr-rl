# VBVR-Pro RL embedded-video presentation (2026-08-18)

## Outcome

`scripts/eval/vbvr_pro/build_rl_demo_ppt.py` builds two self-contained 16:9 PowerPoint decks from the reviewed strict
20-case and additional-30-case packages:

- `storage/presentations/vbvr_rl_demo_20260818/vbvr_rl_demo_talk_embedded_video_20260818.pptx`: 17 slides, 10 main
  cases, and 40 embedded native MP4 rollouts.
- `storage/presentations/vbvr_rl_demo_20260818/vbvr_rl_demo_full_50_embedded_video_20260818.pptx`: 58 slides, all 50
  reviewed cases, and 200 embedded native MP4 rollouts.

Every case slide holds the input and ground-truth final frame fixed, exposes the exact checkpoint, seed, and reward,
and places the four videos side by side. The pre-play poster for each MP4 is the raw video's exact decoded frame zero,
with no labels or compositing. Presenter notes preserve the complete prompt, manual audit, seeds, scores, and native
hashes.

The ten main examples cover discrete reasoning, obstacle-grid planning, second-largest selection, light sequences,
sliding puzzles, highest-cost paths, key-door planning, object merging and counting, ball-eating dynamics, and
communicating-vessel physics. The remaining 40 reviewed groups are dynamic appendix slides rather than static index
rows.

## Presentation claim

The deck deliberately separates two kinds of evidence:

- Quantitative: the fixed-contract 500-sample curve improves from 0.472177 to 0.547886 at CPS 0.7, an absolute gain
  of 0.075709 (about 16.0%), with 144 complete cells and zero scorer errors.
- Qualitative: the same-condition, four-seed groups show the semantically different behaviors and correct relative
  rewards that create a GRPO learning signal.

The selected groups are high-contrast examples and are not an unbiased estimate of average model quality. They also
use different samples across checkpoints, so they must not be described as one fixed sample evolving through the
training timeline.

## Validation

The builder reopens both decks, tests each PPTX ZIP package, verifies the PowerPoint `videoFile` and `p14:media`
elements, requires the `video/mp4` content type, and compares every embedded media SHA-256 against its source native
video. The final report is
`storage/presentations/vbvr_rl_demo_20260818/build_report.json`.

The talk deck contains 40 unique source-matching MP4 parts. The complete deck contains 200 unique source-matching MP4
parts. All shape bounds were checked against the 16:9 canvas, all 50 case slides contain four media objects, and the
corrected obstacle-grid example includes rollout 0 seed `2026086101` with rewards 0.380, 0.120, 0.706, and 0.997.

The deck was rasterized with an isolated renderer for visual inspection of the cover, method, quantitative curve,
reading guide, representative case pages, claim boundary, and evidence slide. The renderer's evaluation watermark is
not stored in either PPTX. Explicit East Asian typeface declarations use `Noto Sans CJK SC` to avoid missing-glyph
fallbacks during cross-platform rendering.

Desktop PowerPoint is the delivery target because Google Slides import does not reliably preserve locally embedded
MP4 parts. A future native Google Slides version would need Drive-hosted copies of the videos and Drive-backed media
objects.
