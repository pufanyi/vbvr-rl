# Checkpoint-2200 versus DiffSynth baseline presentation (2026-08-18)

## Outcome

`scripts/eval/vbvr_pro/build_checkpoint2200_vs_baseline_ppt.py` builds a self-contained 56-slide PowerPoint from the
strict native-512 evaluation matrix. It compares:

- DiffSynth step-35500 baseline with 30-step UniPC ODE, CFG 1, and seed 0.
- DanceGRPO checkpoint 2200 with 30-step CPS 0.7, CFG 1, and seed 0.

The output is
`storage/presentations/vbvr_checkpoint2200_vs_baseline_20260818/vbvr_checkpoint2200_vs_diffsynth_baseline_top50_embedded_video_20260818.pptx`.
It contains 50 paired examples and 100 embedded native 512x512x81 MP4s. Every case page labels the two exact EvalKit
scores and their checkpoint-minus-baseline delta. Each movie poster is the raw video's exact decoded frame zero,
without cropping, labels, or compositing.

## Source and selection audit

The two `eval_samples.json` files are byte-identical with SHA-256
`b1a73419d481f039a04cd69bc10199caa569e12412cdd68e7a47935d1c53f9f1`. Both result JSONs contain the same 500
canonical samples and zero scorer errors. They use EvalKit e140 source fingerprint
`4cc7d028d4106a28190a63bc179562d5ac9add9263cb71926dd6385c5714bcf8`.

Across all 500 paired samples, 246 checkpoint scores are higher, 133 are tied, and 121 are lower. The complete Overall
means are 0.473463 for the DiffSynth baseline and 0.547886 for checkpoint 2200. The presentation selects the 50 largest
exact paired deltas with deterministic tie-breaking. These cases cover 37 task types, 27 In-Domain samples, and 23
Out-of-Domain samples; their deltas range from +0.383065 to +1.000000, with mean +0.713082.

Thirteen generated audit pages show frames 0, 20, 40, 60, and 80 for both videos plus the input and GT final frame.
All 50 selected cases were visually reviewed against these sheets, and the score direction was consistent with the
observable completion difference.

## Validation and evidence boundary

The builder validates all selected native videos as 512x512 RGB, 81 frames, and 16 FPS. It reopens the PPTX, tests its
ZIP package, checks the PowerPoint video OOXML and MP4 content type, and compares all 100 embedded MP4 hashes with the
sources. All 100 poster PNGs are checked pixel-for-pixel against their corresponding decoded raw frame zero.
The rasterized `preview/` spot checks carry the renderer's evaluation watermark; the watermark is not present in the
PPTX.

The top-50 examples are deliberately selected for contrast and are not an unbiased estimate of average quality. This
comparison also changes both the model checkpoint and the sampler (CPS versus UniPC); it demonstrates the combined
generation configuration. A same-sampler control comparison is required to attribute the difference to RL alone.
