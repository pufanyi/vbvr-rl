# VBVR G-16 reward failure on 2026-07-25

## Summary

The 32-GPU DanceGRPO job did not fail because of GPU memory, NCCL, or a node/network outage. Rank 22 hit a deterministic bug in the pinned VBVR EvalKit `6fedd9d9` while scoring a G-16 sample. That rank exited first; the later Gloo `Connection closed by peer` messages and SIGTERM exits were consequences of the first failure.

The failure chain was:

`margin-offset grid` → `red endpoint missed` → `optimal-path detection returns None` → `meta_px read before assignment` → `rank 22 exits` → `distributed peers lose their connection`

## Reproduction sample

- Task: `G-16_grid_go_through_block_data-generator`
- Sample: `grid_go_through_block_00006001`
- Sampler position: optimizer step 16, prompt/rank 22 with seed 42
- Metadata start/end: `[0, 7]` → `[1, 7]` (adjacent cells)
- Video: [ground_truth.mp4](assets/vbvr_g16_grid_go_through_block_00006001/ground_truth.mp4)
- First frame: [first_frame.png](assets/vbvr_g16_grid_go_through_block_00006001/first_frame.png)
- Final frame: [final_frame.png](assets/vbvr_g16_grid_go_through_block_00006001/final_frame.png)
- Metadata: [metadata.json](assets/vbvr_g16_grid_go_through_block_00006001/metadata.json)
- Prompt: [prompt.txt](assets/vbvr_g16_grid_go_through_block_00006001/prompt.txt)
- Video properties: MPEG-4, 1024×1024, 70 frames, 16 FPS, 4.375 seconds
- Video SHA-256: `e0f67e7e07bd50c06514aead87e0c9011b22811af312a98c2af0d3a492ee8caf`

The failed generated rollout was stored in a `TemporaryDirectory` with `vbvr_reward_keep_tmp: false`, so it was deleted during exception cleanup and cannot be recovered. It is not needed to reproduce the problem: evaluating the copied ground-truth video against itself produces the same exception.

## Independent prompt-wave confirmation

`logs-acp-20260725T204210.txt.gz` is a second run made before the zero-score fallback was enabled. It used `prompt_wave=16`, two waves, and two ranks per prompt. The run completed optimizer step 15, then encountered the same G-16 sample during step 16.

The second wave's global prompt 22 was split across ranks 6 and 22, with 16 rollouts on each rank. Both ranks independently raised the same `meta_px` exception at 20:34:19 local time. All later Gloo, TCPStore, NCCL, broken-pipe, and SIGTERM messages followed those scorer failures. This confirms that the failure is independent of rollout content and shared-prompt topology.

## Root cause

The first frame contains a valid 10×10 grid, but the drawn grid starts about 48 pixels inside the 1024×1024 canvas. `maze.detect_grid_colors()` divides the entire image into ten cells starting at pixel zero instead of detecting the drawn grid bounds.

Because the green start and red end are adjacent, one inferred detector cell contains enough pixels from both. The detector checks green before red, classifies that mixed cell as green, and finds no red endpoint:

```text
blue=[(5, 6)]
green=[(7, 0), (7, 1)]
red=[]
obstacle=[(5, 2)]
```

`GridGoThroughBlockEvaluator._compute_optimal_path_info()` therefore returns `None`. In `In_Domain_50_part1.py`, `meta_px` is assigned only inside the successful `opt is not None` branch, but is read later outside that branch. Python raises:

```text
cannot access local variable 'meta_px' where it is not associated with a value
```

The relevant code is:

- `storage/evalkits/vbvr-evalkit-interleave-main_v2-6fedd9d9/vbvr_bench/evaluators/utils/maze.py:147`
- `storage/evalkits/vbvr-evalkit-interleave-main_v2-6fedd9d9/vbvr_bench/evaluators/In_Domain_50_part1.py:2733`

## Why the whole job stopped

The training config had `vbvr_reward_fail_on_error: true`. `VBVRRuleReward` converted the EvalKit error into a fatal reward exception on rank 22. Torch Elastic then terminated the other local workers, while remote ranks still inside `all_gather_object()` reported peer-closure errors.

The last completed optimizer step was 15. With `save_steps: 100`, no training checkpoint had been written.

## Temporary operational decision

The two DiffSynth manifest-RL configs now use:

```yaml
vbvr_reward_fail_on_error: false
vbvr_reward_unsupported_score: 0.0
```

Until the pinned evaluator is fixed, any scorer exception in these two runs is converted to reward `0.0` and training continues. This is deliberately broader than the G-16 bug and can hide other scorer failures, so it is a temporary continuity measure rather than a correctness fix.

The proper fix is to load the metadata path before grid-path detection and use metadata start/end/path whenever image-based detection fails. Before restoring fail-fast behavior, add this exact ground-truth self-score as a regression and require a finite, error-free score.
