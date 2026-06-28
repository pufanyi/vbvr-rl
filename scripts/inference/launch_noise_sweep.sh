#!/usr/bin/env bash
# Launch the noise-coefficient sweep across 8 GPUs.
#
#   Checkpoint A (RL / cos)  -> GPUs 0-3, samples sharded in contiguous blocks of 8
#   Checkpoint B (SFT)       -> GPUs 4-7, samples sharded in contiguous blocks of 8
#
# Each process loads its checkpoint once and runs the full (mode, noise) config
# list over its 8 samples. Per-GPU logs land in <OUT_ROOT>/logs/.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

OUT_ROOT="${OUT_ROOT:-storage/outputs/noise_coeff_sweep}"
STEPS="${STEPS:-50}"
ROUNDS="${ROUNDS:-4}"
LATENTS="storage/latents/maze_5b_line_to_ball_v1/webdataset/rl"

CKPT_A="storage/checkpoints/cos_maze_5b_line_to_ball_100k_tau_0.9/checkpoint-epoch4"
NAME_A="cos_maze_epoch4"
CKPT_B="storage/checkpoints/sft_maze_5b_line_to_ball_direct/checkpoint-3000"
NAME_B="sft_direct_3000"

# Trimmed-to-representative noise coefficients: small -> very large. ODE is the
# deterministic (eta=0) baseline; SDE eta is uncapped; CPS noise_level caps at 1.0.
# Each SDE/CPS coefficient is re-sampled ROUNDS times with different seeds.
CONFIGS="ode:0,sde:0.3,sde:0.7,sde:1.0,sde:2.0,cps:0.3,cps:0.7,cps:0.9,cps:1.0"

# Contiguous sample blocks of 8 across the 4 GPUs assigned to each checkpoint.
SHARDS=("0,1,2,3,4,5,6,7" "8,9,10,11,12,13,14,15" "16,17,18,19,20,21,22,23" "24,25,26,27,28,29,30,31")

mkdir -p "$OUT_ROOT/logs"
PIDS=()

launch() {
  local gpu="$1" ckpt="$2" name="$3" samples="$4"
  echo "[launch] GPU $gpu -> $name samples=$samples"
  PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/inference/noise_coeff_sweep.py \
    --checkpoint "$ckpt" --ckpt_name "$name" --device cuda:0 \
    --latent_webdataset_dir "$LATENTS" \
    --sample_indices "$samples" --configs "$CONFIGS" --rounds "$ROUNDS" \
    --num_sampling_steps "$STEPS" --output_root "$OUT_ROOT" \
    >"$OUT_ROOT/logs/${name}_gpu${gpu}.log" 2>&1 &
  PIDS+=($!)
}

for i in 0 1 2 3; do
  launch "$i" "$CKPT_A" "$NAME_A" "${SHARDS[$i]}"
done
for i in 0 1 2 3; do
  launch "$((i + 4))" "$CKPT_B" "$NAME_B" "${SHARDS[$i]}"
done

echo "[launch] 8 processes started; waiting..."
FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then FAIL=1; fi
done
echo "[launch] all done (fail=$FAIL)"
exit "$FAIL"
