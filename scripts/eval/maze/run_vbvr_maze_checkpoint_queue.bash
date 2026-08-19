#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LMMS_MAZE_ROOT="${LMMS_MAZE_ROOT:-$ROOT/../lmms-eval-maze}"
LMMS_PYTHON="${LMMS_PYTHON:-$ROOT/../lmms-eval/.venv/bin/python}"
WAN_PYTHON="${WAN_PYTHON:-$ROOT/.venv/bin/python}"

DATA_PARALLEL="${DATA_PARALLEL:-8}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT/storage/eval_runs/vbvr_maze_checkpoint_queue_$RUN_ID/logs}"
mkdir -p "$LOG_DIR"

TARGET_CKPTS=(
  "storage/checkpoints/cos_maze_line_to_ball_100k_0.9_from_vbvr/checkpoint-2000"
  "storage/checkpoints/cos_maze_line_to_ball_100k_0.9_from_vbvr/checkpoint-epoch0"
  "storage/checkpoints/cos_maze_line_to_ball_100k_0.9_from_vbvr/checkpoint-epoch1"
)

ALL_CKPTS=(
  "${TARGET_CKPTS[@]}"
  "storage/checkpoints/cos_maze_line_to_ball_100k/checkpoint-2000"
  "storage/checkpoints/cos_maze_line_to_ball_100k/checkpoint-4000"
  "storage/checkpoints/cos_maze_line_to_ball_100k/checkpoint-6000"
  "storage/checkpoints/cos_maze_line_to_ball_100k_tau_0.8/checkpoint-1000"
  "storage/checkpoints/dancegrpo_maze_4_rl/checkpoint-200"
  "storage/checkpoints/dancegrpo_vbvr_rl_2/checkpoint-500"
  "storage/checkpoints/dancegrpo_vbvr_rule_kl/checkpoint-1000"
  "storage/checkpoints/dancegrpo_vbvr_rule_kl/checkpoint-500"
  "storage/checkpoints/sft_maze_4/checkpoint-2000"
  "storage/checkpoints/sft_maze_4/checkpoint-epoch0"
  "storage/checkpoints/sft_maze_5/checkpoint-2000"
  "storage/checkpoints/sft_maze_5/checkpoint-epoch0"
  "storage/checkpoints/sft_vbvr_fixed_384_4/checkpoint-12000"
  "storage/checkpoints/sft_vbvr_fixed_384_4/checkpoint-8000"
  "storage/checkpoints/sft_vbvr_fixed_5e-6/checkpoint-2000"
  "storage/checkpoints/sft_vbvr_fixed_5e-6/checkpoint-4000"
  "storage/checkpoints/sft_vbvr_fixed_5e-6/checkpoint-6000"
)

ckpt_name() {
  local ckpt="$1"
  ckpt="${ckpt#storage/checkpoints/}"
  ckpt="${ckpt//\//_}"
  printf '%s\n' "$ckpt"
}

converted_dir() {
  printf '%s/storage/models/dcp_converted/%s\n' "$ROOT" "$(ckpt_name "$1")"
}

abs_ckpt() {
  printf '%s/%s\n' "$ROOT" "$1"
}

run_logged() {
  local label="$1"
  shift
  local log="$LOG_DIR/${label}.log"
  echo "[$(date '+%F %T')] START $label"
  {
    echo "[$(date '+%F %T')] START $label"
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
  } >> "$log"
  set +e
  "$@" >> "$log" 2>&1
  local status=$?
  set -e
  echo "[$(date '+%F %T')] END $label status=$status"
  echo "[$(date '+%F %T')] END $label status=$status" >> "$log"
  return "$status"
}

vbvr_done() {
  local model_abs="$1"
  ROOT="$ROOT" MODEL_ABS="$model_abs" python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
model_abs = str(Path(os.environ["MODEL_ABS"]).resolve())

def from_results(path: Path) -> bool:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    model = data.get("model_name") or data.get("model")
    if not model:
        return False
    if str(Path(model).resolve()) != model_abs:
        return False
    results = data.get("results") or {}
    r = results.get("vbvr") if isinstance(results, dict) else None
    if isinstance(r, dict) and isinstance(r.get("vbvr_overall,none"), (int, float)):
        return True
    s = data.get("summary")
    return isinstance(s, dict) and s.get("n") == 500 and isinstance(s.get("overall"), (int, float))

for path in (root / "storage/lmms_eval").glob("**/*_results.json"):
    if from_results(path):
        raise SystemExit(0)

model_name = Path(model_abs).name
for sub in (root / "storage/lmms_eval").glob("*/submissions/vbvr_eval_results.json"):
    try:
        data = json.loads(sub.read_text())
    except Exception:
        continue
    s = data.get("summary", data)
    if not (isinstance(s, dict) and s.get("n") == 500 and isinstance(s.get("overall"), (int, float))):
        continue
    generated = sub.parents[1] / "generated_videos" / model_name
    if generated.is_dir():
        raise SystemExit(0)
raise SystemExit(1)
PY
}

maze_done() {
  local model_abs="$1"
  ROOT="$ROOT" MODEL_ABS="$model_abs" python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
model_name = Path(os.environ["MODEL_ABS"]).name

for sub in list((root / "storage/lmms_eval_maze").glob("*/submissions/maze_line_eval_results.json")) + list((root / "storage/lmms_eval_maze").glob("*/submissions/maze_line_topology_eval_results.json")):
    try:
        data = json.loads(sub.read_text())
    except Exception:
        continue
    s = data.get("summary")
    if not isinstance(s, dict):
        continue
    if s.get("n") != 100 or not isinstance(s.get("overall"), (int, float)):
        continue
    if "topology_score" not in s and "correct_prefix_ratio" not in s:
        continue
    generated = sub.parents[1] / "generated_videos" / model_name / "maze_line"
    if generated.is_dir():
        raise SystemExit(0)
raise SystemExit(1)
PY
}

convert_missing_many() {
  local label="$1"
  shift
  local args=(
    -m src.cli.convert_dcp_to_diffusers
    --base_model "$ROOT/storage/models/Wan2.2-I2V-A14B-Diffusers"
    --torch_dtype bfloat16
    --device cuda
    --max_shard_size 10GB
  )
  local count=0
  for ckpt in "$@"; do
    local conv
    conv="$(converted_dir "$ckpt")"
    if [[ -f "$conv/model_index.json" ]]; then
      echo "[skip] converted exists: $conv"
      continue
    fi
    if [[ -d "$conv" ]] && [[ -n "$(find "$conv" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
      echo "[error] incomplete converted output exists: $conv" >&2
      echo "        remove it or set a fresh CONVERTED_ROOT before rerunning" >&2
      return 1
    fi
    args+=(--checkpoint "$(abs_ckpt "$ckpt")" --output "$conv")
    count=$((count + 1))
  done
  if [[ "$count" -eq 0 ]]; then
    return 0
  fi
  (
    cd "$ROOT"
    PYTHONPATH="$ROOT:${PYTHONPATH:-}" run_logged "convert_${label}" "$WAN_PYTHON" "${args[@]}"
  )
}

run_vbvr_one() {
  local ckpt="$1"
  local name conv out
  name="$(ckpt_name "$ckpt")"
  conv="$(converted_dir "$ckpt")"
  if vbvr_done "$conv"; then
    echo "[skip] VBVR already complete: $name"
    return 0
  fi
  out="$ROOT/storage/lmms_eval/vbvr_fastvideo_${name}_${RUN_ID}"
  (
    cd "$ROOT"
    DATA_PARALLEL="$DATA_PARALLEL" \
      EVAL_OUTPUT_DIR="$out" \
      run_logged "vbvr_${name}" fish scripts/eval/lmms/lmms_eval_checkpoint.fish "$ckpt"
  )
}

run_maze_one() {
  local ckpt="$1"
  local name conv out
  name="$(ckpt_name "$ckpt")"
  conv="$(converted_dir "$ckpt")"
  if maze_done "$conv"; then
    echo "[skip] Maze already complete: $name"
    return 0
  fi
  if [[ ! -f "$conv/model_index.json" ]]; then
    convert_missing_many "for_maze_${name}" "$ckpt"
  fi
  out="$ROOT/storage/lmms_eval_maze/maze_fastvideo_${name}_${RUN_ID}"
  (
    cd "$LMMS_MAZE_ROOT"
    PYTHON_BIN="$LMMS_PYTHON" \
      MODEL_DIR="$conv" \
      OUTPUT_DIR="$out" \
      DATA_PARALLEL="$DATA_PARALLEL" \
      run_logged "maze_${name}" tools/run_maze_fastvideo.sh
  )
}

echo "Run id:        $RUN_ID"
echo "Log dir:       $LOG_DIR"
echo "Data parallel: $DATA_PARALLEL"
echo "Target checkpoints first:"
printf '  - %s\n' "${TARGET_CKPTS[@]}"
echo

convert_missing_many target "${TARGET_CKPTS[@]}"
for ckpt in "${TARGET_CKPTS[@]}"; do
  run_vbvr_one "$ckpt"
done
for ckpt in "${TARGET_CKPTS[@]}"; do
  run_maze_one "$ckpt"
done

convert_missing_many all "${ALL_CKPTS[@]}"
for ckpt in "${ALL_CKPTS[@]}"; do
  run_vbvr_one "$ckpt"
done
for ckpt in "${ALL_CKPTS[@]}"; do
  run_maze_one "$ckpt"
done

echo "[$(date '+%F %T')] queue complete"
