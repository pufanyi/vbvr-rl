#!/usr/bin/env fish

# Evaluate checkpoints 100--500 from the 512x512x81 manifest-RL run. The first
# four checkpoints run concurrently on disjoint two-GPU groups. Checkpoint 500
# starts on all eight GPUs only after the first wave succeeds.

source (dirname (status filename))/../../../lib/env.fish

function _fail
    echo "[error] $argv" >&2
    exit 1
end

set -l script_dir (dirname (status filename))
set -g _manifest_rl_sweep_launcher $script_dir/vbvr_pro_5b_dancegrpo_manifest_rl_checkpoint_cps0p7_main_v2.fish
test -f $_manifest_rl_sweep_launcher
or _fail "launcher does not exist: $_manifest_rl_sweep_launcher"

set -q CHECKPOINT_ROOT[1]
or set -gx CHECKPOINT_ROOT storage/checkpoints/dancegrpo_vbvr_pro_5b_512x512x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_1e-6_manifest_rl_evalkit_6fedd9d9_reward1024_fps16_8_nodes
set -q EVAL_LOG_DIR[1]
or set -g EVAL_LOG_DIR storage/eval_logs/vbvr_pro_main_v2_512x512x81_manifest_rl_cps0p7
set -g _manifest_rl_sweep_log_dir $EVAL_LOG_DIR
mkdir -p $_manifest_rl_sweep_log_dir
or _fail "could not create log directory: $_manifest_rl_sweep_log_dir"
set -q OUTPUT_BASE[1]
or set -gx OUTPUT_BASE storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_manifest_326f7bda_evalkit_eb977da6
set -g _manifest_rl_sweep_output_base $OUTPUT_BASE

set -l first_wave_steps 100 200 300 400
set -l first_wave_devices 0,1 2,3 4,5 6,7
set -l final_steps 500

for step in $first_wave_steps $final_steps
    test -f $CHECKPOINT_ROOT/checkpoint-$step/high/.metadata
    or _fail "checkpoint-$step is missing high/.metadata under $CHECKPOINT_ROOT"
end

function _checkpoint_complete
    if set -q DRY_RUN[1]
        return 0
    end

    set -l step $argv[1]
    set -l output_root $_manifest_rl_sweep_output_base/dancegrpo_vbvr_pro_5b_checkpoint-$step-cps-noise-0.7
    .venv/bin/python -c '
import json
import sys
from pathlib import Path

from src.eval.vbvr_runtime import validate_vbvr_scorer_runtime

root = Path(sys.argv[1])
result_path = root / "scores" / "eval_1024x1024_81f_fps16_5p0625s_vbvr_results.json"
score_provenance_path = root / "score-provenance.json"
required_files = (
    result_path,
    score_provenance_path,
    root / "scores" / "eval_1024x1024_81f_fps16_5p0625s_task_scores.xlsx",
    root / "final_scores.txt",
)
missing = [str(path) for path in required_files if not path.is_file()]
if missing:
    raise SystemExit("missing completion artifacts: " + ", ".join(missing))

result = json.loads(result_path.read_text())
samples = result.get("samples")
if not isinstance(samples, list) or len(samples) != 500:
    raise SystemExit(f"expected 500 samples, found {len(samples) if isinstance(samples, list) else type(samples).__name__}")
errors = [sample for sample in samples if sample.get("error")]
if errors:
    raise SystemExit(f"score result contains {len(errors)} scorer errors")
tasks = {sample.get("task_name") for sample in samples}
if len(tasks) != 100:
    raise SystemExit(f"expected 100 tasks, found {len(tasks)}")
summary = result.get("summary", {})
for key, expected_count in (("In_Domain", 250), ("Out_of_Domain", 250), ("overall", 500)):
    if summary.get(key, {}).get("num_samples") != expected_count:
        raise SystemExit(f"invalid {key} sample count")

provenance = json.loads(score_provenance_path.read_text())
if provenance.get("values", {}).get("state") != "complete":
    raise SystemExit("score provenance is not complete")
dependencies = json.loads(provenance["values"]["scorer_dependencies"])
expected_runtime = validate_vbvr_scorer_runtime()
if dependencies.get("contract") != expected_runtime["contract"]:
    raise SystemExit(
        f"unexpected scorer runtime contract {dependencies.get('contract')!r}, "
        f"expected {expected_runtime['contract']!r}"
    )
if dependencies.get("sha256") != expected_runtime["sha256"]:
    raise SystemExit(
        f"unexpected scorer runtime fingerprint {dependencies.get('sha256')!r}, "
        f"expected {expected_runtime['sha256']!r}"
    )
' "$output_root"
end

function _run_checkpoint
    set -l step $argv[1]
    set -l devices $argv[2]
    set -l num_gpus (count (string split , -- $devices))
    set -l log_path $_manifest_rl_sweep_log_dir/checkpoint-$step-cps-noise-0.7.log

    echo "[start] "(date --iso-8601=seconds)" checkpoint-$step GPUs=$devices log=$log_path"
    env \
        CHECKPOINT_STEP=$step \
        CHECKPOINT_ROOT=$CHECKPOINT_ROOT \
        NUM_GPUS=$num_gpus \
        CUDA_DEVICES=$devices \
        PREP_WORKERS=4 \
        SCORE_WORKERS=2 \
        SCORE_THREADS_PER_WORKER=8 \
        fish $_manifest_rl_sweep_launcher >$log_path 2>&1
    set -l run_status $status
    if test $run_status -ne 0
        echo "[error] "(date --iso-8601=seconds)" checkpoint-$step failed with status $run_status; log=$log_path" >&2
        tail -n 120 $log_path >&2
        return $run_status
    end
    if not _checkpoint_complete $step
        echo "[error] "(date --iso-8601=seconds)" checkpoint-$step did not produce a complete strict result; log=$log_path" >&2
        tail -n 120 $log_path >&2
        return 1
    end
    echo "[done]  "(date --iso-8601=seconds)" checkpoint-$step log=$log_path"
end

set -l first_wave_pids
set -l first_wave_logs
for index in (seq (count $first_wave_steps))
    set -l step $first_wave_steps[$index]
    set -l devices $first_wave_devices[$index]
    set -l num_gpus (count (string split , -- $devices))
    set -l log_path $_manifest_rl_sweep_log_dir/checkpoint-$step-cps-noise-0.7.log

    echo "[start] "(date --iso-8601=seconds)" checkpoint-$step GPUs=$devices log=$log_path"
    env \
        CHECKPOINT_STEP=$step \
        CHECKPOINT_ROOT=$CHECKPOINT_ROOT \
        NUM_GPUS=$num_gpus \
        CUDA_DEVICES=$devices \
        PREP_WORKERS=4 \
        SCORE_WORKERS=2 \
        SCORE_THREADS_PER_WORKER=8 \
        fish $_manifest_rl_sweep_launcher >$log_path 2>&1 &
    set -a first_wave_pids $last_pid
    set -a first_wave_logs $log_path
end

set -l first_wave_failed 0
for index in (seq (count $first_wave_pids))
    wait $first_wave_pids[$index]
    set -l wait_status $status
    # Fish's `wait` reports whether it successfully waited, not the child's
    # exit status. Treat the validated artifacts as the completion contract.
    if test $wait_status -ne 0; or not _checkpoint_complete $first_wave_steps[$index]
        set first_wave_failed 1
        echo "[error] "(date --iso-8601=seconds)" checkpoint-$first_wave_steps[$index] did not produce a complete strict result; log=$first_wave_logs[$index]" >&2
        tail -n 120 $first_wave_logs[$index] >&2
    else
        echo "[done]  "(date --iso-8601=seconds)" checkpoint-$first_wave_steps[$index] log=$first_wave_logs[$index]"
    end
end
test $first_wave_failed -eq 0
or _fail "at least one first-wave checkpoint failed; checkpoint-500 was not started"

for step in $final_steps
    _run_checkpoint $step 0,1,2,3,4,5,6,7
    or exit $status
end

echo "[done] all 512x512x81 manifest-RL CPS 0.7 checkpoint evaluations completed"
