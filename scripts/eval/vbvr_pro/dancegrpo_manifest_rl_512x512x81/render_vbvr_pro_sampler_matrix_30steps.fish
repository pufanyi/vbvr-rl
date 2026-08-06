#!/usr/bin/env fish

# Render one fixed formal sample for every model x sampler cell in the matched
# native-512 matrix. Each output contains step_00..step_29, a contact sheet,
# and a 30-cell grid video. Step 30/final is copied byte-for-byte from the
# quantitative run that EvalKit scored.

source (dirname (status filename))/../../../lib/env.fish

function _fail
    echo "[error] $argv" >&2
    exit 1
end

set -q CHECKPOINT_ROOT[1]
or set -gx CHECKPOINT_ROOT storage/checkpoints/dancegrpo_vbvr_pro_5b_512x512x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl_fujian_new_evalkit_e140038f
set -q CONVERTED_BASE[1]
or set -g CONVERTED_BASE storage/models/dcp_converted_5b
set -q CONVERTED_PREFIX[1]
or set -g CONVERTED_PREFIX dancegrpo_vbvr_pro_5b_512x512x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl_fujian_new_evalkit_e140038f
set -q BASELINE_MODEL[1]
or set -g BASELINE_MODEL storage/models/diffsynth_converted_5b/wan2.2-TI2V-5B_260715_vbvr_pro_step-35500
set -q EVAL_OUTPUT_BASE[1]
or set -g EVAL_OUTPUT_BASE storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_fujian_new_e140_lr5e6_eval500_181e2010_manifest_afab352e_evalkit_4cc7d028
set -q TRAJECTORY_ROOT[1]
or set -g TRAJECTORY_ROOT storage/eval_out/vbvr_pro_sampler_matrix_30step_trajectories
set -q TRAJECTORY_LOG_DIR[1]
or set -g TRAJECTORY_LOG_DIR storage/eval_logs/vbvr_pro_sampler_matrix_30step_trajectories
set -q SAMPLE_INDEX[1]
or set -g SAMPLE_INDEX 0

set -g _trajectory_converted_base $CONVERTED_BASE
set -g _trajectory_converted_prefix $CONVERTED_PREFIX
set -g _trajectory_baseline_model $BASELINE_MODEL
set -g _trajectory_eval_base $EVAL_OUTPUT_BASE
set -g _trajectory_root $TRAJECTORY_ROOT
set -g _trajectory_log_dir $TRAJECTORY_LOG_DIR
mkdir -p $_trajectory_root $_trajectory_log_dir; or _fail "could not create trajectory output/log directories"

set -l checkpoint_steps
for checkpoint_dir in (find $CHECKPOINT_ROOT -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
    test -f $checkpoint_dir/high/.metadata; or continue
    set -a checkpoint_steps (string replace 'checkpoint-' '' -- (basename $checkpoint_dir))
end
set -l model_ids baseline $checkpoint_steps
set -l sampler_ids cps0p1 cps0p3 cps0p7 cps0p9 euler unipc

function _sampler_parts
    switch $argv[1]
        case cps0p1
            printf '%s\n' cps 0.1 cps-noise-0.1
        case cps0p3
            printf '%s\n' cps 0.3 cps-noise-0.3
        case cps0p7
            printf '%s\n' cps 0.7 cps-noise-0.7
        case cps0p9
            printf '%s\n' cps 0.9 cps-noise-0.9
        case euler
            printf '%s\n' euler unused euler-ode-30steps-cfg1
        case unipc
            printf '%s\n' unipc unused unipc-ode-30steps-cfg1
        case '*'
            return 1
    end
end

function _model_path
    if test "$argv[1]" = baseline
        echo $_trajectory_baseline_model
    else
        echo $_trajectory_converted_base/$_trajectory_converted_prefix"_checkpoint-$argv[1]"
    end
end

function _eval_root
    set -l model_id $argv[1]
    set -l sampler_id $argv[2]
    set -l parts (_sampler_parts $sampler_id); or return 1
    set -l label $parts[3]
    if test "$model_id" = baseline
        if test "$sampler_id" = cps0p7
            echo $_trajectory_eval_base/diffsynth_step35500-baseline-cps0p7-30steps-cfg1
        else
            echo $_trajectory_eval_base/diffsynth_step35500-baseline-$label
        end
    else
        echo $_trajectory_eval_base/dancegrpo_vbvr_pro_5b_checkpoint-$model_id-$label
    end
end

function _trajectory_dir
    echo $_trajectory_root/$argv[1]-$argv[2]-sample(string pad -w 5 -c 0 $SAMPLE_INDEX)
end

function _formal_video
    set -l eval_root (_eval_root $argv[1] $argv[2])
    .venv/bin/python -c '
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
index = int(sys.argv[2])
data = json.loads((root / "eval_samples.json").read_text())
if not 0 <= index < len(data):
    raise SystemExit(f"sample index {index} outside {len(data)} samples")
item = data[index]
name = next((str(item[key]) for key in ("name", "id") if item.get(key) is not None and str(item[key])), str(index))
print(root / "generated_512x512x81" / Path(name).with_suffix(".mp4"))
' $eval_root $SAMPLE_INDEX
end

function _trajectory_complete
    set -l model_id $argv[1]
    set -l sampler_id $argv[2]
    set -l output_dir (_trajectory_dir $model_id $sampler_id)
    set -l formal (_formal_video $model_id $sampler_id); or return 1
    .venv/bin/python -c '
import hashlib, json, sys
from pathlib import Path
root, formal = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
manifest_path = root / "manifest.json"
required = [root / f"step_{i:02d}.mp4" for i in range(30)]
required += [root / "final_00.mp4", root / "steps_grid.mp4", root / "step_contact_sheet.jpg", manifest_path, formal]
if not all(path.is_file() for path in required):
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text())
if len(manifest.get("step_previews", [])) != 30:
    raise SystemExit(1)
digest = hashlib.sha256(formal.read_bytes()).hexdigest()
if hashlib.sha256((root / "step_29.mp4").read_bytes()).hexdigest() != digest:
    raise SystemExit(1)
if hashlib.sha256((root / "final_00.mp4").read_bytes()).hexdigest() != digest:
    raise SystemExit(1)
binding = manifest.get("formal_final_binding") or {}
if binding.get("source") != str(formal) or binding.get("sha256") != digest:
    raise SystemExit(1)
' $output_dir $formal
end

set -l tasks
for sampler_id in $sampler_ids
    if set -q SAMPLER_FILTER[1]; and test "$SAMPLER_FILTER" != "$sampler_id"
        continue
    end
    for model_id in $model_ids
        if set -q MODEL_FILTER[1]; and test "$MODEL_FILTER" != "$model_id"
            continue
        end
        set -a tasks "$model_id,$sampler_id"
    end
end
test (count $tasks) -gt 0; or _fail "filters selected no trajectory tasks"

function _launch_wave
    set -l wave_tasks $argv
    set -l running_pids
    set -l running_tasks
    set -l running_logs
    for slot in (seq (count $wave_tasks))
        set -l task (string split , -- $wave_tasks[$slot])
        set -l model_id $task[1]
        set -l sampler_id $task[2]
        set -l sampler (_sampler_parts $sampler_id); or return 1
        set -l mode $sampler[1]
        set -l level $sampler[2]
        set -l eval_root (_eval_root $model_id $sampler_id)
        set -l eval_json $eval_root/eval_samples.json
        set -l formal (_formal_video $model_id $sampler_id); or return 1
        set -l model_path (_model_path $model_id)
        set -l output_dir (_trajectory_dir $model_id $sampler_id)
        set -l log_path $_trajectory_log_dir/$model_id-$sampler_id.log
        set -l device (math $slot - 1)

        test -f $eval_json; or begin
            echo "[error] quantitative eval JSON missing: $eval_json" >&2
            return 1
        end
        test -f $formal; or begin
            echo "[error] formal final video missing: $formal" >&2
            return 1
        end
        if _trajectory_complete $model_id $sampler_id
            echo "[skip] $model_id/$sampler_id trajectory already complete"
            continue
        end

        set -l noise_args
        if test "$mode" = cps
            set noise_args --noise_level $level
        end
        echo "[start] "(date --iso-8601=seconds)" $model_id/$sampler_id GPU=$device log=$log_path"
        env CUDA_VISIBLE_DEVICES=$device \
            .venv/bin/python -m src.cli.render_vbvr_i2v_steps \
                --eval_json $eval_json \
                --model_path $model_path \
                --output_dir $output_dir \
                --sampler $mode \
                $noise_args \
                --sample_index $SAMPLE_INDEX \
                --height 512 --width 512 --num_frames 81 \
                --num_inference_steps 30 --guidance_scale 1.0 \
                --fps 16 --seed 0 --device cuda:0 \
                --grid_cols 6 --grid_thumb_width 160 \
                --formal_final_video $formal \
                >$log_path 2>&1 &
        set -a running_pids $last_pid
        set -a running_tasks $wave_tasks[$slot]
        set -a running_logs $log_path
    end

    set -l failed 0
    for slot in (seq (count $running_pids))
        wait $running_pids[$slot]
        set -l rc $status
        set -l task (string split , -- $running_tasks[$slot])
        if test $rc -ne 0; or not _trajectory_complete $task[1] $task[2]
            set failed 1
            echo "[error] $task[1]/$task[2] trajectory failed; log=$running_logs[$slot]" >&2
            tail -n 120 $running_logs[$slot] >&2
        else
            echo "[done]  "(date --iso-8601=seconds)" $task[1]/$task[2]"
        end
    end
    test $failed -eq 0
end

echo "[trajectory] fixed sample index: $SAMPLE_INDEX"
echo "[trajectory] selected tasks: "(count $tasks)
set -l start 1
while test $start -le (count $tasks)
    set -l wave
    for offset in 0 1 2 3 4 5 6 7
        set -l position (math $start + $offset)
        if test $position -le (count $tasks)
            set -a wave $tasks[$position]
        end
    end
    echo "[wave] $wave"
    _launch_wave $wave; or _fail "trajectory wave failed: $wave"
    set start (math $start + 8)
end

echo "[done] all 30-step trajectory displays are complete under $_trajectory_root"
