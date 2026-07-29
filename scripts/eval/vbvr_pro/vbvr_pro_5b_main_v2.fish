#!/usr/bin/env fish

# End-to-end VBVR-Pro main_v2 evaluation for a Wan2.2 TI2V 5B DCP checkpoint.
# All generated artifacts stay under this repository's ignored storage/ tree.

source (dirname (status filename))/../../lib/env.fish

# Ignore ambient multinode launcher state; torchrun owns the local DP group.
set -e RANK
set -e WORLD_SIZE
set -e LOCAL_RANK
set -e LOCAL_WORLD_SIZE
set -e NODE_RANK

set -q CHECKPOINT[1]; or set CHECKPOINT storage/checkpoints/dancegrpo_vbvr_pro_5b_256x256x161_rule_cps_from_nsft_bs32/checkpoint-1200
set -q BASE_MODEL[1]; or set BASE_MODEL storage/models/Wan2.2-TI2V-5B-Diffusers
set -q CONVERTED_MODEL[1]; or set CONVERTED_MODEL storage/models/dcp_converted_5b/dancegrpo_vbvr_pro_5b_256x256x161_rule_cps_from_nsft_bs32_checkpoint-1200
set -q GT_BASE[1]; or set GT_BASE /mnt/aigc/xujunxiang/VR_Data/VBVR-Bench_Pro-video
set -q SPLIT_MANIFEST[1]; or set SPLIT_MANIFEST /mnt/aigc/xujunxiang/Code/VBVR-Pro/scripts/split_manifest.json

set -q EVALKIT_DIR[1]; or set EVALKIT_DIR storage/evalkits/vbvr-evalkit-interleave-main_v2-6fedd9d9
set -q EVALKIT_REPO[1]; or set EVALKIT_REPO git@github.com:xujunxiangwork/VBVR-Evalkit-Interleave.git
set -q EVALKIT_REV[1]; or set EVALKIT_REV 6fedd9d9edb8daafa56aca8e53885aa8ad6f6037
set -q EVALKIT_SOURCE_SHA256[1]; or set EVALKIT_SOURCE_SHA256 eb977da60e95456734063ba018b14d805680179fdf0e3e3b2ba6f603f27a935c
set -q EVALKIT_INSTALL_LOCK[1]; or set EVALKIT_INSTALL_LOCK $EVALKIT_DIR.install.lock
set -q EVALKIT_INSTALL_WAIT_SECONDS[1]; or set EVALKIT_INSTALL_WAIT_SECONDS 1800
set -q EVALKIT_INSTALL_POLL_SECONDS[1]; or set EVALKIT_INSTALL_POLL_SECONDS 2
set -q EASYOCR_SOURCE_MODELS[1]; or set EASYOCR_SOURCE_MODELS /mnt/aigc/xujunxiang/Code/VBVR-Bench/VBVR-EvalKit/easyocr_models

set -q OUTPUT_ROOT[1]; or set OUTPUT_ROOT storage/eval_out/vbvr_pro_main_v2_evalkit_eb977da6/dancegrpo_vbvr_pro_5b_checkpoint-1200
set -q EVAL_JSON[1]; or set EVAL_JSON $OUTPUT_ROOT/eval_samples.json
set -q GENERATED_DIR[1]; or set GENERATED_DIR $OUTPUT_ROOT/generated_256x256x161
set -q PREPARED_DIR[1]; or set PREPARED_DIR $OUTPUT_ROOT/eval_1024x1024_161f_5s
set -q SCORE_DIR[1]; or set SCORE_DIR $OUTPUT_ROOT/scores
set -q EASYOCR_ROOT[1]; or set EASYOCR_ROOT storage/evalkits/easyocr-shared
set -q PRECONVERTED_MODEL[1]; or set PRECONVERTED_MODEL 0
if test "$PRECONVERTED_MODEL" = 1
    # Direct Diffusers inputs must carry their own immutable conversion/import
    # record. Generation provenance fingerprints both this file and the full
    # model tree.
    set -q CONVERSION_PROVENANCE[1]
    or set CONVERSION_PROVENANCE $CONVERTED_MODEL/conversion_metadata.json
else
    set -q CONVERSION_PROVENANCE[1]
    or set CONVERSION_PROVENANCE $CONVERTED_MODEL.wan-trainer-provenance.json
end
set -q GENERATION_PROVENANCE[1]; or set GENERATION_PROVENANCE $OUTPUT_ROOT/generation-provenance.json
set -q PREPARATION_PROVENANCE[1]; or set PREPARATION_PROVENANCE $OUTPUT_ROOT/preparation-provenance.json
set -q SCORE_PROVENANCE[1]; or set SCORE_PROVENANCE $OUTPUT_ROOT/score-provenance.json
set -q CONVERSION_LOCK[1]; or set CONVERSION_LOCK $CONVERSION_PROVENANCE.lock

set -q EXPECTED_VIDEOS[1]; or set EXPECTED_VIDEOS 500
set -q NUM_GPUS[1]; or set NUM_GPUS 8
set -q CUDA_DEVICES[1]; or set CUDA_DEVICES 0,1,2,3,4,5,6,7
set -q NUM_FRAMES[1]; or set NUM_FRAMES 161
set -q USE_ITEM_NUM_FRAMES[1]; or set USE_ITEM_NUM_FRAMES 0
set -q TEMPORAL_ALIGNMENT[1]; or set TEMPORAL_ALIGNMENT 4
set -q HEIGHT[1]; or set HEIGHT 256
set -q WIDTH[1]; or set WIDTH 256
set -q INFER_FPS[1]; or set INFER_FPS 16
set -q NUM_INFERENCE_STEPS[1]; or set NUM_INFERENCE_STEPS 50
set -q GUIDANCE_SCALE[1]; or set GUIDANCE_SCALE 5.0
set -q SEED[1]; or set SEED 0
set -q GENERATION_MODE[1]; or set GENERATION_MODE ode
set -q ODE_SOLVER[1]; or set ODE_SOLVER unipc
set -q CPS_NOISE_LEVEL[1]; or set CPS_NOISE_LEVEL 0.7

set -q PREPARED_HEIGHT[1]; or set PREPARED_HEIGHT 1024
set -q PREPARED_WIDTH[1]; or set PREPARED_WIDTH 1024
set -q MAX_DURATION[1]; or set MAX_DURATION 5
set -q PREP_WORKERS[1]; or set PREP_WORKERS 8
set -q PREP_CRF[1]; or set PREP_CRF 12
set -q SCORE_WORKERS[1]; or set SCORE_WORKERS 8
set -q SCORE_THREADS_PER_WORKER[1]; or set SCORE_THREADS_PER_WORKER 16
set -q CONVERSION_PROVENANCE_WAIT_SECONDS[1]; or set CONVERSION_PROVENANCE_WAIT_SECONDS 1800
set -q CONVERSION_PROVENANCE_POLL_SECONDS[1]; or set CONVERSION_PROVENANCE_POLL_SECONDS 2
set -q CONVERSION_STABILITY_SECONDS[1]; or set CONVERSION_STABILITY_SECONDS 2

set PYTHON .venv/bin/python
set TORCHRUN .venv/bin/torchrun

function _fail
    echo "[error] $argv" >&2
    exit 1
end

function _count_mp4
    set -l root $argv[1]
    if not test -d $root
        echo 0
        return
    end
    find $root -type f -name '*.mp4' ! -name '.*.tmp-rank*-pid*.mp4' | wc -l | string trim
end

contains -- $GENERATION_MODE ode cps
or _fail "GENERATION_MODE must be ode or cps, got $GENERATION_MODE"
contains -- $ODE_SOLVER unipc euler
or _fail "ODE_SOLVER must be unipc or euler, got $ODE_SOLVER"
contains -- $PRECONVERTED_MODEL 0 1
or _fail "PRECONVERTED_MODEL must be 0 or 1, got $PRECONVERTED_MODEL"
contains -- $USE_ITEM_NUM_FRAMES 0 1
or _fail "USE_ITEM_NUM_FRAMES must be 0 or 1, got $USE_ITEM_NUM_FRAMES"
if test "$USE_ITEM_NUM_FRAMES" = 1; and test "$GENERATION_MODE" != ode
    _fail "USE_ITEM_NUM_FRAMES currently requires GENERATION_MODE=ode"
end
if test "$USE_ITEM_NUM_FRAMES" = 1
    $PYTHON -c 'import sys; value=int(sys.argv[1]); raise SystemExit(0 if value > 0 else 1)' \
        $TEMPORAL_ALIGNMENT
    or _fail "TEMPORAL_ALIGNMENT must be a positive integer, got $TEMPORAL_ALIGNMENT"
end
if test $GENERATION_MODE = cps
    $PYTHON -c 'import sys; value=float(sys.argv[1]); raise SystemExit(0 if 0.0 <= value <= 1.0 else 1)' \
        $CPS_NOISE_LEVEL
    or _fail "CPS_NOISE_LEVEL must be in [0, 1], got $CPS_NOISE_LEVEL"
end

if set -q DRY_RUN[1]
    echo "[dry-run] mode=$GENERATION_MODE checkpoint=$CHECKPOINT"
    echo "[dry-run] converted_model=$CONVERTED_MODEL output_root=$OUTPUT_ROOT"
    echo "[dry-run] preconverted_model=$PRECONVERTED_MODEL conversion_provenance=$CONVERSION_PROVENANCE"
    echo "[dry-run] evalkit=$EVALKIT_DIR revision=$EVALKIT_REV source_sha256=$EVALKIT_SOURCE_SHA256"
    if test $GENERATION_MODE = cps
        echo "[dry-run] cps_noise_level=$CPS_NOISE_LEVEL steps=$NUM_INFERENCE_STEPS cfg=$GUIDANCE_SCALE"
    else
        echo "[dry-run] ode_solver=$ODE_SOLVER steps=$NUM_INFERENCE_STEPS cfg=$GUIDANCE_SCALE"
    end
    exit 0
end

function _valid_score_result
    set -l result_path $argv[1]
    test -f $result_path; or return 1
    $PYTHON -c '
import json, sys
data = json.load(open(sys.argv[1]))
samples = data.get("samples", [])
errors = [sample for sample in samples if sample.get("error")]
raise SystemExit(0 if len(samples) == int(sys.argv[2]) and not errors else 1)
' $result_path $EXPECTED_VIDEOS
end

set generation_negative_prompt_sha ($PYTHON -c '
import hashlib
from src.cli.eval_i2v import parse_args
args = parse_args(["--eval_json", "unused", "--output_dir", "unused"])
print(hashlib.sha256(args.negative_prompt.encode("utf-8")).hexdigest())
')
test -n "$generation_negative_prompt_sha"; or _fail "could not fingerprint the generation negative prompt"
set diffusers_version ($PYTHON -c 'import diffusers; print(diffusers.__version__)')
set torch_version ($PYTHON -c 'import torch; print(torch.__version__)')
set ffmpeg_version (ffmpeg -version 2>/dev/null | head -n 1 | string trim)

function _conversion_provenance
    set -l mode $argv[1]
    set -l state $argv[2]
    set -l extra_args $argv[3..-1]
    set -l output_args
    if test $mode = promote
        set output_args --output-tree converted_model=$CONVERTED_MODEL
    else if contains -- $mode check refresh-outputs; and test $state = complete
        set output_args --output-tree converted_model=$CONVERTED_MODEL
    end
    $PYTHON -m src.eval.evaluation_provenance $mode \
        --manifest $CONVERSION_PROVENANCE \
        --stage vbvr-pro-conversion \
        --value state=$state \
        --value torch_dtype=bfloat16 \
        --value use_ema=false \
        --value merge_lora=true \
        --value safe_serialization=true \
        --value max_shard_size=10GB \
        --value fastvideo_compat=false \
        --file converter=src/cli/convert_dcp_to_diffusers.py \
        --tree checkpoint=$CHECKPOINT \
        --tree base_model=$BASE_MODEL \
        $output_args \
        $extra_args
end

set -g _vbvr_conversion_lock_held 0
set -g _vbvr_evalkit_install_lock_held 0

function _release_evalkit_install_lock --on-event fish_exit
    if test "$_vbvr_evalkit_install_lock_held" != 1
        return 0
    end
    command rm -f -- "$EVALKIT_INSTALL_LOCK/owner"
    if not command rmdir -- "$EVALKIT_INSTALL_LOCK" 2>/dev/null
        echo "[warn] could not remove EvalKit install lock directory: $EVALKIT_INSTALL_LOCK" >&2
        return 1
    end
    set -g _vbvr_evalkit_install_lock_held 0
end

function _try_acquire_evalkit_install_lock
    command mkdir -- "$EVALKIT_INSTALL_LOCK" 2>/dev/null; or return 1
    set -g _vbvr_evalkit_install_lock_held 1
    printf 'host=%s\npid=%s\nrevision=%s\nstarted_at=%s\n' \
        (hostname) "$fish_pid" "$EVALKIT_REV" (date --iso-8601=seconds) \
        >"$EVALKIT_INSTALL_LOCK/owner"
    or begin
        _release_evalkit_install_lock
        return 1
    end
    return 0
end

function _release_conversion_lock --on-event fish_exit
    if test "$_vbvr_conversion_lock_held" != 1
        return 0
    end
    command rm -f -- "$CONVERSION_LOCK/owner"
    if not command rmdir -- "$CONVERSION_LOCK" 2>/dev/null
        echo "[warn] could not remove conversion lock directory: $CONVERSION_LOCK" >&2
        return 1
    end
    set -g _vbvr_conversion_lock_held 0
end

function _try_acquire_conversion_lock
    command mkdir -- "$CONVERSION_LOCK" 2>/dev/null; or return 1
    set -g _vbvr_conversion_lock_held 1
    set -l lock_host (hostname)
    printf 'host=%s\npid=%s\ncheckpoint=%s\nstarted_at=%s\n' \
        "$lock_host" "$fish_pid" "$CHECKPOINT" (date --iso-8601=seconds) \
        >"$CONVERSION_LOCK/owner"
    or begin
        _release_conversion_lock
        return 1
    end
    return 0
end

function _converted_model_fingerprint
    $PYTHON -c '
import sys
from pathlib import Path
from src.eval.evaluation_provenance import fingerprint_tree
fingerprint = fingerprint_tree(Path(sys.argv[1]))
print("{}:{}:{}".format(fingerprint["entries"], fingerprint["total_size"], fingerprint["sha256"]))
' "$CONVERTED_MODEL"
end

function _converted_model_is_stable
    set -l before (_converted_model_fingerprint); or return 1
    sleep $CONVERSION_STABILITY_SECONDS
    set -l after (_converted_model_fingerprint); or return 1
    if test "$before" != "$after"
        echo "[error] converted model changed during the $CONVERSION_STABILITY_SECONDS-second stability check: $CONVERTED_MODEL" >&2
        return 1
    end
end

function _validate_converted_model
    $PYTHON -m src.eval.validate_diffusers_model "$CONVERTED_MODEL" $argv
end

function _generation_provenance
    set -l mode $argv[1]
    set -l state $argv[2]
    set -l extra_args $argv[3..-1]
    set -l output_args
    if test $mode = promote
        set output_args --media-tree generated_videos=$GENERATED_DIR
    else if test $mode = check; and test $state = complete
        set output_args --media-tree generated_videos=$GENERATED_DIR
    end
    set -l generator_source src/cli/eval_i2v.py
    set -l sampler_args
    set -l generator_file_args
    set -l frame_contract_args
    if test "$USE_ITEM_NUM_FRAMES" = 1
        set frame_contract_args \
            --value num_frames_mode=gt_duration \
            --value temporal_alignment=$TEMPORAL_ALIGNMENT
    end
    if test $GENERATION_MODE = cps
        set generator_source src/cli/eval_i2v_cps.py
        set sampler_args \
            --value generation_mode=cps \
            --value cps_noise_level=$CPS_NOISE_LEVEL
    else if test $ODE_SOLVER = euler
        set generator_source src/cli/eval_i2v_euler.py
        set generator_file_args --file generator_base=src/cli/eval_i2v.py
        set sampler_args \
            --value generation_mode=ode \
            --value ode_solver=flowmatch_euler
    end
    $PYTHON -m src.eval.evaluation_provenance $mode \
        --manifest $GENERATION_PROVENANCE \
        --stage vbvr-pro-generation \
        --value state=$state \
        --value height=$HEIGHT \
        --value width=$WIDTH \
        --value num_frames=$NUM_FRAMES \
        $frame_contract_args \
        --value fps=$INFER_FPS \
        --value num_inference_steps=$NUM_INFERENCE_STEPS \
        --value guidance_scale=$GUIDANCE_SCALE \
        --value seed=$SEED \
        --value num_gpus=$NUM_GPUS \
        --value negative_prompt_sha256=$generation_negative_prompt_sha \
        --value diffusers_version=$diffusers_version \
        --value torch_version=$torch_version \
        --file conversion_provenance=$CONVERSION_PROVENANCE \
        --file eval_json=$EVAL_JSON \
        --file split_manifest=$SPLIT_MANIFEST \
        --file generator=$generator_source \
        $generator_file_args \
        --tree converted_model=$CONVERTED_MODEL \
        --tree eval_source=$GT_BASE \
        $sampler_args \
        $output_args \
        $extra_args
end

function _preparation_provenance
    set -l mode $argv[1]
    set -l state $argv[2]
    set -l extra_args $argv[3..-1]
    set -l output_args
    if test $mode = promote
        set output_args --media-tree prepared_videos=$PREPARED_DIR
    else if test $mode = check; and test $state = complete
        set output_args --media-tree prepared_videos=$PREPARED_DIR
    end
    $PYTHON -m src.eval.evaluation_provenance $mode \
        --manifest $PREPARATION_PROVENANCE \
        --stage vbvr-pro-preparation \
        --value state=$state \
        --value height=$PREPARED_HEIGHT \
        --value width=$PREPARED_WIDTH \
        --value max_duration=$MAX_DURATION \
        --value crf=$PREP_CRF \
        --value ffmpeg_version=$ffmpeg_version \
        --file generation_provenance=$GENERATION_PROVENANCE \
        --file preparer=src/cli/prepare_vbvr_eval_videos.py \
        $output_args \
        $extra_args
end

function _score_provenance
    set -l mode $argv[1]
    set -l state $argv[2]
    set -l result_file $argv[3]
    set -l current_evalkit_source_sha256 ($PYTHON -c '
import sys
from src.eval.vbvr_run_evaluation_parallel import evalkit_source_sha256
print(evalkit_source_sha256(sys.argv[1]))
' $EVALKIT_DIR); or return 1
    if test "$current_evalkit_source_sha256" != "$EVALKIT_SOURCE_SHA256"
        echo "[error] EvalKit source fingerprint changed or does not match the pin: expected=$EVALKIT_SOURCE_SHA256 actual=$current_evalkit_source_sha256" >&2
        return 1
    end
    set -l output_args
    if test $mode = promote
        set output_args --output-file result=$result_file
    else if test $mode = check; and test $state = complete
        set output_args --output-file result=$result_file
    end
    $PYTHON -m src.eval.evaluation_provenance $mode \
        --manifest $SCORE_PROVENANCE \
        --stage vbvr-pro-score \
        --value state=$state \
        --value evalkit_revision=$EVALKIT_REV \
        --value evalkit_revision_actual=$evalkit_revision_actual \
        --value evalkit_source_sha256=$current_evalkit_source_sha256 \
        --value scorer_dependencies=$scorer_dependency_versions \
        --value device=cpu \
        --value workers=$SCORE_WORKERS \
        --value threads_per_worker=$SCORE_THREADS_PER_WORKER \
        --file preparation_provenance=$PREPARATION_PROVENANCE \
        --file scorer_entrypoint=$EVALKIT_DIR/run_evaluation.py \
        --file scorer_requirements=$EVALKIT_DIR/requirements.txt \
        --file scorer_runtime=src/eval/vbvr_runtime.py \
        --file easyocr_craft=$EASYOCR_ROOT/model/craft_mlt_25k.pth \
        --file easyocr_english=$EASYOCR_ROOT/model/english_g2.pth \
        --file scorer_wrapper=src/eval/vbvr_run_evaluation_parallel.py \
        --tree prepared_videos=$PREPARED_DIR \
        --tree ground_truth=$GT_BASE \
        $output_args
end

set -l required_paths $GT_BASE $SPLIT_MANIFEST $EASYOCR_SOURCE_MODELS
if test "$PRECONVERTED_MODEL" = 0
    set -a required_paths $CHECKPOINT $BASE_MODEL
end
for required in $required_paths
    test -e $required; or _fail "required path does not exist: $required"
end

set -l visible_count (count (string split , -- $CUDA_DEVICES))
test $visible_count -ge $NUM_GPUS; or _fail "CUDA_DEVICES exposes $visible_count devices, but NUM_GPUS=$NUM_GPUS"
set -gx CUDA_VISIBLE_DEVICES $CUDA_DEVICES

if not test -f $EVALKIT_DIR/run_evaluation.py
    set -l evalkit_parent (dirname $EVALKIT_DIR)
    mkdir -p $evalkit_parent; or exit 1
    set -l evalkit_waited_seconds 0
    set -l evalkit_wait_announced 0
    while not test -f $EVALKIT_DIR/run_evaluation.py
        if test -e $EVALKIT_DIR
            _fail "EvalKit directory exists but is incomplete: $EVALKIT_DIR"
        end
        if _try_acquire_evalkit_install_lock
            if test -f $EVALKIT_DIR/run_evaluation.py
                _release_evalkit_install_lock; or exit 1
                break
            end
            set -l evalkit_staging $EVALKIT_DIR.tmp-$fish_pid
            test ! -e $evalkit_staging
            or _fail "temporary EvalKit checkout already exists: $evalkit_staging"
            echo "[evalkit] fetching exact revision $EVALKIT_REV from $EVALKIT_REPO"
            git init -q $evalkit_staging; or exit 1
            git -C $evalkit_staging remote add origin $EVALKIT_REPO; or exit 1
            git -C $evalkit_staging fetch --depth 1 origin $EVALKIT_REV; or exit 1
            git -C $evalkit_staging checkout -q --detach FETCH_HEAD; or exit 1
            command mv -T -- $evalkit_staging $EVALKIT_DIR; or exit 1
            _release_evalkit_install_lock; or exit 1
            break
        end
        if test $evalkit_wait_announced -eq 0
            echo "[wait] another job owns the EvalKit install lock: $EVALKIT_INSTALL_LOCK"
            set evalkit_wait_announced 1
        end
        if test $evalkit_waited_seconds -ge $EVALKIT_INSTALL_WAIT_SECONDS
            test -f "$EVALKIT_INSTALL_LOCK/owner"; and cat "$EVALKIT_INSTALL_LOCK/owner" >&2
            _fail "timed out after $EVALKIT_INSTALL_WAIT_SECONDS seconds waiting for EvalKit install lock"
        end
        sleep $EVALKIT_INSTALL_POLL_SECONDS
        set evalkit_waited_seconds (math $evalkit_waited_seconds + $EVALKIT_INSTALL_POLL_SECONDS)
    end
end

set -g evalkit_source_sha256 ($PYTHON -c '
import sys
from src.eval.vbvr_run_evaluation_parallel import evalkit_source_sha256
print(evalkit_source_sha256(sys.argv[1]))
' $EVALKIT_DIR)
test "$evalkit_source_sha256" = "$EVALKIT_SOURCE_SHA256"
or _fail "EvalKit source fingerprint mismatch: expected=$EVALKIT_SOURCE_SHA256 actual=$evalkit_source_sha256 path=$EVALKIT_DIR"

set -g evalkit_revision_actual unavailable
if git -C $EVALKIT_DIR rev-parse --is-inside-work-tree >/dev/null 2>&1
    set evalkit_revision_actual (git -C $EVALKIT_DIR rev-parse HEAD); or exit 1
    test "$evalkit_revision_actual" = "$EVALKIT_REV"
    or _fail "EvalKit Git revision mismatch: expected=$EVALKIT_REV actual=$evalkit_revision_actual path=$EVALKIT_DIR"
end

mkdir -p $EASYOCR_ROOT/model $EASYOCR_ROOT/user_network; or exit 1
for model_file in craft_mlt_25k.pth english_g2.pth
    test -f $EASYOCR_SOURCE_MODELS/$model_file
    or _fail "missing pre-populated EasyOCR weight: $EASYOCR_SOURCE_MODELS/$model_file"
    set -l source_model $EASYOCR_SOURCE_MODELS/$model_file
    set -l installed_model $EASYOCR_ROOT/model/$model_file
    if not test -f $installed_model; or not cmp -s $source_model $installed_model
        set -l temporary_model $installed_model.tmp-$fish_pid
        cp $source_model $temporary_model; or exit 1
        mv -f $temporary_model $installed_model; or exit 1
    end
end

set -l evalkit_easyocr_models $EVALKIT_DIR/easyocr_models
set -l personal_easyocr_models (realpath $EASYOCR_ROOT/model)
if test -e $evalkit_easyocr_models; and not test -L $evalkit_easyocr_models
    _fail "expected a symlink at $evalkit_easyocr_models; refusing to replace an existing directory"
end
if not test -L $evalkit_easyocr_models; or test (realpath $evalkit_easyocr_models) != $personal_easyocr_models
    set -l temporary_link $evalkit_easyocr_models.tmp-$fish_pid
    ln -s $personal_easyocr_models $temporary_link; or exit 1
    mv -Tf $temporary_link $evalkit_easyocr_models; or exit 1
end

set -g scorer_dependency_versions (env \
    OMP_NUM_THREADS=$SCORE_THREADS_PER_WORKER \
    MKL_NUM_THREADS=$SCORE_THREADS_PER_WORKER \
    OPENBLAS_NUM_THREADS=$SCORE_THREADS_PER_WORKER \
    NUMEXPR_NUM_THREADS=$SCORE_THREADS_PER_WORKER \
    $PYTHON -m src.eval.vbvr_runtime --json
); or _fail "main_v2 scorer runtime contract failed; run uv sync --frozen and restart"
test -n "$scorer_dependency_versions"
or _fail "main_v2 scorer dependency version record is empty"

if test "$PRECONVERTED_MODEL" = 1
    test -f "$CONVERTED_MODEL/model_index.json"
    or _fail "preconverted Diffusers model is incomplete: $CONVERTED_MODEL"
    test -f "$CONVERSION_PROVENANCE"
    or _fail "preconverted model provenance is missing: $CONVERSION_PROVENANCE"
    _validate_converted_model
    or _fail "preconverted model failed structural validation: $CONVERTED_MODEL"
    _converted_model_is_stable
    or _fail "preconverted model is still being modified: $CONVERTED_MODEL"
    echo "[skip] validated stable preconverted model and provenance: $CONVERTED_MODEL"
else
    mkdir -p (dirname "$CONVERTED_MODEL") (dirname "$CONVERSION_PROVENANCE"); or exit 1
    set -l conversion_action skipped
    set -l conversion_waited_seconds 0
    set -l conversion_wait_announced 0
    while true
        if test -f "$CONVERTED_MODEL/model_index.json"; and _conversion_provenance check complete --quiet
            break
        end

        if _try_acquire_conversion_lock
            # The winner must re-check after taking the lock: another process may
            # have completed conversion between the first check and mkdir.
            if test -f "$CONVERTED_MODEL/model_index.json"; and _conversion_provenance check complete --quiet
                _release_conversion_lock; or exit 1
                break
            end

            if test -f "$CONVERTED_MODEL/model_index.json"
                # Recover only the known concurrent-writer failure mode. Inputs
                # must still be byte-for-byte identical to the recorded complete
                # provenance, and the final model must be valid and stable.
                if not _conversion_provenance check-inputs complete --quiet
                    _conversion_provenance check complete
                    or _fail "converted model provenance inputs are missing or stale; use a fresh CONVERTED_MODEL or reconvert explicitly"
                end
                echo "[repair] validating stable converted model before refreshing stale output fingerprints"
                _validate_converted_model; or _fail "converted model failed structural validation: $CONVERTED_MODEL"
                _converted_model_is_stable; or _fail "converted model is still being modified: $CONVERTED_MODEL"
                _conversion_provenance refresh-outputs complete
                or _fail "could not refresh converted model output provenance: $CONVERSION_PROVENANCE"
                set conversion_action repaired
                _release_conversion_lock; or exit 1
                break
            end

            if test -d "$CONVERTED_MODEL"; and test -n (find "$CONVERTED_MODEL" -mindepth 1 -print -quit 2>/dev/null)
                _fail "converted model directory is incomplete: $CONVERTED_MODEL"
            end

            echo "[convert] $CHECKPOINT -> $CONVERTED_MODEL (raw weights, no EMA)"
            _conversion_provenance write in_progress_resume; or exit 1
            $PYTHON -m src.cli.convert_dcp_to_diffusers \
                --checkpoint "$CHECKPOINT" \
                --output "$CONVERTED_MODEL" \
                --base_model "$BASE_MODEL" \
                --torch_dtype bfloat16 \
                --device cuda:0 \
                --no-use_ema \
                --merge_lora \
                --safe_serialization \
                --max_shard_size 10GB \
                --no-fastvideo_compat
            or exit 1
            _validate_converted_model; or _fail "newly converted model failed structural validation: $CONVERTED_MODEL"
            _converted_model_is_stable; or _fail "newly converted model is still being modified: $CONVERTED_MODEL"
            _conversion_provenance promote in_progress_resume; or exit 1
            set conversion_action converted
            _release_conversion_lock; or exit 1
            break
        end

        if test $conversion_wait_announced -eq 0
            echo "[wait] another job owns the conversion lock: $CONVERSION_LOCK"
            set conversion_wait_announced 1
        end
        if test $conversion_waited_seconds -ge $CONVERSION_PROVENANCE_WAIT_SECONDS
            if test -f "$CONVERSION_LOCK/owner"
                echo "[error] conversion lock owner:" >&2
                command cat "$CONVERSION_LOCK/owner" >&2
            end
            _fail "timed out after $CONVERSION_PROVENANCE_WAIT_SECONDS seconds waiting for conversion lock: $CONVERSION_LOCK"
        end
        sleep $CONVERSION_PROVENANCE_POLL_SECONDS
        set conversion_waited_seconds (math "$conversion_waited_seconds + $CONVERSION_PROVENANCE_POLL_SECONDS")
    end

    switch $conversion_action
        case converted
            echo "[done] converted model and provenance are complete: $CONVERTED_MODEL"
        case repaired
            echo "[done] refreshed converted model output provenance: $CONVERSION_PROVENANCE"
        case '*'
            echo "[skip] converted model and provenance are complete: $CONVERTED_MODEL"
    end
end

if set -q CONVERSION_ONLY[1]
    echo "[done] CONVERSION_ONLY requested; stopping before generation"
    exit 0
end

echo "[manifest] validating the flattened bench against $SPLIT_MANIFEST"
set -l eval_json_args \
    --gt_base $GT_BASE \
    --split_manifest $SPLIT_MANIFEST \
    --output $EVAL_JSON \
    --layout domain \
    --expected_samples $EXPECTED_VIDEOS
if test "$USE_ITEM_NUM_FRAMES" = 1
    set -a eval_json_args \
        --generation_fps $INFER_FPS \
        --temporal_alignment $TEMPORAL_ALIGNMENT
end
$PYTHON -m src.eval.build_vbvr_eval_json $eval_json_args
or exit 1

set -l item_num_frames_args
if test "$USE_ITEM_NUM_FRAMES" = 1
    set item_num_frames_args --use_item_num_frames
end

set -l generated_valid 0
if $PYTHON -m src.cli.eval_i2v \
        --eval_json $EVAL_JSON \
        --output_dir $GENERATED_DIR \
        --height $HEIGHT \
        --width $WIDTH \
        --num_frames $NUM_FRAMES \
        --fps $INFER_FPS \
        $item_num_frames_args \
        --validate_only >/dev/null 2>&1
    set generated_valid 1
end

set -l generated_count (_count_mp4 $GENERATED_DIR)
set -l generation_state missing
set -l force_generation 0
set -l generation_provenance_complete 0
if _generation_provenance check-inputs complete --quiet
    set generation_state complete
    if _generation_provenance check complete --quiet
        set generation_provenance_complete 1
    else if test $generated_valid -eq 1
        # Structurally valid media changed behind the complete manifest.
        set force_generation 1
    end
else if _generation_provenance check in_progress_resume --quiet
    set generation_state in_progress_resume
else if _generation_provenance check in_progress_rewrite --quiet
    set generation_state in_progress_rewrite
    set force_generation 1
else if test $generated_count -gt 0
    if set -q FORCE_REGENERATE[1]
        set force_generation 1
    else
        _fail "generated videos do not match current checkpoint/config provenance; set FORCE_REGENERATE=1 or use a fresh OUTPUT_ROOT"
    end
end

if test $generated_valid -eq 1; and test $generation_provenance_complete -eq 1
    echo "[skip] generated videos passed provenance, exact path, and media validation"
else
    if test "$USE_ITEM_NUM_FRAMES" = 1
        echo "[generate] resuming or repairing $generated_count/$EXPECTED_VIDEOS GT-duration-matched videos with $NUM_GPUS-GPU DP"
    else
        echo "[generate] resuming or repairing $generated_count/$EXPECTED_VIDEOS videos with $NUM_GPUS-GPU DP"
    end
    set -l generation_run_state in_progress_resume
    if test $force_generation -eq 1
        set generation_run_state in_progress_rewrite
    end
    _generation_provenance write $generation_run_state; or exit 1
    set -l generation_args \
        --eval_json $EVAL_JSON \
        --model_path $CONVERTED_MODEL \
        --output_dir $GENERATED_DIR \
        --height $HEIGHT \
        --width $WIDTH \
        --num_frames $NUM_FRAMES \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --guidance_scale $GUIDANCE_SCALE \
        --fps $INFER_FPS \
        --seed $SEED
    set -a generation_args $item_num_frames_args
    set -l generation_module src.cli.eval_i2v
    if test $GENERATION_MODE = cps
        set generation_module src.cli.eval_i2v_cps
        set -a generation_args --noise_level $CPS_NOISE_LEVEL
    else if test $ODE_SOLVER = euler
        set generation_module src.cli.eval_i2v_euler
        set -a generation_args --disable_progress_bar
    else
        set -a generation_args --disable_progress_bar
    end
    if test $force_generation -eq 1
        set -a generation_args --force
    end
    $TORCHRUN --standalone --nproc_per_node=$NUM_GPUS -m $generation_module $generation_args
    or exit 1
    $PYTHON -m src.cli.eval_i2v \
        --eval_json $EVAL_JSON \
        --output_dir $GENERATED_DIR \
        --height $HEIGHT \
        --width $WIDTH \
        --num_frames $NUM_FRAMES \
        --fps $INFER_FPS \
        $item_num_frames_args \
        --validate_only
    or _fail "generated video set failed exact path or media validation"
    _generation_provenance promote $generation_run_state; or exit 1
end

echo "[prepare] retaining every frame, resizing without crop, and retiming to <= $MAX_DURATION seconds"
set -l prepared_count (_count_mp4 $PREPARED_DIR)
set -l preparation_run_state in_progress_resume
set -l force_preparation 0
if _preparation_provenance check complete --quiet
    set force_preparation 0
else if _preparation_provenance check-inputs complete --quiet
    # A complete output fingerprint changed, so rebuild every prepared video.
    set force_preparation 1
else if _preparation_provenance check in_progress_resume --quiet
    set force_preparation 0
else if _preparation_provenance check in_progress_rewrite --quiet
    set force_preparation 1
else if test $prepared_count -gt 0
    if set -q FORCE_REPREPARE[1]; or test $force_generation -eq 1
        set force_preparation 1
    else
        _fail "prepared videos do not match current preparation provenance; set FORCE_REPREPARE=1 or use a fresh OUTPUT_ROOT"
    end
end

if test $force_preparation -eq 1
    set preparation_run_state in_progress_rewrite
end
_preparation_provenance write $preparation_run_state; or exit 1
set -l preparation_args \
    --input-dir $GENERATED_DIR \
    --output-dir $PREPARED_DIR \
    --width $PREPARED_WIDTH \
    --height $PREPARED_HEIGHT \
    --max-duration $MAX_DURATION \
    --workers $PREP_WORKERS \
    --crf $PREP_CRF \
    --expected-videos $EXPECTED_VIDEOS
if test $force_preparation -eq 1
    set -a preparation_args --force
end
$PYTHON -m src.cli.prepare_vbvr_eval_videos $preparation_args
or exit 1

set prepared_count (_count_mp4 $PREPARED_DIR)
test $prepared_count -eq $EXPECTED_VIDEOS
or _fail "video preparation ended with $prepared_count/$EXPECTED_VIDEOS videos"
_preparation_provenance promote $preparation_run_state; or exit 1

set -l prepared_name (basename (string trim -r -c / -- $PREPARED_DIR))
set -l result_path $SCORE_DIR/$prepared_name"_vbvr_results.json"
rm -f $result_path $SCORE_PROVENANCE
_score_provenance write in_progress_rewrite $result_path; or exit 1
echo "[score] EvalKit $EVALKIT_DIR with $SCORE_WORKERS CPU workers"
env CUDA_VISIBLE_DEVICES= EASYOCR_MODULE_PATH=(realpath $EASYOCR_ROOT) \
    OMP_NUM_THREADS=$SCORE_THREADS_PER_WORKER \
    MKL_NUM_THREADS=$SCORE_THREADS_PER_WORKER \
    OPENBLAS_NUM_THREADS=$SCORE_THREADS_PER_WORKER \
    NUMEXPR_NUM_THREADS=$SCORE_THREADS_PER_WORKER \
    $PYTHON -m src.eval.vbvr_run_evaluation_parallel \
    --model_path $PREPARED_DIR \
    --gt_base $GT_BASE \
    --output_dir $SCORE_DIR \
    --evalkit_dir $EVALKIT_DIR \
    --expected_evalkit_source_sha256 $EVALKIT_SOURCE_SHA256 \
    --expected_videos $EXPECTED_VIDEOS \
    --device cpu \
    --num_workers $SCORE_WORKERS \
    --threads_per_worker $SCORE_THREADS_PER_WORKER
or exit 1

_valid_score_result $result_path
or _fail "score JSON is partial or contains scorer errors: $result_path"
_score_provenance promote in_progress_rewrite $result_path; or exit 1

$PYTHON -c '
import json, sys
summary = json.load(open(sys.argv[1]))["summary"]
for key in ("In_Domain", "Out_of_Domain", "overall"):
    item = summary[key]
    print("{}: {:.6f} ({} samples)".format(key, item["mean_score"], item["num_samples"]))
' $result_path

echo "[done] result: $result_path"
