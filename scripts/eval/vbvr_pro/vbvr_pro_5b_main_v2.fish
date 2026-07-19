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

set -q EVALKIT_DIR[1]; or set EVALKIT_DIR storage/evalkits/vbvr-evalkit-interleave-main_v2
set -q EVALKIT_REPO[1]; or set EVALKIT_REPO git@github.com:xujunxiangwork/VBVR-Evalkit-Interleave.git
set -q EVALKIT_REV[1]; or set EVALKIT_REV 42a1593d8e493370c768be8e43646f0e0a9d8525
set -q EASYOCR_SOURCE_MODELS[1]; or set EASYOCR_SOURCE_MODELS /mnt/aigc/xujunxiang/Code/VBVR-Bench/VBVR-EvalKit/easyocr_models

set -q OUTPUT_ROOT[1]; or set OUTPUT_ROOT storage/eval_out/vbvr_pro_main_v2/dancegrpo_vbvr_pro_5b_checkpoint-1200
set -q EVAL_JSON[1]; or set EVAL_JSON $OUTPUT_ROOT/eval_samples.json
set -q GENERATED_DIR[1]; or set GENERATED_DIR $OUTPUT_ROOT/generated_256x256x161
set -q PREPARED_DIR[1]; or set PREPARED_DIR $OUTPUT_ROOT/eval_1024x1024_161f_5s
set -q SCORE_DIR[1]; or set SCORE_DIR $OUTPUT_ROOT/scores
set -q EASYOCR_ROOT[1]; or set EASYOCR_ROOT $OUTPUT_ROOT/easyocr
set -q CONVERSION_PROVENANCE[1]; or set CONVERSION_PROVENANCE $CONVERTED_MODEL.wan-trainer-provenance.json
set -q GENERATION_PROVENANCE[1]; or set GENERATION_PROVENANCE $OUTPUT_ROOT/generation-provenance.json
set -q PREPARATION_PROVENANCE[1]; or set PREPARATION_PROVENANCE $OUTPUT_ROOT/preparation-provenance.json
set -q SCORE_PROVENANCE[1]; or set SCORE_PROVENANCE $OUTPUT_ROOT/score-provenance.json

set -q EXPECTED_VIDEOS[1]; or set EXPECTED_VIDEOS 500
set -q NUM_GPUS[1]; or set NUM_GPUS 8
set -q CUDA_DEVICES[1]; or set CUDA_DEVICES 0,1,2,3,4,5,6,7
set -q NUM_FRAMES[1]; or set NUM_FRAMES 161
set -q HEIGHT[1]; or set HEIGHT 256
set -q WIDTH[1]; or set WIDTH 256
set -q INFER_FPS[1]; or set INFER_FPS 16
set -q NUM_INFERENCE_STEPS[1]; or set NUM_INFERENCE_STEPS 50
set -q GUIDANCE_SCALE[1]; or set GUIDANCE_SCALE 5.0
set -q SEED[1]; or set SEED 0
set -q GENERATION_MODE[1]; or set GENERATION_MODE ode
set -q CPS_NOISE_LEVEL[1]; or set CPS_NOISE_LEVEL 0.7

set -q PREPARED_HEIGHT[1]; or set PREPARED_HEIGHT 1024
set -q PREPARED_WIDTH[1]; or set PREPARED_WIDTH 1024
set -q MAX_DURATION[1]; or set MAX_DURATION 5
set -q PREP_WORKERS[1]; or set PREP_WORKERS 8
set -q PREP_CRF[1]; or set PREP_CRF 12
set -q SCORE_WORKERS[1]; or set SCORE_WORKERS 8
set -q SCORE_THREADS_PER_WORKER[1]; or set SCORE_THREADS_PER_WORKER 16

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
if test $GENERATION_MODE = cps
    $PYTHON -c 'import sys; value=float(sys.argv[1]); raise SystemExit(0 if 0.0 <= value <= 1.0 else 1)' \
        $CPS_NOISE_LEVEL
    or _fail "CPS_NOISE_LEVEL must be in [0, 1], got $CPS_NOISE_LEVEL"
end

if set -q DRY_RUN[1]
    echo "[dry-run] mode=$GENERATION_MODE checkpoint=$CHECKPOINT"
    echo "[dry-run] converted_model=$CONVERTED_MODEL output_root=$OUTPUT_ROOT"
    if test $GENERATION_MODE = cps
        echo "[dry-run] cps_noise_level=$CPS_NOISE_LEVEL steps=$NUM_INFERENCE_STEPS cfg=$GUIDANCE_SCALE"
    else
        echo "[dry-run] steps=$NUM_INFERENCE_STEPS cfg=$GUIDANCE_SCALE"
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
    set -l output_args
    if test $mode = promote
        set output_args --output-tree converted_model=$CONVERTED_MODEL
    else if test $mode = check; and test $state = complete
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
        $output_args
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
    if test $GENERATION_MODE = cps
        set generator_source src/cli/eval_i2v_cps.py
        set sampler_args \
            --value generation_mode=cps \
            --value cps_noise_level=$CPS_NOISE_LEVEL
    end
    $PYTHON -m src.eval.evaluation_provenance $mode \
        --manifest $GENERATION_PROVENANCE \
        --stage vbvr-pro-generation \
        --value state=$state \
        --value height=$HEIGHT \
        --value width=$WIDTH \
        --value num_frames=$NUM_FRAMES \
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
        --value device=cpu \
        --value workers=$SCORE_WORKERS \
        --value threads_per_worker=$SCORE_THREADS_PER_WORKER \
        --file preparation_provenance=$PREPARATION_PROVENANCE \
        --file scorer_entrypoint=$EVALKIT_DIR/run_evaluation.py \
        --file scorer_wrapper=src/eval/vbvr_run_evaluation_parallel.py \
        --tree prepared_videos=$PREPARED_DIR \
        --tree ground_truth=$GT_BASE \
        $output_args
end

for required in $CHECKPOINT $BASE_MODEL $GT_BASE $SPLIT_MANIFEST $EASYOCR_SOURCE_MODELS
    test -e $required; or _fail "required path does not exist: $required"
end

set -l visible_count (count (string split , -- $CUDA_DEVICES))
test $visible_count -ge $NUM_GPUS; or _fail "CUDA_DEVICES exposes $visible_count devices, but NUM_GPUS=$NUM_GPUS"
set -gx CUDA_VISIBLE_DEVICES $CUDA_DEVICES

if not test -f $EVALKIT_DIR/run_evaluation.py
    if test -e $EVALKIT_DIR
        _fail "EvalKit directory exists but is incomplete: $EVALKIT_DIR"
    end
    echo "[evalkit] cloning main_v2 from $EVALKIT_REPO"
    git clone --depth 1 --branch main_v2 $EVALKIT_REPO $EVALKIT_DIR; or exit 1
end

mkdir -p $EASYOCR_ROOT/model $EASYOCR_ROOT/user_network; or exit 1
for model_file in craft_mlt_25k.pth english_g2.pth
    test -f $EASYOCR_SOURCE_MODELS/$model_file
    or _fail "missing pre-populated EasyOCR weight: $EASYOCR_SOURCE_MODELS/$model_file"
    cp -f $EASYOCR_SOURCE_MODELS/$model_file $EASYOCR_ROOT/model/$model_file; or exit 1
end

set -l evalkit_easyocr_models $EVALKIT_DIR/easyocr_models
set -l personal_easyocr_models (realpath $EASYOCR_ROOT/model)
if test -L $evalkit_easyocr_models
    if test (realpath $evalkit_easyocr_models) != $personal_easyocr_models
        rm $evalkit_easyocr_models; or exit 1
    end
else if test -e $evalkit_easyocr_models
    _fail "expected a symlink at $evalkit_easyocr_models; refusing to replace an existing directory"
end
if not test -e $evalkit_easyocr_models
    ln -s $personal_easyocr_models $evalkit_easyocr_models; or exit 1
end

$PYTHON -c 'import easyocr, norfair, scipy, skimage' 2>/dev/null
or _fail "main_v2 dependencies are missing; install norfair and easyocr into .venv before scoring"

if not test -f $CONVERTED_MODEL/model_index.json
    if test -d $CONVERTED_MODEL; and test -n (find $CONVERTED_MODEL -mindepth 1 -print -quit 2>/dev/null)
        _fail "converted model directory is incomplete: $CONVERTED_MODEL"
    end
    echo "[convert] $CHECKPOINT -> $CONVERTED_MODEL (raw weights, no EMA)"
    _conversion_provenance write in_progress_resume; or exit 1
    $PYTHON -m src.cli.convert_dcp_to_diffusers \
        --checkpoint $CHECKPOINT \
        --output $CONVERTED_MODEL \
        --base_model $BASE_MODEL \
        --torch_dtype bfloat16 \
        --device cuda:0 \
        --no-use_ema \
        --merge_lora \
        --safe_serialization \
        --max_shard_size 10GB \
        --no-fastvideo_compat
    or exit 1
    _conversion_provenance promote in_progress_resume; or exit 1
else
    _conversion_provenance check complete
    or _fail "converted model provenance is missing or stale; use a fresh CONVERTED_MODEL or reconvert explicitly"
    echo "[skip] converted model and provenance are complete: $CONVERTED_MODEL"
end

echo "[manifest] validating the flattened bench against $SPLIT_MANIFEST"
$PYTHON -m src.eval.build_vbvr_eval_json \
    --gt_base $GT_BASE \
    --split_manifest $SPLIT_MANIFEST \
    --output $EVAL_JSON \
    --layout domain \
    --expected_samples $EXPECTED_VIDEOS
or exit 1

set -l generated_valid 0
if $PYTHON -m src.cli.eval_i2v \
        --eval_json $EVAL_JSON \
        --output_dir $GENERATED_DIR \
        --height $HEIGHT \
        --width $WIDTH \
        --num_frames $NUM_FRAMES \
        --fps $INFER_FPS \
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
    echo "[generate] resuming or repairing $generated_count/$EXPECTED_VIDEOS videos with $NUM_GPUS-GPU DP"
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
    set -l generation_module src.cli.eval_i2v
    if test $GENERATION_MODE = cps
        set generation_module src.cli.eval_i2v_cps
        set -a generation_args --noise_level $CPS_NOISE_LEVEL
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
