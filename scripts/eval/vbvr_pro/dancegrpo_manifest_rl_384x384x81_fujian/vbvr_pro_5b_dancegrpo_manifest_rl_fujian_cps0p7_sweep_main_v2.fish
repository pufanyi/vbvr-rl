#!/usr/bin/env fish

# Evaluate every complete checkpoint in the Fujian 384x384x81 manifest-RL run
# at a configurable native inference resolution (384x384 by default). Waves
# use all eight local GPUs: four two-GPU jobs, two four-GPU jobs, or one
# eight-GPU job depending on the number of checkpoints left in the final wave.

source (dirname (status filename))/../../../lib/env.fish

function _fail
    echo "[error] $argv" >&2
    exit 1
end

set -l script_dir (dirname (status filename))
set -g _fujian_sweep_launcher $script_dir/vbvr_pro_5b_dancegrpo_manifest_rl_fujian_checkpoint_cps0p7_main_v2.fish
test -f $_fujian_sweep_launcher
or _fail "launcher does not exist: $_fujian_sweep_launcher"

set -q CHECKPOINT_ROOT[1]
or set -gx CHECKPOINT_ROOT storage/checkpoints/dancegrpo_vbvr_pro_5b_384x384x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl_fujian
set -q GT_BASE[1]
or set -gx GT_BASE (realpath storage/datasets/vbvr-pro-eval-500)
set -q SPLIT_MANIFEST[1]
or set -gx SPLIT_MANIFEST $GT_BASE/split_manifest.json
set -q EVALKIT_REV[1]
or set -gx EVALKIT_REV e140038f2aee76ca518f464755fa8bc19b783ba5
set -q EVALKIT_SOURCE_SHA256[1]
or set -gx EVALKIT_SOURCE_SHA256 4cc7d028d4106a28190a63bc179562d5ac9add9263cb71926dd6385c5714bcf8
set -q EASYOCR_ROOT[1]
or set -gx EASYOCR_ROOT storage/evalkits/easyocr-shared
set -q EASYOCR_SOURCE_MODELS[1]
or set -gx EASYOCR_SOURCE_MODELS $EASYOCR_ROOT/model
set -q HEIGHT[1]
or set -gx HEIGHT 384
set -q WIDTH[1]
or set -gx WIDTH 384
set -g _fujian_native_shape "$HEIGHT"x"$WIDTH"x81
set -q EVAL_LOG_DIR[1]
or set -g EVAL_LOG_DIR storage/eval_logs/vbvr_pro_main_v2_$_fujian_native_shape"_manifest_rl_fujian_cps0p7_evalkit_4cc7d028"
set -q OUTPUT_BASE[1]
or set -gx OUTPUT_BASE storage/eval_out/vbvr_pro_main_v2_$_fujian_native_shape"_manifest_rl_fujian_eval500_181e2010_manifest_afab352e_evalkit_4cc7d028"
set -q CONVERTED_BASE[1]
or set -g CONVERTED_BASE storage/models/dcp_converted_5b
set -q CONVERTED_PREFIX[1]
or set -g CONVERTED_PREFIX dancegrpo_vbvr_pro_5b_384x384x81_rule_cps0p7_from_diffsynth_step35500_bs32_lr_5e-6_manifest_rl_fujian

set -g _fujian_sweep_log_dir $EVAL_LOG_DIR
set -g _fujian_sweep_output_base $OUTPUT_BASE
set -g _fujian_sweep_converted_base $CONVERTED_BASE
set -g _fujian_sweep_converted_prefix $CONVERTED_PREFIX
mkdir -p $_fujian_sweep_log_dir
or _fail "could not create log directory: $_fujian_sweep_log_dir"

test -d $CHECKPOINT_ROOT; or _fail "checkpoint root does not exist: $CHECKPOINT_ROOT"
test -d $GT_BASE; or _fail "GT_BASE does not exist: $GT_BASE"
test -f $SPLIT_MANIFEST; or _fail "split manifest does not exist: $SPLIT_MANIFEST"
test ! -e $GT_BASE/.cache
or _fail "move the Hugging Face local-dir .cache outside GT_BASE before provenance fingerprinting: $GT_BASE/.cache"

set -g _fujian_manifest_sha256 (sha256sum $SPLIT_MANIFEST | awk '{print $1}')
test "$_fujian_manifest_sha256" = afab352e08c590c9f4b480ef314b37f6896eef6430f42ea6c0ce0494f2aa8c4e
or _fail "unexpected sanitized split manifest fingerprint: $_fujian_manifest_sha256"
set -l checksums_sha256 (sha256sum $GT_BASE/SHA256SUMS | awk '{print $1}')
test "$checksums_sha256" = a67c534293724ddfc6657af755ab65e9b1354879deb2cfc47de22ede43942861
or _fail "unexpected dataset checksum-manifest fingerprint: $checksums_sha256"

echo "[dataset] verifying the complete downloaded VBVR-Pro eval snapshot"
pushd $GT_BASE >/dev/null; or exit 1
sha256sum -c SHA256SUMS --quiet
set -l checksum_status $status
popd >/dev/null; or exit 1
test $checksum_status -eq 0; or _fail "VBVR-Pro eval snapshot failed SHA-256 verification"
set -gx WAN_TRAINER_VBVR_EVAL_DATA_VERIFIED 1

set -l checkpoint_steps
for checkpoint_dir in (find $CHECKPOINT_ROOT -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
    test -f $checkpoint_dir/high/.metadata; or begin
        echo "[skip] incomplete checkpoint without high/.metadata: $checkpoint_dir" >&2
        continue
    end
    set -a checkpoint_steps (string replace 'checkpoint-' '' -- (basename $checkpoint_dir))
end
test (count $checkpoint_steps) -gt 0
or _fail "no complete checkpoints found under $CHECKPOINT_ROOT"

function _converted_model_for_step
    set -l step $argv[1]
    echo $_fujian_sweep_converted_base/$_fujian_sweep_converted_prefix"_checkpoint-$step"
end

function _output_root_for_step
    set -l step $argv[1]
    echo $_fujian_sweep_output_base/dancegrpo_vbvr_pro_5b_checkpoint-$step-cps-noise-0.7
end

function _checkpoint_complete
    if set -q DRY_RUN[1]
        return 0
    end

    set -l step $argv[1]
    set -l output_root (_output_root_for_step $step)
    set -l converted_model (_converted_model_for_step $step)
    .venv/bin/python -c '
import json
import sys
from pathlib import Path

from src.eval.evaluation_provenance import verify_recorded_manifest
from src.eval.vbvr_runtime import validate_vbvr_scorer_runtime

root = Path(sys.argv[1])
converted_model = Path(sys.argv[2])
expected_manifest_sha256 = sys.argv[3]
expected_evalkit_revision = sys.argv[4]
expected_evalkit_source_sha256 = sys.argv[5]
expected_gt_base = Path(sys.argv[6])
expected_height = sys.argv[7]
expected_width = sys.argv[8]
result = root / "scores" / "eval_1024x1024_81f_fps16_5p0625s_vbvr_results.json"
generation_provenance = root / "generation-provenance.json"
preparation_provenance = root / "preparation-provenance.json"
score_provenance = root / "score-provenance.json"
workbook = root / "scores" / "eval_1024x1024_81f_fps16_5p0625s_task_scores.xlsx"
summary_path = root / "final_scores.txt"
required = (
    result,
    generation_provenance,
    preparation_provenance,
    score_provenance,
    workbook,
    summary_path,
)
if not all(path.is_file() for path in required):
    raise SystemExit(1)

data = json.loads(result.read_text())
samples = data.get("samples")
if not isinstance(samples, list) or len(samples) != 500 or any(sample.get("error") for sample in samples):
    raise SystemExit(1)
tasks = {sample.get("task_name") for sample in samples}
if len(tasks) != 100:
    raise SystemExit(1)
result_summary = data.get("summary", {})
for key, expected_count in (("In_Domain", 250), ("Out_of_Domain", 250), ("overall", 500)):
    if result_summary.get(key, {}).get("num_samples") != expected_count:
        raise SystemExit(1)
if len(result_summary.get("overall", {}).get("by_task", {})) != 100:
    raise SystemExit(1)

generation = json.loads(generation_provenance.read_text())
preparation = json.loads(preparation_provenance.read_text())
score = json.loads(score_provenance.read_text())
for manifest_path, stage in (
    (generation_provenance, "vbvr-pro-generation"),
    (preparation_provenance, "vbvr-pro-preparation"),
    (score_provenance, "vbvr-pro-score"),
):
    matches, detail = verify_recorded_manifest(manifest_path, expected_stage=stage, require_complete=True)
    if not matches:
        print(f"[incomplete] {detail}", file=sys.stderr)
        raise SystemExit(1)

recorded_manifest = generation.get("files", {}).get("split_manifest", {})
if recorded_manifest.get("sha256") != expected_manifest_sha256:
    raise SystemExit(1)
generation_values = generation.get("values", {})
expected_generation_values = {
    "state": "complete",
    "height": expected_height,
    "width": expected_width,
    "num_frames": "81",
    "fps": "16",
    "num_inference_steps": "30",
    "guidance_scale": "1.0",
    "seed": "0",
    "generation_mode": "cps",
    "cps_noise_level": "0.7",
}
if any(str(generation_values.get(key)) != value for key, value in expected_generation_values.items()):
    raise SystemExit(1)
preparation_values = preparation.get("values", {})
if preparation_values.get("state") != "complete" or str(preparation_values.get("max_duration")) != "5.0625":
    raise SystemExit(1)
score_values = score.get("values", {})
if score_values.get("state") != "complete":
    raise SystemExit(1)
if score_values.get("evalkit_revision") != expected_evalkit_revision:
    raise SystemExit(1)
if score_values.get("evalkit_source_sha256") != expected_evalkit_source_sha256:
    raise SystemExit(1)
dependencies = json.loads(score_values["scorer_dependencies"])
expected_runtime = validate_vbvr_scorer_runtime()
if dependencies.get("contract") != expected_runtime["contract"]:
    raise SystemExit(1)
if dependencies.get("sha256") != expected_runtime["sha256"]:
    raise SystemExit(1)

generated = root / f"generated_{expected_height}x{expected_width}x81"
prepared = root / "eval_1024x1024_81f_fps16_5p0625s"
bindings = (
    (generation, "media_trees", "generated_videos", generated),
    (generation, "trees", "converted_model", converted_model),
    (generation, "trees", "eval_source", expected_gt_base),
    (preparation, "media_trees", "prepared_videos", prepared),
    (score, "trees", "prepared_videos", prepared),
    (score, "trees", "ground_truth", expected_gt_base),
    (score, "output_files", "result", result),
)
for manifest, section, name, expected_path in bindings:
    recorded_path = manifest.get(section, {}).get(name, {}).get("path")
    if recorded_path != str(expected_path.resolve()):
        raise SystemExit(1)
if sum(1 for _ in generated.rglob("*.mp4")) != 500:
    raise SystemExit(1)
if sum(1 for _ in prepared.rglob("*.mp4")) != 500:
    raise SystemExit(1)
' $output_root $converted_model $_fujian_manifest_sha256 $EVALKIT_REV $EVALKIT_SOURCE_SHA256 $GT_BASE $HEIGHT $WIDTH
end

function _launch_wave
    set -l wave_steps $argv
    set -l wave_devices
    switch (count $wave_steps)
        case 1
            set wave_devices 0,1,2,3,4,5,6,7
        case 2
            set wave_devices 0,1,2,3 4,5,6,7
        case 3
            set wave_devices 0,1 2,3 4,5,6,7
        case 4
            set wave_devices 0,1 2,3 4,5 6,7
        case '*'
            echo "[error] invalid wave size: "(count $wave_steps) >&2
            return 1
    end

    set -l running_steps
    set -l running_pids
    set -l running_logs
    for index in (seq (count $wave_steps))
        set -l step $wave_steps[$index]
        set -l devices $wave_devices[$index]
        set -l num_gpus (count (string split , -- $devices))
        set -l output_root (_output_root_for_step $step)
        set -l converted_model (_converted_model_for_step $step)
        set -l log_path $_fujian_sweep_log_dir/checkpoint-$step-cps-noise-0.7.log

        if not set -q DRY_RUN[1]; and _checkpoint_complete $step
            echo "[skip] checkpoint-$step is already complete: $output_root"
            continue
        end

        echo "[start] "(date --iso-8601=seconds)" checkpoint-$step GPUs=$devices log=$log_path"
        env \
            CHECKPOINT_STEP=$step \
            CHECKPOINT_ROOT=$CHECKPOINT_ROOT \
            CONVERTED_MODEL=$converted_model \
            OUTPUT_BASE=$OUTPUT_BASE \
            OUTPUT_ROOT=$output_root \
            GT_BASE=$GT_BASE \
            SPLIT_MANIFEST=$SPLIT_MANIFEST \
            EVALKIT_REV=$EVALKIT_REV \
            EVALKIT_SOURCE_SHA256=$EVALKIT_SOURCE_SHA256 \
            EASYOCR_ROOT=$EASYOCR_ROOT \
            EASYOCR_SOURCE_MODELS=$EASYOCR_SOURCE_MODELS \
            WAN_TRAINER_VBVR_EVAL_DATA_VERIFIED=1 \
            HEIGHT=$HEIGHT \
            WIDTH=$WIDTH \
            NUM_GPUS=$num_gpus \
            CUDA_DEVICES=$devices \
            PREP_WORKERS=4 \
            SCORE_WORKERS=2 \
            SCORE_THREADS_PER_WORKER=8 \
            fish $_fujian_sweep_launcher >$log_path 2>&1 &
        set -a running_steps $step
        set -a running_pids $last_pid
        set -a running_logs $log_path
    end

    if test (count $running_pids) -eq 0
        return 0
    end

    set -l wave_failed 0
    for index in (seq (count $running_pids))
        wait $running_pids[$index]
        set -l wait_status $status
        set -l step $running_steps[$index]
        if test $wait_status -ne 0; or not _checkpoint_complete $step
            set wave_failed 1
            echo "[error] "(date --iso-8601=seconds)" checkpoint-$step did not produce a complete strict result; log=$running_logs[$index]" >&2
            tail -n 120 $running_logs[$index] >&2
        else
            echo "[done]  "(date --iso-8601=seconds)" checkpoint-$step log=$running_logs[$index]"
        end
    end
    test $wave_failed -eq 0
end

echo "[sweep] checkpoints: $checkpoint_steps"
echo "[sweep] sampler: 30-step Flow-CPS 0.7, CFG 1.0, seed 0"
echo "[sweep] native media: $_fujian_native_shape at 16 FPS"
echo "[sweep] GT snapshot: $GT_BASE"
echo "[sweep] split manifest: $SPLIT_MANIFEST ($_fujian_manifest_sha256)"
echo "[sweep] output base: $OUTPUT_BASE"

set -l wave_start 1
set -l checkpoint_count (count $checkpoint_steps)
while test $wave_start -le $checkpoint_count
    set -l wave_steps
    for offset in 0 1 2 3
        set -l position (math $wave_start + $offset)
        if test $position -le $checkpoint_count
            set -a wave_steps $checkpoint_steps[$position]
        end
    end
    echo "[wave] checkpoints: $wave_steps"
    _launch_wave $wave_steps
    or _fail "checkpoint wave failed: $wave_steps"
    set wave_start (math $wave_start + 4)
end

if not set -q DRY_RUN[1]
    .venv/bin/python -c '
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for run in root.glob("dancegrpo_vbvr_pro_5b_checkpoint-*-cps-noise-0.7"):
    step = int(run.name.split("checkpoint-", 1)[1].split("-", 1)[0])
    result = run / "scores" / "eval_1024x1024_81f_fps16_5p0625s_vbvr_results.json"
    summary = json.loads(result.read_text())["summary"]
    categories = summary["overall"]["by_category"]
    rows.append(
        (
            step,
            summary["overall"]["mean_score"],
            summary["In_Domain"]["mean_score"],
            summary["Out_of_Domain"]["mean_score"],
            categories["Abstraction"],
            categories["Perception"],
            categories["Spatiality"],
            categories["Transformation"],
            categories["Knowledge"],
        )
    )
header = (
    "checkpoint\toverall\tin_domain\tout_of_domain\tabstraction\tperception"
    "\tspatiality\ttransformation\tknowledge"
)
lines = [header]
for row in sorted(rows):
    lines.append("\t".join((str(row[0]), *(f"{value:.6f}" for value in row[1:]))))
table = "\n".join(lines) + "\n"
summary_path = root / "checkpoint_scores.tsv"
summary_path.write_text(table)
print(table, end="")
print(f"[done] checkpoint summary: {summary_path}")
' $OUTPUT_BASE
    or _fail "could not write checkpoint summary: $OUTPUT_BASE/checkpoint_scores.tsv"
end

echo "[done] all Fujian $_fujian_native_shape manifest-RL CPS 0.7 checkpoint evaluations completed"
