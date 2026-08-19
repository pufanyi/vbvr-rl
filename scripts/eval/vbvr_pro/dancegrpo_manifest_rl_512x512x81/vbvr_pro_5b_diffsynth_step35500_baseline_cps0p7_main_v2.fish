#!/usr/bin/env fish

# Sampler-matched baseline for the native-512/e140 production checkpoint curve.
# The default is Flow-CPS 0.7; callers may select another CPS coefficient or
# matched Euler/UniPC ODE while retaining the same 30-step/CFG/seed/media and
# pinned-e140 scoring contract.

source (dirname (status filename))/../../../lib/env.fish

function _fail
    echo "[error] $argv" >&2
    exit 1
end

set -lx PRECONVERTED_MODEL 1
set -q CONVERTED_MODEL[1]
or set -lx CONVERTED_MODEL storage/models/diffsynth_converted_5b/wan2.2-TI2V-5B_260715_vbvr_pro_step-35500
set -q CONVERSION_PROVENANCE[1]
or set -lx CONVERSION_PROVENANCE $CONVERTED_MODEL/conversion_metadata.json
set -q CHECKPOINT[1]
or set -lx CHECKPOINT $CONVERTED_MODEL

set -q GT_BASE[1]
or set -lx GT_BASE (realpath storage/datasets/vbvr-pro-eval-500)
set -q SPLIT_MANIFEST[1]
or set -lx SPLIT_MANIFEST $GT_BASE/split_manifest.json
set -q EVALKIT_DIR[1]
or set -lx EVALKIT_DIR storage/evalkits/vbvr-evalkit-interleave-main_v2-e140038f
set -q EVALKIT_REV[1]
or set -lx EVALKIT_REV e140038f2aee76ca518f464755fa8bc19b783ba5
set -q EVALKIT_SOURCE_SHA256[1]
or set -lx EVALKIT_SOURCE_SHA256 4cc7d028d4106a28190a63bc179562d5ac9add9263cb71926dd6385c5714bcf8
set -q EASYOCR_ROOT[1]
or set -lx EASYOCR_ROOT storage/evalkits/easyocr-shared
set -q EASYOCR_SOURCE_MODELS[1]
or set -lx EASYOCR_SOURCE_MODELS $EASYOCR_ROOT/model

set -q GENERATION_MODE[1]
or set -lx GENERATION_MODE cps
set -q ODE_SOLVER[1]
or set -lx ODE_SOLVER unipc
set -q CPS_NOISE_LEVEL[1]
or set -lx CPS_NOISE_LEVEL 0.7
contains -- $GENERATION_MODE cps ode
or _fail "GENERATION_MODE must be cps or ode: $GENERATION_MODE"
contains -- $ODE_SOLVER euler unipc
or _fail "ODE_SOLVER must be euler or unipc: $ODE_SOLVER"
set -l sampler_label
if test "$GENERATION_MODE" = cps
    .venv/bin/python -c 'import sys; x=float(sys.argv[1]); raise SystemExit(0 if 0 <= x <= 1 else 1)' $CPS_NOISE_LEVEL
    or _fail "CPS_NOISE_LEVEL must be in [0, 1]: $CPS_NOISE_LEVEL"
    set sampler_label cps-noise-$CPS_NOISE_LEVEL
else
    set sampler_label $ODE_SOLVER-ode-30steps-cfg1
end

set -l default_output_root storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_e140_lr5e6_eval500_181e2010_manifest_afab352e_evalkit_4cc7d028/diffsynth_step35500-baseline-$sampler_label
if test "$GENERATION_MODE" = cps; and test "$CPS_NOISE_LEVEL" = 0.7
    # Keep the already-audited historical path reusable.
    set default_output_root storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_e140_lr5e6_eval500_181e2010_manifest_afab352e_evalkit_4cc7d028/diffsynth_step35500-baseline-cps0p7-30steps-cfg1
end

set -q OUTPUT_ROOT[1]
or set -lx OUTPUT_ROOT $default_output_root
set -q EVAL_JSON[1]
or set -lx EVAL_JSON $OUTPUT_ROOT/eval_samples.json
set -q GENERATED_DIR[1]
or set -lx GENERATED_DIR $OUTPUT_ROOT/generated_512x512x81
set -q PREPARED_DIR[1]
or set -lx PREPARED_DIR $OUTPUT_ROOT/eval_1024x1024_81f_fps16_5p0625s
set -q SCORE_DIR[1]
or set -lx SCORE_DIR $OUTPUT_ROOT/scores

set -lx NUM_INFERENCE_STEPS 30
set -lx GUIDANCE_SCALE 1.0
set -lx SEED 0
set -lx HEIGHT 512
set -lx WIDTH 512
set -lx NUM_FRAMES 81
set -lx INFER_FPS 16
set -lx MAX_DURATION 5.0625
set -q PREP_WORKERS[1]
or set -lx PREP_WORKERS 4
set -q SCORE_WORKERS[1]
or set -lx SCORE_WORKERS 2
set -q SCORE_THREADS_PER_WORKER[1]
or set -lx SCORE_THREADS_PER_WORKER 8

test -d $GT_BASE; or _fail "GT_BASE does not exist: $GT_BASE"
test -f $SPLIT_MANIFEST; or _fail "split manifest does not exist: $SPLIT_MANIFEST"
test ! -e $GT_BASE/.cache
or _fail "move the Hugging Face local-dir .cache outside GT_BASE before provenance fingerprinting: $GT_BASE/.cache"

set -l manifest_sha256 (sha256sum $SPLIT_MANIFEST | awk '{print $1}')
test "$manifest_sha256" = afab352e08c590c9f4b480ef314b37f6896eef6430f42ea6c0ce0494f2aa8c4e
or _fail "unexpected sanitized split manifest fingerprint: $manifest_sha256"
set -l checksums_sha256 (sha256sum $GT_BASE/SHA256SUMS | awk '{print $1}')
test "$checksums_sha256" = a67c534293724ddfc6657af755ab65e9b1354879deb2cfc47de22ede43942861
or _fail "unexpected dataset checksum-manifest fingerprint: $checksums_sha256"

echo "[dataset] verifying the complete downloaded VBVR-Pro eval snapshot"
pushd $GT_BASE >/dev/null; or exit 1
sha256sum -c SHA256SUMS --quiet
set -l checksum_status $status
popd >/dev/null; or exit 1
test $checksum_status -eq 0; or _fail "VBVR-Pro eval snapshot failed SHA-256 verification"
set -lx WAN_TRAINER_VBVR_EVAL_DATA_VERIFIED 1

set -l script_dir (dirname (status filename))
fish $script_dir/../vbvr_pro_5b_main_v2.fish $argv
set -l pipeline_status $status
if test $pipeline_status -ne 0
    exit $pipeline_status
end
if set -q DRY_RUN[1]; or set -q CONVERSION_ONLY[1]
    exit 0
end

set -l prepared_name (basename (string trim -r -c / -- $PREPARED_DIR))
set -l result_json $SCORE_DIR/$prepared_name"_vbvr_results.json"
set -q TASK_SCORE_XLSX[1]
or set TASK_SCORE_XLSX $SCORE_DIR/$prepared_name"_task_scores.xlsx"
set -q FINAL_SCORES_TXT[1]
or set FINAL_SCORES_TXT $OUTPUT_ROOT/final_scores.txt

.venv/bin/python -m src.cli.export_vbvr_task_scores $result_json \
    --output $TASK_SCORE_XLSX \
    --summary-output $FINAL_SCORES_TXT \
    --expected-samples 500 \
    --expected-tasks 100
or exit 1

.venv/bin/python -c '
import json
import math
import sys
from pathlib import Path

from src.eval.evaluation_provenance import verify_recorded_manifest
from src.eval.vbvr_runtime import validate_vbvr_scorer_runtime

root = Path(sys.argv[1])
converted_model = Path(sys.argv[2])
gt_base = Path(sys.argv[3])
expected_revision = sys.argv[4]
expected_source_sha256 = sys.argv[5]
result = Path(sys.argv[6])
workbook = Path(sys.argv[7])
summary_path = Path(sys.argv[8])
required = (result, workbook, summary_path)
if not all(path.is_file() for path in required):
    raise SystemExit(1)

data = json.loads(result.read_text())
samples = data.get("samples")
if not isinstance(samples, list) or len(samples) != 500:
    raise SystemExit(1)
if any(sample.get("error") for sample in samples):
    raise SystemExit(1)
if not all(math.isfinite(float(sample["score"])) for sample in samples):
    raise SystemExit(1)
if len({sample.get("task_name") for sample in samples}) != 100:
    raise SystemExit(1)
if sum(sample.get("split") == "In_Domain" for sample in samples) != 250:
    raise SystemExit(1)
if sum(sample.get("split") == "Out_of_Domain" for sample in samples) != 250:
    raise SystemExit(1)

manifests = {
    "generation": (root / "generation-provenance.json", "vbvr-pro-generation"),
    "preparation": (root / "preparation-provenance.json", "vbvr-pro-preparation"),
    "score": (root / "score-provenance.json", "vbvr-pro-score"),
}
loaded = {}
for name, (path, stage) in manifests.items():
    matches, detail = verify_recorded_manifest(path, expected_stage=stage, require_complete=True)
    if not matches:
        print(detail, file=sys.stderr)
        raise SystemExit(1)
    loaded[name] = json.loads(path.read_text())

generation_values = loaded["generation"].get("values", {})
expected_generation = {
    "state": "complete",
    "height": "512",
    "width": "512",
    "num_frames": "81",
    "fps": "16",
    "num_inference_steps": "30",
    "guidance_scale": "1.0",
    "seed": "0",
    "generation_mode": sys.argv[9],
}
if sys.argv[9] == "cps":
    expected_generation["cps_noise_level"] = sys.argv[10]
elif sys.argv[9] == "ode":
    expected_generation["ode_solver"] = "flowmatch_euler" if sys.argv[11] == "euler" else "unipc"
if any(str(generation_values.get(key)) != value for key, value in expected_generation.items()):
    raise SystemExit(1)
if loaded["generation"].get("trees", {}).get("converted_model", {}).get("path") != str(converted_model.resolve()):
    raise SystemExit(1)
if loaded["generation"].get("trees", {}).get("eval_source", {}).get("path") != str(gt_base.resolve()):
    raise SystemExit(1)

score_values = loaded["score"].get("values", {})
if score_values.get("evalkit_revision") != expected_revision:
    raise SystemExit(1)
if score_values.get("evalkit_source_sha256") != expected_source_sha256:
    raise SystemExit(1)
dependencies = json.loads(score_values["scorer_dependencies"])
runtime = validate_vbvr_scorer_runtime()
if dependencies.get("contract") != runtime["contract"] or dependencies.get("sha256") != runtime["sha256"]:
    raise SystemExit(1)

generated = root / "generated_512x512x81"
prepared = root / "eval_1024x1024_81f_fps16_5p0625s"
if sum(1 for _ in generated.rglob("*.mp4")) != 500:
    raise SystemExit(1)
if sum(1 for _ in prepared.rglob("*.mp4")) != 500:
    raise SystemExit(1)
' $OUTPUT_ROOT $CONVERTED_MODEL $GT_BASE $EVALKIT_REV $EVALKIT_SOURCE_SHA256 \
    $result_json $TASK_SCORE_XLSX $FINAL_SCORES_TXT $GENERATION_MODE $CPS_NOISE_LEVEL $ODE_SOLVER
or exit 1

echo "[done] task score workbook: $TASK_SCORE_XLSX"
echo "[done] concise scores: $FINAL_SCORES_TXT"
echo "[done] sampler-matched DiffSynth step-35500 baseline passed strict audit ($sampler_label)"
