#!/usr/bin/env fish

# Sequentially evaluate every complete strict In-Domain checkpoint with the
# four standard VBVR-Pro modes. Completed outputs are skipped unless
# STRICT_REEVALUATE_COMPLETE=1 is set.

source (dirname (status filename))/../../../lib/env.fish

function _fail
    echo "[error] $argv" >&2
    exit 1
end

set -q CHECKPOINT_ROOT[1]
or set CHECKPOINT_ROOT storage/checkpoints/dancegrpo_vbvr_pro_5b_256x256x161_rule_cps_from_nsft_bs32_lr_1e-6_indomain_strict
set -q SPLIT_MANIFEST[1]
or set SPLIT_MANIFEST /mnt/aigc/xujunxiang/Code/VBVR-Pro/scripts/split_manifest.json
set -q EVALKIT_REV[1]
or set EVALKIT_REV 6fedd9d9edb8daafa56aca8e53885aa8ad6f6037
set -q EVALKIT_SOURCE_SHA256[1]
or set EVALKIT_SOURCE_SHA256 eb977da60e95456734063ba018b14d805680179fdf0e3e3b2ba6f603f27a935c
test -f $SPLIT_MANIFEST; or _fail "split manifest does not exist: $SPLIT_MANIFEST"
set -g _strict_manifest_sha256 (sha256sum $SPLIT_MANIFEST | awk '{print $1}')
set -g _strict_manifest_sha256_prefix (string sub -s 1 -l 8 -- $_strict_manifest_sha256)
set -g _strict_evalkit_sha256_prefix (string sub -s 1 -l 8 -- $EVALKIT_SOURCE_SHA256)
set -q OUTPUT_BASE[1]
or set OUTPUT_BASE storage/eval_out/vbvr_pro_main_v2_indomain_strict_manifest_$_strict_manifest_sha256_prefix"_evalkit_"$_strict_evalkit_sha256_prefix
set -q STRICT_EVAL_LOG_DIR[1]
or set STRICT_EVAL_LOG_DIR storage/eval_logs/vbvr_pro_main_v2_indomain_strict
set -q STRICT_REEVALUATE_COMPLETE[1]
or set STRICT_REEVALUATE_COMPLETE 0

set -g _strict_generic_dir scripts/eval/vbvr_pro/dancegrpo_bs32
set -g _strict_unipc_launcher $_strict_generic_dir/vbvr_pro_5b_dancegrpo_checkpoint_main_v2.fish
set -g _strict_euler_launcher $_strict_generic_dir/vbvr_pro_5b_dancegrpo_checkpoint_euler_main_v2.fish
set -g _strict_cps_launcher $_strict_generic_dir/vbvr_pro_5b_dancegrpo_checkpoint_cps_main_v2.fish

for required in $CHECKPOINT_ROOT $_strict_unipc_launcher $_strict_euler_launcher $_strict_cps_launcher
    test -e $required; or _fail "required path does not exist: $required"
end

set -l checkpoint_steps
for checkpoint_dir in (find $CHECKPOINT_ROOT -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
    test -f $checkpoint_dir/high/.metadata; or begin
        echo "[skip] incomplete checkpoint without high/.metadata: $checkpoint_dir" >&2
        continue
    end
    set -a checkpoint_steps (string replace 'checkpoint-' '' -- (basename $checkpoint_dir))
end
test (count $checkpoint_steps) -gt 0; or _fail "no complete checkpoints found under $CHECKPOINT_ROOT"

mkdir -p $STRICT_EVAL_LOG_DIR; or exit 1

function _result_complete
    set -l output_root $argv[1]
    set -l expected_manifest_sha256 $argv[2]
    set -l expected_evalkit_revision $argv[3]
    set -l expected_evalkit_source_sha256 $argv[4]
    .venv/bin/python -c '
import json
import sys
from pathlib import Path

from src.eval.evaluation_provenance import verify_recorded_manifest

root = Path(sys.argv[1])
expected_manifest_sha256 = sys.argv[2]
expected_evalkit_revision = sys.argv[3]
expected_evalkit_source_sha256 = sys.argv[4]
result = root / "scores" / "eval_1024x1024_161f_5s_vbvr_results.json"
generation_provenance = root / "generation-provenance.json"
preparation_provenance = root / "preparation-provenance.json"
score_provenance = root / "score-provenance.json"
workbook = root / "scores" / "eval_1024x1024_161f_5s_task_scores.xlsx"
summary = root / "final_scores.txt"
required = (
    result,
    generation_provenance,
    preparation_provenance,
    score_provenance,
    workbook,
    summary,
)
if not all(path.is_file() for path in required):
    raise SystemExit(1)
data = json.loads(result.read_text())
samples = data.get("samples", [])
if len(samples) != 500 or any(sample.get("error") for sample in samples):
    raise SystemExit(1)
if len(data.get("summary", {}).get("overall", {}).get("by_task", {})) != 100:
    raise SystemExit(1)
generation = json.loads(generation_provenance.read_text())
preparation = json.loads(preparation_provenance.read_text())
score = json.loads(score_provenance.read_text())
if any(item.get("values", {}).get("state") != "complete" for item in (generation, preparation, score)):
    raise SystemExit(1)
for manifest_path, stage in (
    (generation_provenance, "vbvr-pro-generation"),
    (preparation_provenance, "vbvr-pro-preparation"),
    (score_provenance, "vbvr-pro-score"),
):
    matches, detail = verify_recorded_manifest(
        manifest_path,
        expected_stage=stage,
        require_complete=True,
    )
    if not matches:
        print(f"[incomplete] {detail}", file=sys.stderr)
        raise SystemExit(1)
recorded_manifest_sha256 = generation.get("files", {}).get("split_manifest", {}).get("sha256")
if recorded_manifest_sha256 != expected_manifest_sha256:
    raise SystemExit(1)
score_values = score.get("values", {})
if score_values.get("evalkit_revision") != expected_evalkit_revision:
    raise SystemExit(1)
if score_values.get("evalkit_source_sha256") != expected_evalkit_source_sha256:
    raise SystemExit(1)
generated = root / "generated_256x256x161"
prepared = root / "eval_1024x1024_161f_5s"
bindings = (
    (generation, "media_trees", "generated_videos", generated),
    (preparation, "media_trees", "prepared_videos", prepared),
    (score, "trees", "prepared_videos", prepared),
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
' $output_root $expected_manifest_sha256 $expected_evalkit_revision $expected_evalkit_source_sha256
end

function _run_mode
    set -l step $argv[1]
    set -l mode $argv[2]
    set -l checkpoint_slug dancegrpo_vbvr_pro_5b_256x256x161_rule_cps_from_nsft_bs32_lr_1e-6_indomain_strict_checkpoint-$step
    set -l converted_model storage/models/dcp_converted_5b/$checkpoint_slug
    set -l output_root
    set -l launcher
    set -e CPS_NOISE_LEVEL

    switch $mode
        case unipc
            set output_root $OUTPUT_BASE/dancegrpo_vbvr_pro_5b_checkpoint-$step
            set launcher $_strict_unipc_launcher
        case euler
            set output_root $OUTPUT_BASE/dancegrpo_vbvr_pro_5b_checkpoint-$step-euler
            set launcher $_strict_euler_launcher
        case cps-0.3
            set output_root $OUTPUT_BASE/dancegrpo_vbvr_pro_5b_checkpoint-$step-cps-noise-0.3
            set launcher $_strict_cps_launcher
        case cps-0.7
            set output_root $OUTPUT_BASE/dancegrpo_vbvr_pro_5b_checkpoint-$step-cps-noise-0.7
            set launcher $_strict_cps_launcher
        case '*'
            echo "[error] unsupported mode: $mode" >&2
            return 1
    end
    if string match -q 'cps-*' -- $mode
        set --function --export CPS_NOISE_LEVEL (string replace 'cps-' '' -- $mode)
    end

    if test "$STRICT_REEVALUATE_COMPLETE" != 1; and _result_complete \
            $output_root $_strict_manifest_sha256 $EVALKIT_REV $EVALKIT_SOURCE_SHA256
        echo "[skip] checkpoint-$step $mode is already complete: $output_root"
        return 0
    end

    set -lx CHECKPOINT_STEP $step
    set -lx CHECKPOINT_ROOT $CHECKPOINT_ROOT
    set -lx CONVERTED_MODEL $converted_model
    set -lx OUTPUT_ROOT $output_root
    set -lx SPLIT_MANIFEST $SPLIT_MANIFEST
    set -lx EVALKIT_REV $EVALKIT_REV
    set -lx EVALKIT_SOURCE_SHA256 $EVALKIT_SOURCE_SHA256
    set -lx NUM_GPUS 8
    set -lx CUDA_DEVICES 0,1,2,3,4,5,6,7

    # A non-empty incomplete output may have provenance from an older manifest
    # or an interrupted rewrite. Explicitly regenerate it instead of silently
    # mixing media from two configurations.
    set -e FORCE_REGENERATE
    if test -d $output_root; and test -n (find $output_root -mindepth 1 -print -quit 2>/dev/null)
        set -lx FORCE_REGENERATE 1
    end

    set -l log_path $STRICT_EVAL_LOG_DIR/checkpoint-$step-$mode.log
    echo "[start] "(date --iso-8601=seconds)" checkpoint-$step $mode -> $output_root"
    fish $launcher >$log_path 2>&1
    set -l run_status $status
    if test $run_status -ne 0
        echo "[error] checkpoint-$step $mode failed with status $run_status; tail of $log_path:" >&2
        tail -n 120 $log_path >&2
        return $run_status
    end
    if set -q DRY_RUN[1]
        echo "[dry-run] checkpoint-$step $mode resolved successfully; log=$log_path"
        return 0
    end
    if not _result_complete \
            $output_root $_strict_manifest_sha256 $EVALKIT_REV $EVALKIT_SOURCE_SHA256
        echo "[error] checkpoint-$step $mode exited successfully but its result contract is incomplete" >&2
        tail -n 120 $log_path >&2
        return 1
    end
    echo "[done]  "(date --iso-8601=seconds)" checkpoint-$step $mode; log=$log_path"
end

echo "[sweep] checkpoints: $checkpoint_steps"
echo "[sweep] modes: unipc euler cps-0.3 cps-0.7"
echo "[sweep] split manifest: $SPLIT_MANIFEST ($_strict_manifest_sha256)"
echo "[sweep] output base: $OUTPUT_BASE"
echo "[sweep] re-evaluate complete outputs: $STRICT_REEVALUATE_COMPLETE"

for step in $checkpoint_steps
    for mode in unipc euler cps-0.3 cps-0.7
        _run_mode $step $mode; or exit $status
    end
end

echo "[done] strict In-Domain VBVR-Pro sweep is complete"
