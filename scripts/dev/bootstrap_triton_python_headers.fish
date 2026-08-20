#!/usr/bin/env fish
# Ensure the locked Pixi Python headers are present and prove that Triton's
# CUDA driver helper compiles from a fresh cache.

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root; or exit 1

if not command -q pixi
    echo "[error] Pixi is required to provision the project environment" >&2
    exit 1
end

pixi install --environment default --locked; or exit 1
source scripts/lib/env.fish; or exit 1

set -l python_include (python -c 'import sysconfig; print(sysconfig.get_path("include"))')
if not test -f "$python_include/Python.h"
    echo "[bootstrap] Python.h is missing; reinstalling the locked Pixi Python package"
    pixi reinstall --environment default --locked python; or exit 1
    set python_include (python -c 'import sysconfig; print(sysconfig.get_path("include"))')
end

if not test -f "$python_include/Python.h"
    echo "[error] locked Pixi Python still lacks Python.h: $python_include" >&2
    exit 1
end
set -gx WAN_TRAINER_PYTHON_INCLUDE $python_include
if not contains -- $python_include $CPATH
    set -gx CPATH $python_include $CPATH
end

set -l fresh_cache (mktemp -d /tmp/wan-trainer-triton-bootstrap.XXXXXX); or exit 1
set -lx TRITON_CACHE_DIR $fresh_cache
python -c \
    'from triton.runtime import driver; print(f"[bootstrap] Triton target: {driver.active.get_current_target()}")'
or exit 1

echo "[bootstrap] Fresh-cache Triton compilation passed with Pixi headers from $WAN_TRAINER_PYTHON_INCLUDE"
