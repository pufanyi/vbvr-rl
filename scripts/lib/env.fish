# Shared project environment setup for fish launchers.

set -gx WAN_TRAINER_ROOT (realpath (dirname (status filename))/../..)

cd $WAN_TRAINER_ROOT; or exit 1

if not command -q pixi
    echo "[error] Pixi is required; install it and run 'pixi install --locked'." >&2
    exit 1
end

# Launchers can be called either through `pixi run` or directly. Direct calls
# activate the locked default environment through Pixi before doing any work.
if not set -q PIXI_IN_SHELL; or not set -q PIXI_ENVIRONMENT_NAME; or test "$PIXI_ENVIRONMENT_NAME" != default
    pixi shell-hook \
        --manifest-path "$WAN_TRAINER_ROOT/pyproject.toml" \
        --environment default \
        --locked \
        --shell fish | source
    or exit 1
end

if not command -q python; or not command -q torchrun
    echo "[error] the locked Pixi default environment is incomplete; run 'pixi install --locked'." >&2
    exit 1
end

set -gx PYTHONPATH $WAN_TRAINER_ROOT $PYTHONPATH

# Triton compiles a small CUDA driver extension lazily on the first compiled
# forward. Pixi's Python package normally includes matching headers. Prefer an
# explicit operator override, then the active Pixi interpreter's include
# directory, then an existing CPATH entry. Never download at launch.
set -l wan_python_include (python -c 'import sysconfig; print(sysconfig.get_path("include"))')
set -l wan_python_abi (python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
set -l wan_python_header_include

for candidate in $WAN_TRAINER_PYTHON_INCLUDE $wan_python_include $CPATH
    if test -n "$candidate"; and test -f "$candidate/Python.h"
        set wan_python_header_include $candidate
        break
    end
end

if test -n "$wan_python_header_include"
    if not contains -- $wan_python_header_include $CPATH
        set -gx CPATH $wan_python_header_include $CPATH
    end
    set -gx WAN_TRAINER_PYTHON_INCLUDE $wan_python_header_include

    if test "$wan_python_header_include" != "$wan_python_include"
        if test -f "$wan_python_include/Python.h"
            echo "[env] Using configured Python headers from $wan_python_header_include via CPATH"
        else
            echo "[env] Python headers missing under $wan_python_include; using $wan_python_header_include via CPATH"
        end
    end
else
    echo "[warning] Python.h for Python $wan_python_abi was not found; torch.compile/Triton will fail." >&2
    echo "[warning] Run 'pixi reinstall --locked python' or export WAN_TRAINER_PYTHON_INCLUDE/CPATH." >&2
    echo "[warning] Then run fish scripts/dev/bootstrap_triton_python_headers.fish to verify a fresh compile." >&2
end
