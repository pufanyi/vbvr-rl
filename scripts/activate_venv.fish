# Shared project environment setup for fish launchers.

set -gx WAN_TRAINER_ROOT (realpath (dirname (status filename))/..)

cd $WAN_TRAINER_ROOT; or exit 1

if not test -f .venv/bin/activate.fish
    echo "[error] missing .venv/bin/activate.fish under $WAN_TRAINER_ROOT" >&2
    exit 1
end

source .venv/bin/activate.fish; or exit 1
set -gx PYTHONPATH $WAN_TRAINER_ROOT $PYTHONPATH
