# Evaluation Scripts

Evaluation launchers are grouped by benchmark and runtime so checkpoint sweeps
do not crowd a single directory:

- `lmms/`: DCP conversion plus lmms-eval/FastVideo launchers.
- `maze/`: Maze rendering and queued Maze/VBVR checkpoint evaluation.
- `vbvr/`: legacy VBVR generation, scoring, monitors, and compatibility wrappers.
- `vbvr_pro/`: VBVR-Pro `main_v2` evaluation and reporting.

The VBVR-Pro DanceGRPO wrappers are further separated by training run:

- `vbvr_pro/dancegrpo_bs32/`: the original bs32 checkpoints at steps 300–2700.
- `vbvr_pro/dancegrpo_bs32_lr_1e-6/`: lr=1e-6 checkpoints at steps 100–800;
  every checkpoint has ODE, CPS 0.3, and CPS 0.7 entrypoints.
- `vbvr_pro/dancegrpo_indomain_strict/`: the strict In-Domain checkpoint series.

Run every launcher from the repository root. Most fish launchers source
`scripts/lib/env.fish`, which activates `.venv` and sets `PYTHONPATH`.
