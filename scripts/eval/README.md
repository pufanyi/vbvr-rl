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
  every checkpoint has UniPC ODE, deterministic FlowMatch Euler ODE, CPS 0.3,
  and CPS 0.7 entrypoints.
- `vbvr_pro/dancegrpo_indomain_strict/`: the strict In-Domain checkpoint series.
- `vbvr_pro/dancegrpo_manifest_rl_512x512x81/`: the 512x512x81 manifest-RL
  checkpoint series evaluated with the matching 30-step CPS 0.7 rollout policy;
  its sweep launcher evaluates checkpoints 100--400 on four disjoint two-GPU
  groups, then evaluates checkpoint 500 on all eight GPUs.

Run every launcher from the repository root. Most fish launchers source
`scripts/lib/env.fish`, which activates `.venv` and sets `PYTHONPATH`.

VBVR-Pro training rewards and offline scoring deliberately use the same pinned
scorer runtime. Check it before a launch with:

```bash
.venv/bin/python -m src.eval.vbvr_runtime
```

The common VBVR-Pro launcher records that runtime in score provenance, and both
the parent scorer and every spawned worker validate it before loading EvalKit.
Do not move only offline evaluation into a separate environment: that creates a
real train/eval gap. If scorer isolation becomes necessary, route both the
training reward and offline scoring through the same isolated runtime.
