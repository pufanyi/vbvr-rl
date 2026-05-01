# Project Memory

- For long-running tasks, keep monitoring the process instead of leaving it hanging. Poll logs or command output, verify the exit status or completion signal, and make sure no required background session is still running before reporting the task as done.
- For VBVR/lmms-eval checkpoint evaluation, read `docs/vbvr_lmms_eval.md` first. It documents the DCP-to-Diffusers conversion step, the lmms-eval command, hardware overrides, output paths, and lessons from the 2026-04-29 partial run.
