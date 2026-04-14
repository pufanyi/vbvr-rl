# Draft: Unify Shared Checkpoint/Resume Logic Between SFT and RL

## Goal

Reduce duplication between `BaseTrainer` and `BaseRLTrainer`, but only for the
parts that are actually the same in behavior.

The immediate target is:

- checkpoint discovery
- DCP save
- DCP load
- EP checkpoint layout handling
- EP checkpoint -> non-EP weights-only init
- dataloader state restore
- optimizer/step reset after load

The goal is not to force SFT and RL into one base class.

## Why This Is Worth Doing

Right now the checkpoint/resume path exists in two places:

- `src/trainer/base_trainer.py`
- `src/trainer/base_rl_trainer.py`

They are nearly identical, and the recent GRPO bug came from exactly this:

- GRPO was running with `expert_parallel: false`
- `resume_from` pointed to an expert-parallel SFT checkpoint
- the fix had to be patched into both trainer trees

This is the right signal that the checkpoint/resume path should be shared.

## Non-Goals

Do not unify these yet:

- `_compute_total_steps`
- `_build_dataset`
- expert-parallel data semantics
- training loop bodies
- RL-specific reference policy logic
- SFT/RL class hierarchy as a whole

Reason:

- SFT and RL already differ in optimizer-step accounting
- SFT supports `expert_parallel_data_mode=duplicate/split`
- RL currently uses different batch semantics and is likely to evolve further

Trying to unify all of that now would mix stable shared code with code that is
still changing.

## Recommended Shape

Do **not** put this into `utils.py`.

This logic is not a pure utility layer. It depends on trainer instance state:

- `self.model`
- `self.ema`
- `self.train_state`
- `self.dataloader`
- `self.rank`
- `self.dp_rank`
- `self.expert_parallel`
- `self._dp_pg`
- `self._barrier()`
- `self._build_optimizers(...)`
- `self._compute_total_steps()`

If moved to `utils.py`, the code will either:

- take too many parameters, or
- end up passing the entire trainer object around anyway

Better options:

1. `CheckpointRuntimeMixin`
2. `src/trainer/checkpoint_runtime.py` with a mixin class inside

Recommended name:

- `CheckpointRuntimeMixin`

## Proposed Refactor Boundary

### Keep in `src/trainer/checkpoint.py`

This file should remain the stateless DCP helper layer:

- `TrainState`
- checkpoint layout detection
- load into pipeline helpers
- low-level single-expert load helpers

This is already the right place for model-state loading primitives.

### Move to a Shared Mixin

Extract these methods from both trainer bases into one shared mixin:

- `_find_latest_checkpoint`
- `_save_checkpoint`
- `_load_checkpoint`

Also consider extracting one small internal helper:

- `_reset_training_state_after_load()`

That helper would centralize:

- reset `step/epoch/batch_idx`
- rebuild optimizers
- reattach optimizers to `TrainState`
- recompute `total_steps`

This helper is currently duplicated inside both trainer bases.

## Suggested File Layout

### New file

- `src/trainer/checkpoint_runtime.py`

### Suggested contents

- `class CheckpointRuntimeMixin:`
  - `_find_latest_checkpoint`
  - `_save_checkpoint`
  - `_load_checkpoint`
  - `_reset_training_state_after_load`

### Existing files updated

- `src/trainer/base_trainer.py`
- `src/trainer/base_rl_trainer.py`

Both would inherit the mixin and delete their local checkpoint methods.

## Assumptions the Mixin Can Require

The mixin can assume the host trainer defines:

- `self.cfg`
- `self.model`
- `self.ema`
- `self.train_state`
- `self.dataloader`
- `self.rank`
- `self.world_size`
- `self.expert_parallel`
- `self.expert_group`
- `self.dp_rank`
- `self._dp_pg`
- `self._reset_on_load`
- `self._barrier()`
- `self._build_optimizers(cfg)`
- `self._compute_total_steps()`

That is acceptable because this is a trainer mixin, not a general utility.

## Step-by-Step Implementation Plan

### Step 1

Create `src/trainer/checkpoint_runtime.py` and move the current shared logic
there without changing behavior.

Start from the newer implementation that already includes:

- EP checkpoint detection
- EP -> non-EP weights-only init

### Step 2

Extract `_reset_training_state_after_load()`.

Target behavior:

- set `step=0`
- set `epoch=0`
- set `batch_idx=0`
- rebuild optimizers
- rebind optimizers to `self.train_state`
- recompute `self.total_steps`

This should remove the ugliest duplicated block.

### Step 3

Make both base classes inherit the mixin:

- `class BaseTrainer(CheckpointRuntimeMixin):`
- `class BaseRLTrainer(CheckpointRuntimeMixin):`

Delete their local implementations of:

- `_find_latest_checkpoint`
- `_save_checkpoint`
- `_load_checkpoint`

### Step 4

Keep the rest of the trainer code separate.

Do not touch:

- `_compute_total_steps`
- `_build_dataset`
- train loops

### Step 5

Run smoke validation for all checkpoint layouts.

## Validation Matrix

### Case 1: Flat checkpoint -> flat trainer

Expected:

- full DCP resume works
- dataloader state restores
- step/epoch/batch_idx restore unless reset requested

### Case 2: EP checkpoint -> EP trainer

Expected:

- each group loads its local expert subdir
- DCP process group handling still works

### Case 3: EP checkpoint -> non-EP trainer, reset load

Expected:

- no attempt to fully resume optimizer/dataloader
- load model weights only
- EMA reinitialized
- optimizer reset
- step/epoch/batch_idx reset to zero

This is the case that fixed the GRPO bug.

### Case 4: non-EP trainer with `auto_resume: true`

Expected:

- latest flat checkpoint still found correctly

### Case 5: EP trainer with `auto_resume: true`

Expected:

- latest checkpoint dir found if either `high/.metadata` or `low/.metadata` exists

## Risks

### Risk 1: Hidden Behavioral Drift

Even if the code is duplicated today, the two trainer trees may rely on small
differences later.

Mitigation:

- only share checkpoint/resume code
- keep dataset/step-count/training logic separate

### Risk 2: Mixin Coupling Gets Too Wide

If the mixin starts reaching into unrelated trainer logic, it becomes just
another disguised base class.

Mitigation:

- keep the mixin narrow
- only include checkpoint/resume lifecycle logic

### Risk 3: Auto-resume Picking Incomplete Checkpoints

Current auto-detection mainly checks for `.metadata`.

Possible future improvement:

- ignore directories missing expected shard files
- ignore checkpoints still being written

This is a follow-up, not required for the first refactor.

## Open Questions

1. Should `auto_resume` explicitly ignore directories newer than some grace
period to avoid partially written checkpoints?
2. Should EP -> non-EP load optionally support EMA weights too, or is model
weights only enough?
3. Should the same weights-only init path also be supported for non-EP -> EP,
or can that wait?

## Tomorrow's Minimal Deliverable

If time is limited, only do this:

1. Add `CheckpointRuntimeMixin`
2. Move `_find_latest_checkpoint`
3. Move `_save_checkpoint`
4. Move `_load_checkpoint`
5. Add `_reset_training_state_after_load`
6. Wire both base classes to use it
7. Run smoke tests for flat resume and EP -> non-EP init

That is already a good, contained refactor.

## Nice-to-Have Follow-Up

After the checkpoint refactor is done, revisit whether a second shared mixin is
worth it for:

- common model build
- common optimizer build
- common DCP state setup

But only do that after checkpoint code is stable again.
