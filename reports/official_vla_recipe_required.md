# Official VLA Recipe Required

Date: 2026-07-09 KST

## Decision

`OFFICIAL_VLA_RECIPE_REPRODUCTION_REQUIRED`

The next valid VLA step is official baseline reproduction, not a new method and not another custom adapter tweak.

## Why This Is Required

The local custom SmolVLA 7D adapter stack has passed several infrastructure gates, but still fails the control-transfer gate:

- interface fixed,
- feature schema fixed,
- expert replay stabilized,
- action range fixed,
- learned adapter still failed replay/progress,
- clip-only matched or beat the trained range-fixed adapter.

That means the project cannot distinguish method weakness from local recipe mismatch until an official training/evaluation recipe is reproduced.

## Required Baseline Ingredients

- Official SmolVLA/LeRobot/OpenVLA-style training script or recipe.
- Official observation preprocessing.
- Official action normalization and denormalization.
- Official gripper convention.
- Official train/eval split semantics.
- Official rollout/evaluation stack or a documented faithful local equivalent.
- Successful baseline behavior before any method modification.

## Resume Gate

Only resume VLA method work if the official baseline:

- runs without large unapproved downloads,
- has documented preprocessing and action conventions,
- succeeds or matches expected behavior on an official/standard task split,
- beats or clearly contextualizes simple baselines,
- provides a stable control baseline before method changes.

If this cannot be achieved under local constraints, use `STOP_VLA_METHOD_SEARCH_UNDER_CURRENT_SETUP`.
