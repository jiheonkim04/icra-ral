# CAVM-VLA Preregistration

Date: 2026-07-13 KST

Proposal hash: `849A98B2F137FC43EAA68C7B7D7DB246FEF58DD2EDBBD1F8869C4BA092DE68F2`

## Fixed Identity Policy

`CAVM_RESET_IDENTITY_BASE = 20260901`.

Exact official LIBERO initial-state index is:

`identity - CAVM_RESET_IDENTITY_BASE`.

The finite official initial-state vectors may overlap prior campaigns, but CAVM uses fresh episode keys and does not choose identities from observed CAVM outcomes.

## Tasks

Use exactly two hard tasks:

1. `libero_spatial/task_4`
2. `libero_10/task_4`

These are reused because they are the long-standing hard local SmolVLA axes and because CAVM needs mixed frozen success/failure traces. They are not selected from CAVM outcomes.

## Splits

- acquisition identities: `20260901..20260912`
- calibration identities: `20260913..20260916`
- Stage 2A identities: `20260917..20260921`
- Stage 2B identities: `20260922..20260941`

No acquisition or calibration trace may appear in Stage 2A or Stage 2B memory evaluation.

## Features

Inference key:

`z_t = [q_t, a_t, a_{t-1}, rho_t, task_one_hot]`

where:

- `q_t` is official 8D proprioceptive state;
- `a_t` is current frozen SmolVLA queued action;
- `a_{t-1}` is previous executed action, zero at episode start;
- `rho_t` is chunk-index fraction;
- `task_one_hot` has dimension `2`.

Feature dimension: `8 + 7 + 7 + 1 + 2 = 25`.

Forbidden inference fields:

- simulator object poses;
- rewards before terminal success logging;
- BDDL predicates;
- ground-truth task progress;
- held-out identity membership;
- future actions;
- success labels from held-out episodes.

## Stage 0 / Stage 1 Acquisition And Calibration

Run frozen queued SmolVLA on acquisition and calibration identities.

Record every executed step:

- split;
- task key;
- identity;
- step;
- state;
- frozen action;
- previous action;
- chunk-index fraction;
- terminal episode success label copied to each trace row only after the episode finishes.

Stage 0 hard kill if any holds:

1. any selected task has fewer than `2` successful acquisition episodes or fewer than `2` failed acquisition episodes;
2. fewer than `10%` of calibration records have both success and failure neighborhoods available under the fixed retrieval rule;
3. median success/failure action-mean separation among gateable calibration records is below `0.05` in 7D action L2;
4. any privileged inference feature is required.

If Stage 0 passes, Stage 1 saves a memory artifact and fixed calibration config.

Calibration rules:

- standardize features using acquisition records only;
- same-task retrieval only;
- `k_success = 8`;
- `k_failure = 8`;
- retrieval bandwidth `sigma` is the median positive same-task nearest-neighbor distance over acquisition records, clipped to `[1e-3, 10.0]`;
- margin threshold `eta` is the `25th` percentile of gateable calibration separations, but not below `0.05`;
- margin scale `gamma` is `max(q75 - q25, 0.05)` over gateable calibration separations;
- `alpha = 0.35`;
- `beta = 0.50`;
- action clipping uses the same conservative bounds as existing official rollout postprocessed actions.

## Stage 2 Variants

Use exactly five variants:

1. `frozen_smolvla`
2. `success_only_memory_proxy`
3. `nearest_success_replay`
4. `cavm_no_contrast_ablation`
5. `cavm_full`

Definitions:

- `frozen_smolvla`: unmodified queued SmolVLA.
- `success_only_memory_proxy`: weighted success-action mean with success-density/confidence only. This is the local Retrieve-then-Steer proxy and is not an official reproduction.
- `nearest_success_replay`: single nearest successful action blended with the same maximum `alpha`.
- `cavm_no_contrast_ablation`: uses both success/failure density and margin gate, but action target is `mu+` only.
- `cavm_full`: action target is `mu+ + beta (mu+ - mu-)`.

All memory variants use the same memory, standardization, retrieval bandwidth, identity splits, clipping, and frozen policy calls.

## Stage 2A

Run `5` identities x `2` tasks x `5` variants = `50` episodes.

Stage 2A catastrophic kill if:

- `cavm_full` has `0 / 10` success while any paired baseline has at least `4 / 10`;
- `cavm_full` is at least `30` absolute percentage points below the strongest baseline or key ablation;
- the CAVM gate never activates on held-out episodes;
- any forbidden inference signal is used.

Proceed to Stage 2B when CAVM is tied, narrowly negative, or positive.

## Stage 2B

Run `20` identities x `2` tasks x `5` variants = `200` episodes.

Stage 2B prototype GO requires:

- `cavm_full` beats the strongest baseline and `cavm_no_contrast_ablation`;
- absolute task-balanced gain is at least `10` points at prototype scale, or paired evidence is consistently positive with meaningful failure-rate reduction;
- CAVM gate activates on held-out episodes;
- no privileged inference signal is used;
- CAVM heavy policy calls per step are not above frozen SmolVLA, since CAVM must not replan.

Stage 2B permanent kill if:

- `success_only_memory_proxy`, `nearest_success_replay`, or `cavm_no_contrast_ablation` matches or beats `cavm_full`;
- `cavm_full` is clearly worse than frozen SmolVLA;
- the upper confidence bound versus the strongest baseline excludes a useful `+0.10` improvement;
- mechanism activation is absent;
- privileged inference is required.

If Stage 2B is unresolved but non-killed, one expansion to at most `80` paired episodes per policy is allowed by active governance. No third expansion is allowed.

## Reporting

Report:

- successes/counts;
- task-balanced success;
- per-task success;
- paired wins/losses/ties versus full;
- paired bootstrap confidence intervals;
- gate activation rate;
- mean action delta from frozen;
- heavy policy calls per step;
- latency and CUDA memory;
- identity-overlap validation.
