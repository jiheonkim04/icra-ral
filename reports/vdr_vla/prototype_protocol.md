# VDR-VLA Prototype Protocol

Date: 2026-07-16 KST

Proposal SHA-256:
`0229EBC15901F4FE1EDD3839AB6B984AFA3E0E99836B5C88CF21F2C7DE2B3E72`.

Decision: `VDR_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0A_PENDING`

## Purpose

Run a bounded development-only audit before any expensive training, validation
search, manifest freeze, or rollout. The protocol tests whether VDR has legal
data, noncollapsed dynamic residual targets, usable action-conditioned
headroom, identity-preserving integration, and finite gradients.

## Frozen Stage 0A Inputs

- task sources:
  - `libero_spatial/task_3`
  - `libero_object/task_3`
  - `libero_goal/task_5`
  - `libero_10/task_5`
- discovery demos: `0..7`
- validation demos: `8..9`
- horizons: `{4,12}`
- projection dimension: `32`
- ridge coefficient: `1e-4`
- sampled audit subset: deterministic, task-balanced, maximum `256`
  validation windows per horizon
- confirmatory simulator identities read: `0`
- closed-loop episodes: `0`

## Stage 0A Required Artifacts

The runner must persist:

- `reports/vdr_vla/stage_0a_manifest.json`
- `reports/vdr_vla/stage_0a_partial.json`
- `reports/vdr_vla/stage_0a_result.json`
- `reports/vdr_vla/stage_0a_result.md`
- `reports/vdr_vla/stage_0a_validation.json`
- `reports/vdr_vla/stage_0a_adjudication.md`
- `reports/vdr_vla/stage_0a_pid.txt`
- `reports/vdr_vla/stage_0a_heartbeat.json`
- `reports/vdr_vla/stage_0a_status.json`
- stdout/stderr logs and exit-code files for any detached execution

Partial results must be valid JSON after every accepted row. Resume may add
only missing `(partition,suite,task_identity,source_hash,demo_id,frame_index,horizon)`
keys and may not repeat completed keys.

## Acceptance Gates

Stage 0A passes only if every preregistered hard gate in
`reports/vdr_vla/preregistration.md` passes. In particular:

- duplicate/manifest/split-overlap counts are zero;
- actionless static predictor has measurable validation signal;
- generated-action-conditioned residual probe beats the actionless probe;
- FutureVLA proxy leaves residual headroom;
- VDR gradients are finite and nonzero in expected trainable parameters;
- Base hash, identity, disk reload, and action validity pass;
- exceptions are zero.

## Stage 0A Decisions

- `VDR_STAGE_0A_PASS_STAGE_0B_ALLOWED`
- `VDR_STAGE_0A_DATA_OR_SUPERVISION_FAILURE`
- `VDR_STAGE_0A_NO_USABLE_HEADROOM`
- `VDR_STAGE_0A_DESIGN_FAILURE`
- `VDR_STAGE_0A_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`

No Stage 0A decision is a scientific closed-loop kill.

## After Stage 0A

If Stage 0A passes, run only the frozen Stage 0B micro-fit. If Stage 0A fails,
do not repair by changing residual targets, PCA dimension, horizons,
thresholds, task sources, coefficient set, action validity policy, or target
construction. Archive the failure class and continue to the next method cycle.

## First Closed-Loop Policy List

The first serious comparison is frozen to:

1. `smolvla_base`
2. `futurevla_latent_alignment_proxy`
3. `vdr_full`
4. `vdr_no_action_residual`
5. `standard_lora`

No additional baseline may be added before Stage A unless it tests a concrete
reviewer objection, is decision-relevant, and is cheaper than proceeding under
the frozen comparison.
