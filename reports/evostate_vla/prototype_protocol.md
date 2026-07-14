# EvoState-VLA Prototype Protocol

Date: 2026-07-14 KST

Proposal hash: `A44ED68CC8E1F296DB8B0B3E16FF84D7D5BBE684EAF63EAE29E7CC91DCFD93C9`

Current decision: `EVOSTATE_STAGE_0_IMPLEMENTATION_PENDING`

## Required Order

1. Implement Stage 0 development audit.
2. Run Stage 0 on `reports/cavm_vla/acquisition_records.jsonl`.
3. If Stage 0 passes, run the six-config validation search.
4. Freeze exactly one selected configuration.
5. Implement Stage A runner for the five policies.
6. Run Stage A only after tests and validators pass.
7. Run Stage B only if Stage A is non-catastrophic and not a valid permanent kill.

## Five Policies

`faulted_base_smolvla`:

- frozen SmolVLA action under the shared `translation_lag_scale_fault` wrapper.

`dream_lite_proxy`:

- uses the learned transition model to choose between the current queued/base action and a fresh-policy action by predicted mismatch reduction;
- no inverse-dynamics correction;
- labeled as a local transparent proxy, not official DREAM-Chunk.

`evostate_full`:

- persistent action-evolved predicted state;
- validation-calibrated reliability gate;
- bounded damped inverse-dynamics correction.

`evostate_no_state_prior_ablation`:

- same correction machinery;
- predicted state is reset to observed state every step;
- tests whether persistent action-evolved state matters.

`static_inverse_dynamics`:

- simple ridge/static inverse correction using the same development partition;
- no learned reliability gate;
- strongest simple reviewer-killer baseline.

## Artifacts

Expected Stage 0 artifacts:

- `reports/evostate_vla/development_audit.json`
- `reports/evostate_vla/development_audit.md`

Expected validation artifacts:

- `reports/evostate_vla/validation_search.json`
- `reports/evostate_vla/validation_search.md`
- `reports/evostate_vla/selected_config.json`
- `reports/evostate_vla/checkpoints/<selected>.pt` or equivalent coefficient file

Expected rollout artifacts:

- `reports/evostate_vla/stage_a_partial_result.json`
- `reports/evostate_vla/stage_a_result.json`
- `reports/evostate_vla/stage_a_result.md`
- `reports/evostate_vla/stage_b_partial_result.json`
- `reports/evostate_vla/stage_b_result.json`
- `reports/evostate_vla/stage_b_result.md`

## Validation Before Rollout

Required local checks:

- governance validator passes;
- focused EvoState unit tests pass;
- no confirmatory identity appears in training/validation;
- checkpoint or coefficient reload is exact within numeric tolerance;
- selected config exists and is frozen;
- action validity is `1.0` on validation;
- p95 action delta does not exceed preregistered cap;
- static inverse and no-state-prior ablations are implemented.

## No-Retrofit Rules

Do not:

- use FANG Stage B identities for development;
- tune `alpha`, gate threshold, damping, or fault constants after Stage A/B;
- add extra policies before the first serious comparison;
- add a new simple baseline after seeing Stage A/B;
- reinterpret static inverse dominance as EvoState success;
- use object pose, success labels, reward, or future state at inference;
- call the DREAM-lite proxy official DREAM-Chunk.
