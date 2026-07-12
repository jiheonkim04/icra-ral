# Autonomous RA-L Reviewer Log

## 2026-07-12 KST

Cycle 1 method: `DICD-VLA`

Reviewer position:

- The first decisive question is not offline fit; it is closed-loop task success under the preregistered delay condition.
- The direct chunk-index delayed baseline is the key reviewer-killer baseline.
- The no-history ablation must be beaten, or the executed-action-history component is not useful.
- Clean success is not the primary claim in Stage A, but obvious clean degradation would block scale-up.
- No privileged simulator state, success labels, future observations, reset identity, or held-out outcome labels may be used at inference.

Fixed Stage A decisions:

- `PROTOTYPE_GO` only if full DICD beats the strongest delayed baseline by at least `5` absolute task-balanced success points and beats both key comparisons.
- `SIMPLE_BASELINE_EXPLAINS_METHOD` if direct chunk indexing matches or beats full.
- `KEY_COMPONENT_NOT_USEFUL` if no-history ablation matches or beats full.
- `GENUINE_METHOD_KILL` if the method is active but loses to delayed baselines.
- `UNDERPOWERED_ONE_EXPANSION_ALLOWED` only for a positive but underpowered signal not matched by baseline or ablation.

## 2026-07-12 KST Cycle 2

Cycle 2 method: `FEDO-VLA`

Reviewer position:

- APEX is the closest direct prior and must be treated as a decisive baseline.
- Static inverse-gain compensation is the simplest reviewer-killer baseline under the controlled fault.
- The no-feedback/no-phase ablation must be beaten or FEDO's claimed feedback component is not useful.
- Clean no-fault retention must be checked.

Fixed Stage A decisions:

- `PROTOTYPE_GO` only if full FEDO beats the strongest faulted baseline by at least `5` absolute task-balanced success points and beats static inverse, APEX-style feedback, and ablation.
- `SIMPLE_BASELINE_EXPLAINS_METHOD` if static inverse gain matches or beats full.
- `DIRECT_PRIOR_EXPLAINS_METHOD` if APEX-style feedback matches or beats full.
- `KEY_COMPONENT_NOT_USEFUL` if no-feedback ablation matches or beats full.
- `CLEAN_RETENTION_FAILURE` if clean no-fault retention materially degrades.
