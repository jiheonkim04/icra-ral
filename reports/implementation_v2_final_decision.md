# Implementation V2 Final Decision

Date: 2026-07-11 KST

Final decision: `TWO_IMPLEMENTED_METHODS_KILLED`

## Reclassification

The prior terminal decision `NO_METHOD_AFTER_3_VALID_CYCLES` is preserved as literature triage but reclassified as `PREMATURE_LITERATURE_ONLY_TERMINATION`.

Reason: the three earlier cycles did not include code implementation, frozen prototype execution, baseline comparison, ablation, or closed-loop task-success evidence.

## Implemented Cycle 1

Method: `PhaseBarrier-VLA`

Mechanism: phase-conditioned feasibility-field action projection over postprocessed SmolVLA actions.

Evidence:

- code: `tca_map/smolvla/phase_barrier_vla.py`
- runner: `scripts/run_phase_barrier_vla_prototype.py`
- protocol: `reports/phase_barrier_vla_prototype_protocol.md`
- result: `reports/phase_barrier_vla_prototype_result.json`
- final decision: `PHASE_BARRIER_VALID_KILL`
- training happened: `true`
- closed-loop experiment happened: `true`
- variants: frozen SmolVLA, Pre-VLA-style halt proxy, simple global damping, no-phase ablation, full PhaseBarrier

Result summary:

- frozen SmolVLA task-balanced success: `0.0`
- strongest non-ablation baseline: `0.0`
- no-phase ablation: `0.0`
- full PhaseBarrier: `0.0`
- GO: `false`

Kill reason: full method did not improve task-balanced closed-loop success and did not establish value of the phase-conditioned component.

## Implemented Cycle 2

Method: `CensorCredit-VLA`

Mechanism: intervention-censored temporal credit with action-history hold/blend at inference.

Evidence:

- code: `tca_map/smolvla/censored_credit_vla.py`
- runner: `scripts/run_censor_credit_vla_prototype.py`
- protocol: `reports/censor_credit_vla_prototype_protocol.md`
- result: `reports/censor_credit_vla_prototype_result.json`
- final decision: `CENSOR_CREDIT_VALID_KILL`
- training happened: `true`
- closed-loop experiment happened: `true`
- variants: frozen SmolVLA, VLA-Corrector-style jump proxy, simple temporal EMA, uncensored recovery ablation, full CensorCredit

Result summary:

- frozen SmolVLA task-balanced success: `0.0`
- strongest non-ablation baseline: `0.0`
- uncensored recovery ablation: `0.5`
- full CensorCredit: `0.5`
- absolute gain over strongest non-ablation baseline: `50.0` percentage points
- GO: `false`

Kill reason: full method improved over frozen/simple baselines, but the key uncensored ablation matched it. The censored temporal-credit component was therefore not proven necessary.

## Final Status

Paper-ready status: `false`

The campaign now satisfies the user's implementation requirement for a terminal no-method result:

- at least two genuinely distinct methods were implemented;
- both had frozen protocols;
- both had unmodified backbone baselines;
- both had direct-prior/local-proxy baselines;
- both had simple killer baselines;
- both had key ablations;
- both ran closed-loop task-success evaluations;
- both produced fixed GO/KILL decisions.

No main-branch update was made.

## Reopen Condition

Reopen only for a stronger repeat of CensorCredit if the user wants to investigate the weak positive signal despite the ablation kill, or for a genuinely new method with a different mechanism and a predeclared implementation path.
