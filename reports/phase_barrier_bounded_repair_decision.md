# PhaseBarrier-VLA Bounded Repair Decision

Date: 2026-07-12 KST

Final decision: `PHASEBARRIER_COMPONENT_NOT_USEFUL`

## Decision Rationale

The valid bounded repair reused the original saved PhaseBarrier weights and evaluated `100` paired closed-loop episodes on the original two targeted tasks. The full method completed `20/20` assigned episodes with no exceptions and materially changed actions, but it scored `0/20`.

Decisive comparison:

- frozen SmolVLA: `8/20`
- strongest non-ablation baseline: frozen SmolVLA, `8/20`
- simple global damping: `0/20`
- no-phase ablation: `9/20`
- full PhaseBarrier: `0/20`

The no-phase ablation beat the full phase-conditioned component by `9` successes and `45` task-balanced percentage points. Paired full-versus-ablation accounting was `0/9/11` W/L/T for full. Therefore the phase-conditioned component is not useful in this implementation.

## Why Not Another Decision

- Not `PHASEBARRIER_PROTOTYPE_GO`: full did not improve over the strongest baseline and did not beat ablation.
- Not `PHASEBARRIER_GENUINE_METHOD_KILL`: the sharper cause is ablation domination, not merely failure against baselines.
- Not `PHASEBARRIER_KILLED_BY_SIMPLE_BASELINE`: simple global damping matched full at `0/20`, but the key ablation beat full and is the more specific reviewer-relevant failure.
- Not `PHASEBARRIER_IMPLEMENTATION_FAILURE`: full modified actions in `20/20` episodes with mean action delta `0.105796`.
- Not `PHASEBARRIER_RESULT_STILL_INCONCLUSIVE`: the full method scored `0/20` while the key ablation scored `9/20`; this is resolved enough for final PhaseBarrier adjudication.

## Automatic Next Action

Archive PhaseBarrier-VLA and do not rescue it.

CensorCredit-VLA remains classified as `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` from the postmortem. Its exact documented failure is that censored and uncensored generated labels were identical for `24/24` records, yielding identical model weights. This run does not repair CensorCredit because the prompt explicitly forbids repairing CensorCredit in this run.

Permitted later action, only if explicitly reopened: a bounded CensorCredit implementation repair that first demonstrates differentiated censored versus uncensored labels before any closed-loop repeat.
