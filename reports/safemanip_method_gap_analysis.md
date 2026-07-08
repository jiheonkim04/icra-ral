# SafeManip Method Gap Analysis

Date: 2026-07-08

This document identifies method gaps only after reading the official SafeManip
paper/code surface. It does not propose or implement a method.

## Anchor Interpretation

SafeManip is an evaluation benchmark. It makes temporal manipulation safety
measurable through LTLf properties, simulator-state predicate extraction, DFA
monitoring, and analysis metrics. It does not optimize a policy, train a safety
head, correct actions, or intervene at deployment time.

The benchmark is valuable because it makes task completion and temporal safety
separable. The paper's main result is that success improvements can still leave
policies in high violation regimes.

## Gap Matrix

| Gap requested by scout | SafeManip status | Evidence/interpretation | Method implication |
| --- | --- | --- | --- |
| Evaluation-only gap | Present | SafeManip is a benchmark and monitor pipeline | It diagnoses, but does not improve, policies |
| No training/improvement method | Present | Paper states evaluated policies are externally provided and not trained/fine-tuned in the work | Any improvement method would be outside SafeManip |
| No preference optimization | Present | No DPO/preference or pairwise safety optimization is official | Generic DPO would be a mandatory future baseline, not assumed novel |
| No multi-model correction | Present | No critic/corrector ensemble or multi-model intervention is provided | Any correction route must beat simple safety-only and no-op baselines |
| No deployment-time intervention | Present | Monitors evaluate rollouts; they do not modify policy actions | A runtime route remains open only after official subset reproduction |
| No utility-preserving safety optimization | Present | Safety prompts reduce success sharply while only modestly reducing violations | Safe-success, not violation rate alone, is the core target |

## Strongest RA-L Topic Gap

The strongest topic gap is utility-preserving temporal safety improvement on
SafeManip:

- Increase success-and-safe rollouts.
- Reduce success-but-unsafe rollouts.
- Reduce property/category violation and exposure.
- Preserve task success relative to base policy and official prompt variants.

This is a real gap because SafeManip's own prompt pilot demonstrates a common
failure mode: conservative safety guidance can lower violation rate mostly by
destroying task success. That leaves room for a method that uses temporal safety
signals without collapsing utility.

However, this is not yet a go signal for a method. The local project reset
requires official anchor reproduction first. Under current constraints, even a
minimal official SafeManip subset is too heavy locally.

## Simple Baseline Risks

Before any SafeManip method claim, these baselines would be mandatory:

| Baseline | Why it is dangerous |
| --- | --- |
| Base policy | SafeManip already reports multiple base policies; any method must beat the relevant one |
| Official safety prompts | The paper's short/long prompts are direct conservative-safety baselines |
| No-op/abort | Can create fail-safe rollouts and reduce violations without task utility |
| Stop-on-risk | Can reduce exposure by halting, but may lower success |
| Safety-only filter | Can optimize the safety metric without preserving task completion |
| Clipping-only | May reduce contact magnitude or extreme actions without understanding temporal constraints |
| Generic reward penalty | If training is allowed later, this is the obvious non-novel safety baseline |
| Generic DPO/preference tuning | If pairwise labels exist, this is the obvious preference baseline |

The key scoring guardrail is:

```text
safe_success_rate = fraction of rollouts with task_success = 1 and no property violation
```

Violation-only gains are insufficient.

## What Simple Baselines Are Unlikely To Solve

Simple conservative baselines can probably reduce some violation counts, but
they are unlikely to solve utility-preserving temporal safety by themselves:

- No-op/abort and stop-on-risk likely increase fail-but-safe rollouts.
- Prompt-only safety guidance already shows a success collapse in SafeManip.
- Clipping-only does not address temporal ordering, contamination, delayed
  settling, or enclosure sequencing.
- Reward penalties and DPO require labels/training and would need full
  benchmark validation to avoid overfitting frequent categories.

These are hypotheses for evaluation design, not a new method proposal.

## Method-Level Opportunity

No method-level route should be started now.

A narrow method direction may become valid only if all three gates pass:

1. SafeManip official subset reproduction is feasible on GPU/cloud.
2. The reproduced subset confirms a clear failure gap in safe-success, not just
   aggregate violation rate.
3. Safety-only, prompt-only, stop/abort, clipping, and no-op baselines fail to
   close that gap without unacceptable task-success loss.

If those gates pass, the acceptable method topic would be narrowly constrained
to improving SafeManip metrics while preserving task utility. The scout does
not invent that method.

## Current Gap Verdict

SafeManip has a clear benchmark-level gap and a plausible RA-L method opening,
but the project should not proceed to method design until a reproducible
official SafeManip subset is established. Because that subset is too heavy
locally under current constraints, the immediate route is blocked by resources,
not by lack of a conceptual gap.
