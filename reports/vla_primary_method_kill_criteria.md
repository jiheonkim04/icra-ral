# ECHO-VLA Fixed Kill Criteria

Date: 2026-07-11 KST

These criteria are frozen before implementation. They are prototype decision criteria, not a demand for perfect RA-L certainty.

## Prototype Kill Criteria

Kill or fundamentally redesign ECHO-VLA if any decisive condition holds:

- the counterfactual effect component does not beat its ablation,
- a simple heuristic predicate-progress baseline matches it,
- a progress/value head matches it,
- a Pre-VLA-style validity/advantage head matches it,
- closed-loop task success does not improve,
- task-balanced success gain is less than `5` absolute percentage points on the targeted hard condition,
- improvement exists only in offline action, effect, or calibration metrics,
- the method needs privileged simulator state at inference,
- gains depend on favorable test seeds or one task only,
- latency or forward-pass overhead is disproportionate to success gain,
- the effect labels are too coarse to explain or predict task success,
- a stronger OpenVLA-OFT INT4 baseline removes the claimed problem in the full matrix,
- novelty collapses into confidence, progress, generic verification, generic candidate selection, generic replanning, or adaptive chunking.

## Non-Kill Conditions

Do not kill solely because:

- standard LIBERO overall SOTA is not achieved,
- the first prototype uses only SmolVLA,
- simulator labels are used during training,
- the method requires a second-backbone follow-up after prototype success,
- some tasks show no gain while the predeclared effect-critical condition improves.

## Redesign Directions If Killed

- If progress/value matches ECHO: make effect representation more predicate-specific or abandon the method.
- If heuristic matches ECHO: reduce architecture complexity or abandon.
- If labels are noisy: improve predicate extraction before any model scaling.
- If latency dominates: switch from candidate reranking to single-sample differentiable guidance or abandon.
- If OpenVLA-OFT removes the effect: reframe as backbone-specific engineering and stop paper push.
