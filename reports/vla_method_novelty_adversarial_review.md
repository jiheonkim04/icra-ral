# VLA Method Novelty Adversarial Review

Date: 2026-07-11 KST

Role: hostile RA-L reviewer. No implementation or experiment occurred.

## Weighted Scores

| Candidate | Novelty 40 | Method depth 20 | Experimental strength 20 | Feasibility 10 | Review robustness 10 | Weighted score | Estimated RA-L strength | Kill probability | Decisive weakness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| ECHO-VLA | 34 | 17 | 17 | 8 | 7 | 83 | Strong if effect labels are clean | 0.35 | Could collapse to a progress/value reranker if effect representation is too coarse |
| BARRIER-VLA | 29 | 17 | 15 | 6 | 5 | 72 | Borderline-strong | 0.55 | Simple clipping/geometric filters and VeriSpace comparisons may erase novelty |
| SEMAPHORE-VLA | 25 | 14 | 14 | 8 | 4 | 65 | Weak-to-borderline | 0.60 | SPR/ProgressVLA/ProgVLA proximity is severe |
| IRIS-VLA | 28 | 16 | 16 | 5 | 5 | 70 | Borderline | 0.50 | VLA-Corrector and static visual-change baselines are dangerous |

## Candidate 1 - ECHO-VLA

- Problem actually important: yes. It directly targets action-loss versus task-success mismatch.
- Already solved by recent preprint: not found. Pre-VLA predicts safety/advantage, ProgressVLA predicts progress, AFIL uses failure negatives, CoVer/VeriSpace verify candidates. None centers interventional predicate-effect credit.
- Renamed known method: not if the paper keeps `P(effect | do(action), context)` and matched counterfactual labels as the core mechanism.
- Stronger backbone eliminates issue: no. Stronger backbones can improve actions but not necessarily calibrate physical effect under perturbation.
- More frequent replanning solves it: no. Replanning without effect credit can repeat high-likelihood low-effect actions.
- Data augmentation solves it: partly but not decisively. Augmentation does not identify which action caused which predicate effect.
- Simple MLP/static threshold/action filter solves it: possible if labels are weak. Must include value/progress/verifier baselines.
- Privileged inference signals: no if predicate/effect estimators are distilled.
- Evaluation-only: no, it changes training objective and action selection/guidance.
- Can improve task success: plausible under perturbations and hard phase transitions.
- Gap large enough: yes.
- Two backbones: yes, SmolVLA and OpenVLA-OFT INT4.
- Two conditions: yes, standard LIBERO and controlled perturbation/LIBERO-Plus.
- Path to beating strongest baseline: credible if effect labels distinguish action-equivalent failures.
- Verdict: keep and select.

## Candidate 2 - BARRIER-VLA

- Problem actually important: yes, especially contact-rich manipulation.
- Already solved: not exactly, but VeriSpace and Pre-VLA occupy spatial/safety verification.
- Renamed known method: risk of being seen as a learned safety filter or CBF wrapper.
- Stronger backbone: may reduce obvious violations, but not all constraints.
- More frequent replanning: helps but does not enforce constraint residuals.
- Data augmentation: strong baseline.
- Simple filter: high risk, especially for LIBERO.
- Privileged inference: no if distilled, but signed labels are privileged during training.
- Evaluation-only: no.
- Task success: plausible but may improve validity without success.
- Two backbones/conditions: credible.
- Path to beating strongest baseline: uncertain.
- Verdict: reject as primary due high baseline-kill and prior-art proximity.

## Candidate 3 - SEMAPHORE-VLA

- Problem actually important: yes for long-horizon tasks.
- Already solved: partially by SPR, ProgressVLA, ProgVLA, Gemini Robotics-ER style reasoning.
- Renamed known method: high risk as progress/phase prediction.
- Stronger backbone: could reduce semantic phase errors.
- More frequent replanning: may help.
- Data augmentation: likely competitive.
- Simple MLP/static phase head: dangerous baseline.
- Privileged inference: avoidable.
- Evaluation-only: no, but method novelty is fragile.
- Task success: plausible on long horizon.
- Two backbones/conditions: credible.
- Path to beating strongest baseline: weak.
- Verdict: reject as primary.

## Candidate 4 - IRIS-VLA

- Problem actually important: yes, irreversible errors are physically meaningful.
- Already solved: VLA-Corrector is very close in adaptive correction; Pre-VLA is close in preemptive action validity.
- Renamed known method: risk if recoverability margin is just confidence under another name.
- Stronger backbone: disturbances can still create irreversibility.
- More frequent replanning: only helps before boundary crossing.
- Data augmentation: could be competitive.
- Simple visual-change detector: dangerous baseline.
- Privileged inference: avoidable but recoverability labels are expensive.
- Evaluation-only: no.
- Task success: plausible under disturbance protocol.
- Two backbones/conditions: credible but heavier.
- Path to beating strongest baseline: possible but less immediate than ECHO.
- Verdict: reject as primary.

## Selection

Selected primary method: `ECHO-VLA`.

Why exactly one: it has the clearest method-level novelty, the strongest bridge from literature gap to closed-loop success, a feasible local prototype path, and the cleanest distinction from recent verification/progress/correction papers.

Rejected candidates:

- `BARRIER-VLA`: likely killed by simple action clipping/geometric filters or reframed as VeriSpace/Pre-VLA with a differentiable wrapper.
- `SEMAPHORE-VLA`: novelty collapses toward SPR/ProgressVLA/ProgVLA and manual phase baselines.
- `IRIS-VLA`: too close to VLA-Corrector unless recoverability labels prove a large difference, and prototype labels are costly.
