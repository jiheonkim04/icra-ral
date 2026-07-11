# VLA Method Design Decision

Date: 2026-07-11 KST

Branch: `codex/paper-first-vla-ral-method-design`

Base main commit at goal start: `5c2a3645fce4920340e2155ee252c4e7821b47f2`

## Process Compliance

- experiments happened: `no`
- training happened: `no`
- GPU/model inference happened: `no`
- simulator execution happened: `no`
- large download happened: `no`
- implementation happened: `no`
- old killed routes revived: `no`

## Literature Reviewed

Primary-source count reviewed: `34`.

Close recent methods that shaped the decision include OpenVLA-OFT, SmolVLA, pi0, GR00T N1, Gemini Robotics, VLAConf, CoVer, VeriSpace, Pre-VLA, VLA-Corrector, AAC, SEAM, Legato, DREAM-Chunk, SPR, ProgressVLA, ProgVLA, AFIL, PriorVLA, CLARE, VLA-GSE, LIBERO-Plus, LIBERO-Occ, LIBERO-CF/CAG, STRONG-VLA, and CRT.

## Ten Implicit Opportunities

1. Counterfactual action-effect credit for action chunks.
2. Phase-conditioned predicate effects instead of scalar progress.
3. Effect-equivalence classes for action candidate diversity.
4. Privileged training labels with non-privileged deployment.
5. Effect-aware closed-loop calibration.
6. Nonmonotonic long-horizon credit.
7. Contact-phase effect learning from visual proxies.
8. Risk-sensitive effect selection under intervention cost.
9. Counterfactual language-control coupling.
10. Robustness as effect invariance rather than observation invariance.

## Four Method Candidates

| Candidate | Score | Decision |
| --- | ---: | --- |
| ECHO-VLA: Counterfactual Effect Credit | 83 | selected |
| BARRIER-VLA: Phase-Conditioned Barrier Residuals | 72 | rejected |
| SEMAPHORE-VLA: Semantic Phase Tokens | 65 | rejected |
| IRIS-VLA: Irreversibility-Aware Intervention | 70 | rejected |

## Rejected Candidates And Exact Reasons

- `BARRIER-VLA`: high risk of being killed by clipping, simple geometric filters, VeriSpace, or Pre-VLA comparisons.
- `SEMAPHORE-VLA`: novelty collapses toward SPR, ProgressVLA, ProgVLA, or manual phase/progress baselines.
- `IRIS-VLA`: too close to VLA-Corrector and Pre-VLA unless expensive recoverability labels prove a large separation.

## Selected Primary Method

Selected primary method: `ECHO-VLA`.

Technical novelty: phase-conditioned counterfactual action-effect credit, estimating `P(effect | do(action_chunk), observation, instruction, phase)` and using it as a training/guidance objective for VLA action chunks.

Closest papers:

- Pre-VLA
- ProgressVLA / ProgVLA
- CoVer / VeriSpace
- AFIL
- OpenVLA-OFT

Exact method difference:

- ECHO predicts vector physical predicate effects, not scalar confidence, scalar progress, geometric validity, failure guidance, or action L1.
- ECHO trains with matched-context counterfactual action chunks, so the label isolates the action's expected effect under the same observation/instruction/phase.
- ECHO uses privileged predicates only for training labels and deploys from images/proprioception/language/action chunks.

## First Decisive Prototype

Backbone: SmolVLA.

Tasks:

- `libero_spatial/task_0`: pick up the black bowl between the plate and the ramekin and place it on the plate.
- `libero_object/task_4`: pick up the ketchup and place it in the basket.
- `libero_goal/task_0`: open the middle drawer of the cabinet.
- `libero_10/task_0`: put both the alphabet soup and the tomato sauce in the basket.

Baselines:

- frozen backbone,
- standard adaptation if relevant,
- heuristic effect/progress,
- progress/value head,
- Pre-VLA-style validity/advantage head,
- ECHO without counterfactual ranking.

Success threshold: at least `5` absolute task-balanced success points over the strongest simple baseline, while beating the no-counterfactual ablation.

Kill threshold: no closed-loop gain, simple baseline match, no ablation gain, privileged inference dependency, or latency/compute disproportion.

## Full RA-L Experiment Plan

Backbones:

1. SmolVLA.
2. Quantized OpenVLA-OFT INT4.

Conditions:

1. Standard LIBERO.
2. Controlled execution perturbation or LIBERO-Plus condition selected to test action-effect robustness.

Primary metric: closed-loop task success.

Secondary metrics: task-balanced success, predicate-effect F1, calibration, latency, VRAM, forward passes, perturbation degradation.

Expected SOTA axis: closed-loop success under controlled execution perturbation with calibrated phase-required action-effect prediction.

Estimated RA-L strength: strong if first prototype and second-backbone condition pass.

Estimated kill probability: `0.35`.

## Final Decision

READY_TO_IMPLEMENT_PRIMARY_VLA_METHOD

## Exact Next Implementation Prompt

Implement the first ECHO-VLA prototype only. Use official SmolVLA-LIBERO as the first backbone; do not use OpenVLA-OFT INT4 until the SmolVLA gate passes. Build a bounded effect-label dataset for `libero_spatial/task_0`, `libero_object/task_4`, `libero_goal/task_0`, and `libero_10/task_0`; train only the lightweight visual predicate/effect heads and ECHO ranking objective; compare against frozen SmolVLA, a heuristic predicate-progress baseline, a progress/value head, a Pre-VLA-style validity/advantage head, and the no-counterfactual ECHO ablation. Run no full benchmark. Kill the method if ECHO full fails to beat the strongest simple baseline by at least 5 absolute task-balanced success points or fails to beat its no-counterfactual ablation.
