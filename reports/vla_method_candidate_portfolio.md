# VLA Method Candidate Portfolio

Date: 2026-07-11 KST

Generated exactly four candidates. No experiment, model inference, training, GPU work, download, simulator execution, or implementation occurred.

## Candidate 1 - ECHO-VLA: Counterfactual Effect Credit for Closed-Loop VLA Manipulation

Class: A, training objective / credit assignment.

1. Working title: `ECHO-VLA`.
2. Exact problem: VLA policies are trained to imitate actions, but closed-loop success depends on whether each action chunk causes the required physical state transition.
3. Physical importance: a visually plausible grasp, lift, insertion, or placement action can have high action likelihood while producing no useful effect, causing compounding failure.
4. Gap: OpenVLA-OFT, SmolVLA, and pi0 optimize action regression/flow/token likelihood; ProgVLA/ProgressVLA use progress; Pre-VLA/CoVer/VeriSpace verify candidates. None trains a phase-conditioned interventional action-effect representation.
5. One-sentence novelty: learn and use `P(effect | do(action_chunk), observation, instruction, phase)` as a VLA training and inference signal.
6. Architecture: frozen or lightly adapted VLA backbone; phase parser from instruction and visual predicate memory; visual predicate estimator; effect critic `E_phi(o_t, l, z_t, a_{t:t+H})`; optional small candidate generator/guidance head.
7. New representation/objective/signal: counterfactual effect code `Delta p` over task predicates and pairwise effect-credit loss under matched context.
8. Mathematical formulation: for state predicate vector `p(s)`, label `e_t = p(s_{t+K}) - p(s_t)`. Learn `E_phi(e | do(a), o, l, z)`. Required phase effect is `r_z`. Effect advantage is `A_e(a)=r_z^T E_phi[a] - beta cost(a)`. Training loss is `L = L_BC + lambda_e CE(E_phi,e) + lambda_rank max(0, m - A_e(a+) + A_e(a-)) + lambda_inv ||E_phi(a) - e||` on intervention pairs.
9. Training data: official demonstrations, short simulator-labeled counterfactual chunks, BDDL/success predicate labels, visual observations, proprioception, language, action chunks.
10. Inference inputs: current images, proprioception, instruction, non-privileged phase memory, generated action chunk candidates.
11. Privileged simulator information: yes, during training labels only.
12. Removal at deployment: distill predicates/effects into visual and proprioceptive estimators; no simulator state or success oracle is queried online.
13. Closest five papers: OpenVLA-OFT, ProgVLA, ProgressVLA, AFIL, Pre-VLA. Also close: CoVer, VeriSpace, VLAConf.
14. Method-level differences: versus OpenVLA-OFT, adds interventional effect objective beyond L1 action regression; versus ProgVLA/ProgressVLA, predicts vector predicate effects rather than scalar progress; versus AFIL, contrasts actions by causal effect rather than failure distribution; versus Pre-VLA, trains effect credit, not pre-execution safety/advantage verification; versus CoVer/VeriSpace, learns phase-conditioned physical effects rather than alignment/geometric validity scores.
15. Stronger-backbone objection: a stronger backbone can improve action likelihood but still has no explicit training pressure to distinguish high-likelihood low-effect chunks from lower-likelihood success-critical chunks.
16. Reviewer-killer baseline: train a small progress/value head or Pre-VLA-style validity head and use it to rerank the same candidates.
17. Prototype experiment: SmolVLA on a small LIBERO subset with short counterfactual labels; compare ECHO-guided candidate choice against frozen SmolVLA, standard adaptation, progress/value head, and heuristic action filter.
18. Full two-backbone experiment: SmolVLA and quantized OpenVLA-OFT INT4, same effect labels and deployment interface.
19. Second benchmark/condition: LIBERO-Plus controlled initial-state/action-disturbance perturbation, because effect credit should preserve task predicate transitions under execution shift.
20. Expected compute: prototype fits RTX 5080 with a lightweight effect head and small label set; full paper may require longer CPU/simulator label generation but not unavailable hardware.
21. Expected success axis: closed-loop success under controlled execution perturbation and effect-validity success tradeoff.
22. Novelty risk: medium, because Pre-VLA and progress papers are close but not causal effect-credit methods.
23. Implementation risk: medium, mainly label quality and counterfactual data generation.
24. Probability simple baseline kills it: `0.35`.

## Candidate 2 - BARRIER-VLA: Phase-Conditioned Barrier Residuals Inside VLA Action Generation

Class: B, control/physics-structured VLA method.

1. Working title: `BARRIER-VLA`.
2. Exact problem: VLA chunks may violate implicit kinematic, collision, support, or contact constraints even when semantically correct.
3. Physical importance: contact-rich manipulation fails when trajectories scrape, collide, lose support, or cross unrecoverable contact boundaries.
4. Gap: VeriSpace verifies spatial validity and Pre-VLA predicts safety, but neither changes the action-generation objective with a learned phase-conditioned barrier residual.
5. One-sentence novelty: embed learned control-barrier residuals as differentiable constraints inside VLA action generation rather than post-hoc filtering.
6. Architecture: VLA backbone plus barrier residual network `B_theta(o,l,z,a,tau)` trained from signed constraint labels and used as a differentiable penalty/guidance term during action decoding or flow sampling.
7. New representation/objective/signal: phase-conditioned barrier margin for chunk prefixes, including support, collision, joint-limit, and contact transition margins.
8. Mathematical formulation: learn `h_i(o,l,z,a_{1:k}) >= 0` for constraint `i`; train `L_bar = sum_i softplus(-h_i)` plus imitation; guide action by minimizing `L = L_action + alpha sum_i softplus(-h_i)`.
9. Training data: demonstrations plus simulator-derived signed distances, collisions, support/contact labels, and failed perturbations.
10. Inference inputs: observations, language, phase, candidate chunk or flow latent.
11. Privileged simulator information: training only.
12. Removal at deployment: barrier residual is visual/proprioceptive and does not query simulator geometry online.
13. Closest five papers: VeriSpace, Pre-VLA, VLA-Corrector, Legato, OpenVLA-OFT.
14. Differences: unlike VeriSpace/Pre-VLA, it modifies generation not just verification; unlike VLA-Corrector, it prevents barrier crossings rather than correcting visual drift; unlike Legato, it targets physical constraints not smoothness; unlike OpenVLA-OFT, it adds structured robotics loss.
15. Stronger-backbone objection: more data can reduce constraint violations, but hidden contact/support constraints remain sparse and phase-specific.
16. Reviewer-killer baseline: hard action clipping plus simple collision/height thresholds.
17. Prototype experiment: LIBERO drawer/placement tasks with synthetic perturbations and barrier-label ablations.
18. Full two-backbone experiment: SmolVLA and OpenVLA-OFT INT4 with identical residual head.
19. Second condition: controlled action disturbance or LIBERO-Plus initial-state perturbation.
20. Expected compute: label extraction may be heavier than ECHO; training head is light.
21. Expected success axis: constraint-valid success under physical perturbation.
22. Novelty risk: medium-high due VeriSpace/Pre-VLA proximity.
23. Implementation risk: high because reliable signed barriers may be hard in LIBERO.
24. Probability simple baseline kills it: `0.55`.

## Candidate 3 - SEMAPHORE-VLA: Semantic Phase Tokens for Continuous Control

Class: C, representation/reasoning linking temporally extended semantics to control.

1. Working title: `SEMAPHORE-VLA`.
2. Exact problem: long-horizon instructions describe semantic phases, but continuous action heads often condition on one global instruction.
3. Physical importance: a policy can perform a locally plausible action for the wrong subgoal or switch phases before the object state supports it.
4. Gap: SPR uses language-to-spatial milestones and progress/recovery; ProgressVLA and ProgVLA add progress. They do not bind language clauses to continuous-control phase tokens with predicate-gated action objectives.
5. One-sentence novelty: induce semantic phase tokens whose validity is gated by visual predicate state and whose action loss is restricted to phase-relevant effects.
6. Architecture: instruction decomposer; phase-token encoder; predicate-gated memory; VLA action head conditioned on active phase token; phase transition model.
7. New representation/objective/signal: phase-token/action-effect alignment and phase-transition contrastive loss.
8. Mathematical formulation: infer `z_t in {1..M}`; train `p(z_t | o_{<=t}, l)` with weak labels from predicate transitions; action objective `L = sum_t L_action(a_t | o_t,l,z_t) + gamma CE(T(z_t,o_t),z_{t+1}) + eta InfoNCE(z_t,e_t)`.
9. Training data: demonstrations, language, BDDL predicate traces, optional automatic phase labels.
10. Inference inputs: image/proprio history, language, active phase memory.
11. Privileged simulator information: optional for phase labels during training.
12. Removal at deployment: visual predicate estimator and learned phase transition model.
13. Closest five papers: SPR, ProgressVLA, ProgVLA, CoVer, Gemini Robotics-ER.
14. Differences: unlike SPR, no external waypoint/rewind loop; unlike progress papers, phase is discrete semantic-control state; unlike CoVer, it conditions generation; unlike Gemini Robotics-ER, it is a trainable VLA control representation.
15. Stronger-backbone objection: a larger VLM may parse semantics better but still may not know when a physical predicate justifies phase transition.
16. Reviewer-killer baseline: add a standard progress head or manually segment phases by time.
17. Prototype experiment: LIBERO long-horizon tasks with phase labels from BDDL predicate transitions.
18. Full two-backbone experiment: SmolVLA and OpenVLA-OFT INT4.
19. Second condition: counterfactual language or LIBERO-CF-like instruction perturbations.
20. Expected compute: moderate; phase labels are cheaper than simulator counterfactuals.
21. Expected success axis: long-horizon phase consistency and task success.
22. Novelty risk: high because SPR/ProgressVLA/ProgVLA are close.
23. Implementation risk: medium.
24. Probability simple baseline kills it: `0.60`.

## Candidate 4 - IRIS-VLA: Irreversibility-Aware Deployment Intervention

Class: D, robustness/deployment-time method with specific intervention.

1. Working title: `IRIS-VLA`.
2. Exact problem: generic replanning or correction can be too late when a chunk is about to cause irreversible negative physical effects such as losing a grasp, pushing an object out of reach, or closing a wrong container.
3. Physical importance: not all errors are equal; some low-level deviations are recoverable and others destroy task feasibility.
4. Gap: VLA-Corrector reacts to latent visual drift; Pre-VLA verifies before execution; AAC adapts horizon. None explicitly models irreversible negative effect boundaries.
5. One-sentence novelty: predict whether continuing the current chunk crosses a learned irreversible-effect boundary and intervene only at that boundary.
6. Architecture: irreversible-effect predictor over observation history, current chunk prefix, and phase; intervention policy that either continues, truncates, or requests a short recovery chunk.
7. New representation/objective/signal: recoverability margin `rho_t`, trained from simulator rollouts that determine whether the state remains recoverable to the required predicate.
8. Mathematical formulation: `rho(o_t,h_t,a_{k:H},z)=P(exists policy pi recovers r_z within B steps)`. Intervene if `rho < tau_z` and predicted effect loss exceeds cost. Loss is calibrated BCE plus intervention-cost risk.
9. Training data: demonstrations, bounded perturbation rollouts, recoverability labels.
10. Inference inputs: observation history, current chunk prefix, language/phase.
11. Privileged simulator information: training labels only.
12. Removal at deployment: recoverability predictor runs from images/proprio/history only.
13. Closest five papers: VLA-Corrector, Pre-VLA, AAC, DREAM-Chunk, VLAConf.
14. Differences: unlike VLA-Corrector, it predicts irreversible effect boundary rather than visual deviation; unlike Pre-VLA, it operates mid-chunk; unlike AAC, horizon changes are risk-triggered; unlike DREAM-Chunk, no candidate latent future matching; unlike VLAConf, not generic task confidence.
15. Stronger-backbone objection: strong policies still face disturbances and partially observed contact events; irreversibility is a property of environment dynamics.
16. Reviewer-killer baseline: static early-stop threshold or visual-change detector.
17. Prototype experiment: controlled action disturbance on a few LIBERO tasks with recoverability labels.
18. Full two-backbone experiment: SmolVLA and OpenVLA-OFT INT4.
19. Second condition: controlled latency/disturbance protocol.
20. Expected compute: label generation may be expensive; head is light.
21. Expected success axis: recovery success per intervention and success under disturbances.
22. Novelty risk: medium-high due VLA-Corrector and Pre-VLA.
23. Implementation risk: high because recoverability labels are costly.
24. Probability simple baseline kills it: `0.50`.

## Portfolio Summary

| Candidate | Broad class | Novelty risk | Implementation risk | Simple-baseline kill probability | Keep for hostile review |
| --- | --- | --- | --- | --- | --- |
| ECHO-VLA | A | Medium | Medium | 0.35 | Yes |
| BARRIER-VLA | B | Medium-high | High | 0.55 | Yes |
| SEMAPHORE-VLA | C | High | Medium | 0.60 | Yes |
| IRIS-VLA | D | Medium-high | High | 0.50 | Yes |

The portfolio intentionally avoids LoRA, SmolVLA, quantization, generic verification, generic confidence, generic progress, generic replanning, and adapter routing as contributions.
