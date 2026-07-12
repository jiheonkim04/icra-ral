# Epoch 2 Cycle 2 Candidate Generation

Date: 2026-07-12 KST

Governance: `reports/current_research_governance.md`

Cycle 2 pivots away from PTC-VLA. PTC was a direct policy-input-state transition-conditioned action head and was permanently killed in Stage A. The candidates below must therefore change at least two core dimensions relative to PTC and the Epoch 1 routes. This set avoids post-hoc delay adapters, low-level feedback residual correction, image hold-last or edge repair, selectors/rankers/verifiers, barriers/filters/damping, generic progress/value/confidence heads, generic DPO, simple action reweighting, and another state-transition direct head.

Latest-literature pressure points:

- CAST argues that counterfactual labels/actions can reduce language posterior collapse in VLA policies: https://arxiv.org/html/2508.13446v2
- LIBERO-CF/CAG shows counterfactual language failures and vision shortcuts under feasible alternative instructions: https://arxiv.org/html/2602.17659v1
- FineVLA reports that process-level instruction supervision can improve steerable manipulation behavior: https://arxiv.org/html/2605.27284v1
- RoboSemanticBench diagnoses a persistent gap between semantic competence and action prediction: https://arxiv.org/html/2606.02277v1
- IGAR proposes train-free attention recalibration for linguistic blindness: https://arxiv.org/abs/2603.06001
- CF-VLA shows the action-generation literature is crowded around efficient coarse-to-fine flow starts: https://arxiv.org/html/2604.24622v1
- TAG shows target/distractor guidance is already an active inference-time baseline family: https://arxiv.org/html/2603.24584v1

Local feasibility facts:

- Standard LeRobot SmolVLA-LIBERO metadata covers 40 LIBERO tasks.
- Local LIBERO HDF5/BDDL assets also include LIBERO-90, but using LIBERO-90 as training data would need careful disclosure.
- The first prototype should use the standard 40-task official closed-loop stack unless a method technically requires a different suite.
- The LeRobot preprocessing path injects task text through `add_envs_task`, and task text can be overridden in a controlled runner for counterfactual guidance baselines.

## Candidate 1: SACF-VLA

Name: `SACF-VLA`, Same-scene Action Counterfactual Factorization VLA

Hidden assumption: same-scene LIBERO task families contain an exploitable action factorization: a phase-local shared manipulation component plus an instruction-semantic component that changes target object, source relation, or destination while preserving much of the visuomotor scaffold.

Precise novelty: train a lightweight semantic-prefix action generator using same-scene counterfactual task families. Instead of generating synthetic counterfactual labels/actions with a VLM, SACF uses existing official LIBERO task families to factor demonstrations into shared phase action and semantic-slot action components. At inference, a fixed early prefix from the factorized head is followed by the frozen VLA, testing whether a learned semantic action factor can steer the policy into the correct physical branch without candidate selection or feedback residual correction.

Equations:

- semantic slot: `z_l = h(l)` from instruction tokens and suite-local slot parser
- phase: `p_t = floor(t / T * P)`
- shared component: `u_t = f_shared(s_t, p_t, family)`
- semantic component: `v_t = f_sem(s_t, p_t, family, z_l)`
- action: `a_t = clip(u_t + v_t, -1, 1)`
- reconstruction: `L_bc = ||a_t - a_demo_t||_2^2`
- same-family counterfactual contrast: `L_cf = ||(v_i - v_j) - stopgrad(a_i - a_j)||_2^2`
- shared invariance: `L_inv = ||u_i - u_j||_2^2` for matched phase/family pairs

Representation: policy-input robot state, fixed phase, suite/family code, and semantic slot features parsed from task text.

Objective: behavior cloning plus same-family counterfactual factorization; not preference learning, ranking, progress prediction, or transition modeling.

Supervision: official LIBERO HDF5 demonstration actions for standard task families. LIBERO-90 is reserved for later second-condition design only if Reviewer B approves.

Inference: fixed semantic prefix for a preregistered first fraction of the episode, then frozen SmolVLA. No learned selector, no online success signal, no simulator privileged state, no residual added to frozen action.

Direct baseline: plain same-data BC prefix head without counterfactual factorization.

Simple killer baseline: task/phase mean-action prefix with the same fixed handoff.

Closest papers:

| Paper | Overlap | Difference |
| --- | --- | --- |
| CAST | counterfactual language/action supervision | SACF uses existing same-scene task families, no VLM-generated counterfactual action chunks |
| CAG / LIBERO-CF | counterfactual feasible instructions in LIBERO | SACF trains a factorized action generator rather than dual-branch inference guidance |
| FineVLA | process-level instruction/action alignment | SACF is a local counterfactual action-factor objective, not large fine-grained annotation |
| IGAR | language-grounding mitigation | SACF is trained action factorization, not train-free attention recalibration |
| TAG | object/distractor guidance | SACF does not erase targets or use residual guidance from counterfactual images |

Core dimensions changed relative to PTC:

- core problem: semantic counterfactual language/action grounding rather than short-horizon transition under-modeling;
- representation: semantic slots and same-scene task family factors rather than transition latents;
- supervision: expert demonstration family contrasts rather than frozen-policy state transitions;
- objective: counterfactual factorization rather than transition-conditioned MSE;
- inference: fixed semantic prefix plus frozen handoff rather than direct full-episode state-transition head.

Prototype tasks: standard LIBERO semantic families, initially two tasks from `libero_spatial` and/or `libero_object` with paired reset identities.

Second-backbone path: if Stage B/GO exists, fit the same semantic prefix using Quantized OpenVLA-OFT INT4 traces or shared demonstration data, then compare OpenVLA-OFT INT4 against OpenVLA-OFT INT4 + SACF fixed prefix.

Second-condition path: held-out same-scene LIBERO-90 or LIBERO-CF-style task variants, only after primary prototype GO.

Failure risk: high. A plain BC prefix or phase-mean prefix may explain all gains, or the fixed prefix may disrupt frozen VLA recovery.

## Candidate 2: LCAR-VLA

Name: `LCAR-VLA`, Language-Contrast Action Recalibration VLA

Hidden assumption: the frozen policy already encodes useful language sensitivity in its action distribution, but the signal is weak and can be exposed by comparing full-instruction and null/counterfactual-instruction action predictions.

Precise novelty attempt: calibrate a low-rank action-space language direction from training tasks, then apply a fixed-dimensional action recalibration at inference.

Equations:

- full action: `a_l = pi(o, l)`
- null action: `a_0 = pi(o, "")`
- language direction: `d_l = P_k(a_l - a_0)`
- action: `a = clip(a_l + gamma d_l, -1, 1)`

Representation: frozen VLA actions under full and null task text, plus low-rank action-language basis.

Objective: train-free or lightly calibrated language-action direction exposure.

Direct baseline: CAG-style dual-branch guidance.

Simple killer baseline: uncalibrated fixed `a_l + gamma(a_l - a_0)`.

Primary weakness: this is very close to CAG, IGAR, and TAG-style guidance. It risks collapsing into an inference-time guidance baseline, and prior ECHO found no useful candidate headroom on its tested same-state action alternatives.

Decision: keep as a baseline or fallback diagnostic, not the selected contribution.

## Candidate 3: ECAF-VLA

Name: `ECAF-VLA`, Endpoint-Conditioned Action Flow VLA

Hidden assumption: the main issue is not semantic grounding but poor action-generation initialization; predicting a coarse endpoint before local action refinement can improve low-NFE SmolVLA action chunks.

Precise novelty attempt: add a tiny endpoint-conditioned action initializer trained from demonstration chunks, then refine with a local one-step head.

Equations:

- endpoint prior: `e_t = f_e(o_t, l)`
- local refinement: `a_t = f_r(o_t, l, e_t)`
- loss: `||e_t - a_{t+H}||^2 + ||a_t - a_demo_t||^2`

Representation: action endpoint and local refinement code.

Objective: coarse-to-fine action generation.

Direct baseline: CF-VLA-style coarse-to-fine local proxy.

Simple killer baseline: plain BC chunk head.

Primary weakness: CF-VLA already occupies the coarse-to-fine action-generation claim strongly, and this route is adjacent to recent latent-action/flow work plus PTC's failed direct action-head neighborhood.

Decision: not selected for Cycle 2.

## Selection

Selected method: `SACF-VLA`.

Reason: SACF is the strongest locally feasible pivot that changes core problem, representation, supervision, objective, and inference structure relative to PTC. It attacks a currently active literature gap, uses standard LIBERO semantic task families, admits strong reviewer-killer baselines, and can be implemented as a bounded Stage A without downloads or full-model fine-tuning.
