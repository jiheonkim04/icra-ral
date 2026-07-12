# Epoch 3 Cycle 2 Candidate Generation

Date: 2026-07-12 KST

Decision: select exactly one new Epoch 3 Cycle 2 method after `CBFD-VLA` was killed.

Cycle 2 must not be a cosmetic variant of cross-backbone teacher-trace distillation. It should target a different claim axis and change at least two of core problem, representation, supervision, objective, inference intervention, data source, and claim.

Primary sources used:

- TTT-VLA latent prompt optimization: https://arxiv.org/html/2606.03127v1
- Domain Arithmetic one-shot VLA adaptation: https://arxiv.org/html/2607.00666v1
- VISTA physics-validated visual adaptation: https://arxiv.org/html/2606.04708v1
- One-shot environmental shift adaptation: https://arxiv.org/html/2607.00666v1
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- SmolVLA: https://arxiv.org/abs/2506.01844

## Candidate 1: SCVC-VLA

Name: `SCVC-VLA`, Sensor-Canonicalized VLA Control.

Hidden assumption: under controlled deployment sensor shifts, the VLA's action generator may still be adequate if its visual/proprio inputs are mapped back into the training-domain statistics.

Precise novelty: learn and apply a lightweight per-camera/per-proprio canonicalization layer from calibration identities before the frozen policy. Unlike GCAP, this does not inpaint occlusions or hold frames; unlike TTT-VLA, it does not optimize latent prompt tokens or policy parameters.

Equations: for image stream `x'_c = A_c x_c + b_c`, estimate target calibration stats `(mu_c, sigma_c)` and transform

`c_phi(x'_c) = clip((x'_c - m_c(x'_c)) / (s_c(x'_c)+eps) * sigma_c + mu_c)`.

For proprio shift `q'`, use the same calibration form on the state vector when enabled. The policy executes `a_t = pi(c_phi(o'_t, q'_t), l)`.

Representation: calibrated image/state statistics, not action labels or teacher traces.

Objective: minimize deployment-to-calibration statistic mismatch before inference; no action imitation.

Supervision: clean calibration observations from disjoint train identities, plus controlled sensor-shifted held-out evaluation.

Inference: frozen SmolVLA after canonicalization; no teacher, no action head, no ranking.

Required data: official SmolVLA, official LIBERO exact resets, calibration observations from predeclared train identities.

Closest five papers: TTT-VLA, Domain Arithmetic, VISTA, PAD/test-time training, GCAP-VLA.

Overlap matrix:

| Axis | SCVC-VLA | Closest overlap? |
| --- | --- | --- |
| Core problem | controlled deployment sensor shift | high with TTT/domain adaptation |
| Representation | image/proprio canonical stats | partial |
| Supervision | calibration observations | partial |
| Objective | statistic canonicalization | partial |
| Inference | preprocessing only | distinct from action adapters |
| Data source | train-identity calibration observations | distinct from CBFD |
| Claim | shift robustness with frozen VLA | distinct from clean-task repair |

Direct baseline: TTT/domain-adaptation local proxy as per-frame canonicalization without temporal calibration memory.

Simple killer baseline: true static inverse-affine correction for the predeclared synthetic shift; if this matches full SCVC, the method is only a known-shift preprocessor.

Key ablation: SCVC without temporal/running calibration.

Implementation plan: implement image shift and canonicalization on preprocessed SmolVLA image tensors; collect calibration stats on train identities; Stage A on shifted held-out identities.

Prototype tasks: `libero_spatial/task_4` and `libero_10/task_4`.

Second-backbone path: apply the same canonicalizer before Quantized OpenVLA-OFT INT4 if preprocessing can be safely patched.

Second-condition path: proprio bias/scale shift after visual shift.

Compute estimate: low to moderate, SmolVLA-only Stage A.

Failure risk: high. Simple inverse-affine or per-frame mean/std may match it.

## Candidate 2: KSC-VLA

Name: `KSC-VLA`, Kinematic State Canonicalization for VLAs.

Hidden assumption: some deployment failures come from proprio calibration mismatch rather than visual-language grounding or action generation.

Precise novelty: calibrate the 8D proprio input using reset and motion-constraint statistics before frozen VLA inference.

Equations: estimate `q = D(q' - beta)` from calibration identities and execute `pi(o, q, l)`.

Representation: proprio affine calibration parameters.

Objective: state-domain calibration, not action loss.

Supervision: reset observations and controlled proprio shifts.

Inference: frozen VLA with canonicalized state.

Required data: exact-reset LIBERO observations.

Closest five papers: Domain Arithmetic, TTT-VLA, TT-VLA, OpenVLA-OFT proprio conditioning, SmolVLA.

Overlap matrix:

| Axis | KSC-VLA | Closest overlap? |
| --- | --- | --- |
| Core problem | proprio shift | partial |
| Representation | 8D state affine | partial |
| Supervision | calibration states | partial |
| Objective | state consistency | low |
| Inference | preprocessing only | distinct |
| Data source | reset calibration | distinct |
| Claim | proprio robustness | distinct |

Direct baseline: known inverse bias/scale.

Simple killer baseline: no state input or zero-bias correction.

Key ablation: no scale correction.

Implementation plan: patch state tensor in preprocessed batch and run controlled Stage A.

Prototype tasks: same two hard tasks with proprio bias.

Second-backbone path: OpenVLA-OFT has proprio input; patch its input if safe.

Second-condition path: combined visual/proprio shift.

Compute estimate: low.

Failure risk: too narrow and likely solved by known inverse affine.

## Candidate 3: CCFT-VLA

Name: `CCFT-VLA`, Calibration-Conditioned Frozen-Token VLA.

Hidden assumption: a calibration vector can steer a frozen VLA if inserted as task metadata, without weight updates.

Precise novelty: convert sensor calibration statistics into a textual/latent condition prepended to the instruction, not an action or image edit.

Equations: `l' = l || text(mu, sigma, shift_type)` and execute `pi(o', q', l')`.

Representation: calibration metadata tokens.

Objective: no training; deployment conditioning.

Supervision: calibration observations.

Inference: frozen VLA with calibration-conditioned instruction.

Required data: calibration observations and prompt template.

Closest five papers: TTT-VLA, human-assisted prompt steering, Domain Arithmetic, VLS, OpenVLA prompt tuning.

Overlap matrix:

| Axis | CCFT-VLA | Closest overlap? |
| --- | --- | --- |
| Core problem | deployment shift | partial |
| Representation | text metadata | high with prompt steering |
| Supervision | calibration stats | partial |
| Objective | none | low |
| Inference | prompt-only | high |
| Data source | calibration observations | distinct |
| Claim | frozen prompt steering | weak |

Direct baseline: original instruction under shift.

Simple killer baseline: hand-written "image is darker/brighter" instruction.

Key ablation: remove numeric calibration.

Implementation plan: modify task string during rollout.

Prototype tasks: visual shift tasks.

Second-backbone path: prompt both SmolVLA and OpenVLA-OFT.

Second-condition path: proprio shift is hard to express in text.

Compute estimate: low.

Failure risk: likely too close to prompt steering and may have no effect in SmolVLA.

## Selection

Selected method: `SCVC-VLA`.

Reason: SCVC changes the core problem to controlled deployment sensor shift, changes representation to calibrated sensor statistics, changes objective/supervision to calibration observation matching, and changes inference by preprocessing observations rather than editing actions. It is the most locally feasible non-CBFD pivot and has strong simple baselines that can kill it cleanly.
