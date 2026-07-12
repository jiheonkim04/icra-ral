# Epoch 3 Cycle 1 Candidate Generation

Date: 2026-07-12 KST

Decision: select exactly one Epoch 3 method after the related Epoch 2 kills in `reports/epoch_2_failure_synthesis.md`.

Epoch 3 must change at least two core dimensions relative to `PTC-VLA`, `SACF-VLA`, and `OCFN-VLA`. The selected method must not be another thin action-surface intervention, fixed flow-noise prior, semantic prefix, ranker, verifier, barrier, damping rule, or simple action-statistic baseline.

Recent primary sources checked for this selection:

- VLA-OPD, On-Policy VLA Distillation: https://arxiv.org/html/2603.26666v1
- TTT-VLA, latent prompt test-time optimization: https://arxiv.org/html/2606.03127v1
- Domain Arithmetic for one-shot VLA adaptation: https://arxiv.org/html/2607.00666v1
- RouterVLA, smoke-test-supervised heterogeneous VLA selection: https://arxiv.org/html/2606.27355v1
- Retrieve-then-Steer online success memory: https://arxiv.org/html/2605.10094v1
- DreamAvoid critical-phase test-time dreaming: https://arxiv.org/html/2605.11750v1
- Learning Action Priors for cross-embodiment robot manipulation: https://arxiv.org/html/2606.26095v1
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- SmolVLA: https://arxiv.org/abs/2506.01844
- Shallow-pi knowledge distillation for flow-based VLAs: https://arxiv.org/abs/2601.20262

## Candidate 1: CBFD-VLA

Name: `CBFD-VLA`, Cross-Backbone Failure-Set Distillation for VLAs.

Hidden assumption: frozen SmolVLA fails on some LIBERO states because its compact action expert lacks successful closed-loop state-action support, not because the task is intrinsically unsolvable or the official interface is broken. A locally validated stronger backbone can provide successful target-domain traces, and the useful part of that evidence is concentrated in states where SmolVLA fails.

Precise novelty: use a second VLA backbone as an autonomous simulator teacher only during training, then distill its successful traces into a compact SmolVLA adapter through a failure-set objective with retention replay. Unlike VLA-OPD, the teacher is a heterogeneous architecture and the supervision is not dense token-level guidance on the student's self-generated trajectory. Unlike RouterVLA or success memory, no teacher, router, scorer, memory retrieval, or candidate selector is used at inference.

Equations:

Let `pi_S` be frozen SmolVLA, `pi_T` be Quantized OpenVLA-OFT INT4, and `F` be train identities where `pi_S` failed or where the task is known to be SmolVLA-hard. Collect teacher success traces

`D_T = {(o_t, q_t, l, a^T_t, h, r=1) : tau_T is successful on identity i in F}`.

Collect retention rows

`D_R = {(o_t, q_t, l, a^S_t)}` from successful SmolVLA or official demonstration rows outside the failure set.

Train a small adapter `theta` initialized from `pi_S` with

`L_CBFD(theta) = E_{D_T}[w_F ||a_theta(o,q,l) - a^T||_1] + lambda_R E_{D_R}[||a_theta(o,q,l) - a^S||_1] + lambda_D E_{D_T}[||Delta_theta(o,q,l)||_2^2]`.

`w_F` is larger for identities/tasks where frozen SmolVLA failed and the teacher succeeded. The final policy is `pi_S+theta`; the teacher is not loaded at inference.

Representation: official two-camera RGB, official 8D proprio state, task language, and teacher action chunks saved as supervision. No success label is visible at inference.

Objective: failure-set teacher-action imitation plus retention regularization to keep non-target behavior close to frozen SmolVLA.

Supervision: Quantized OpenVLA-OFT INT4 closed-loop successful traces on train identities, plus frozen SmolVLA/demo retention rows.

Inference: one SmolVLA adapter policy call per control step. No OpenVLA call, no routing, no candidate ranking, no online memory.

Required data: existing SmolVLA checkpoint, official LIBERO dataset, existing OpenVLA-OFT INT4 checkpoint and WSL evaluation stack, new bounded teacher rollout traces on predeclared train identities.

Closest five papers and exact difference:

| Paper | Overlap | CBFD difference |
| --- | --- | --- |
| VLA-OPD | teacher distillation for VLA robustness | heterogeneous teacher-to-student trace distillation on failure sets; no dense token-level reverse-KL on student self-trajectories |
| Shallow-pi | distills VLA behavior | compression/latency distillation, not cross-backbone failure-set success transfer |
| Exp2VLA | expert-to-VLA distillation | domain is UAV navigation; no targeted cross-backbone failure-set SmolVLA repair |
| OpenVLA-OFT | strong continuous-action VLA recipe | used only as a quantized teacher data source, not claimed as our method |
| Learning Action Priors | action module prior training | broad motion prior before VLA alignment, not closed-loop teacher-success failure-set distillation |

Overlap matrix:

| Axis | CBFD-VLA | Closest overlap? |
| --- | --- | --- |
| Core problem | transfer successful closed-loop behavior from strong VLA to compact VLA failure sets | partial with VLA-OPD |
| Representation | official observations plus heterogeneous teacher action chunks | partial with distillation methods |
| Supervision | successful teacher traces and retention replay | partial |
| Objective | failure-weighted imitation plus retention regularization | partial |
| Inference | single student adapter call, no teacher | distinct from memory/routing/test-time steering |
| Data source | locally validated Quantized OpenVLA-OFT INT4 rollouts | distinct from Epoch 2 |
| Claim | compact-backbone recovery from cross-backbone success evidence | distinct from action-surface prior calibration |

Direct baseline: local VLA-OPD proxy, implemented as teacher-trace distillation without failure weighting and without retention replay.

Simple reviewer-killer baseline: replay or nearest-neighbor teacher-trace action memory on matched train traces, plus a demo-only adapter if training is cheap.

Key ablation: CBFD without failure weighting and CBFD without retention regularization.

Implementation plan: first write trace schema and offline synthetic tests; collect bounded OpenVLA-OFT INT4 teacher traces on two hard tasks and train identities; train a rank-4 or smaller adapter/head-only student with teacher labels and retention rows; run Stage A on held-out identities with frozen SmolVLA, direct distillation proxy, simple replay/memory baseline, key ablation, and full CBFD.

Prototype tasks: `libero_spatial/task_4` and `libero_10/task_4`, because they are validated SmolVLA-hard and validated OpenVLA-solvable under exact initial states.

Second-backbone path: if SmolVLA student reaches GO, reverse the direction is not a valid second-backbone comparison. The scale-up path must instead test `Quantized OpenVLA-OFT INT4` against `Quantized OpenVLA-OFT INT4 + CBFD-style compact adapter` using a different teacher or held-out OpenVLA self-distillation split, or narrow the claim to compact-student distillation and add a second compact VLA if locally available.

Second-condition path: controlled visual/proprio deployment shift or held-out LIBERO task family where teacher succeeds and SmolVLA has failures, with predeclared train/test identity separation.

Compute estimate: moderate. OpenVLA teacher acquisition is the expensive part but the checkpoint and INT4 path are already local. Adapter training should stay within the verified SmolVLA LoRA budget.

Failure risk: high novelty risk due to VLA-OPD and generic distillation. High final-paper risk because the teacher is stronger than the student and reviewers may call this engineering transfer unless failure-set objective and retention evidence matter.

## Candidate 2: SCVC-VLA

Name: `SCVC-VLA`, Shift-Calibrated Visual/State Canonicalization for VLAs.

Hidden assumption: deployment failures under camera or proprio shifts come from input-domain mismatch that can be corrected before the policy, rather than from action-generation weakness.

Precise novelty: learn a lightweight visual/proprio canonicalizer from one calibration trace and a self-supervised consistency loss, then feed canonicalized observations to a frozen VLA. Unlike GCAP, this is not hold-last or edge repair under occlusion; it is a deployment-shift canonicalization interface with explicit train/test domain separation.

Equations:

Let `c_phi` map shifted observations to canonical observations and states. Train

`L_SCVC(phi) = ||g(c_phi(o'_t, q'_t)) - q_t||_1 + alpha ||stats(c_phi(o'_t)) - stats(o_t)||_2^2 + beta TV(c_phi(o'_t))`

on paired or one-shot calibration samples, then execute `pi(a|c_phi(o',q'), l)`.

Representation: camera images, proprio state, calibration statistics, and a small canonicalization parameter vector.

Objective: self-supervised state grounding and canonical-domain matching, not action imitation.

Supervision: controlled deployment-shift calibration traces in simulation.

Inference: frozen VLA after canonicalization; no action adapter.

Required data: official LIBERO rollouts under controlled camera/proprio perturbations.

Closest five papers and exact difference:

| Paper | Overlap | SCVC difference |
| --- | --- | --- |
| TTT-VLA | test-time adaptation with latent prompt and proxy state grounding | input canonicalizer, not latent prompt inside policy |
| Domain Arithmetic | one-shot adaptation under environmental shift | no weight arithmetic; canonicalizes inputs before policy |
| VISTA | physics-validated visual adaptation | local calibration for VLA control, not UMI physics validation |
| PAD/test-time training | proxy adaptation | VLA-specific frozen policy canonicalization |
| GCAP-VLA | perception repair | not occlusion inpainting, hold-last, or edge restoration |

Overlap matrix:

| Axis | SCVC-VLA | Closest overlap? |
| --- | --- | --- |
| Core problem | deployment input shift | high with TTT/domain adaptation |
| Representation | canonicalizer parameters | partial |
| Supervision | calibration/state consistency | partial |
| Objective | canonical-domain matching | partial |
| Inference | preprocessing only | distinct from action adapters |
| Data source | shifted simulator traces | distinct from Epoch 2 |
| Claim | robustness to controlled input shifts | distinct from clean-task success |

Direct baseline: TTT-VLA-style latent prompt proxy is not locally available; use one-shot finetune/canonicalizer proxy and report as local proxy.

Simple reviewer-killer baseline: histogram matching, center crop/color normalization, and identity/no-canonicalizer.

Key ablation: no proprio canonicalization or no visual canonicalization.

Implementation plan: implement image/state perturbation wrappers, canonicalizer, offline calibration, and Stage A under predeclared visual/proprio shifts.

Prototype tasks: hard and easy paired tasks under controlled visual brightness/color and proprio bias shifts.

Second-backbone path: apply the same canonicalizer before Quantized OpenVLA-OFT INT4 if the preprocessing stack can be safely patched.

Second-condition path: different perturbation family, such as camera crop or proprio scale.

Compute estimate: low to moderate.

Failure risk: high because simple normalization may match it; also risks being perceived as domain adaptation rather than a VLA method.

## Candidate 3: BSTA-VLA

Name: `BSTA-VLA`, Boundary-State Transition Awareness for VLAs.

Hidden assumption: VLA failures are concentrated near a small number of physical boundary states, and learning a boundary-state transition representation can improve recovery without global action retuning.

Precise novelty: learn a boundary-state representation from short simulator continuations of success/failure states, then use it to decide when to execute a predeclared recovery action template or policy continuation. Unlike DreamAvoid, this would avoid learned video dreaming and use LIBERO state transition evidence directly.

Equations:

For boundary states `b_t`, learn `z_t = f_phi(o_t, q_t, l)` and a transition-risk model

`p_phi(fail | z_t, a_t, h)`.

A boundary trigger uses `p_phi` to select between base continuation and a fixed recovery continuation.

Representation: boundary state embeddings and short-horizon transition labels.

Objective: contrastive success/failure transition separation plus calibrated boundary risk.

Supervision: simulator short-horizon rollouts from boundary states.

Inference: trigger plus fixed recovery. No full policy fine-tuning.

Required data: success/failure rollouts and boundary resets.

Closest five papers and exact difference:

| Paper | Overlap | BSTA difference |
| --- | --- | --- |
| DreamAvoid | critical-phase test-time dreaming | no generated visual dreaming; uses direct transition labels |
| TT-VLA | test-time reinforcement adaptation | no online RL parameter update |
| Online Success Memory | uses successful segments | uses boundary transition labels, not retrieval steering |
| ECHO-VLA | effect mediator/candidate credit | does not rank sampled candidate actions |
| Progress/value heads | transition-risk estimation | must avoid becoming a generic value head |

Overlap matrix:

| Axis | BSTA-VLA | Closest overlap? |
| --- | --- | --- |
| Core problem | boundary-state failure awareness | high with DreamAvoid |
| Representation | transition-risk boundary embedding | partial |
| Supervision | short-horizon failure labels | partial |
| Objective | failure transition separation | partial |
| Inference | trigger/recovery | high risk of verifier/ranker equivalence |
| Data source | boundary rollouts | distinct from Epoch 2 |
| Claim | critical-state robustness | partial |

Direct baseline: DreamAvoid local proxy would be difficult; use trigger-only and short-horizon risk proxy.

Simple reviewer-killer baseline: always execute fixed recovery in the same phase, or no trigger.

Key ablation: representation without transition labels.

Implementation plan: mine boundary states, generate short continuations, train transition-risk model, Stage A on boundary-heavy tasks.

Prototype tasks: tasks with known contact/placement boundary failures.

Second-backbone path: same trigger around OpenVLA-OFT INT4, if the boundary failure exists.

Second-condition path: controlled action disturbance near boundary states.

Compute estimate: moderate.

Failure risk: highest. It is close to DreamAvoid, generic value/progress heads, and verifier-like intervention, which active governance discourages.

## Selection

Selected method: `CBFD-VLA`.

Reason: `CBFD-VLA` changes at least four core dimensions relative to Epoch 2: the data source becomes a heterogeneous successful teacher backbone rather than the same SmolVLA evidence stream; the supervision becomes cross-backbone successful closed-loop traces; the objective becomes failure-set distillation with retention rather than action-prefix/noise-prior intervention; and the claim becomes compact-student recovery from validated cross-backbone success evidence. It is also the most locally actionable because Quantized OpenVLA-OFT INT4 is already validated on the exact hard tasks.

Selection caveat: Reviewer B must attack VLA-OPD and generic distillation overlap. Implementation may proceed only if the proposal survives as not near-exact duplication and not mathematically equivalent to a simple teacher-trace BC baseline.
