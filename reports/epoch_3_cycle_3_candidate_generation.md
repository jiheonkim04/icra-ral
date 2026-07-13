# Epoch 3 Cycle 3 Candidate Generation

Date: 2026-07-12 KST

Decision: select exactly one method after `CBFD-VLA` and `SCVC-VLA` were killed.

## Candidate 1: PSE-VLA

Name: `PSE-VLA`, Photometric Sensor-Ensemble VLA.

Hidden assumption: VLA action predictions may be sensitive to nuisance photometric style. Averaging actions over a fixed sensor-style bank may stabilize decisions without training, teacher traces, action correction, or success labels.

Novelty: sensor-space test-time augmentation for frozen VLA control, with the action produced by the same frozen policy under multiple observation styles and averaged in 7D environment-action space. Unlike SCVC, PSE does not try to restore images to clean statistics. Unlike verification methods, it does not score or choose candidates.

Equations:

`a_t = (1 / K) sum_k pi(T_k(o_t), q_t, l)`.

Representation: transformed observation bank.

Objective/supervision: none at inference; fixed predeclared transform bank.

Inference: multiple frozen SmolVLA action-chunk predictions, averaged.

Required data: no training data beyond official model/assets.

Closest papers: test-time perturbation learning, scaling verification, TTT-VLA, robustness analyses of VLA perturbations, Domain Arithmetic.

Direct baseline: best single photometric transform.

Simple killer baseline: clean duplicate ensemble, which tests whether averaging mechanics alone matters.

Key ablation: average only clean duplicate predictions.

Prototype tasks: `libero_spatial/task_4` and `libero_10/task_4`, held-out identities not used by SCVC Stage B.

Second-backbone path: same transform bank around Quantized OpenVLA-OFT INT4 if its evaluation wrapper can safely call action prediction without stateful queue corruption.

Second-condition path: controlled lighting/background shift.

Risk: high. Best single transform may dominate or action averaging may blur decisive gripper behavior.

## Candidate 2: KSC-VLA

Name: `KSC-VLA`, Kinematic State Canonicalization for VLAs.

Hidden assumption: proprio input calibration can matter independently of visual grounding.

Novelty: reset-calibrated 8D proprio preprocessing before frozen VLA inference.

Equations: `q = D(q' - beta)`, `a = pi(o, q, l)`.

Representation: proprio affine calibration.

Objective/supervision: calibration-state consistency.

Inference: state preprocessing only.

Required data: exact reset observations.

Closest papers: TTT-VLA, Domain Arithmetic, OpenVLA-OFT proprio conditioning, TT-VLA, VLA robustness audits.

Direct baseline: known inverse affine.

Simple killer baseline: no-state or bias-only correction.

Key ablation: no scale correction.

Prototype tasks: controlled proprio bias on the two hard tasks.

Second-backbone path: OpenVLA-OFT proprio input patch.

Second-condition path: combined visual/proprio shift.

Risk: likely too narrow and killed by known inverse affine.

## Candidate 3: BSTA-VLA

Name: `BSTA-VLA`, Boundary-State Transition Awareness.

Hidden assumption: failures are concentrated near boundary states, and a transition representation can identify recovery moments.

Novelty: boundary-state short-horizon transition labels rather than action-surface correction.

Equations: learn `p(fail | f(o,q,l), h)` from short continuations.

Representation: boundary embeddings.

Objective/supervision: short-horizon success/failure transition labels.

Inference: trigger fixed recovery or base continuation.

Required data: boundary rollouts.

Closest papers: DreamAvoid, online success memory, ECHO-VLA, test-time verification, progress heads.

Direct baseline: fixed recovery trigger.

Simple killer baseline: always recover at same phase.

Key ablation: no transition labels.

Prototype tasks: contact-heavy hard tasks.

Second-backbone path: only if the boundary failure exists for OpenVLA-OFT.

Second-condition path: controlled action disturbance near boundary.

Risk: too close to verifier/value-head methods and active governance discourages that route.

## Selection

Selected method: `PSE-VLA`.

Reason: PSE is the cleanest non-CBFD, non-SCVC pivot. It changes inference to sensor-space ensemble averaging, avoids teacher data and canonicalization, and can be killed cheaply by best-single-transform and duplicate-clean baselines.
