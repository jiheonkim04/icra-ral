# CBFD-VLA Reviewer B Attack

Date: 2026-07-12 KST

Reviewed fixed proposal hash: `D355F0FC8C728320D448E572E3CB3D7F8D823EAE7C8C3E91078D1376CEE526E2`

## Closest Current Papers

1. VLA-OPD: https://arxiv.org/html/2603.26666v1
2. Shallow-pi: https://arxiv.org/abs/2601.20262
3. OpenVLA-OFT: https://arxiv.org/abs/2502.19645
4. Retrieve-then-Steer online success memory: https://arxiv.org/html/2605.10094v1
5. RouterVLA: https://arxiv.org/html/2606.27355v1
6. DreamAvoid: https://arxiv.org/html/2605.11750v1
7. Learning Action Priors: https://arxiv.org/html/2606.26095v1

## Main Novelty Attack

The dangerous overlap is VLA-OPD. VLA-OPD also uses an expert teacher to give dense supervision to improve VLA closed-loop robustness. It explicitly frames itself as bridging offline SFT and online RL by using teacher guidance on policy-induced states.

`CBFD-VLA` is not a near-exact duplicate only if the paper claim remains narrower:

- heterogeneous teacher-to-compact-student transfer;
- failure-set concentration rather than generic on-policy distillation;
- local closed-loop teacher success traces rather than token-level dense reverse-KL;
- retention replay as a necessary component;
- no teacher or router at inference.

If the implementation collapses to "collect OpenVLA actions and behavior-clone SmolVLA," the contribution is too ordinary.

## Simplest Equivalent Method

The simplest equivalent method is not CBFD. It is teacher-trace behavior cloning:

`L(theta) = E_{D_T}[||a_theta - a_T||_1]`.

If this direct distillation proxy matches CBFD, the failure weighting and retention story contributes nothing.

The next simplest method is nearest-neighbor teacher-trace memory. If replaying or retrieving train teacher chunks matches CBFD on held-out resets, the result is not a learned cross-backbone transfer method.

## Simple Killer Baseline

Required killer:

- `teacher_trace_memory`: nearest train trace by task and early proprio distance; execute the corresponding teacher action chunk or phase-aligned mean chunk.

This is deliberately crude. If it works, the improvement is trajectory memorization or reset similarity, not CBFD.

## Leakage Risks

- OpenVLA teacher acquisition identities must be disjoint from student held-out identities.
- If Stage A identities reuse the exact teacher reset states, the result is invalid.
- If the teacher is called during student evaluation, the result becomes routing or online teacher control.
- If failure weighting is computed using held-out Stage A outcomes, the result is cherry-picked.
- If an adapter checkpoint is selected after held-out rollout, the result is invalid.

## Resource Risks

OpenVLA-OFT INT4 is locally validated, but loading teacher and student together is not allowed. Teacher trace acquisition and student training/evaluation must be separated so multiple large VLAs are not resident simultaneously.

If OpenVLA teacher acquisition beyond the already verified exact identities requires new large downloads, stop that path. Current plan should reuse the local checkpoint only.

## Required Prototype Baselines

1. `frozen_smolvla`
2. `direct_distill_proxy`
3. `teacher_trace_memory`
4. `cbfd_no_retention`
5. `cbfd_full`

The optional `quantized_openvla_oft_int4_teacher` may be reported as a ceiling but cannot replace a student baseline.

## Pre-Implementation Decision

Decision: `IMPLEMENTATION_ALLOWED_WITH_STRICT_BASELINES`

Reason: the proposal is close to VLA-OPD and generic distillation, but it is not a near-exact duplicate across problem, representation, supervision, objective, policy component, inference, data, and claim. It is not mathematically identical to the direct distillation baseline until measured, and the required killer baselines can decide the issue cheaply.

Reviewer B condition: the prototype must kill CBFD if `direct_distill_proxy`, `teacher_trace_memory`, or `cbfd_no_retention` matches or beats `cbfd_full`.
