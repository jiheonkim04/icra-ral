# CBFD-VLA Researcher Proposal

Date: 2026-07-12 KST

Method: `CBFD-VLA`, Cross-Backbone Failure-Set Distillation for VLAs.

## Claim

Compact VLAs can fail on deployment states that are solvable by a stronger VLA backbone in the same simulator and embodiment. Instead of treating that stronger backbone only as a baseline, `CBFD-VLA` uses it as a bounded autonomous teacher to create successful state-action supervision for the compact backbone's failure set. The final policy is still a compact SmolVLA adapter; the teacher is absent at inference.

## Core Mechanism

Let `pi_S` be frozen SmolVLA and `pi_T` be Quantized OpenVLA-OFT INT4.

On predeclared train identities, execute both policies where possible and build:

`D_F = {(o_t, q_t, l, a^T_t, k, task, identity)}` from successful `pi_T` traces on tasks/identities where frozen SmolVLA is known to be weak or fails in acquisition.

`D_R = {(o_t, q_t, l, a^S_t)}` from frozen SmolVLA successful traces or official demonstration rows outside the failure-set identities.

Train a small SmolVLA adapter `theta` with:

`L(theta) = E_{D_F}[w(task,identity) ||a_theta(o,q,l) - a^T||_1] + lambda_R E_{D_R}[||a_theta(o,q,l) - a^S||_1] + lambda_D E_{D_F}[||a_theta(o,q,l) - a_S(o,q,l)||_2^2]`.

The failure weight `w` is fixed before held-out rollout. It is not tuned on Stage A outcomes.

## What Changes Relative To Epoch 2

- Core problem: transfer validated cross-backbone success evidence into a compact student.
- Data source: Quantized OpenVLA-OFT INT4 closed-loop successful traces, not only SmolVLA demonstrations or SmolVLA rollouts.
- Supervision: teacher action chunks on successful target-domain states plus retention rows.
- Objective: failure-weighted distillation with retention, not direct transition heads, semantic prefixes, or flow-noise priors.
- Inference: single student adapter call, no teacher, no router, no memory retrieval, no candidate scoring.

## Baselines

Required prototype policies:

1. `frozen_smolvla`: unmodified SmolVLA.
2. `direct_distill_proxy`: same teacher traces, no failure weighting, no retention replay.
3. `teacher_trace_memory`: simple nearest-neighbor or open-loop teacher-trace memory baseline that tests whether the method is only replaying train trajectories.
4. `cbfd_no_retention`: failure-weighted teacher imitation without retention.
5. `cbfd_full`: failure-weighted teacher imitation with retention.

Optional diagnostic:

- `quantized_openvla_oft_int4_teacher`: teacher upper-bound on train/acquisition identities only, clearly labeled as quantized teacher and not the student method.

## Prototype

Train/acquisition tasks:

- `libero_spatial/task_4`
- `libero_10/task_4`

Train identities:

- initial identities mapped to official LIBERO init states `0..4` for acquisition where previous OpenVLA evidence already showed success;
- additional predeclared identities may be used only if written before collection.

Stage A held-out identities:

- predeclared identities disjoint from teacher acquisition identities.

Stage A GO/non-GO follows `reports/current_research_governance.md`.

## Leakage Guards

- Teacher acquisition identities are disjoint from held-out Stage A identities.
- No Stage A outcome may change training weights, retention lambda, identity list, or selected adapter checkpoint.
- The teacher is not loaded during student Stage A evaluation except in an optional separately labeled upper-bound run.
- Any nearest-neighbor memory baseline may use only train traces.

## Expected Evidence

The method must show that `cbfd_full` beats frozen SmolVLA, the direct distillation proxy, the trace-memory killer baseline, and `cbfd_no_retention`. If direct distillation or trace memory matches it, the method is killed.

## Final-Paper Path If Prototype GO

The final paper cannot claim generic VLA improvement from SmolVLA-only evidence. After a prototype GO, the next steps are:

- larger SmolVLA confirmation with confidence intervals;
- stronger direct distillation baselines, including a faithful VLA-OPD-style local proxy;
- retention and failure-weight ablations;
- efficiency report;
- second condition based on held-out deployment shift or task family;
- a second-backbone comparison that is not circular teacher-student reuse, or a narrowed compact-student claim with another locally feasible compact backbone.
