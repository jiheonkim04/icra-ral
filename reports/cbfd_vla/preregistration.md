# CBFD-VLA Preregistration

Date: 2026-07-12 KST

Decision: `IMPLEMENTATION_PREREGISTERED`

## Fixed Method

Full method: `cbfd_full`.

Baselines and ablations:

1. `frozen_smolvla`
2. `direct_distill_proxy`
3. `teacher_trace_memory`
4. `cbfd_no_retention`
5. `cbfd_full`

Optional diagnostic:

- `quantized_openvla_oft_int4_teacher` on acquisition identities only.

## Train/Acquisition

Tasks:

- `libero_spatial/task_4`
- `libero_10/task_4`

Teacher acquisition identities:

- `20260711..20260715`, mapped to official LIBERO init states `0..4`.

These are the identities where prior repository evidence already showed Quantized OpenVLA-OFT INT4 `5 / 5` and frozen SmolVLA `1 / 5` per task on the two hard tasks.

Student train rows:

- teacher action rows from successful Quantized OpenVLA-OFT INT4 traces;
- retention rows from official demonstration data or frozen SmolVLA traces, disjoint from held-out Stage A identities.

No Stage A result may alter train identities, loss weights, retention source, checkpoint selection, or baseline list.

## Stage A

Held-out identities:

- `20260716..20260720`, mapped to official LIBERO init states `5..9`.

Policies:

- `frozen_smolvla`
- `direct_distill_proxy`
- `teacher_trace_memory`
- `cbfd_no_retention`
- `cbfd_full`

Episode count:

- 2 tasks x 5 identities x 5 policies = `50` paired closed-loop episodes.

Primary metric:

- task-balanced closed-loop success.

Mechanism metrics:

- mean action delta `cbfd_full - frozen_smolvla`;
- mean action delta `cbfd_full - direct_distill_proxy`;
- retention loss on held-out retention rows;
- train/held-out identity disjointness.

## Stage A Decision

Permanent Stage A kill only when active governance permits:

- implementation/data mechanism invalid;
- full method at least `0.30` absolute below strongest baseline or ablation;
- full method `0 / 10` while a paired baseline has at least `4 / 10`;
- oracle or upper bound proves no usable headroom;
- exact trivial equivalence is demonstrated.

Otherwise:

- positive, tie, small negative, one- or two-episode difference, or noisy cross-task result goes to Stage B.

## Stage B

If Stage A does not permanently kill, run at least `40` paired episodes per key policy with task-balanced held-out identities.

If Stage B remains unresolved, one expansion to at most `80` paired episodes per policy is allowed under current governance.

## GO Criteria

Prototype GO requires:

- `cbfd_full` beats strongest baseline and ablation;
- absolute gain at least `0.10` at prototype scale, or consistently positive paired evidence with meaningful failure-rate reduction;
- mechanism active;
- no teacher at inference;
- no held-out leakage.

## Kill Criteria

Kill CBFD when:

- `direct_distill_proxy` matches or beats `cbfd_full`;
- `teacher_trace_memory` matches or beats `cbfd_full`;
- `cbfd_no_retention` matches or beats `cbfd_full`;
- full is clearly worse;
- upper confidence bound after valid Stage B excludes useful improvement;
- teacher acquisition cannot be performed without unavailable resources;
- inference requires OpenVLA or any other teacher.
