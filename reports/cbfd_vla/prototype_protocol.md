# CBFD-VLA Prototype Protocol

Date: 2026-07-12 KST

## Purpose

Test whether heterogeneous teacher success traces can be distilled into a compact SmolVLA student in a way that beats generic teacher behavior cloning, train-trace memory, and no-retention ablation on held-out closed-loop LIBERO identities.

## Stages

1. `synthetic`: verify failure weighting, retention, and teacher-memory baselines on toy data.
2. `teacher-acquisition`: collect or load Quantized OpenVLA-OFT INT4 teacher traces for predeclared train identities.
3. `student-train`: train the direct distillation proxy, no-retention ablation, and full CBFD adapter.
4. `stage-a`: run `50` paired closed-loop held-out episodes.
5. `stage-b`: run `40` paired episodes per policy if required.

## Fixed Identity Split

Teacher acquisition:

- `20260711..20260715`.

Stage A:

- `20260716..20260720`.

Stage B initial:

- `20260721..20260740`.

Stage B expansion:

- `20260741..20260760`, only if preregistered expansion is required.

Identity mapping:

- `identity - 20260711` maps to official LIBERO initial states.

## Policies

- `frozen_smolvla`
- `direct_distill_proxy`
- `teacher_trace_memory`
- `cbfd_no_retention`
- `cbfd_full`

## Artifact Paths

- proposal: `reports/cbfd_vla/researcher_proposal.md`
- proposal hash: `reports/cbfd_vla/proposal_hash.txt`
- reviewer attack: `reports/cbfd_vla/reviewer_attack.md`
- preregistration: `reports/cbfd_vla/preregistration.md`
- synthetic result: `reports/cbfd_vla/synthetic_result.json`
- teacher acquisition result: `reports/cbfd_vla/teacher_acquisition_result.json`
- student training result: `reports/cbfd_vla/student_train_result.json`
- Stage A result: `reports/cbfd_vla/stage_a_result.json`
- Stage B result: `reports/cbfd_vla/stage_b_result.json`

## Resource Rules

- Do not keep SmolVLA and OpenVLA loaded together.
- Do not use CPU or disk offload.
- Do not download new checkpoints for this prototype.
- Stop after repeated identical CUDA OOM.
- Save partial results after every episode during closed-loop stages.

## Validity Rules

- Teacher acquisition identities must not overlap Stage A or Stage B identities.
- Teacher actions may supervise training only.
- No teacher calls during student evaluation.
- No checkpoint selection after held-out rollout results.
- `teacher_trace_memory` may use train traces only.
