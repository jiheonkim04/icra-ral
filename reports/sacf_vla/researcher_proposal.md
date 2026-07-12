# SACF-VLA Researcher Proposal

Date: 2026-07-12 KST

Method: `SACF-VLA`, Same-scene Action Counterfactual Factorization VLA

Governance: `reports/current_research_governance.md`

Researcher A freezes this proposal before Reviewer B begins.

## Claim

VLA language failures are not only a matter of missing better instruction text. In same-scene manipulation families, actions can be decomposed into a shared phase-local manipulation scaffold and a semantic action factor that changes source, object, or destination. A small factorized semantic-prefix generator trained from existing task-family demonstrations can steer a frozen VLA into the correct semantic branch before handing control back to the frozen policy.

## Mechanism

SACF trains two lightweight heads:

- `f_shared(s, p, family)` predicts the phase-local action scaffold common to a task family.
- `f_sem(s, p, family, z_l)` predicts the instruction-semantic action component.

The prefix action is:

`a = clip(f_shared(s, p, family) + f_sem(s, p, family, z_l), -1, 1)`.

The method runs this action generator for a fixed preregistered prefix fraction, then hands off to frozen SmolVLA. The fixed handoff is part of the method and is not tuned from test outcomes.

## Training Signal

Training uses official local LIBERO HDF5 demonstrations from standard task families only:

- `libero_spatial` for same object/destination but different source relation;
- `libero_object` for same destination but different target object;
- optionally `libero_goal` if the first two families do not provide enough rows.

For matched family/phase pairs, SACF minimizes:

- reconstruction MSE to demonstration actions;
- counterfactual semantic difference loss aligning semantic-component differences with action differences;
- shared-component invariance loss within a family/phase.

No simulator success labels or object state are used at inference. HDF5 object states may not be used as default inference features.

## Baselines

Stage A compares:

1. `frozen_smolvla`
2. `task_phase_mean_prefix`
3. `plain_bc_prefix`
4. `cag_null_guidance`
5. `sacf_full`

The key ablation is `plain_bc_prefix`, trained on the same rows but without the counterfactual factorization losses/components.

The simple reviewer-killer baseline is `task_phase_mean_prefix`, using the same fixed handoff.

The closest direct-prior proxy is `cag_null_guidance`, implemented by querying frozen SmolVLA under full task text and null task text, then applying a fixed guidance scale chosen before Stage A.

## Prototype Tasks

Primary Stage A tasks:

- `libero_spatial/task_4`
- `libero_object/task_4`

Reset identities:

- `20260713`
- `20260714`
- `20260715`
- `20260716`
- `20260717`

Stage A total:

- 5 variants
- 2 tasks
- 5 identities
- 50 closed-loop episodes

## GO And Kill Rules

Stage A is directional and may permanently kill only under `reports/current_research_governance.md`.

Permanent kill in Stage A if:

- implementation/data mechanism is invalid;
- `sacf_full` is at least 30 absolute task-balanced points below the strongest baseline or the `plain_bc_prefix` ablation;
- `sacf_full` has `0 / 10` while a paired baseline has at least `4 / 10`;
- exact trivial equivalence to `plain_bc_prefix`, `task_phase_mean_prefix`, or `cag_null_guidance` is demonstrated.

Advance to Stage B if:

- `sacf_full` beats frozen and the key ablation, or
- the result is noisy/tied/small-negative but mechanism activation is valid and no Stage A permanent kill holds.

Stage B would use at least 40 paired episodes per key policy.

## Resource Plan

- no downloads;
- no full VLA fine-tuning;
- no OpenVLA-OFT work before prototype GO;
- tiny CPU/GPU PyTorch heads only;
- official WSL SmolVLA/LIBERO rollout stack for Stage A.

## Expected Failure Modes

- plain BC prefix explains any gain;
- prefix disrupts the frozen policy and reduces success;
- phase/task means are already as good as the factorized head;
- frozen SmolVLA is insensitive to null-language CAG guidance, making that baseline weak but also exposing limited language action headroom;
- demonstration action conventions differ enough from the official rollout action space to invalidate the prefix.
