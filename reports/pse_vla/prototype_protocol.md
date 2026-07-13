# PSE-VLA Prototype Protocol

Date: 2026-07-12 KST

## Modes

1. `synthetic`: verify transform math, duplicate-clean equivalence, action averaging, and privileged-field rejection.
2. `stage-a`: run held-out Stage A closed-loop evaluation.
3. `stage-b`: run paired Stage B if Stage A is not a permanent kill.

## Artifacts

- proposal: `reports/pse_vla/researcher_proposal.md`
- proposal hash: `reports/pse_vla/proposal_hash.txt`
- Stage A: `reports/pse_vla/stage_a_result.json`
- Stage B 40-paired result: `reports/pse_vla/stage_b_40_result.json`
- Stage B expanded result: `reports/pse_vla/stage_b_result.json`

## Validity

- Transform parameters are fixed before rollout.
- Stage A identities are disjoint from SCVC Stage B identities.
- If the 40-paired Stage B result is unresolved, the one allowed expansion uses the full contiguous exact-init range `20260721..20260760`; identities `20260721..20260740` overlap SCVC's evaluation identity range but have no PSE outcomes before expansion and are added only as the governance-permitted non-cherry-picked expansion.
- The policy is frozen.
- No training, success labels, teacher traces, reward, simulator state, object pose, or future observation is used at inference.
- Multiple transformed observations at one environment step must use a stateless first-action-chunk path.
