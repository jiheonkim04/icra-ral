# SCVC-VLA Prototype Protocol

Date: 2026-07-12 KST

## Modes

1. `synthetic`: verify that SCVC canonicalization reduces shifted-image error.
2. `calibration`: collect clean calibration stats on train identities.
3. `stage-a`: run shifted held-out Stage A.

## Artifacts

- proposal: `reports/scvc_vla/researcher_proposal.md`
- proposal hash: `reports/scvc_vla/proposal_hash.txt`
- calibration: `reports/scvc_vla/calibration_result.json`
- Stage A: `reports/scvc_vla/stage_a_result.json`

## Validity

- Calibration identities and Stage A identities are disjoint.
- Sensor shift parameters are fixed before Stage A.
- Canonicalization uses only image tensor statistics and calibration stats.
- No success, reward, simulator state, object pose, or future observation is used.
