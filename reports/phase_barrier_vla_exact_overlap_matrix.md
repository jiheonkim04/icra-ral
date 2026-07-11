# Exact Overlap Matrix For Prior Literature-Only Kills

Date: 2026-07-11 KST

Legend: `same`, `partial`, `different`, `unavailable`.

## Action Conditioning Route

| Closest paper | Problem | Inputs | Representation | Supervision | Objective | Modified component | Inference intervention | Claim | Exact duplicate? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAC-VLA | partial | partial | partial | partial | different | partial | different | partial | no |
| ActionMap | partial | partial | partial | different | different | partial | different | partial | no |
| AEM/LAWM/LARA | partial | partial | partial | partial | partial | different | different | partial | no |

Result: prior kill was too broad, but local ECHO and local ActionMap evidence still make this route lower priority for immediate implementation.

## Censored Correction Route

| Closest paper | Problem | Inputs | Representation | Supervision | Objective | Modified component | Inference intervention | Claim | Exact duplicate? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TORL-VLA | partial | different | partial | different | partial | different | different | partial | no |
| SDP | partial | partial | different | partial | partial | different | partial | partial | no |
| VLA-Corrector | partial | partial | different | different | different | different | partial | partial | no |

Result: not an exact duplicate, but the strongest version requires intervention or correction data not available locally. Kept as second-cycle candidate if PhaseBarrier fails.

## Contact Barrier Route

| Closest paper | Problem | Inputs | Representation | Supervision | Objective | Modified component | Inference intervention | Claim | Exact duplicate? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VeriSpace | partial | partial | different | partial | different | different | different | partial | no |
| Pre-VLA | partial | partial | different | partial | different | different | partial | partial | no |
| VLA-Corrector | partial | partial | different | different | different | different | different | partial | no |
| SEAM/AAC/Legato | partial | partial | different | different | different | partial | partial | partial | no |

Result: a technically distinct survivor exists: phase-conditioned feasibility-field action projection. It changes the physical/control representation and the action-generation distribution without candidate ranking or generic correction.
