# Post-R2P Archive LIBERO-Spatial Task5 Residual Confirmation

Decision: `LIBERO_SPATIAL_TASK5_REPEATED_SHARED_RESIDUAL_CONFIRMED_CANDIDATE_GENERATION_AUTHORIZED_NO_TRAINING`

The `libero_spatial/task5` residual now repeats across three independent identities: `20260727`, `20260730`, and `20260733`. Each has clean Base, X-VLA first-prior, and OpenVLA-OFT INT4 second-prior failure, with task-level expert headroom positive and same-reset expert headroom unavailable.

| Identity | X-VLA | Base | Expert headroom | OpenVLA-OFT INT4 | Decision |
| ---: | --- | --- | --- | --- | --- |
| 20260727 | failure | failure | task-level positive | failure | known shared residual |
| 20260730 | failure | failure | task-level positive | failure | shared residual |
| 20260731 | success | not needed | not needed | not needed | not a residual |
| 20260732 | success | not needed | not needed | not needed | not a residual |
| 20260733 | failure | failure | task-level positive | failure | shared residual |

Authorization:

- Candidate generation is now authorized, bounded to at most two new candidates.
- Training is not authorized.
- Ours rollout is not authorized.
- R2P-XVLA remains archived and must not be reopened or retuned.
- Closed methods remain closed: R2R-OFT, BR-XVLA, MPR-XVLA, PRC-XVLA, CR-LightVLA, ATCD, MCI-VLA, CSPR-VLA, and R2P-XVLA.

Key artifacts:

- X-VLA screen: `runs/xvla_prior/repeated_residual_spatial_task5_id20260731_33_xvla_prior_20260718T1115KST`
- Base gate for `20260733`: `runs/xvla_prior/diagnostic_smolvla_base_libero_spatial_task5_id20260733_officialenv_20260718T1119KST`
- Headroom gate for `20260733`: `runs/xvla_prior/diagnostic_libero_spatial_task5_expert_headroom_20260733_20260718T1121KST`
- OpenVLA gate for `20260733`: `runs/openvla_oft_int4/diagnostic_spatial_task5_openvla_int4_20260733_openvlaenv_20260718T1122KST`

Comparator-role interpretation: Base establishes repeated backbone failure, X-VLA establishes repeated closest-prior failure, OpenVLA-OFT INT4 establishes repeated second-prior failure, and expert replay establishes task-level recoverability. This supports a narrow residual-specific ideation step only; it is not a paper-candidate result.

Next action: generate at most two new candidates around the exact repeated `libero_spatial/task5` residual, with no training or Ours rollout until a new candidate-specific frozen preregistration/gate exists.
