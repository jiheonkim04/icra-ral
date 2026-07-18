# CVLR-XVLA v1 Archive Decision

- Archive decision: `CVLR_XVLA_V1_ARCHIVED_NOT_STAGE_A_READY`
- Frozen protocol decision: `CVLR_XVLA_STAGE0_DESIGN_FAILURE`
- False-negative safeguard: `ROBUST_EMPIRICAL_DESIGN_FAILURE` for Stage A safety eligibility of this exact formulation only
- Stage A authorized/launched: `false / false`

CVLR v1 learned a genuine offline mechanism. Validation wrist-latent MSE was `0.44344`, versus `0.99419` for zero-fill and `1.55907` for deterministic AWF, and it beat both controls on all three tasks. Training completed `96/96` steps with finite gradients, changed weights, exact checkpoint reload, and a bit-exact clean X-VLA bypass.

The direct insertion was nevertheless outside every relevant frozen safety envelope. All `9/9` dropout rows had raw-gripper violations and discrete flips (`42` flips total), `8/9` violated rotation safety, and `5/9` violated translation safety. Maximum translation RMS was `0.03582 > 0.02`, rotation RMS `3.15253 > 0.05`, and raw-gripper delta `0.97756 > 0.1`.

The strongest fair reading is that departure from a known-failed dropout Base could conceivably be useful; Stage 0 contains no closed-loop efficacy evidence. The frozen safety decision still bars Stage A and cannot be repaired or relaxed post hoc. Thus the exact uncalibrated direct-token-insertion v1 is archived, without claiming that cross-view latent reconstruction as a family is impossible.
