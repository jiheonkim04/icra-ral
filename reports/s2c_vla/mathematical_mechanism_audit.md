# S2C-VLA Mathematical Mechanism Audit

Date: 2026-07-16 KST

Decision: `S2C_MATHEMATICAL_AUDIT_PREREGISTERED`

Proposal: `reports/s2c_vla/researcher_proposal.md`

Proposal SHA-256:
`399A3960F9FF9AFA8EDA7C3F743A95C3FD4DC711644C2398630F1E68486DC5B3`

Reviewer attack: `reports/s2c_vla/reviewer_attack.md`

Researcher rebuttal: `reports/s2c_vla/researcher_rebuttal.md`

No S2C implementation, training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened before this audit.

## Fixed Variables And Shapes

Action dimension: `D = 7`.

Chunk horizon: `H = 50`.

Replanning stride for Stage 0 adjacent chunks: `s = 10`.

Overlap edit length: `K = 10`.

Action groups:

- translation dimensions: `0:3`;
- rotation dimensions: `3:6`;
- gripper dimension: `6`.

For row `i`:

- current frozen Base chunk: `B_i in R^{H x D}`;
- previous committed policy chunk: `P_i in R^{H x D}`;
- previous overlap tail: `T_i = P_i[s:s+K] in R^{K x D}`;
- current Base overlap head: `O_i = B_i[:K] in R^{K x D}`;
- optional demonstration overlap label: `Y_i in R^{K x D}`;
- deployment-observable features: `X_i`, built only from current SmolVLA
  features, current proprioception, current Base chunk, task/phase metadata on
  development partitions, and the previous committed tail;
- no-previous-tail indicator: `r_i in {0,1}`, where `r_i = 0` means S2C must
  output exact Base.

No future observation, object pose, reward, success flag, done flag, expert
future action at inference, or confirmatory identity may enter `X_i`.

## Zones

The current chunk is partitioned as:

- editable overlap zone: `E = {0, ..., K-1}`;
- future passthrough zone: `F = {K, ..., H-1}`.

For every `j in F`, S2C output is frozen:

`A_i[j, d] = B_i[j, d]` for all action dimensions `d`.

Within `E`, unselected cells are also exact Base passthrough.

## Bridge Target

For each row and action dimension, define a deterministic tail-anchored bridge
target `C_i in R^{K x D}` as the minimizer:

`C_i[:, d] = argmin_c L_bridge(c; O_i[:, d], T_i[:, d])`

where:

`L_bridge = ||c - O||_2^2 + lambda_tail ||c - T||_2^2 + lambda_d1 ||D1 c - D1 T||_2^2 + lambda_d2 ||D2 c - D2 T||_2^2`

with fixed coefficients:

- `lambda_tail = 1.0`;
- `lambda_d1 = 0.25`;
- `lambda_d2 = 0.10`.

`D1` is the first-difference matrix over the `K` overlap steps.

`D2` is the second-difference matrix over the `K` overlap steps.

The bridge target is deterministic and nonlearned in Stage 0. It is not by
itself S2C; deterministic bridge behavior belongs to the ChunkFlow proxy or
ablation unless the learned edit mask selects a distinct bounded intervention.

## Edit Mask And Identity Preservation

The learned effective edit mask is:

`G_theta(X_i) = gamma_theta * sigmoid(Z_theta(X_i)) in [0, 1]^{K x D}`

where:

- `Z_theta` is a lightweight adapter/head output with shape `K x D`;
- `gamma_theta in [0, 1]` is a scalar or per-group gate initialized to `0`;
- `gamma_theta = 0` at initialization and after identity reload smoke, so the
  initialized policy is exact Base.

The raw edit is:

`R_i = clip_group(C_i - O_i, -c, c) in R^{K x D}`.

Group caps in normalized action units:

- translation cap: `c_trans = 0.02`;
- rotation cap: `c_rot = 0.05`;
- gripper cap: `c_grip = 0.25`.

For gripper event cells, the effective gripper cap is `0` unless a later
preregistered Stage 0 diagnostic explicitly proves that preserving the event
requires a bounded same-sign edit. The default is Base gripper passthrough.

The output chunk is:

`A_i[:K] = B_i[:K] + r_i * G_theta(X_i) * R_i`

`A_i[K:] = B_i[K:]`

All multiplication is elementwise. No gradient flows into `B_i`, `P_i`,
`T_i`, `C_i`, or `Y_i`; gradients flow only into `theta` through
`G_theta(X_i)` and any lightweight feature adapter used to compute it.

## Objective Terms

All objective terms are computed on development partitions only.

### 1. Demonstration Overlap Huber

Variable: `A_i[:K]`, `Y_i`, both `K x D`.

Formula:

`L_demo = mean_{i,k,d} Huber_delta((A_i[k,d] - Y_i[k,d]) / sigma_d)`

Scale:

- `sigma_trans = 0.02`;
- `sigma_rot = 0.05`;
- `sigma_grip = 1.0`;
- Huber `delta = 1.0` in normalized units.

Units: dimensionless normalized action error.

Gradient path: `A_i -> G_theta -> theta`.

Intended effect: permit boundary edits only where they reduce demonstration
overlap error.

Simpler alternative: standard LoRA on the same demonstrations.

Required ablation: `standard_lora`.

### 2. Tail Continuity Loss

Variable: `A_i[:K]`, `T_i`, both `K x D`.

Formula:

`L_tail = mean Huber_delta((A_i[:K] - T_i) / sigma)`

Scale and units: same as `L_demo`.

Gradient path: `A_i -> G_theta -> theta`.

Intended effect: reduce previous-tail/current-head discontinuity.

Simpler alternative: deterministic ChunkFlow overlap proxy.

Required ablation: `chunkflow_overlap_proxy`.

### 3. First- And Second-Order Continuity

Variables:

- `D1 A_i[:K]`, `D1 T_i`, shape `(K-1) x D`;
- `D2 A_i[:K]`, `D2 T_i`, shape `(K-2) x D`.

Formula:

`L_d1 = mean Huber_delta((D1 A_i[:K] - D1 T_i) / sigma)`

`L_d2 = mean Huber_delta((D2 A_i[:K] - D2 T_i) / sigma)`

Scale and units: normalized action-difference error.

Gradient path: `A_i -> G_theta -> theta`.

Intended effect: reduce boundary jump and high-frequency discontinuity without
editing future-zone cells.

Simpler alternative: first/second-order smoothness loss without learned mask.

Required ablation: `s2c_no_learned_overlap_mask_ablation`.

### 4. Mask Sparsity And Locality

Variable: `G_theta(X_i)`, shape `K x D`.

Formula:

`L_mask = mean(G_theta)`

Scale: dimensionless.

Gradient path: `G_theta -> theta`.

Intended effect: keep S2C local and Base-preserving.

Simpler alternative: deterministic all-overlap blending.

Required ablation: deterministic ChunkFlow proxy and no-mask ablation.

### 5. Event Preservation

Let `E_grip_i[k] = 1` when either `O_i[k,6]` or `T_i[k,6]` crosses the frozen
gripper event threshold relative to its neighboring step.

Formula:

`L_event = mean_{i,k:E_grip_i[k]=1} |A_i[k,6] - O_i[k,6]|`

Scale: normalized gripper units.

Gradient path: `A_i -> G_theta -> theta`.

Intended effect: prevent smoothness from erasing discrete gripper events.

Simpler alternative: exact Base gripper passthrough.

Required diagnostic: gripper-event preservation report.

## Total Objective

The Stage 0 training objective for a small development fit is:

`L_total = L_demo + 0.5 L_tail + 0.25 L_d1 + 0.10 L_d2 + 0.02 L_mask + 1.0 L_event`

Before any training, Stage 0 must estimate term magnitudes and gradient norms
on a small batch. If any weighted term has gradient norm more than `20x` the
median weighted term norm, coefficients must not be silently changed. The
result must stop as an objective-scale implementation failure or a bounded
validation-only search must be preregistered before use.

No KL divergence is used. Deterministic 7D actions and SmolVLA flow vectors are
not treated as probability distributions.

## Stage 0 Pass And Stop Gates

Stage 0 may pass to bounded validation only if all are true:

- proposal hash matches
  `399A3960F9FF9AFA8EDA7C3F743A95C3FD4DC711644C2398630F1E68486DC5B3`;
- no reward, success, done, simulator result, or confirmatory identity is read;
- adjacent chunk manifest has no duplicate, missing, extra, or split-overlap
  keys;
- Base boundary Huber disagreement is at least `0.0025` mean or `0.005` p75 in
  normalized units;
- ChunkFlow proxy leaves at least `2%` residual boundary headroom or measurable
  clean-retention headroom;
- identity reload max absolute error is `<= 1e-6`;
- effective mask positive fraction is between `0.02` and `0.80`;
- future-zone drift max absolute error is exactly `0`;
- action validity is `1.0` under official SmolVLA action semantics;
- S2C full improves the preregistered Stage 0 boundary-retention score over
  deterministic ChunkFlow proxy by at least `2%`;
- S2C full improves the same score over no-learned-overlap-mask ablation by at
  least `5%`;
- standard LoRA does not explain the same boundary-retention score;
- gripper event destruction count is `0`.

Stage 0 must stop before validation search for:

- no adjacent boundary headroom;
- collapsed masks;
- S2C equivalence to ChunkFlow proxy;
- S2C equivalence to no-mask ablation;
- global smoothing;
- future-zone edits;
- gripper event destruction;
- invalid action semantics;
- identity reload failure;
- expert future-tail inference;
- reward, success, done, or confirmatory-record access.

## First Serious Comparison

The first serious comparison remains:

1. `smolvla_base`
2. `chunkflow_overlap_proxy` or official ChunkFlow if installed and verified
3. `s2c_full`
4. `s2c_no_learned_overlap_mask_ablation`
5. `standard_lora`

This audit authorizes only preregistration and prototype protocol drafting next.
It does not authorize implementation, validation search, training, rollout, or
confirmatory testing until those artifacts are frozen.
