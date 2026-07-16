# S2C-VLA Researcher A Proposal

Date: 2026-07-16 KST

Decision: `S2C_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

Method: `S2C-VLA`, Seam-Supervised Chunk Consistency for Base-preserving
SmolVLA execution.

Cycle: Epoch 4 Cycle 31

Previous method: `URF-VLA`

Previous fixed result: `URF_STAGE_0_NO_USABLE_HEADROOM`

URF is preserved unchanged. No URF repair, rescue, threshold change, proxy
change, task change, or reinterpretation is allowed.

## Claim

SmolVLA may fail not because its whole action chunk is far from the
demonstration chunk, but because independently decoded adjacent chunks disagree
at execution boundaries. A Base-preserving overlap edit mask and tail-anchored
bridge can reduce cross-chunk discontinuity while changing only selected
boundary cells and preserving clean Base behavior elsewhere.

## Closest Prior

Closest prior: ChunkFlow

Primary source: `https://arxiv.org/html/2607.12992v1`

Project page: `https://cytoderm-ai.github.io/chunkflow`

Positive result: ChunkFlow reports a seam-aware training-and-execution
framework for chunked policies. It partitions chunks into frozen, editable, and
future zones; uses deterministic overlap blending; adds boundary and continuity
losses; and reports `93.4%` LIBERO long-horizon success with improved boundary
jump, high-frequency energy, smoothness, and low-latency inference.

Secondary prior: SEAM

Primary source: `https://arxiv.org/abs/2607.04609`

Positive result: SEAM reports a training-free previous-tail consistency method
for flow-matching VLAs that reduces boundary jerk and transition discontinuity
on LIBERO-10 while preserving baseline-level task success.

## What Is New

S2C is not a new VLA backbone, a full ChunkFlow reproduction, or a generic
smoothness penalty.

S2C adds one mechanism to frozen SmolVLA action chunks:

`deployment-available previous tail + current Base chunk -> learned overlap
edit mask -> bounded tail-anchored bridge -> edited boundary cells only`.

The mechanism is distinct from ChunkFlow because:

- ChunkFlow trains a chunked policy with seam losses, history corruption, and
  optional advantage-weighted fine-tuning.
- S2C keeps SmolVLA as Base and attaches an identity-initialized overlap edit
  layer.
- S2C changes only selected boundary cells; unselected cells and non-overlap
  future zones remain exact Base.
- S2C uses existing LIBERO demonstrations for overlap supervision and does not
  require rollout rewards, success labels, done flags, object poses, or future
  observations at inference.

LoRA, if used, is only implementation infrastructure for the overlap mask or
bridge parameterization. The scientific mechanism is execution-indexed chunk
consistency.

## Mechanism Sketch

Let `B_t in R^{50 x 7}` be the frozen SmolVLA Base chunk decoded at replanning
time `t`.

Let `E_{t-1}` be the previously executed or committed chunk state available at
deployment. Its unexecuted or overlap tail is `T_{t-1}`.

Let `O_t` be the current chunk head over the same overlap horizon.

S2C computes:

- an overlap discrepancy feature `D_t = phi(T_{t-1}, O_t, proprio_t, phase_t)`;
- an edit mask `M_t in [0,1]^{K x 7}` over the overlap cells;
- a tail-anchored bridge target `C_t`, constructed from the previous tail and
  current Base head with first- and second-order continuity terms;
- an edited chunk:

`A_t[:K] = B_t[:K] + M_t * clip(C_t - B_t[:K], -c, c)`

and:

`A_t[K:] = B_t[K:]`

unless a later mathematical audit permits a strictly bounded future-zone edit.

The initial mask is zero, so initialized and reloaded S2C equals Base exactly.

## Development Data

Use only existing LIBERO development demonstrations and SmolVLA Base decoded
chunks.

Allowed training/development inputs:

- current observation-derived SmolVLA features;
- current proprioception;
- past executed actions or previous Base chunk tail;
- current Base chunk;
- action chunks from the same demonstration for supervised overlap targets;
- task identity and phase on discovery/validation partitions.

Forbidden inference inputs:

- future observations;
- object poses;
- rollout rewards;
- success flags;
- done flags;
- confirmatory-test identities;
- any task/result signal from held-out confirmatory evaluation.

## First Serious Comparison

The first serious comparison must include exactly:

1. `smolvla_base`
2. `chunkflow_overlap_proxy` or official ChunkFlow if locally installed and
   verified
3. `s2c_full`
4. `s2c_no_learned_overlap_mask_ablation`
5. `standard_lora`

The closest prior enters as policy 2. If official ChunkFlow assets are not
locally installed and verified, policy 2 must be a transparent local proxy and
must not be a strawman.

Standard LoRA is required because S2C trains a small adapter/head on
demonstrations.

## Stage 0 Audit

Stage 0 is development-only. It is not a closed-loop scientific result.

Required checks:

- proposal hash and source artifacts;
- manifest duplicate/missing/extra/split-overlap integrity;
- no reward/success/done/confirmatory records;
- paired neighboring chunk availability;
- previous-tail/current-head boundary disagreement headroom;
- ChunkFlow/SEAM proxy implementation sanity;
- S2C identity passthrough and disk reload within `1e-6`;
- mask noncollapse and locality;
- selected edit cells differ from Base while unselected cells remain Base;
- boundary jump reduction versus Base and ChunkFlow proxy;
- high-frequency and first/second-order discontinuity reduction;
- action validity under official SmolVLA semantics;
- clean retention on non-boundary cells;
- `s2c_full` beats `s2c_no_learned_overlap_mask_ablation`;
- `s2c_full` is not explained by deterministic blending alone;
- standard LoRA does not explain the same effect.

Do not proceed to bounded validation if:

- adjacent chunk boundary disagreement is absent;
- mask labels collapse all-zero or all-one;
- S2C acts globally instead of locally at boundaries;
- ChunkFlow proxy dominates S2C;
- the no-mask ablation explains the effect;
- action deltas are globally destructive;
- identity reload fails;
- any privileged inference input or confirmatory record is used.

## Expected Evidence If It Works

S2C should show:

- nontrivial Base boundary disagreement on development rows;
- lower boundary jump than Base;
- lower boundary jump or better success-proxy score than the closest prior
  proxy;
- lower discontinuity than the no-learned-overlap-mask ablation;
- exact preservation of unselected Base cells;
- bounded deltas in translation, rotation, and gripper dimensions;
- no clean-retention collapse;
- no confirmatory-test tuning.

## Current Status

No S2C implementation, training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened before this proposal.

Immediate next stage: Reviewer B attack on novelty, prior boundary,
mathematical validity, data viability, identity preservation, and decisive
experiment feasibility.
