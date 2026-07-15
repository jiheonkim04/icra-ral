# HASTE-VLA Prototype Protocol

Date: 2026-07-15 KST

Proposal hash:
`5415BC1533A24EC55CC511DDEB014BB11D9C19F603C59D1F1D3E151E15B930A6`.

Decision: `HASTE_PROTOTYPE_PROTOCOL_FROZEN`

## Stage 0A Execution

The Stage 0A runner must create durable:

- PID;
- heartbeat and status JSON;
- preflight JSON;
- source/row manifest;
- atomic partial result;
- stdout and stderr logs;
- exit-code file;
- result JSON and Markdown;
- independent validation JSON;
- adjudication Markdown.

Resume accepts only missing manifest keys after verifying method, proposal,
source, split, and manifest hashes. A complete result refuses duplicate
execution.

## Fixed Stage 0A Sampling

For each source demonstration and each horizon `H_e`, enumerate every frame
with at least one observable future interval. Persist all label statistics.

For expensive frozen SmolVLA inference, choose a deterministic equal-task
subset separately for discovery and validation:

- event-near: `tau <= 10`;
- event-far: `tau > 10`;
- censored;
- up to `32` rows per stratum per task and partition;
- evenly spaced indices after sorting by `(demo_id, frame_index, H_e)`;
- stable first-use deduplication;
- no replacement across splits.

The exact subset is frozen in the manifest before model inference. No reward,
success, done, video outcome, or confirmatory identity is read.

## Probe Contract

Input is the one audited existing `960`-dimensional SmolVLA representation.
The hook name and tensor position must be persisted before feature extraction.

Linear hazard probe:

- one `Linear(960,H_e)` layer;
- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- exactly `100` discovery steps;
- fixed seed `20262200`;
- no validation early stopping.

Linear displacement probe:

- one `Linear(960,6)` layer;
- same optimizer and fixed 100 steps;
- uncensored discovery rows only;
- no validation early stopping.

These probes test observability only. They are not HASTE checkpoints and may
not enter rollout.

## Base Headroom Contract

Decode one ordinary Base action chunk per sampled frame under the canonical
disk-loaded checkpoint and processor. Compare its aligned first action with the
recorded postprocessed demonstration action at that frame.

Report arm Huber/L2 and gripper-sign error by event-near, event-far, censored,
task, and split. Offline errors are development diagnostics only.

## Identity Contract

Instantiate the frozen rank-4 LoRA target set and both auxiliary heads. Before
optimization:

- compare Base and initialized-HASTE flow vectors on identical noisy actions;
- compare decoded actions under fixed generator state;
- save, reload, and repeat;
- hash Base parameters before and after;
- require maximum error at most `1e-6`.

No query token, processor, normalization, action head, or solver modification
is permitted in Stage 0A.

## Stage 0B Contract

Only after Stage 0A passes, implement matched 20-step micro-fits for the four
trainable policies. Save objective magnitudes, gradient norms/conflicts,
checkpoints, reload checks, event-near/far action effects, and action validity.

No Stage 0B threshold or implementation choice may be selected from Stage 0A
outcomes beyond the automatic pass/fail authorization.

## Automatic Decisions

- Stage 0A pass: implement and run only frozen Stage 0B;
- Stage 0A failure: adjudicate, commit, push, and continue to Cycle 23 without
  HASTE repair or rescue;
- Stage 0B pass: run the six-config validation-only search;
- Stage 0B failure: adjudicate and continue without rescue;
- validation pass: freeze one checkpoint/config and paired manifests;
- empirical GO: immediately verify Quantized OpenVLA-OFT INT4 and a second
  claim-specific condition;
- empirical kill: preserve it and continue by campaign governance.

Routine WSL monitoring, resume, validation, commit, and push require no user
approval.
