# A2C2 Official-Prior-First Problem Verification Protocol

Date: `2026-07-19 KST`

Current decision: `A2C2_PROBLEM_VERIFICATION_PROTOCOL_FROZEN_READY_FOR_PREFLIGHT`

This preregistration freezes the problem-verification stage for the selected
`ASYNC_DELAY_REACTIVITY` thesis. It executes Base and the closest external
Prior only. It does not design, train, or rollout Ours.

## Fidelity boundary

The implementation is always labelled
`MECHANISM_FAITHFUL_A2C2_LOCAL_PORT`, never an official reproduction. The
official source is pinned to `k1000dai/a2c2-libero` commit
`54dd088302a0ef3f50c4add3ec927ab94d76a406`.

The released repository uses LeRobot `0.2.0`; it cannot parse the runnable
official SmolVLA checkpoint's LeRobot `0.4.x` fields. The local port therefore
keeps the validated LeRobot `0.4.4` SmolVLA graph and captures the same first
prefix VLM token with a read-only forward hook. It preserves the released
correction graph: two latest RGB views through a shared frozen ImageNet
ResNet-18, latest 8D state, selected base action, full stale `50x7` chunk,
sin/cos offset, VLM latent, released SHA1 task scalar, a 512-dimensional
8-head 6-layer transformer, and an additive normalized-action residual.

No official correction-head checkpoint is available. The local Prior is
trained for `40,000` AdamW steps at microbatch `4`, learning rate `1e-5`,
weight decay `1e-5`, and gradient clip `10`. This is below both the paper's
`200,000` steps and the README's inconsistent `400,000` steps. It is Prior
module training, not VLA training.

## Frozen data and panel

Prior training uses exactly the first four ascending episode identities for
each LIBERO Spatial global task index `30..39`, 40 episodes total. Anchors have
stride `8`; each anchor uses the deterministic unique offsets
`{0, floor(max/3), floor(2max/3), max}` with `max=min(49, remaining steps)`.
Expert actions are training supervision only.

Closed-loop problem verification uses the result-independent panel:

- LIBERO Spatial task `0`, `4`, and `8`;
- official init-state identities `0..4` for every task;
- `15` matched episodes per arm;
- official success and a `220`-step cap;
- no expert action, reward, done flag, future observation, or reset identity at
  live policy inference.

The three arms are `BASE_STANDARD_E10_D0`, `BASE_DELAYED_E40_D10`, and
`PRIOR_DELAYED_E40_D10`. The queue follows the released A2C2 evaluator. The
first chunk executes `[0:e)` and retains `[e:e+d)`. Every later chunk first
executes `d` retained actions, then new actions `[d:e)`, and retains the new
`[e:e+d)` tail.

## Frozen decision rules

Evaluation is valid only if all 45 rows complete on identical identities with
finite actions, zero exceptions, nonzero Base forwards, and nonzero Prior
forwards in the Prior arm.

- Base competence: at least `8/15` standard successes and at least one success
  on every task.
- Repeated problem: standard minus delayed Base is at least `3/15`, with at
  least three matched standard-success/delayed-failure identities spanning at
  least two tasks.
- Prior improvement: Prior minus delayed Base is at least `2/15`, at least two
  delayed failures recover, no more than one delayed success regresses, and
  live correction is nonzero.
- Saturation: improvement passes and Prior is within one success of standard,
  or at most one matched standard-success/Prior-failure identity remains.
- Residual: improvement passes and standard minus Prior is at least `2/15`,
  with at least two matched residual identities spanning at least two tasks.

Adjudication returns exactly one of the seven decisions required by the steer.
`VERIFIED_PRIOR_RESIDUAL` alone authorizes generating at most two Ours
candidates. `PRIOR_SATURATES_PROBLEM` closes this thesis without Ours.

## Execution and resources

Jobs execute in this exact order: `SETUP_PREFLIGHT`, `CACHED_FEATURE_PROBE`,
`PRIOR_MODULE_TRAINING`, Base `VLA_CLOSED_LOOP_ROLLOUT`, trained-Prior
`VLA_CLOSED_LOOP_ROLLOUT`, and `REPORT_ONLY` adjudication.

Reserved VRAM may not exceed `88%`; system RAM use may not exceed `82%`.
Swap and CPU offload are forbidden. Partial results and heartbeat/status files
are mandatory. The complete machine-readable contract is
`reports/a2c2_prior/problem_verification_protocol.json`.

## Preserved infrastructure repair

The first frozen preflight exposed one `INFRASTRUCTURE_NULL_DEFECT`: LeRobot
`0.4.4` calls `vlm_with_expert.forward(...)` directly, bypassing PyTorch's
`Module.__call__` hook dispatcher. The failed attempt is preserved in
`reports/a2c2_prior/preflight_failed_attempt_1.json`. The single narrow repair
temporarily wraps the same bound forward method, records its unchanged prefix
return, and restores it after use. No scientific contract changed.

The first cached-feature attempt then stopped with a distinct
`DATA_PIPELINE_DEFECT` before any cache row or model forward. LeRobot `0.4.4`
does not expose the LeRobot `0.2` `episode_data_index` attribute. The bounded
repair derives identical half-open subset-local bounds from the authoritative
`hf_dataset["episode_index"]` column and rejects any missing, duplicated, or
noncontiguous frozen episode. The failed attempt is preserved in
`reports/a2c2_prior/cache_failed_attempt_1.json`; no scientific field changed.

The repaired cache run was then stopped on a distinct
`RESOURCE_COMPATIBILITY_DEFECT`: retained WSL2 host memory put Windows RAM at
87.93%, even though the WSL-local view was 23.3%. The valid durable cache was
preserved at 384 anchors and 1,525 rows. A clean idle-WSL shutdown returned
host RAM to 65.34%; the identical command resumes only missing anchors. The
interruption is preserved in
`reports/a2c2_prior/cache_resource_interruption_1.json`.

A fresh VM exposed a distinct default-allocation root: WSL2's default 50% host
memory limit raised Windows RAM from 65.78% to 88.93% on the minimum live
model path. The cache was again preserved, now at 533 anchors and 2,115 rows.
The previously absent `.wslconfig` is bounded to 3,584 MiB with swap disabled
and immediate cache reclaim, following Microsoft's official WSL interface.
The exact attempt is
`reports/a2c2_prior/cache_wsl_default_memory_failure_1.json`.

The first Base rollout stopped before any episode or model forward on a
distinct simulator-path `RESOURCE_COMPATIBILITY_DEFECT`. The 3,584 MiB cache
cap OOM-killed Python at about 2.77 GiB anonymous RSS plus 256 MiB WSLg shared
memory. The single simulator-path correction raises the cap to 4,096 MiB,
keeps swap disabled, and disables unused WSL GUI support. The failed attempt
is preserved in `reports/a2c2_prior/base_rollout_oom_failed_attempt_1.json`.
