# HEST-VLA Prototype Protocol

Date: 2026-07-15 KST

Proposal hash:
`E56B4717BDF949E1A4371457058DFC662E0D79C70D9E2FBEF35A5415FD0F0527`.

Decision: `HEST_PROTOTYPE_PROTOCOL_FROZEN`

## Stage 0A Execution

Entry point: `scripts/run_hest_vla_stage0a.py`.

Implementation module: `tca_map/smolvla/hest_vla.py`.

Required durable artifacts:

- `reports/hest_vla/stage_0a_pair_manifest.json`;
- `reports/hest_vla/stage_0a_partial.json`;
- `reports/hest_vla/stage_0a_heartbeat.json`;
- `reports/hest_vla/stage_0a_status.json`;
- `reports/hest_vla/stage_0a_pid.txt`;
- `reports/hest_vla/stage_0a_stdout.log`;
- `reports/hest_vla/stage_0a_stderr.log`;
- `reports/hest_vla/stage_0a_exit_code.txt`;
- `reports/hest_vla/stage_0a_result.json`;
- `reports/hest_vla/stage_0a_result.md`;
- `reports/hest_vla/stage_0a_validation.json`.

The runner is CPU-only. It must not import or load SmolVLA, CUDA, LIBERO
simulator environments, rewards, done flags, videos, or confirmatory manifests.

The partial file is rewritten atomically after each completed window. Resume
accepts only missing manifest keys and refuses any proposal-hash, manifest-hash,
method, stage, or row mismatch.

## Stage 0A Computation

For every frozen source window:

1. read the original `50 x 7` demonstration action block;
2. compute discovery support only from discovery rows;
3. run Base identity, SplineProxy, HEST at alpha `1.0`, NoEndpoint, and
   MovingAverage;
4. compute shape, finite, support, endpoint, first-action, gripper,
   second-difference energy, and pairwise-equivalence metrics;
5. persist source and output hashes rather than full image or state arrays;
6. persist only legal action data and audit summaries.

The reference runner must independently reload its persisted result and
recompute validation gates before adjudication.

## Stage 0B Execution

Stage 0B is specified but not authorized unless Stage 0A passes. Its eventual
entry point must use the official WSL LIBERO environment and exact saved
simulator states. Every key is:

`(policy, suite, task_identity, demo_id, start, stop)`.

The simulator is synchronous. There is no timeout-based success metric and no
change to task identity or action semantics. Exact expert replay is diagnostic
only.

## Queue Integration Smoke

After Stage 0B and before validation rollout, one real SmolVLA queue-refill
smoke must verify:

- one Base model call per queue refill;
- Base input and HEST output shapes `[1,50,7]`;
- the first queued action matches transformed row `0` exactly;
- queue length remains `50`;
- gripper values match Base exactly;
- no processor or postprocessor is bypassed;
- fallback returns the full Base chunk exactly;
- checkpoint-free implementation reloads deterministically.

## Validation And Confirmatory Execution

Validation may evaluate exactly three HEST alpha configurations. Comparators do
not receive a search. The chosen alpha is frozen before Stage A.

Stage A uses the five preregistered policies and ten paired reset identities per
policy. Stage B uses forty paired reset identities per key policy. Results
include success, paired effects, task breakdown, mechanism activation,
fallback, clean retention, and action validity.

Timing and resource metrics are reported only from uncontended intervals with
known boundaries. They never decide task-success validity.

## Automatic Decisions

- Stage 0A pass: freeze Stage 0B execution details and continue.
- Stage 0A data/implementation/no-headroom/design failure: adjudicate honestly,
  commit, push, and continue to Cycle 22 without HEST rescue.
- Stage 0B pass: run queue smoke and bounded validation search.
- Stage 0B failure: adjudicate, commit, push, and continue without rescue.
- validation failure: archive all three configurations and continue.
- Stage A catastrophic valid failure: archive under the frozen rule.
- unresolved Stage A: advance to Stage B.
- Stage B GO: continue to larger SmolVLA and Quantized OpenVLA-OFT INT4.
- Stage B kill: archive and continue by campaign governance.

Routine monitoring, resume, validation, commit, and push do not require user
approval.
