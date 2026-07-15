# KITE-VLA Prototype Protocol

Date: 2026-07-15 KST

Proposal SHA-256:
`FA00DE56D14E4C69388BE1642F7D52153841D58E77FD5A3F5C68B6C624A152B8`.

## Stage 0A Execution Contract

Stage 0A performs source, label, operator, frozen-Base headroom, objective
gradient, identity, and persistence audits only. It may load the frozen
SmolVLA checkpoint. It may not optimize an adapter, load the simulator, read a
reward/success/done flag, or access confirmatory reset identities.

Required durable artifacts:

- `stage_0a_pid.txt`;
- `stage_0a_heartbeat.json`;
- `stage_0a_status.json`;
- `stage_0a_preflight.json`;
- `stage_0a_manifest.json`;
- `stage_0a_partial.json`;
- `stage_0a_result.json` and `.md`;
- `stage_0a_validation.json`;
- `stage_0a_implementation_blocker.json` on exception;
- stdout, stderr, and exit-code files.

Before detached launch, a foreground preflight must construct a small manifest
fixture containing ordinary-list operator/normalization values, canonical-hash
it, write it, parse it, and reproduce the hash. The full runner must use a
NumPy-aware structured serializer and must catch failures before and after
manifest construction.

## Manifest And Resume

Enumerate full discovery/validation labels before model inference. Freeze
source edge hashes and all keys. Select deterministic model-audit rows by
task, split, and horizon after sorting `(demo_id,frame_index,horizon)`: up to
`8` per task/split/horizon, `128` maximum.

Persist a partial after every model row. Resume only missing manifest keys
after verifying method, proposal, source, split, manifest, and cached-feature
hashes. Refuse duplicate execution when a final result exists. A stale
heartbeat alone never proves death.

## Realization Operator Audit

Fit `F_5` and `F_20` from full discovery rows only. Persist:

- discovery command/state mean and standard deviation;
- ridge coefficient;
- coefficients and intercept;
- rank and singular values;
- discovery and validation row counts;
- global and per-task validation MSE;
- discovery-mean validation MSE;
- relative improvement.

No validation refit is allowed.

## Frozen Base Headroom

On deterministic validation model rows, use fixed seed `20262300` and flow
time `u=0.5`. Reconstruct Base clean actions from the native velocity field,
apply fixed differentiable unnormalization, and compute realization Huber at
both horizons. Compare against demonstrated-action operator residual on the
same rows. Persist Base action, target action, predicted displacement, target
displacement, and both errors.

## Gradient Audit

Instantiate zero-effect rank-4 LoRA and compute `L_flow` and `L_kite` on one
fixed discovery batch without stepping the optimizer. Persist term magnitudes,
expected trainable parameter names, per-term gradient norms, gradient ratio,
gradient cosine, frozen-parameter gradient count, and finite fractions.

## Identity Audit

On identical observation, noise, flow time, and solver state compare:

- Base native velocity;
- initialized KITE native velocity;
- reloaded initialized KITE native velocity;
- Base decoded action chunk;
- initialized and reloaded decoded chunks.

Require maximum error `<=1e-6`, unchanged Base hash, valid action shape, and
disk-reloadable adapter. The realization operator must not appear in the
inference policy call graph.

## Decision And Continuation

- Stage 0A pass: implement and run only frozen Stage 0B.
- Stage 0A failure: classify honestly, validate, commit, push, and continue to
  Cycle 24 without KITE repair or rescue, except that an independently proven
  pre-confirmatory rank-4 capacity bottleneck follows the one rank-8 diagnostic
  rule.
- Never change thresholds, tasks, splits, horizons, operator fitting, or
  baselines from partial outcomes.
