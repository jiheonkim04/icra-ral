# PCAV-VLA Executable Prototype Protocol

Date: 2026-07-15 KST

Decision: `PCAV_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0A_PENDING`

Proposal hash:
`E8B23C755C6D4E450FD193101CC0B15F88AAFE20E137A0F86830ED6D421E12AA`.

## Authorized Scope

Only the following may be implemented and run:

- pure PCAV helpers;
- Stage 0A row, partition, candidate, validity, and headroom audits;
- unit/governance tests;
- audit mode;
- detached Stage 0A execution;
- automatic 24-to-96 discovery expansion only when the frozen result is
  `PCAV_STAGE_0A_UNRESOLVED_EXPANSION_REQUIRED`.

No head training, validation decoding, confirmatory decoding, closed-loop
rollout, FAMR checkpoint loading, threshold change, or alternate candidate
distribution is authorized.

## Expected Campaign Stage

Both campaign state files must report:

`epoch_4_cycle_18_pcav_stage_0a_implementation_pending`.

The proposal file hash and `proposal_hash.txt` must both match the frozen hash.

## Audit Command

```bash
/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python \
  scripts/run_pcav_vla_stage0.py \
  --mode audit \
  --checkpoint /mnt/c/assets/checkpoints/smolvla_libero \
  --libero-data-root /mnt/c/assets/data/libero \
  --run-root /mnt/c/Users/jiheo/tca_map/runs/pcav_vla/stage0a \
  --report-root /mnt/c/Users/jiheo/tca_map/reports/pcav_vla
```

Audit mode loads no model and decodes no image. It verifies state, proposal
hash, source paths, result absence, partial JSON parse, PID/worker state, CUDA,
disk space, resource registry, and forbidden gates.

## Stage 0A Command

```bash
/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python \
  scripts/run_pcav_vla_stage0.py \
  --mode stage0a \
  --checkpoint /mnt/c/assets/checkpoints/smolvla_libero \
  --libero-data-root /mnt/c/assets/data/libero \
  --run-root /mnt/c/Users/jiheo/tca_map/runs/pcav_vla/stage0a \
  --report-root /mnt/c/Users/jiheo/tca_map/reports/pcav_vla
```

The runner starts with the fixed 24-row manifest. It writes one atomic partial
row after all four candidates for that row complete. If and only if the frozen
24-row headroom decision is unresolved, it appends only missing keys from the
96-row expansion manifest.

## Durable Runtime Files

Under `runs/pcav_vla/stage0a`:

- `worker.pid`
- `heartbeat.json`
- `status.json`
- `partial_result.json`
- `stdout.log`
- `stderr.log`
- `exit_code.txt`
- `checkpoint_snapshot.json`

Under `reports/pcav_vla`:

- `stage_0a_row_manifest.json`
- `stage_0a_candidate_manifest.json`
- `stage_0a_result.json`
- `stage_0a_result.md`
- `stage_0a_adjudication.md`

Writes are temporary-file plus atomic replace. The partial schema contains
proposal hash, manifest hash, planned row count, completed row count, completed
row keys, rows, exception count, and update timestamp.

## Resume Contract

Before launch:

1. inspect newest PID, heartbeat, status, partial, result, log, and exit code;
2. verify PID liveness in Linux;
3. parse and validate partial JSON;
4. compare completed/planned counts and exception count;
5. never launch if the worker is alive;
6. adjudicate an existing final result without rerun;
7. if dead with a valid partial, resume only missing row keys;
8. verify logs before treating a stale heartbeat as death.

Completed row keys are immutable. A duplicate completed key is a hard
implementation failure.

## Pure Helper Contract

`tca_map/smolvla/pcav_vla.py` must provide deterministic helpers for:

- stable seed and row identity;
- discovery row selection and phase quotas;
- partition overlap and duplicate audit;
- candidate chunk hashing and diversity;
- grouped action error and oracle headroom;
- action validity;
- partial payload validation;
- Stage 0A decision logic;
- result rendering inputs.

Pure helpers load no model, simulator, or external file and receive unit tests.

## Stage 0A Result Schema

Required top-level fields:

- method and proposal hash;
- start/end timestamps;
- resource-contamination audit;
- source/provenance audit;
- partition and duplicate audit;
- raw mapping audit;
- row manifest path/hash;
- candidate manifest path/hash;
- planned/completed row counts;
- candidate count per row;
- exception count;
- Base identity error;
- candidate diversity metrics;
- action-validity metrics;
- per-task and per-phase coverage;
- candidate-oracle headroom metrics;
- expansion used and missing-key counts;
- checkpoint Base hash before/after;
- peak CUDA allocation as diagnostic only;
- confirmatory observations decoded;
- confirmatory actions computed;
- final decision and failure class;
- Stage 0B allowed boolean.

## Frozen Decisions

`PCAV_STAGE_0A_PASS_STAGE_0B_ALLOWED` requires valid Base actions, more than
half the rows with at least one valid unique alternative, and every other
frozen gate. Invalid alternatives are recorded and made ineligible, never
clipped.

`PCAV_STAGE_0A_UNRESOLVED_EXPANSION_REQUIRED` is transient and must
automatically expand to 96 rows in the same run without repeating completed
rows.

`PCAV_STAGE_0A_NO_USABLE_HEADROOM` records a diagnostic no-headroom result and
starts Cycle 19; it does not retune candidates or load an adapted generator.

`PCAV_STAGE_0A_DESIGN_FAILURE_CANDIDATES_COLLAPSED` requires verified distinct
noise, correct implementation, and collapsed postprocessed candidates.

`PCAV_STAGE_0A_IMPLEMENTATION_OR_DATA_FAILURE` covers mapping, data, identity,
validity, serialization, duplicate, exception, or resume failures.

No Stage 0A decision is a closed-loop scientific kill.

## Automatic Continuation

After a valid final result:

1. verify final JSON, manifest hashes, completed/planned counts, duplicates,
   exception count, Base hash, and confirmatory zeros;
2. independently adjudicate under the frozen protocol;
3. commit and push result, adjudication, tests, and campaign state;
4. on pass, implement Stage 0B exactly as preregistered;
5. on no-headroom/design failure, begin the next candidate cycle without
   rescuing PCAV;
6. on implementation/data failure, record the class and begin a new method
   cycle unless a purely mechanical correction is explicitly permitted by
   current governance.

Do not stop after documentation or Stage 0A adjudication.
