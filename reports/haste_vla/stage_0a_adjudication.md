# HASTE-VLA Stage 0A Adjudication

Decision: `HASTE_STAGE_0A_IMPLEMENTATION_FAILURE`.

## Durable Audit

- Linux PID: `295`, dead at adjudication.
- Matching HASTE workers: `0`.
- Exit code: `1`.
- Captured exceptions: `1` (`TypeError`).
- Stderr SHA-256:
  `BDD0BE9546B11ED9F82FDDA234BB8B34D81EBAD843A437A803D763628381A738`.
- Manifest persisted: `false`.
- Partial persisted: `false`.
- Persisted rows: `0`.
- Feature, adapter, and auxiliary-head artifacts: absent.

The traceback is deterministic and precedes SmolVLA loading:
`canonical_json_sha256(manifest_payload)` called standard JSON serialization
on NumPy displacement-normalization arrays. The runner did not reach status,
heartbeat, manifest persistence, model inference, probe fitting, identity
audit, or result writing.

## Integrity Decision

There is no valid partial from which to resume and no completed key to repeat.
Duplicate, missing, extra, and manifest-key acceptance are marked inapplicable
because neither a manifest nor any row was persisted. No row is accepted as
scientific evidence.

This is the frozen protocol's implementation-failure branch. It is not a
closed-loop result and does not test the HASTE mechanism. Do not repair or
rerun HASTE, do not authorize Stage 0B, and do not reinterpret this as a
scientific kill. Commit and push the preserved failure, then continue
automatically to Epoch 4 Cycle 23 candidate generation.
