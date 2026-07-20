# Epoch 8 Route Evidence Audit

Generated: 2026-07-20T21:25:29+09:00

## Adjudication

The inherited statement that 95 routes were 95 completed closed-loop experiments is false. The source contains a mixture of exact methods, variants, controls, diagnostics, static candidates, implementation-invalid attempts, and resource/artifact blocks. Only evidence classes 1 and 2 below are scientific closures, and an exact-method closure never closes its whole problem axis.

- Whole program: `INDEPENDENT_ROUTE_AUDIT_REQUIRED`
- Exact frozen four-shard sub-study: `HARD_EXTERNAL_BLOCKER`
- Current paper: `PAPER_NOT_AUTHORIZED`

## Counts

- Raw inherited entries: **95**
- Corrective and Epoch 8 addenda: **14**
- Unique empirically adjudicated problem groups: **6**
- Unique empirically adjudicated method formulations after explicit deduplication: **34**
- Known lower-bound official closed-loop role counts: Base **438**, Prior **345**, Ours **577**, controls **528**
- Entries whose legacy summaries do not support role-specific closed-loop counts: **6**

The role totals are conservative lower bounds, not a global unique-episode total: old diagnostics sometimes reuse rows and many summaries omit role-specific denominators. Those cases remain `UNVERIFIED` in the JSON.

## Evidence classes

| Class | Entries | Scientific closure? |
|---|---:|---|
| 1. `EMPIRICAL_PROBLEM_FALSIFICATION` | 4 | Yes, scoped |
| 2. `EMPIRICAL_EXACT_METHOD_FALSIFICATION` | 34 | Yes, scoped |
| 3. `UNDERPOWERED_OR_AMBIGUOUS` | 11 | No |
| 4. `IMPLEMENTATION_INVALID` | 33 | No |
| 5. `RESOURCE_BLOCKED` | 7 | No |
| 6. `ARTIFACT_BLOCKED` | 0 | No |
| 7. `POSITIONING_OR_OVERLAP_RISK` | 1 | No |
| 8. `STATIC_IDEA_REJECTION` | 8 | No |
| 9. `POSITIVE_PROBLEM_EVIDENCE` | 5 | No |
| 10. `POSITIVE_METHOD_EVIDENCE` | 6 | No |

## Candidate A and Candidate R raw-evidence audit

All 15 records in `epoch7_evidence_index.json` match their supplied SHA-256 hashes. Candidate A's ten unique X-VLA-format HDF5 demonstrations match the hashes embedded in the repaired falsifier. Candidate R's 16 Stage-0 run artifacts and every explicit path/hash pair used by the closed-loop resource blocker are present and match.

One historical caveat is preserved: the invalid-gripper Candidate A attempt references a protocol path with pre-repair SHA `BB595F...`, while that same path now contains the documented repaired protocol with SHA `CB3B9E...`. Therefore the invalid attempt's original protocol bytes are absent at the referenced path. This does not invalidate the repaired final falsifier, whose result, protocol, manifest, and demonstrations match; it limits independent reconstruction of the discarded invalid attempt.

Candidate A supports `EMPIRICAL_EXACT_METHOD_FALSIFICATION` only for the frozen scalar action-energy ranking formulation. Candidate R supports `POSITIVE_PROBLEM_EVIDENCE` at the action-sequence level and `RESOURCE_BLOCKED` for the exact four-shard closed loop, with **0/40 episodes executed**.

## Deduplication policy

- Repeated reports and repairs are evidence records, not new routes.
- Controls, official-prior diagnostics, and residual scans do not count as Ours methods.
- TCA-Map, TCA-Select, and the ActionMap mini-anchor are retained as three raw entries but one explicitly deduplicated target-prior/action-map campaign family.
- The custom SmolVLA 7-D adapter and TG-7D are retained as two raw entries but one deduplicated adapter family.
- Static idea rejection, overlap, missing artifacts, and resource blocks do not count as scientific falsification.

## Immediate consequence

The verified language problem remains open after two scoped exact-method failures: scalar action-energy ranking and PCAT action transport. The independent two-shard study is also resource-blocked with zero scientific episodes. The active hidden-mass screen produced a narrow front/back discovery gap but failed its overall competence gate, and its first legal-response belief mechanism failed the valid frozen Stage 0. None of these scoped results is a problem-axis closure.

Machine-readable per-route details, evidence paths, counts, supported scope, and reopen conditions are in `reports/epoch8_route_evidence_ledger.json`.
