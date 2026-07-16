# CFR-VLA Executable Prototype Protocol

Date: 2026-07-16 KST

Decision: `CFR_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`

Method: `CFR-VLA`, Continuous Full-Chunk Refinement for VLA action-flow
decoding.

Proposal hash:
`9E2FC510B2D97C869F18BE6C5B339CE034DD98223802078358320AA8BEF3D0AE`

Frozen inputs:

- proposal: `reports/cfr_vla/researcher_proposal.md`
- Reviewer B attack: `reports/cfr_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/cfr_vla/researcher_rebuttal.md`
- mathematical audit: `reports/cfr_vla/mathematical_mechanism_audit.md`
- preregistration: `reports/cfr_vla/preregistration.md`

Runner to implement:

`scripts/run_cfr_vla_stage0.py`

No CFR implementation, training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened before this protocol.

## Stage 0 Purpose

Stage 0 is a development-only implementation/data/mechanism audit. It decides
only whether CFR may proceed to bounded validation search.

It is not a closed-loop scientific result.

## Required Command Contract

The runner must support:

```powershell
wsl.exe -e bash -lc 'cd /mnt/c/Users/jiheo/tca_map && /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_cfr_vla_stage0.py'
```

The runner may also support explicit `--checkpoint`, `--data-root`,
`--output-dir`, `--resume`, and `--max-rows` arguments, but defaults must use the
validated local SmolVLA/LIBERO paths discovered by existing repository helpers.

## Required Artifacts

Stage 0 writes under `reports/cfr_vla/`:

- `stage_0_manifest.json`;
- `stage_0_partial.json`;
- `stage_0_status.json`;
- `stage_0_heartbeat.json`;
- `stage_0_result.json`;
- `stage_0_result.md`;
- `stage_0_adjudication.md`;
- `stage_0_action_semantics.json`;
- `stage_0_official_prior_asset_check.json`;
- `stage_0_serializer_preflight.json`;
- `stage_0_preflight.json`;
- `stage_0_pid.txt`;
- `stage_0_exit_code.txt`;
- `stage_0_stdout.log` and `stage_0_stderr.log` when launched detached.

Stage 0 writes feature/model caches under `runs/cfr_vla/stage0/`.

## Worker Safety And Resume

Before launching a worker, check existing PID, heartbeat/status, partial,
result, logs, and exit-code files. If an existing worker is alive, monitor it
only. If it completed, adjudicate existing results only. If it died and the
partial result parses, resume only missing `row_key`s. Never duplicate completed
rows.

Heartbeats are stale only after checking both PID and logs. Duplicate-key and
manifest checks are mandatory before accepting the final result.

## Data Sources

Use only legal LIBERO demonstrations for the fixed development tasks:

1. `libero_spatial/task_3`;
2. `libero_object/task_3`;
3. `libero_goal/task_5`;
4. `libero_10/task_5`.

Discovery demonstrations: `0..7`.

Validation demonstrations: `8..9`.

Confirmatory task/reset identities, rewards, success flags, done flags, object
poses, and future observations are forbidden.

Minimum accepted final Stage 0 manifest:

- at least `512` discovery windows;
- at least `128` validation windows;
- every task has validation rows;
- no validation task fraction exceeds `0.40`;
- duplicate manifest keys `0`;
- duplicate partial keys `0`;
- missing manifest keys `0`;
- extra partial keys `0`;
- split-overlap keys `0`.

## Required Row Key

Every manifest and partial row must include a stable `row_key` containing:

`partition | suite | task_identity | source_edge_sha256 | demo_id | frame_index | policy_probe`

If multiple CFR/DFM settings are audited in one run, the key must also include
`K` and the proxy/variant label.

## Required Preflight

Before model-row work:

1. verify proposal hash equals
   `9E2FC510B2D97C869F18BE6C5B339CE034DD98223802078358320AA8BEF3D0AE`;
2. verify required source documents exist;
3. persist official DFM-VLA asset/code status;
4. persist official SmolVLA/LIBERO action semantics;
5. verify JSON serialization of manifest rows and NumPy values;
6. verify CUDA and official SmolVLA checkpoint availability when model decoding
   is required;
7. persist preflight failures as implementation blockers without fabricating
   partial rows.

## Required Action Semantics

`stage_0_action_semantics.json` must include:

- model-native action shape;
- postprocessor/unnormalizer class and parameters;
- environment action shape;
- environment action-space low/high if exposed;
- gripper convention;
- finite checks;
- action-space or equivalent official environment validation result for Base;
- the final boolean action-validity definition applied to every policy.

No ad hoc `[-1,1]` validity-only rule is allowed.

## Required Mechanism Audits

Stage 0 must compute, persist, and adjudicate:

1. Base decoded chunk finite fraction and shape;
2. Base-to-expert residual variance by action dimension;
3. task/phase residual baseline validation Huber;
4. deployment-input residual/refinement probe validation Huber;
5. relative and absolute probe improvement;
6. DFM proxy validation Huber;
7. CFR proxy/smoke validation Huber where no optimizer rollout is used;
8. CFR-minus-DFM residual headroom;
9. `cfr_no_iterative_refinement` distinctness from `cfr_full`;
10. standard-LoRA availability or transparent not-yet-trained status;
11. initialized Base passthrough max error;
12. disk reload max error;
13. finite objective terms and gradient norms;
14. frozen-parameter gradient count;
15. Base, DFM proxy, no-iterative ablation, standard-LoRA if instantiated, and
    CFR action validity under the same official semantics;
16. no reward/success/done/confirmatory reads.

## Minimal Local Proxy Definitions

The DFM proxy must be iterative full-sequence refinement. A compliant local
proxy may quantize normalized expert action chunks with discovery-derived
per-dimension bins, fit iterative token or residual velocity updates on
discovery rows, and dequantize to `[50,7]` for validation. It must use legal
current deployment inputs and the full action sequence.

`cfr_no_iterative_refinement` must use one terminal residual update under the
same legal inputs and bounds.

If the implementation cannot make these two paths distinct, Stage 0 is
`CFR_STAGE_0_DESIGN_FAILURE`.

## Stage 0 Decision Rule

Return exactly one:

- `CFR_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- `CFR_STAGE_0_NO_USABLE_HEADROOM`;
- `CFR_STAGE_0_DESIGN_FAILURE`;
- `CFR_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- `CFR_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Hard gates for pass:

- proposal hash ok;
- serializer/preflight ok;
- manifest integrity ok;
- no split overlap;
- minimum discovery/validation windows met;
- all task validation coverage ok;
- residual variance noncollapsed;
- residual/refinement probe beats task/phase residual baseline by at least
  `5%` relative validation Huber or `0.005` absolute normalized Huber;
- DFM proxy leaves at least `5%` relative validation Huber or `0.005` absolute
  normalized Huber residual headroom for CFR;
- iterative CFR path is distinct from no-iterative ablation;
- Base identity and disk reload max error at most `1e-6`;
- finite nonzero expected gradients;
- frozen parameter gradient count `0`;
- weighted gradient ratio at most `100`;
- action validity preserved under official semantics;
- exception count `0`;
- reward/success/done/confirmatory reads all `0`.

If any hard gate fails, bounded validation, Stage A, rollout, and confirmatory
testing are forbidden for this CFR run.

## No Scientific Kill At Stage 0

Stage 0 failures are classified as data, no-headroom, design, or
implementation/optimization failures. They are not closed-loop scientific kills.

If Stage 0 passes, bounded validation search is the only allowed next stage.
If Stage 0 fails, do not rescue by changing thresholds, action semantics, task
selection, proxy definition, or coefficient budget.
