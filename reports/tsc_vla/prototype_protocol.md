# TSC-VLA Executable Prototype Protocol

Date: 2026-07-16 KST

Decision: `TSC_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`

Method: `TSC-VLA`, Temporal-Spatial masked action completion for continuous
VLA chunks.

Proposal hash:
`0DF143D2D8773D7ABF4FC76AB7CC083FE7EE65DF84EA06631E67C2445F6DC941`

Frozen inputs:

- proposal: `reports/tsc_vla/researcher_proposal.md`
- Reviewer B attack: `reports/tsc_vla/reviewer_attack.md`
- Researcher A rebuttal: `reports/tsc_vla/researcher_rebuttal.md`
- mathematical audit: `reports/tsc_vla/mathematical_mechanism_audit.md`
- preregistration: `reports/tsc_vla/preregistration.md`

Runner to implement:

`scripts/run_tsc_vla_stage0.py`

No TSC implementation, training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened before this protocol.

## Stage 0 Purpose

Stage 0 is a development-only data, implementation, and mechanism audit. It
decides only whether TSC may proceed to bounded validation search.

It is not a closed-loop scientific result.

## Required Command Contract

The runner must support:

```powershell
wsl.exe -e bash -lc 'cd /mnt/c/Users/jiheo/tca_map && /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_tsc_vla_stage0.py'
```

The runner may also support explicit `--checkpoint`, `--data-root`,
`--output-dir`, `--resume`, and `--max-rows` arguments. Defaults must use the
validated local SmolVLA/LIBERO paths discovered by existing repository helpers
and prior runner patterns.

## Required Artifacts

Stage 0 writes under `reports/tsc_vla/`:

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

Stage 0 writes feature/model caches under `runs/tsc_vla/stage0/`.

## Worker Safety And Resume

Before launching a worker, check existing PID, heartbeat/status, partial,
result, logs, and exit-code files. If an existing worker is alive, monitor it
only. If it completed, adjudicate existing results only. If it died and the
partial result parses, resume only missing `row_key`s. Never duplicate
completed rows.

Heartbeats are stale only after checking both PID and logs. Duplicate-key and
manifest checks are mandatory before accepting the final result.

## Data Sources

Use only legal LIBERO demonstrations for fixed development tasks:

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

`partition | suite | task_identity | source_edge_sha256 | demo_id | frame_index | proxy_variant | policy_probe`

If multiple mask/threshold or proxy settings are audited in one run, the key
must include those labels as well.

## Required Preflight

Before model-row work:

1. verify proposal hash equals
   `0DF143D2D8773D7ABF4FC76AB7CC083FE7EE65DF84EA06631E67C2445F6DC941`;
2. verify required source documents exist;
3. persist official TS-Mask VLA asset/code status;
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

## Fixed Stage 0 Label Construction

For each legal development row:

- decode `A_B in R^[50,7]` from frozen Base SmolVLA;
- align `A_E in R^[50,7]` from the demonstration action chunk;
- compute `R = A_E - A_B`;
- compute valid-step mask `V in {0,1}^[50,1]`.

Discovery-only statistics:

- `S_d = median_discovery(|R_d|) + 1e-6`;
- `Tau_d = quantile_0.80_discovery(|R_d| / S_d)` over valid steps.

Labels:

- `Y_h,d = 1[V_h = 1 and |R_h,d| / S_d >= Tau_d]`;
- Stage 0 hard mask threshold `eta = 0.5`;
- Stage 0 diagnostic action scale `alpha = 0.1`.

If discovery or validation labels are all zero or all one, Stage 0 decision is
`TSC_STAGE_0_DATA_OR_SUPERVISION_FAILURE`.

## Required Mechanism Audits

Stage 0 must compute, persist, and adjudicate:

1. Base decoded chunk finite fraction and shape;
2. Base-to-expert residual scale `S_d` and threshold `Tau_d`;
3. discovery/validation positive and negative mask-cell counts;
4. task/phase coverage and validation task fraction;
5. trivial-majority mask baseline;
6. magnitude-only mask baseline;
7. structured 2D mask probe validation metric;
8. global residual gate and per-timestep or per-dimension gate diagnostics when
   cheap;
9. `ts_mask_continuous_proxy` validation completion Huber;
10. `tsc_full` validation completion Huber;
11. `tsc_no_targeted_mask_ablation` validation completion Huber;
12. full-minus-prior and full-minus-ablation headroom;
13. unselected-cell Base-clamp max error;
14. changed-cell count and mask positive rate;
15. translation, rotation, and gripper delta mean/p95/max;
16. initialized Base passthrough max error;
17. disk reload max error;
18. finite objective terms and gradient norms;
19. frozen-parameter gradient count;
20. Base, prior proxy, no-targeted-mask ablation, standard-LoRA diagnostic if
    instantiated, and TSC action validity under the same official semantics;
21. no reward/success/done/confirmatory reads.

## Minimal Local Proxy Definitions

`ts_mask_continuous_proxy` must be a transparent local continuous proxy for
TS-Mask-style temporal-spatial masked action modeling. It must use the same
data split, legal deployment inputs, action semantics, and comparable capacity
as TSC while omitting only TSC's Base-error-targeted sparse mask.

A compliant proxy may train or fit a generic temporal-spatial completion model
with non-targeted random/block masks on `A_E` or Base-compatible chunks, then
evaluate completion under the same validation protocol.

`tsc_no_targeted_mask_ablation` must use comparable completion capacity while
removing the targeted Base-error mask, for example by using uniform mask rate,
random structured masks, or non-targeted learned masks frozen before validation.

If the implementation cannot make these paths distinct, Stage 0 is
`TSC_STAGE_0_DESIGN_FAILURE`.

## Stage 0 Decision Rule

Return exactly one:

- `TSC_STAGE_0_DATA_OR_SUPERVISION_FAILURE`;
- `TSC_STAGE_0_NO_USABLE_HEADROOM`;
- `TSC_STAGE_0_DESIGN_FAILURE`;
- `TSC_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- `TSC_STAGE_0_PASS_TO_BOUNDED_VALIDATION`.

Hard gates for pass:

- proposal hash ok;
- serializer/preflight ok;
- manifest integrity ok;
- no split overlap;
- minimum discovery/validation windows met;
- all task validation coverage ok;
- mask labels noncollapsed on discovery and validation;
- structured mask probe beats trivial-majority and magnitude-only baselines;
- `tsc_full` beats `ts_mask_continuous_proxy` by at least `5%` relative
  validation Huber or `0.005` absolute normalized Huber;
- `tsc_full` beats `tsc_no_targeted_mask_ablation` by at least `5%` relative
  validation Huber or `0.005` absolute normalized Huber;
- unselected-cell Base-clamp max error at most `1e-6`;
- Base identity and disk reload max error at most `1e-6`;
- changed-cell fraction is nonzero and less than `0.60` on validation;
- action deltas are finite and not globally destructive;
- finite nonzero expected gradients;
- frozen parameter gradient count `0`;
- weighted gradient ratio at most `100`;
- action validity preserved under official semantics;
- exception count `0`;
- reward/success/done/confirmatory reads all `0`.

If any hard gate fails, bounded validation, Stage A, rollout, and confirmatory
testing are forbidden for this TSC run.

## No Scientific Kill At Stage 0

Stage 0 failures are classified as data, no-headroom, design, or
implementation/optimization failures. They are not closed-loop scientific
kills.

If Stage 0 passes, bounded validation search is the only allowed next stage.
If Stage 0 fails, do not rescue by changing thresholds, action semantics, task
selection, proxy definition, or coefficient budget.
