# Autonomous RA-L Campaign State

Date: 2026-07-16 KST

Active governance: `reports/current_research_governance.md`

Current branch: `codex/autonomous-until-paper-governance-v2`

Current decision:
`AFID_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY`

Current epoch: `4`

Current cycle: `33`

Current stage: `epoch_4_cycle_33_afid_stage_0_implementation_validated`

## Epoch 4 Cycle 33 AFID-VLA Selection

Cycle 33 completed the primary-source prior mechanism map in
`reports/epoch_4_cycle_33_prior_mechanism_map.md` and generated exactly three
candidates in `reports/epoch_4_cycle_33_candidate_generation.md`.

Selected method: `AFID-VLA`, Action-Factor Instruction Densification for
Base-preserving SmolVLA, score `90 / 100`.

Closest prior: FineVLA (`https://arxiv.org/html/2605.27284v1`), with positive
evidence that fine-grained instruction supervision improves steerable control
and can retain goal-level success.

Frozen first comparison for the proposal stage: `smolvla_base`,
`finevla_action_factor_proxy`, `afid_full`, `afid_no_factor_ablation`, and
`standard_lora`.

LoRA is only implementation infrastructure. The scientific mechanism is sparse
action-factor instruction densification with a Base-preserving residual gate
and exact passthrough when predicted factor confidence is low.

The AFID-VLA Researcher A proposal is frozen in
`reports/afid_vla/researcher_proposal.md` with SHA-256
`B5D1EE12FF2D0280511452DA7FE55295740FD9942A8BE293F444C8EB157062BC`.

Reviewer B attack is complete in `reports/afid_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. It requires
FineVLA to remain policy 2, a fair nonprivileged FineVLA proxy, frozen factor
extraction rules, noncollapsed factor/mask health, factor observability above
trivial baselines, no-factor and standard-LoRA controls, exact Base
passthrough, and no deterministic-action KL.

Researcher A rebuttal is complete in `reports/afid_vla/researcher_rebuttal.md`
with decision `AFID_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`, accepting all
Reviewer B conditions.

The AFID mathematical mechanism audit is frozen in
`reports/afid_vla/mathematical_mechanism_audit.md` with decision
`AFID_MATHEMATICAL_AUDIT_PREREGISTERED`.

AFID preregistration is frozen in `reports/afid_vla/preregistration.md` with
decision `AFID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`.

AFID prototype protocol is frozen in
`reports/afid_vla/prototype_protocol.md` with decision
`AFID_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING`.

AFID Stage 0 implementation validation is complete with decision
`AFID_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY`.

Next action: launch the frozen AFID Stage 0 development audit after
worker-safety checks. No AFID Stage 0 launch, training, validation search,
rollout, or confirmatory-test access has happened.

## Epoch 4 Cycle 32 LCG-VLA

Cycle 32 completed the primary-source mechanism map in
`reports/epoch_4_cycle_32_prior_mechanism_map.md` and generated exactly three
candidates in `reports/epoch_4_cycle_32_candidate_generation.md`. S2C remains
closed as `S2C_STAGE_0_DATA_OR_SUPERVISION_FAILURE`; no S2C repair, rescue,
threshold change, proxy change, task change, or reinterpretation is allowed.

`LCG-VLA`, Language-Contrastive Guidance for Base-preserving SmolVLA actions,
is selected at `93 / 100`. Its closest positive prior is Counterfactual Action
Guidance (`https://arxiv.org/abs/2602.17659`), which reports improved
LIBERO-CF language-following and task success plus real-world counterfactual
failure reductions.

LCG's single mechanism is an identity-initialized language-contrast action-cell
gate around frozen SmolVLA Base chunks. It compares original-instruction
chunks with a legal language-null or counterfactual-language branch and permits
bounded edits only where language contrast predicts vision-shortcut risk. LoRA
may only parameterize the gate; it is not the scientific mechanism.

The first serious comparison is frozen at the design level to `smolvla_base`,
`counterfactual_action_guidance_proxy`, `lcg_full`,
`lcg_no_language_contrast_ablation`, and `standard_lora`.

The LCG-VLA Researcher A proposal is frozen in
`reports/lcg_vla/researcher_proposal.md` with SHA-256
`F0D980AA0760F143D781C723DB632BC324C1E18F390D9C33C5DA94F3A897D11E` and
decision `LCG_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`. No LCG implementation,
Stage 0 launch, validation search, training, rollout, simulator access, or
confirmatory-test tuning has happened at proposal-freeze time.

Reviewer B attack is complete in `reports/lcg_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. It preserves
CAG as the closest prior and policy 2, narrows novelty to a frozen-SmolVLA
Base-preserving learned language-contrast action-cell gate, treats the
null-instruction branch as a transparent local proxy, and requires contrast,
residual, and mask noncollapse before any validation search or rollout.

Researcher A rebuttal is complete in
`reports/lcg_vla/researcher_rebuttal.md` with decision
`LCG_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. It accepts all Reviewer B
conditions, keeps CAG as policy 2, defines the null branch as a transparent
proxy, and preserves `B_t - N_t` as a gate signal rather than a residual target.

The mathematical mechanism audit is frozen in
`reports/lcg_vla/mathematical_mechanism_audit.md` with decision
`LCG_MATHEMATICAL_AUDIT_PREREGISTERED`. It fixes `l_null = ""`, `H=50`,
`D=7`, language-contrast mask threshold `0.25`, action-group caps, the CAG
proxy formula, objective coefficients, Stage 0 stop classes, and no
deterministic-action KL.

Preregistration is frozen in `reports/lcg_vla/preregistration.md` with
decision `LCG_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`. It freezes
development tasks, discovery/validation demo IDs, Stage 0 artifacts and gates,
worker resume keys, and the six-configuration bounded validation envelope.

Prototype protocol is frozen in `reports/lcg_vla/prototype_protocol.md` with
decision `LCG_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING`. It
fixes `tca_map/smolvla/lcg_vla.py`, `scripts/run_lcg_vla_stage0.py`,
`tests/test_lcg_vla.py`, required Stage 0 artifacts, serializer preflight, and
worker-safety checks.

LCG Stage 0 implementation is validated as
`LCG_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY`: WSL py_compile passed,
focused LCG tests reported `6 passed`, and
`reports/lcg_vla/stage_0_serializer_preflight.json` has matching fixture and
reproduced hashes. The next step is worker-safety inspection and then LCG Stage
0 launch or adjudication without duplicate execution.

LCG Stage 0 completed as `LCG_STAGE_0_DESIGN_FAILURE` in
`reports/lcg_vla/stage_0_result.json`: `5120 / 5120` model rows, exception
count `0`, duplicate / missing / extra / split-overlap keys all `0`, and exact
key-set equality. The frozen design gates failed because the language gate
activated nearly everywhere (`0.99978125`) and `standard_lora_proxy` explained
or exceeded LCG (`lora_explains=true`). This is a development-only Stage 0
stop, not a closed-loop scientific kill. LCG may not be repaired or rescued.

Current next action: generate exactly three Epoch 4 Cycle 33 candidates under
current governance.

## Epoch 4 Cycle 31 S2C-VLA

Cycle 31 generated exactly three candidates in
`reports/epoch_4_cycle_31_candidate_generation.md` after the primary-source
mechanism map in `reports/epoch_4_cycle_31_prior_mechanism_map.md`. URF remains
closed as `URF_STAGE_0_NO_USABLE_HEADROOM`; no URF repair, rescue, threshold
change, proxy change, task change, or reinterpretation is allowed.

`S2C-VLA`, Seam-Supervised Chunk Consistency for Base-preserving SmolVLA
execution, is selected at `95 / 100`. Its closest positive prior is ChunkFlow
(`https://arxiv.org/html/2607.12992v1`, project page
`https://cytoderm-ai.github.io/chunkflow`), which reports `93.4%` LIBERO
long-horizon success with improved boundary jump, high-frequency energy,
smoothness metrics, and low-latency inference.

S2C's single mechanism is a Base-preserving learned overlap edit mask and
tail-anchored bridge for SmolVLA action-chunk boundary consistency. LoRA may
only be implementation infrastructure; the scientific method is
execution-indexed chunk consistency.

The first serious comparison is frozen at the design level to `smolvla_base`,
`chunkflow_overlap_proxy` or official ChunkFlow if installed, `s2c_full`,
`s2c_no_learned_overlap_mask_ablation`, and `standard_lora`.

The S2C-VLA Researcher A proposal is frozen in
`reports/s2c_vla/researcher_proposal.md` with SHA-256
`399A3960F9FF9AFA8EDA7C3F743A95C3FD4DC711644C2398630F1E68486DC5B3` and
decision `S2C_PROPOSAL_FROZEN_REVIEWER_ATTACK_COMPLETED`.

Reviewer B attack is complete in `reports/s2c_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. It narrows S2C
to a frozen-SmolVLA Base-preserving learned overlap edit layer, keeps ChunkFlow
as policy 2, keeps SEAM secondary, forbids expert future-tail inference, and
requires gripper/event protection plus standard LoRA control. No S2C training,
validation search, rollout, simulator access, or confirmatory-test tuning has
happened.

Researcher A rebuttal is complete in
`reports/s2c_vla/researcher_rebuttal.md` with decision
`S2C_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. It accepts all Reviewer B
conditions, defines the deployment previous-tail rule, and keeps URF plus all
previous closed methods closed.

The mathematical mechanism audit is frozen in
`reports/s2c_vla/mathematical_mechanism_audit.md` with decision
`S2C_MATHEMATICAL_AUDIT_PREREGISTERED`. It fixes `H=50`, stride `10`,
overlap `K=10`, objective coefficients, Stage 0 gates, and forbids
deterministic-action KL.

Preregistration is frozen in `reports/s2c_vla/preregistration.md` with
decision `S2C_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`. It freezes
development tasks, discovery/validation demo IDs, Stage 0 artifacts and gates,
bounded validation search limits, and worker resume keys.

Prototype protocol is frozen in `reports/s2c_vla/prototype_protocol.md` with
decision `S2C_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING`. It
fixes `tca_map/smolvla/s2c_vla.py`, `scripts/run_s2c_vla_stage0.py`,
`tests/test_s2c_vla.py`, required Stage 0 artifacts, serializer preflight, and
worker-safety checks.

S2C Stage 0 implementation is validated as
`S2C_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY`: WSL py_compile passed,
focused S2C tests reported `7 passed`, and
`reports/s2c_vla/stage_0_serializer_preflight.json` has matching fixture and
reproduced hashes.

S2C Stage 0 completed as `S2C_STAGE_0_DATA_OR_SUPERVISION_FAILURE` in
`reports/s2c_vla/stage_0_result.json`: `885 / 885` model rows, exception
count `0`, duplicate / missing / extra / split-overlap keys all `0`, and exact
key-set equality. It found `177` adjacent pairs, but failed the frozen data
coverage gate; secondary gates also failed no-headroom and mask/gripper
criteria. This is development-only, not a closed-loop scientific kill. S2C may
not be repaired or rescued.

Current next action: generate exactly three Cycle 32 candidates under current
governance.

## Epoch 4 Cycle 30 URF-VLA

Cycle 30 generated exactly three candidates in
`reports/epoch_4_cycle_30_candidate_generation.md` after the primary-source
mechanism map in `reports/epoch_4_cycle_30_prior_mechanism_map.md`. CCIF
remains closed as `CCIF_STAGE_0_DESIGN_FAILURE`; no CCIF repair, rescue,
threshold change, proxy change, task change, or reinterpretation is allowed.

`URF-VLA`, Uncertainty-Routed Residual Flow for Base-preserving SmolVLA
chunks, is selected at `92 / 100`. Its closest positive prior is SUREFlow
(`https://arxiv.org/abs/2607.10504`, official repository
`https://github.com/tanvirnwu/SUREFlow`), which reports `92.5%` average LIBERO
success and LIBERO-PRO robustness with a 179M uncertainty-aware residual-flow
VLA.

URF predicts a heteroscedastic residual-flow field around the frozen SmolVLA
Base action chunk and routes bounded residual transport only where predicted
residual uncertainty supports intervention. LoRA may only be implementation
infrastructure; the scientific method is uncertainty-routed residual transport.

The first serious comparison is frozen at the design level to exactly
`smolvla_base`, `sureflow_uncertainty_residual_proxy` or official `sureflow`
if installed, `urf_full`, `urf_no_uncertainty_route_ablation`, and
`standard_lora`.

The URF-VLA Researcher A proposal is frozen in
`reports/urf_vla/researcher_proposal.md` with SHA-256
`E78829E736C3F22451E72574092221904ACBE4C4BE0BDA7FA046832DABED3532` and
decision `URF_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`. No URF training,
validation search, rollout, simulator access, or confirmatory-test tuning has
happened.

Reviewer B attack is complete in `reports/urf_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. It preserves
SUREFlow as the closest prior, adds Guided Action Flow as the closest
frozen-SmolVLA action-intervention prior, treats flow-based and perturbation
uncertainty methods as uncertainty-signal alternatives, keeps
`urf_no_uncertainty_route_ablation` and `standard_lora` live, and narrows URF
to Base-preserving uncertainty-routed bounded residual transport around a
frozen SmolVLA chunk.

Researcher A rebuttal is complete in `reports/urf_vla/researcher_rebuttal.md`
with decision `URF_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. It accepts all
Reviewer B conditions, including the SUREFlow proxy, Guided Action Flow prior
position, flow/perturbation uncertainty alternatives, no-uncertainty ablation,
standard LoRA simple killer, Stage 0 disagreement diagnostics, monotonic
uncertainty strata gate, no global route gate, mathematical log-variance audit,
no deterministic-action KL, no privileged inference inputs, and no rescue of
CCIF/TSC/CFR/AMP/RAP/VDR.

The URF mathematical mechanism audit is frozen in
`reports/urf_vla/mathematical_mechanism_audit.md` with decision
`URF_MATHEMATICAL_AUDIT_PREREGISTERED`. It defines `[B,50,7]` residual and log
variance tensors, normalized residual scales, explicit uncertainty-dependent
route logits, heteroscedastic residual pseudo-NLL, route BCE, clean retention,
gradient and magnitude audits, uncertainty monotonicity diagnostics, no
deterministic-action KL, and the fixed Stage 0 stop classes.

The URF preregistration is frozen in `reports/urf_vla/preregistration.md` with
decision `URF_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`. It freezes
the discovery/validation/confirmatory partitions, fixed LIBERO development
tasks, residual and route construction, Stage 0 artifact paths, pass gates,
stop classes, bounded six-configuration validation search, first five-policy
comparison, and confirmatory tuning prohibition.

The executable URF prototype protocol is frozen in
`reports/urf_vla/prototype_protocol.md` with decision
`URF_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`. It fixes helper, runner, and
test paths; artifact paths; row keys; worker safety and missing-key-only resume
rules; required Stage 0 probes; result metrics; and the frozen decision order.

URF Stage 0 implementation is validated as
`URF_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY`: WSL py_compile passed,
focused URF tests reported `8 passed`, and
`reports/urf_vla/stage_0_serializer_preflight.json` has matching fixture and
reproduced hashes. Current next action is to run worker-safety checks and then
launch the frozen Stage 0 audit only if no live or completed URF worker exists.

URF Stage 0 then completed as `URF_STAGE_0_NO_USABLE_HEADROOM` in
`reports/urf_vla/stage_0_result.json`: `5120 / 5120` model rows, exception
count `0`, duplicate / missing / extra / split-overlap keys all `0`, and exact
key-set equality. Bounded validation is not allowed because Base residual
headroom was below the frozen `0.005` normalized Huber gate
(`0.0033407550543043956`) and the heteroscedastic residual proxy did not beat
homoscedastic or task/phase baselines. This is a development-only no-headroom
stop, not a closed-loop scientific kill; URF rescue is forbidden.

Current next action: generate exactly three Epoch 4 Cycle 31 candidates under
current governance.

## Epoch 4 Cycle 29 CCIF-VLA

Cycle 29 generated exactly three candidates in
`reports/epoch_4_cycle_29_candidate_generation.md` after the primary-source
mechanism map in `reports/epoch_4_cycle_29_prior_mechanism_map.md`. TSC remains
closed as `TSC_STAGE_0_NO_USABLE_HEADROOM`; no TSC repair, rescue, threshold
change, proxy change, task change, or reinterpretation is allowed.

`CCIF-VLA`, Continuous Coarse Intent Field for base-preserving VLA chunks, is
selected at `92 / 100`. Its closest positive prior is Coarse-to-Control
(`https://arxiv.org/abs/2606.07107`), which reports `97.9%` average LIBERO
success and uses action-token planning before executable action generation.

CCIF predicts a deployment-observable continuous coarse motor-intent field from
current inputs and the Base decoded chunk, then conditions a bounded
identity-preserving residual action field on that intent. LoRA may only be
implementation infrastructure; the scientific method is continuous coarse
intent field conditioning.

The first serious comparison is frozen at the design level to exactly
`smolvla_base`, `coarse_to_control_continuous_proxy`, `ccif_full`,
`ccif_no_coarse_intent_ablation`, and `standard_lora`.

The CCIF-VLA Researcher A proposal is frozen in
`reports/ccif_vla/researcher_proposal.md` with SHA-256
`2AFC40F050FD7F0D28507344358CBCB70BF27CC901C57474A501D3EB87E7FAA1` and
decision `CCIF_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`. No CCIF training,
validation search, rollout, simulator access, or confirmatory-test tuning has
happened.

Reviewer B attack is complete in `reports/ccif_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. It narrows CCIF
novelty to Base-preserving continuous coarse-intent residual constraint, keeps
`coarse_to_control_continuous_proxy` as policy 2, requires that proxy not be a
strawman, keeps `ccif_no_coarse_intent_ablation` and `standard_lora` live,
requires task/phase mean and endpoint-only intent diagnostics, freezes intent
construction before implementation, and forbids privileged inference inputs.

Researcher A rebuttal is complete in
`reports/ccif_vla/researcher_rebuttal.md` with decision
`CCIF_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. It accepts narrowed novelty,
the Coarse-to-Control proxy, the no-intent ablation, standard LoRA, cheap
intent diagnostics, intent-construction freeze, identity-preserving reload,
and no privileged inference.

The CCIF mathematical mechanism audit is frozen in
`reports/ccif_vla/mathematical_mechanism_audit.md` with decision
`CCIF_MATHEMATICAL_AUDIT_PREREGISTERED`. It freezes the `m = 31` coarse-intent
vector, waypoint indices `[9, 19, 34, 49]`, intent template, residual/gate
formula, objective terms and gradient paths, no deterministic-action KL,
Coarse-to-Control proxy requirements, no-intent ablation, cheap intent
diagnostics, identity-preserving integration, and Stage 0 stop classes.

The CCIF preregistration is frozen in `reports/ccif_vla/preregistration.md`
with decision `CCIF_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`. It
freezes evidence partitions, fixed source tasks and demonstrations, the
`m = 31` intent definition, Stage 0 artifacts, pass gates, stop classes,
bounded validation-search budget, first comparison policies, and confirmatory
discipline.

The executable CCIF prototype protocol is frozen in
`reports/ccif_vla/prototype_protocol.md` with decision
`CCIF_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`. It freezes the Stage 0 runner
contract, artifacts, worker/resume safety, fixed source data, policy/probe
rows, mechanism constants, required metrics, and decision rule.

CCIF Stage 0 implementation is now validated with helper
`tca_map/smolvla/ccif_vla.py`, runner `scripts/run_ccif_vla_stage0.py`, and
tests `tests/test_ccif_vla.py`. WSL py_compile passed, focused CCIF tests
reported `9 passed`, and serializer preflight wrote
`reports/ccif_vla/stage_0_serializer_preflight.json` with hash
`86D9BC43FC697ECFDEA8C7F12FE0DF98CA5877E74B9D7952D2466787E8998FFE`.
No CCIF training, validation search, rollout, simulator access, or
confirmatory-test tuning has happened.

CCIF Stage 0 completed in `reports/ccif_vla/stage_0_result.json` with decision
`CCIF_STAGE_0_DESIGN_FAILURE`. The fixed audit completed `4480 / 4480` model
rows over `640` unique observation rows, with duplicate/missing/extra key
counts all `0`, final exception count `0`, and two repaired resume blocker
exceptions recorded separately. The deployment intent probe failed to beat the
task/phase mean (`-0.5813931486058512` relative Huber improvement), and the
endpoint-only diagnostic explained the signal. Bounded validation is not
allowed. This is a development-only design failure, not a closed-loop
scientific kill; no rollout, simulator load, training, validation search, or
confirmatory-test access occurred. CCIF may not be rescued under this method
cycle.

Current next action: begin Epoch 4 Cycle 30 candidate search under the current
positive-prior governance.

## Epoch 4 Cycle 28 TSC-VLA

Cycle 28 generated exactly three candidates in
`reports/epoch_4_cycle_28_candidate_generation.md` after the primary-source
mechanism map in `reports/epoch_4_cycle_28_prior_mechanism_map.md`. CFR remains
closed as `CFR_STAGE_0_NO_USABLE_HEADROOM`; no CFR repair, rescue, threshold
change, or reinterpretation is allowed.

`TSC-VLA`, Temporal-Spatial masked action completion for continuous VLA chunks,
is selected at `91 / 100`. Its closest positive prior is TS-Mask VLA
(`https://arxiv.org/abs/2607.09818`), which reports a Discrete Diffusion Action
Expert, Bridge Attention, and 2D temporal-spatial action-token masking with
`95.7%` average LIBERO success and CALVIN average sequence length `4.19`.

TSC predicts a deployment-observable sparse time-dimension action-cell error
mask over the Base decoded `[50,7]` SmolVLA chunk and runs continuous masked
completion that changes only selected cells while clamping all unselected cells
exactly to Base. LoRA is only implementation infrastructure.

The first serious comparison is frozen at the design level to exactly
`smolvla_base`, `ts_mask_continuous_proxy` or official `ts_mask_vla` if
installed, `tsc_full`, `tsc_no_targeted_mask_ablation`, and `standard_lora`.
No TSC training, validation search, rollout, simulator access, or
confirmatory-test tuning has happened.

The TSC-VLA Researcher A proposal is frozen in
`reports/tsc_vla/researcher_proposal.md` with SHA-256
`0DF143D2D8773D7ABF4FC76AB7CC083FE7EE65DF84EA06631E67C2445F6DC941` and
decision `TSC_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`.

Reviewer B attack is complete in `reports/tsc_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. It narrows TSC
novelty to Base-clamped continuous action-cell completion, requires a faithful
TS-Mask proxy as policy 2, keeps `tsc_no_targeted_mask_ablation` and
`standard_lora` live, requires threshold-freeze and simple residual-gate
diagnostics, and forbids privileged inference inputs.

Researcher A rebuttal is complete in
`reports/tsc_vla/researcher_rebuttal.md` with decision
`TSC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. It accepts all Reviewer B
conditions before mathematical audit.

The TSC mathematical mechanism audit is frozen in
`reports/tsc_vla/mathematical_mechanism_audit.md` with decision
`TSC_MATHEMATICAL_AUDIT_PREREGISTERED`. It freezes variables, tensor shapes,
mask-label construction, Base-clamped action formula, objective terms,
gradient paths, no deterministic-action KL, TS-Mask proxy requirements,
required ablations, and Stage 0 stop classes.

The TSC preregistration is frozen in `reports/tsc_vla/preregistration.md` with
decision `TSC_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`. It freezes
discovery/validation/confirmatory separation, fixed Stage 0 data construction,
label thresholds, required outputs, pass/stop classes, bounded validation search
budget, and first closed-loop comparison policy.

The executable TSC prototype protocol is frozen in
`reports/tsc_vla/prototype_protocol.md` with decision
`TSC_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`. It freezes the Stage 0 runner
contract, artifacts, worker/resume safety, fixed data sources, action
semantics, label construction, TS-Mask proxy definitions, mechanism audits,
pass gates, and stop classes.

The TSC Stage 0 runner is implemented in `scripts/run_tsc_vla_stage0.py` with
helper module `tca_map/smolvla/tsc_vla.py`. Focused TSC tests pass
(`8 passed`), `py_compile` passes, and the serializer preflight is persisted in
`reports/tsc_vla/stage_0_serializer_preflight.json`. No TSC training,
validation search, rollout, simulator access, or confirmatory-test tuning has
happened.

Next action: check existing TSC Stage 0 PID, heartbeat/status, partial, result,
logs, and exit-code artifacts before launching or adjudicating Stage 0 under
the frozen resume rules.

TSC Stage 0 completed under the frozen protocol in
`reports/tsc_vla/stage_0_result.json` with decision
`TSC_STAGE_0_NO_USABLE_HEADROOM`. The worker completed `640 / 640` planned
development rows with exception count `0`, exit code `0`, duplicate manifest
keys `0`, duplicate partial keys `0`, missing keys `0`, extra keys `0`,
split-overlap keys `0`, and exact manifest/partial key-set equality.

The no-headroom stop is development-only, not a closed-loop scientific kill.
TSC full improved validation Huber only weakly over the TS-Mask proxy
(`0.010570183991642059` relative, `0.00018146866814880425` absolute Huber) and
over the no-targeted-mask ablation (`0.01502019039472744` relative,
`0.00019155778220941375` absolute Huber), below the frozen `5%` relative or
`0.005` absolute gate. The structured mask probe also lost to both baselines:
structured BCE `3.845927165246039`, trivial BCE `0.5102969729338862`, and
magnitude-only BCE `0.7535353061465139`.

Bounded validation, Stage A rollout, threshold repair, task change, proxy
change, or TSC rescue is forbidden. Next action: generate exactly three Epoch 4
Cycle 29 candidates.

## Epoch 4 Cycle 27 CFR-VLA

Cycle 27 generated exactly three candidates in
`reports/epoch_4_cycle_27_candidate_generation.md` after the primary-source
mechanism map in `reports/epoch_4_cycle_27_prior_mechanism_map.md`.

`CFR-VLA`, Continuous Full-Chunk Refinement for VLA action-flow decoding, is
selected at `92 / 100`. Its closest positive prior is DFM-VLA
(`https://arxiv.org/html/2603.26320v1`) with project page
`https://chris1220313648.github.io/DFM-VLA/`.

CFR learns a bounded continuous residual velocity/refinement field over the
full `[50,7]` SmolVLA action chunk. Starting from a Base decoded chunk, it
applies fixed iterative full-chunk refinement before execution. LoRA may only
be identity-preserving implementation infrastructure; the scientific mechanism
is iterative continuous full-chunk refinement, not LoRA, not adaptive chunk
size, and not AMP-style action-manifold projection.

The first serious comparison is frozen at the design level to exactly
`smolvla_base`, `dfm_vla_continuous_refinement_proxy` or official `dfm_vla` if
installed, `cfr_full`, `cfr_no_iterative_refinement`, and `standard_lora`.

The CFR-VLA Researcher A proposal is frozen in
`reports/cfr_vla/researcher_proposal.md` with SHA-256
`9E2FC510B2D97C869F18BE6C5B339CE034DD98223802078358320AA8BEF3D0AE` and
decision `CFR_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`. No CFR training,
validation search, rollout, simulator access, or confirmatory-test tuning has
happened. AMP remains closed unchanged as
`AMP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`; no AMP repair, rescue,
threshold change, clipping, or reinterpretation is allowed.

Reviewer B attack is complete in `reports/cfr_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. Required
conditions keep the DFM proxy or official DFM-VLA as policy 2, narrow novelty
to continuous Base-start identity-preserving refinement, keep
`cfr_no_iterative_refinement` and `standard_lora` live, require official action
validity semantics before Stage 0, require residual/headroom health, require
mathematical objective and gradient-scale audits, and forbid privileged
inference inputs.

Researcher A rebuttal is complete in `reports/cfr_vla/researcher_rebuttal.md`
with decision `CFR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. It accepts narrowed
novelty, DFM proxy status, key ablation, standard-LoRA simple killer, official
action-validity semantics, data/headroom gates, mathematical audit
requirements, and no privileged inference inputs.

The CFR mathematical mechanism audit is frozen in
`reports/cfr_vla/mathematical_mechanism_audit.md` with decision
`CFR_MATHEMATICAL_AUDIT_PREREGISTERED`. It freezes variables and tensor shapes,
continuous refinement dynamics, Huber/vector-field objectives, gradient paths,
loss-scale audit, official action-validity semantics, no deterministic-action
KL, DFM proxy policy, key ablation, simple killer baseline, and Stage 0 stop
classes.

The CFR preregistration is frozen in `reports/cfr_vla/preregistration.md` with
decision `CFR_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`. It freezes
discovery/validation/confirmatory partitions, fixed development tasks, Stage 0
audit outputs, stop classes, bounded six-configuration validation search, first
five-policy comparison, Stage A/B policy, and confirmatory tuning prohibition.

The executable prototype protocol is frozen in
`reports/cfr_vla/prototype_protocol.md` with decision
`CFR_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`. It freezes the Stage 0 runner
contract, artifact paths, worker/resume safety, data sources, action semantics,
mechanism audits, proxy definitions, hard pass gates, and stop classes.

Next action: implement and validate `scripts/run_cfr_vla_stage0.py` before
execution.

The CFR Stage 0 runner is now implemented in `scripts/run_cfr_vla_stage0.py`
with helper module `tca_map/smolvla/cfr_vla.py`. The focused CFR runner/helper
tests in `tests/test_cfr_vla.py` pass (`8` tests), and no CFR training,
validation search, rollout, simulator access, or confirmatory-test tuning has
happened. Next action: check existing CFR Stage 0 PID/status/heartbeat,
partial/result/log/exit-code artifacts, then launch, monitor, resume, or
adjudicate strictly according to the frozen worker/resume rules.

CFR Stage 0 completed from Linux worker PID `310` with exit code `0`.
The fixed protocol produced `640 / 640` rows, exception count `0`, duplicate
manifest keys `0`, duplicate partial keys `0`, missing manifest keys `0`, extra
partial keys `0`, split overlap keys `0`, and key sets equal `true`.
The final decision is `CFR_STAGE_0_NO_USABLE_HEADROOM`: CFR residual-probe
relative/absolute Huber gain was `-6.04941221711208 / -0.11968147462337628`;
CFR-minus-DFM-proxy headroom was
`-6.068176722319228 / -0.11975307303185317`. The result is a development-only
no-headroom stop, not a closed-loop scientific kill. Bounded validation,
Stage A, rescue, and threshold/method changes are disallowed.

Next action: generate exactly three Epoch 4 Cycle 28 candidates; do not repair
or rescue CFR-VLA.

## Epoch 4 Cycle 26 AMP-VLA

Cycle 26 generated exactly three candidates in
`reports/epoch_4_cycle_26_candidate_generation.md` after the primary-source
mechanism map in `reports/epoch_4_cycle_26_prior_mechanism_map.md`.

`AMP-VLA`, Action-Manifold Projection for VLA action-flow adaptation, is
selected at `95 / 100`. Its closest positive prior is ABot-M0
(`https://arxiv.org/abs/2602.11236`) with official repository
`https://github.com/amap-cvlab/ABot-Manipulation`.

AMP learns a discovery-only low-dimensional action manifold over LIBERO action
chunks and constrains SmolVLA adaptation through an identity-preserving
projection or bounded gated residual. LoRA is only implementation
infrastructure. The first serious comparison is exactly `smolvla_base`,
`abot_m0_action_manifold_proxy`, `amp_full`,
`amp_no_manifold_projection`, and `standard_lora`.

No AMP closed-loop rollout, simulator access, or confirmatory-test tuning has happened.

The AMP-VLA Researcher A proposal is frozen in
`reports/amp_vla/researcher_proposal.md` with SHA-256
`67ACC693C706B76BC9FB84F9E59BA3DF9C0463A0BAFABE539312D0E232DFE9A4`
and decision `AMP_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`.
Reviewer B attack is complete in `reports/amp_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. Next action:
Researcher A rebuttal is complete in `reports/amp_vla/researcher_rebuttal.md`
with decision `AMP_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. The mathematical
mechanism audit is frozen in `reports/amp_vla/mathematical_mechanism_audit.md`
with decision `AMP_MATHEMATICAL_AUDIT_PREREGISTERED`. Preregistration is
frozen in `reports/amp_vla/preregistration.md` with decision
`AMP_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`. The executable
prototype protocol is frozen in `reports/amp_vla/prototype_protocol.md` with
decision `AMP_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`.

AMP Stage 0 completed under the frozen protocol in
`reports/amp_vla/stage_0_result.json`. Worker PID `379` completed `1280 /
1280` planned development rows with exception count `0`, exact
manifest/partial key equality, duplicate partial keys `0`, missing keys `0`,
extra keys `0`, and split-overlap keys `0`. No simulator rollout, reward,
success, done, confirmatory identity access, training, validation search, or
confirmatory-test tuning occurred.

The final Stage 0 decision is
`AMP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`. The development audit
found healthy manifest/source/feature alignment, positive manifold coordinate
variance, distinct projection paths, finite nonzero AMP gradients, base hash
retention, and exact identity/reload (`0.0` max error). The method still failed
the frozen Stage 0 gates because postprocessed Base action validity was false
(`base_action_in_bounds = false`, so `action_validity_ok = false`) and the
deployment-input coordinate/headroom probes were negative: coordinate relative
improvement `-4.947553385520279`, AMP-minus-ABot-proxy relative improvement
`-2.663165575108502`. This is a development-only implementation/optimization
stop, not a closed-loop scientific kill. Bounded validation is not allowed.
Next action: generate exactly three Cycle 27 candidates without AMP repair or
rescue.

## Epoch 4 Cycle 25 RAP-VLA

Cycle 25 generated exactly three candidates in
`reports/epoch_4_cycle_25_candidate_generation.md` after the primary-source
map in `reports/epoch_4_cycle_25_prior_mechanism_map.md`.

`RAP-VLA`, Retrieval-Anchored Prior residualization for VLA action flows, is
selected at `94 / 100`. Its closest positive prior is OptimusVLA
(`https://arxiv.org/abs/2602.20200`) with official repository
`https://github.com/iLearn-Lab/CVPR26-OptimusVLA`. RAP uses retrieved legal
demonstration action anchors from deployment-observable current features, then
learns a bounded residual action-flow path around the anchor. LoRA is only the
identity-preserving implementation scaffold for the residual/gate path.

The first serious comparison is exactly Base, transparent OptimusVLA memory
prior proxy, RAP full, anchor-only/no-residual ablation, and matched standard
LoRA. The RAP Researcher A proposal is frozen in
`reports/rap_vla/researcher_proposal.md` with SHA-256
`E9C3672544E486E4D5BAA883917F8429DB0FB36982F3F5944AC26A85783D1008`. No
training, validation search, rollout, simulator access, or confirmatory-test
tuning has happened for RAP. Reviewer B attack is complete in
`reports/rap_vla/reviewer_attack.md` with decision
`REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. Researcher A rebuttal is
complete in `reports/rap_vla/researcher_rebuttal.md` with decision
`RAP_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. The mathematical mechanism audit is
frozen in `reports/rap_vla/mathematical_mechanism_audit.md` with decision
`RAP_MATHEMATICAL_AUDIT_PREREGISTERED`. Preregistration is frozen in
`reports/rap_vla/preregistration.md` with decision
`RAP_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`. The prototype
protocol is frozen in `reports/rap_vla/prototype_protocol.md` with decision
`RAP_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`.

RAP Stage 0 completed under the frozen protocol in
`reports/rap_vla/stage_0_result.json`. Worker PID `287` completed `640 / 640`
planned development rows with exception count `0`, exact manifest/partial key
equality, duplicate partial keys `0`, missing keys `0`, extra keys `0`, and
split-overlap keys `0`. The OptimusVLA comparison status is fixed as
`optimusvla_memory_prior_proxy`.

The fixed Stage 0 decision is
`RAP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`. This is not a
closed-loop scientific kill: no training, validation search, rollout,
simulator access, or confirmatory-test tuning occurred. The blocking hard
gate is postprocessed action validity (`action_validity_ok = false`,
`base_action_in_bounds = false`). Retrieval-anchor headroom was positive
(`0.23865551292280293` relative MSE improvement), but the residual probe
failed (`-3.830674623085068` relative improvement and
`-0.07385385182729762` absolute Huber improvement). Bounded validation,
RAP rerun, repair, rescue, clipping, threshold changes, and reinterpretation
are forbidden. RAP remains closed while Cycle 26 proceeds with AMP-VLA.

## Epoch 4 Cycle 24 VDR-VLA

Cycle 24 generated exactly three candidates and selected `VDR-VLA` at
`92 / 100`, anchored to FutureVLA. Proposal SHA-256:
`0229EBC15901F4FE1EDD3839AB6B984AFA3E0E99836B5C88CF21F2C7DE2B3E72`.

VDR's scientific method is dynamic future-feature residual alignment: subtract
a discovery-fitted actionless static future-feature predictor, then supervise
the generated-action-conditioned prediction of the remaining residual. Rank-4
LoRA or an equivalent zero-effect adapter is only implementation
infrastructure. The first comparison is Base, transparent FutureVLA proxy, VDR
full, no-action-residual ablation, and standard LoRA.

The complete proposal/review/rebuttal/math/preregistration/protocol package is
frozen under `reports/vdr_vla/`. VDR Stage 0A worker PID `411` completed
`1536 / 1536` planned development rows with runner exception count `0`, exact
manifest/partial key equality, duplicate partial keys `0`, missing keys `0`,
extra keys `0`, and split-overlap keys `0`.

The fixed Stage 0A decision is
`VDR_STAGE_0A_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`. This is a
development-only implementation/optimization failure, not a closed-loop
scientific kill: no training, validation search, rollout, simulator access,
or confirmatory-test tuning occurred. Stage 0B, VDR rerun, repair, rescue,
threshold changes, clipping, and reinterpretation are forbidden.

## Epoch 4 Cycle 22 HASTE-VLA

Cycle 22 generated exactly three candidates and selected `HASTE-VLA` at
`95 / 100`, anchored to StaKe. Proposal hash:
`5415BC1533A24EC55CC511DDEB014BB11D9C19F603C59D1F1D3E151E15B930A6`.

The review, rebuttal, mathematical audit, preregistration, and prototype
protocol are frozen. HASTE supervises a censored time-to-command-event hazard
and relative cumulative arm displacement during training while retaining the
ordinary inference path. Runner commit `3dd76f0` passed unit, real-checkpoint
interface, and zero-effect identity smokes. Frozen execution PID `295` exited
`1` before manifest persistence on NumPy JSON serialization. It produced no
partial, model row, feature cache, adapter, simulator row, or confirmatory
access. Decision: `HASTE_STAGE_0A_IMPLEMENTATION_FAILURE`. Do not repair,
resume, rerun, or rescue HASTE; continue to Cycle 23.

## Epoch 4 Cycle 23 KITE-VLA

Cycle 23 generated exactly three candidates and selected `KITE-VLA` at
`96 / 100`, anchored to GeoPredict. Proposal SHA-256:
`FA00DE56D14E4C69388BE1642F7D52153841D58E77FD5A3F5C68B6C624A152B8`.
KITE fits discovery-only empirical action-to-state realization operators and
backpropagates multi-horizon future-state error through generated clean
actions. Review, rebuttal, mathematical audit, preregistration, and prototype
protocol are frozen. Runner commit `62dbb75` passed the structured serializer,
full data/operator, real-checkpoint action-path gradient, and zero-effect
save/reload identity smokes. Stage 0A is ready for foreground serializer
preflight and one detached launch.

Stage 0A completed `128 / 128` model rows after a missing-key-only resume from
115 rows. Integrity and all 64 feature-cache hashes pass, while the original
atomic persistence exception remains recorded. The frozen action-validity
gate independently failed on all 128 reconstructed rows, with maximum absolute
action `1.1056011915206909`. Final decision:
`KITE_STAGE_0A_IMPLEMENTATION_FAILURE`. This is not a scientific kill. Do not
run Stage 0B, rerun, or rescue KITE; continue to Cycle 24.

## Epoch 4 Cycle 21 HEST-VLA

Cycle 21 generated exactly three candidates and selected `HEST-VLA` at
`93 / 100`, anchored to Spline Policy. Proposal hash:
`E56B4717BDF949E1A4371457058DFC662E0D79C70D9E2FBEF35A5415FD0F0527`.
The review, rebuttal, mathematical audit, preregistration, and prototype
protocol are frozen. Stage 0A completed `160 / 160` action windows with zero
exceptions and exact manifest and persistence integrity. Endpoint, event,
acting, energy, and comparator-distinction gates passed, but the frozen
all-variant support gate failed because one validation Base row and HEST's
whole-Base fallback were outside discovery-defined support.

Decision: `HEST_STAGE_0A_IMPLEMENTATION_FAILURE`. This is a pre-rollout
implementation/prototype support failure, not a scientific kill. Stage 0B,
rerun, repair, and HEST rescue are forbidden. Continue to Cycle 22.

## Epoch 4 Cycle 20 NICE-VLA

Cycle 20 generated exactly three prior-anchored candidates and selected
`NICE-VLA` at `96 / 100`, with VLA-Corrector as the closest positive prior.
The proposal hash is
`898BA577B38966D877E3EEC724EB98751BD8C2685CCD0BBA620EB6B6B9598C0A`.
The review, rebuttal, mathematical audit, preregistration, and prototype
protocol are frozen. Only the 128-pair discovery-only Stage 0A audit is
authorized; validation and confirmatory records remain sealed.

## Epoch 4 Cycle 19 SPARC-VLA

Cycle 19 selected `SPARC-VLA` from exactly three candidates at `96 / 100`,
with COAST as the closest positive prior. Proposal hash:
`CC2F9ACCE2A26EC438C58F2854ADC95134354C245CAD8ED961D29A895DBC697D`.

After one preserved capture-reset failure and the single allowed
implementation repair, final PID `306` completed `2 / 2` Stage 0A smoke rows
with exit `0`, zero exceptions, and zero duplicate, missing, or extra indices.
The hook acted and all Base identity/reload checks passed. Both synthetic
action rows failed the frozen Base-relative range-safety gate.

The decision is
`SPARC_STAGE_0A_IMPLEMENTATION_OR_PROTOTYPE_ACTION_VALIDITY_FAILURE_NO_SCIENTIFIC_KILL`.
No labeled fit, validation, rollout, or confirmatory evaluation occurred.
Stage 0B and SPARC rescue are forbidden. Continue to Cycle 20.

## Corrected Epoch 1 Result

Cycle 1 `DICD-VLA`:

- corrected status: `UNDERPOWERED_STAGE_A_NON_GO_ARCHIVED`
- full: `1 / 10`
- direct chunk-index delay: `2 / 10`
- no-history ablation: `1 / 10`
- ruling: do not rerun or rescue the current formulation; do not treat a one-episode difference at 10 episodes per policy as a permanent scientific family kill.

Cycle 2 `FEDO-VLA`:

- corrected status: `VALID_CURRENT_FORMULATION_KILL`
- faulted full: `1 / 10`
- static inverse gain: `2 / 10`
- APEX-style feedback proxy: `2 / 10`
- no-feedback ablation: `2 / 10`
- clean frozen: `4 / 10`
- clean FEDO: `0 / 10`
- ruling: do not revive the current formulation.

Cycle 3 `GCAP-VLA`:

- corrected status: `UNDERPOWERED_TARGET_AXIS_NON_GO_ARCHIVED`
- occluded full: `3 / 10`
- occluded frozen: `4 / 10`
- Sobel edge boost: `5 / 10`
- no-temporal ablation: `4 / 10`
- clean frozen: `1 / 10`
- clean GCAP: `5 / 10`
- ruling: do not rerun or rescue the current formulation; do not call the whole perception-repair family dead.

## Epoch 2 Result

Epoch 2 Cycle 1 `PTC-VLA` is archived as `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`.

Epoch 2 Cycle 2 `SACF-VLA` is archived as `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`.

Epoch 2 Cycle 3 `OCFN-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: expanded Stage B completed `400 / 400` total episodes with zero exceptions, `80` paired episodes per key policy, active mechanism, OCFN full `26 / 80`, zero-noise SmolVLA `27 / 80`, and paired upper confidence bound versus the strongest baseline `0.0625`.

These three related failures are synthesized in `reports/epoch_2_failure_synthesis.md`.

## Next Action

Epoch 3 Cycle 1 `CBFD-VLA` is archived as `STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE`: Stage A completed `50 / 50` held-out episodes with zero exceptions, frozen SmolVLA reached `7 / 10`, and full CBFD reached `0 / 10` with active mechanism.

Epoch 3 Cycle 2 `SCVC-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: full SCVC reached `11 / 40`, shifted frozen SmolVLA reached `20 / 40`, and the paired bootstrap CI versus shifted frozen was `[-0.425, -0.025]`.

Epoch 3 Cycle 3 `PSE-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: expanded Stage B completed `400 / 400` rows with zero exceptions, full PSE reached `50 / 80`, bright-single reached `51 / 80`, and the paired CI versus bright-single was `[-0.1000, 0.0750]`.

The related Epoch 3 failures are synthesized in `reports/epoch_3_failure_synthesis.md`.

Epoch 4 Cycle 1 `RCV-VLA` is archived as `STAGE_2B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: Stage 2B completed `200 / 200` episodes with zero exceptions, full RCV reached `20 / 40`, no-context ablation reached `24 / 40`, and stateless first-action reached `24 / 40`.

Epoch 4 Cycle 2 `CAVM-VLA` is archived as `STAGE_2B_EXPANDED_NON_GO_NO_THIRD_EXPANSION`: the expanded result completed `290 / 290` rows with zero exceptions, full CAVM reached `24 / 58`, nearest-success replay reached `23 / 58`, frozen SmolVLA reached `22 / 58`, success-only memory proxy reached `20 / 58`, and no-contrast ablation reached `21 / 58`.

## Epoch 4 Cycle 3

`FANG-VLA` is selected and preregistered as an AFIL-anchored identity-preserving failure-aware action-field guidance method.

Artifacts:

- `reports/epoch_4_cycle_3_prior_mechanism_map.md`
- `reports/epoch_4_cycle_3_candidate_generation.md`
- `reports/fang_vla/researcher_proposal.md`
- `reports/fang_vla/proposal_hash.txt`
- `reports/fang_vla/reviewer_attack.md`
- `reports/fang_vla/researcher_rebuttal.md`
- `reports/fang_vla/mathematical_mechanism_audit.md`
- `reports/fang_vla/preregistration.md`
- `reports/fang_vla/prototype_protocol.md`

Development audit passed and the calibrated validation search selected `fang_c01`. The uncalibrated gate failure is preserved as a negative validation result. Stage A completed `50 / 50` episodes with all policies tied at `3 / 10`.

Stage B completed `200 / 200` episodes with zero exceptions. Full FANG reached `11 / 40`, while frozen SmolVLA reached `16 / 40`, AFIL local proxy reached `15 / 40`, nearest-success replay reached `14 / 40`, and the no-failure ablation also reached `11 / 40`. The paired full-minus-base delta was `-0.125` with CI `[-0.250, 0.000]`; full was exactly tied with the key ablation.

Final FANG decision: `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT`. Do not rescue this formulation.

## Epoch 4 Cycle 4

`EvoState-VLA` is selected and preregistered as an EvoScene/DREAM-anchored action-evolved state guidance method.

Artifacts:

- `reports/epoch_4_cycle_4_prior_mechanism_map.md`
- `reports/epoch_4_cycle_4_candidate_generation.md`
- `reports/evostate_vla/researcher_proposal.md`
- `reports/evostate_vla/proposal_hash.txt`
- `reports/evostate_vla/reviewer_attack.md`
- `reports/evostate_vla/researcher_rebuttal.md`
- `reports/evostate_vla/mathematical_mechanism_audit.md`
- `reports/evostate_vla/preregistration.md`
- `reports/evostate_vla/prototype_protocol.md`

Stage 0 development audit completed without closed-loop rollout and stopped as `AUDIT_STOP_DESIGN_FAILURE`. The full transition model improved over a constant predictor by `0.715309`, but improved only `0.024689` over an actionless model, below the preregistered `0.05` action-input improvement threshold. This means the proposed action-conditioned state mechanism is not sufficiently supported for rollout.

## Epoch 4 Cycle 5

`RAC-VLA` is selected and preregistered as a Reflective VLA-anchored action-consequence calibration method for frozen SmolVLA under controlled deployment action-channel shift.

Stage 0 development audit passed with `10769` consequence pairs, `53685` labeled examples, zero duplicate perturbation keys, full validation accuracy `0.585745`, action-only validation accuracy `0.368496`, no-consequence validation accuracy `0.374483`, and full-vs-best-baseline margin `0.211262`.

The bounded six-config validation search selected `rac_h4_a0.05`: history horizon `4`, residual alpha `0.05`, score `0.508926`, full validation accuracy `0.603250`, and full-vs-best-baseline margin `0.244397`.

Stage A completed `50 / 50` episodes with zero exceptions under the frozen hidden `x_attenuate` action-channel shift. RAC full reached `0 / 10`, frozen shifted Base reached `0 / 10`, the no-consequence ablation reached `0 / 10`, the Reflective-history proxy reached `1 / 10`, and the online diagonal inverse-gain baseline reached `1 / 10`. RAC full tied Base and the key ablation, lost by only `1 / 10` to the strongest baseline and simple baseline, and did not satisfy any permanent Stage A kill criterion.

Stage B completed `200 / 200` episodes with zero exceptions and a valid shared task/reset manifest: `200` unique `(variant, task, identity)` keys, duplicate keys `0`, `40` episodes per variant, and identical paired manifests. RAC full reached `1 / 40`; shifted Base reached `1 / 40`; the Reflective-history proxy reached `1 / 40`; the no-consequence ablation reached `2 / 40`; and the online diagonal inverse-gain simple baseline reached `2 / 40`. RAC full tied Base and the closest-prior proxy, but lost to the key ablation and simple baseline.

Final RAC decision: `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT`. Do not rescue RAC, retune `rac_h4_a0.05`, change the hidden shift, or reinterpret the closed result.

## Post-RAC Governance

The post-RAC performance-oriented governance is installed in `reports/current_research_governance.md`, `AGENTS.md`, and `reports/codex_delegation_manual.md`. Future method cycles must maximize the probability of an honest paper-worthy positive result by using positive-prior anchors, usable-headroom audits, data/supervision health gates, identity-preserving integration, bounded development search, mathematical objective engineering, mechanism smoke, and frozen confirmatory tests.

## Epoch 4 Cycle 6

`MTF-VLA` is selected and preregistered as a FrameSkip and StructVLA anchored milestone-transition data-supervision method for identity-preserving SmolVLA adapter training.

Stage 0 development audit passed without training or closed-loop rollout using `reports/official_smolvla_stable_prediction_artifact.json`: `1600` development records, `1200` reserved test records not used, `40` task keys, duplicate sample keys `0`, duplicate frame keys `0`, high-low score gap `0.585702`, gripper-transition fraction `0.341875`, and adapter-init action delta p95 `0.0`.

The bounded six-config validation search selected `mtf_r20_ret100`: retained high-frame ratio `0.20`, retention coefficient `1.00`, validation score `0.643663`, `176` high train frames, and `391` base-retention train frames. The selected config and training manifest are frozen under `reports/mtf_vla/`.

Current stage: adapter training pending. Stage A must not start until disk-reloadable checkpoints exist for MTF full, no-retention ablation, FrameSkip proxy, and uniform retained-ratio LoRA.


The MTF adapter-training runner is implemented and dry-run validated in `scripts/run_mtf_vla_adapter_training.py`. The real selected-training manifest joins cleanly with the official split and stable prediction artifact: MTF full has `567` training events (`176` milestone, `391` frozen-base retention), no-retention ablation has `176`, the FrameSkip proxy has `176`, and uniform retained-ratio LoRA has `240`. Train/validation/test frame overlap is `0 / 0 / 0`; validation remains `400` frames; no training or closed-loop rollout happened in this dry run.

Current stage: adapter-training runner validated. Stage A must still not start until disk-reloadable checkpoints are trained and disk-reload verified for all four trainable policies.

Adapter training completed after the runner dry-run and a development-only FrameSkip proxy repair: all four trainable Stage A policies were trained with seed `101`, saved under `runs/mtf_vla_checkpoints/mtf_r20_ret100`, reloaded from disk, and evaluated on the `400` validation frames. Final decision: `MTF_ALL_ADAPTER_CHECKPOINTS_VERIFIED_STAGE_A_READY`. Validation action L2 means were `0.082590885` for MTF full, `0.082867367` for the no-retention ablation, `0.082553130` for the corrected FrameSkip proxy, and `0.082396918` for uniform retained-ratio LoRA. The corrected FrameSkip proxy uses `240` action-variation-selected train events and is distinct from the no-retention ablation. No closed-loop rollout happened and no confirmatory-test identities were used.

The MTF Stage A matched manifest is frozen in `reports/mtf_vla/stage_a_manifest.json` with canonical payload hash `1BB86A8060F8CD057AF984423021CA582E87661CB5157C072EF34B6F587739E3`. It contains exactly five policies (`frozen_smolvla`, `frameskip_proxy_lora`, `uniform_retained_ratio_lora`, `mtf_no_retention_ablation`, `mtf_full`), five deterministic task keys selected from the official 20-task manifest, fresh reset seeds `20261201` and `20261202`, `10` paired cases per policy, and `50` total planned episodes. `frameskip_proxy_lora` is labeled as a faithful local proxy, not an official FrameSkip reproduction.

Stage A completed `50 / 50` official LIBERO episodes with zero exceptions. Frozen SmolVLA, FrameSkip proxy, and uniform retained-ratio LoRA each reached `8 / 10`; the no-retention ablation and MTF full each reached `7 / 10`. Full MTF tied the key ablation and was only one episode behind the strongest baselines, so this is not a valid Stage A kill. The frozen adjudication is `MTF_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`.

The MTF Stage B matched manifest was frozen in `reports/mtf_vla/stage_b_manifest.json` with canonical payload hash `3C9D9CCF835A3B9753B81C320E9390EC9DA516514563E4850C1DC4F19ACC5743`. It used all `20` official tasks, fresh reset seeds `20261203` and `20261204`, `40` paired cases per policy, and `200` total planned episodes. The five policy identities were unchanged from Stage A.

Stage B completed `200 / 200` official LIBERO episodes with zero exceptions. Frozen SmolVLA reached `28 / 40`, the FrameSkip proxy reached `27 / 40`, uniform retained-ratio LoRA reached `29 / 40`, the no-retention ablation reached `32 / 40`, and MTF full reached `26 / 40`. Full-minus-no-retention paired delta was `-0.15` with CI `[-0.275, -0.025]`.

Final MTF decision: `MTF_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD`. Do not rescue MTF by retuning `mtf_r20_ret100`, changing retention, changing task/reset identities, or reinterpreting Stage B outcomes.

## Epoch 4 Cycle 7

Exactly three post-MTF candidates were generated and scored in `reports/epoch_4_cycle_7_candidate_generation.md` after building the prior mechanism map in `reports/epoch_4_cycle_7_prior_mechanism_map.md`. MTF remains archived and may not be rescued.

`DAGR-VLA` is selected as a DAM-VLA anchored dynamic arm/gripper routing method for frozen SmolVLA adaptation. Proposal hash: `BDE0EC67ACE8EC457CE6495D723EE476064F3D80946151326B11F0B5A1AFEF89`.

The selected first comparison is frozen at the design level to five policies: Base, a DAM-style static component proxy, DAGR full, a no-dynamic-route shared residual ablation, and one gripper-transition heuristic simple killer. No closed-loop rollout, training, or confirmatory-test tuning has happened for DAGR.

Reviewer B attack is complete in `reports/dagr_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. The attack narrows novelty against DAM-VLA, requires `dam_static_component_proxy` to be labeled as a faithful transparent local proxy rather than an official DAM-VLA reproduction, forbids KL over deterministic 7D actions, and makes noncollapsed route-label health, route observability, bounded action deltas, and identity-preserving integration mandatory before rollout.

Researcher A rebuttal is complete in `reports/dagr_vla/researcher_rebuttal.md` with decision `DAGR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. DAGR will not claim broad dynamic arm/gripper routing novelty; its local claim is frozen SmolVLA identity-preserving route-gated residual adaptation. No training, rollout, or confirmatory-test tuning has happened.

The DAGR mathematical audit, preregistration, and prototype protocol are frozen in `reports/dagr_vla/mathematical_mechanism_audit.md`, `reports/dagr_vla/preregistration.md`, and `reports/dagr_vla/prototype_protocol.md`.

Stage 0 development audit passed as `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH` in `reports/dagr_vla/development_audit.json`: `1600` development records, `1200` train, `400` validation, `1200` reserved test records not used, duplicate sample keys `0`, duplicate frame keys `0`, split overlap `0 / 0 / 0`, base action validity `1.0`, validation any-route fraction `0.865`, and no hard stops. Route-probe accuracy margins over validation majority were translation `0.0375`, rotation `0.0725`, and gripper `0.26`.

The bounded six-config validation search completed as `VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING` in `reports/dagr_vla/validation_search.json`. Selected config: `dagr_a020_route_mlp`, residual alpha `0.20`, route architecture `mlp`, validation score `0.8571740870493018`, delta L2 p95 `0.008609326556324959`, clean delta L2 p95 `0.00672802422195673`, and action validity `1.0`.

DAGR policy identity training is complete in `reports/dagr_vla/policy_checkpoint_manifest.json`. Final decision: `DAGR_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY`. The trainable identities `dagr_full`, `dam_static_component_proxy`, and `dagr_no_dynamic_route_ablation` all disk-reload, preserve initial base passthrough, and have validation action validity `1.0`; the nontrainable `gripper_transition_heuristic` config is saved under the same checkpoint root.

Checkpoint root: `runs/dagr_vla_checkpoints/dagr_a020_route_mlp`. DAGR full validation delta L2 p95 is `0.008576558902859688`; DAM static proxy p95 is `0.016259152442216873`; no-dynamic-route ablation p95 is `0.006147781852632761`.

The DAGR Stage A matched manifest is frozen in `reports/dagr_vla/stage_a_manifest.json` with canonical payload hash `8379E47D3C3C73E21ADDD285491750E7406B8389578C0003278E5E187EA27E7B`. It contains exactly five policies (`frozen_smolvla`, `dam_static_component_proxy`, `dagr_full`, `dagr_no_dynamic_route_ablation`, `gripper_transition_heuristic`), five evenly spaced official tasks, fresh reset seeds `20261205` and `20261206`, `10` paired cases per policy, and `50` total planned episodes. `dam_static_component_proxy` remains labeled as a faithful transparent local proxy, not an official DAM-VLA reproduction.

DAGR Stage A policy preflight passed as `DAGR_STAGE_A_PREFLIGHT_PASS_READY_FOR_OFFICIAL_ROLLOUT` in `reports/dagr_vla/stage_a_preflight.json`. Five policies loaded through the official SmolVLA/LIBERO path, four checkpoint identities checksum-verified, no accidental checkpoint reuse was detected, the base policy and learned DAGR heads ran on CUDA, and the wrappers produced finite 7D actions. No rollout, training, or confirmatory-test tuning happened during preflight. Next action: launch the official DAGR Stage A rollout.

Stage A completed `50 / 50` official LIBERO episodes with zero exceptions. Frozen SmolVLA reached `8 / 10`, the gripper-transition heuristic reached `7 / 10`, DAGR full reached `6 / 10`, the no-dynamic-route ablation reached `5 / 10`, and the DAM-style static component proxy reached `2 / 10`. DAGR full beat the closest-prior proxy and key ablation but trailed Base by two episodes, which is noncatastrophic under Stage A governance. Final Stage A decision: `DAGR_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`. Next action: freeze the DAGR Stage B matched manifest without retuning.

The DAGR Stage B matched manifest froze all `20` official tasks, fresh reset seeds `20261207` and `20261208`, `40` paired cases per policy, `200` total planned episodes, and the unchanged five-policy comparison. No checkpoint, threshold, task, or reset was selected from Stage B outcomes.

DAGR Stage B completed `200 / 200` official LIBERO episodes with zero exceptions and no confirmatory-test tuning. Frozen SmolVLA reached `28 / 40`, the DAM-style static component proxy reached `5 / 40`, DAGR full reached `18 / 40`, the no-dynamic-route ablation reached `16 / 40`, and the gripper-transition heuristic reached `24 / 40`. Full-minus-Base paired delta was `-0.25` with CI `[-0.4, -0.1]`; full-minus-gripper paired delta was `-0.15` with CI `[-0.3, 0.0]`.

Final DAGR decision: `DAGR_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD`. This is a valid current-formulation kill because the simple gripper-transition heuristic and Base explain or exceed the full method. Do not rescue DAGR by retuning `dagr_a020_route_mlp`, changing route thresholds, changing task/reset identities, changing the policy list, or reinterpreting partial results.

## Epoch 4 Cycle 8

Exactly three post-DAGR candidates were generated and scored in `reports/epoch_4_cycle_8_candidate_generation.md` after building the prior mechanism map in `reports/epoch_4_cycle_8_prior_mechanism_map.md`. DAGR remains archived and may not be rescued.

`MARC-VLA` is selected as an OpenVLA-OFT anchored median-anchor correction method for frozen SmolVLA flow actions. Proposal hash: `D1F910465D4E415C996B3F8C7CE2B2CF47339EA94D697B06A9DCED49AC1E585A`.

The selected first comparison is frozen at the design level to five policies: Base, OpenVLA-OFT-style L1 proxy, MARC full, no-disagreement-gate ablation, and one static L1 mixture simple killer. No closed-loop rollout or confirmatory-test tuning has happened for MARC.

Reviewer B attack is complete in `reports/marc_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. The attack narrows novelty against OpenVLA-OFT and requires noncollapsed disagreement labels, observable gates, bounded action deltas, identity-preserving integration, and a static-mixture simple killer before rollout.

Researcher A rebuttal is complete in `reports/marc_vla/researcher_rebuttal.md` with decision `MARC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. MARC will not claim that continuous L1 action prediction is novel; its local claim is frozen SmolVLA median-anchor correction. No training rollout or confirmatory-test tuning has happened.

The MARC mathematical audit, preregistration, and prototype protocol are frozen in `reports/marc_vla/mathematical_mechanism_audit.md`, `reports/marc_vla/preregistration.md`, and `reports/marc_vla/prototype_protocol.md`.

Stage 0 development audit passed as `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH` in `reports/marc_vla/development_audit.json`: `1600` development records, `1200` train, `400` validation, `1200` reserved test records not used, duplicate sample keys `0`, duplicate frame keys `0`, split overlap `0 / 0 / 0`, train disagreement fraction `0.4`, validation disagreement fraction `0.44`, gate-probe margin `0.0475`, initial action delta p95 `0.0`, and base action validity `1.0`.

The bounded six-config validation search completed as `VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING` in `reports/marc_vla/validation_search.json`. Selected config: `marc_a020_gate_mlp`, correction alpha `0.20`, gate architecture `mlp`, validation score `0.5457964262366295`, gate accuracy margin `0.0525`, gate predicted-positive fraction `0.3325`, delta L2 p95 `0.011818917468190193`, clean delta L2 p95 `0.010853752493858337`, and action validity `1.0`. Linear configs were stopped for collapsed gates.

MARC full validation action L2 is `0.08665236806523112`; the L1 proxy action L2 is `0.08763420091414227`; full-versus-L1 proxy mean L2 is `0.007010325323790312`; full-versus-no-gate mean L2 is `0.007010325323790312`; full-versus-static mixture mean L2 is `0.0019475044682621956`. The static mixture remains a live reviewer-killer.

MARC policy identity training is complete in `reports/marc_vla/policy_checkpoint_manifest.json`. Final decision: `MARC_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY`. The trainable identities `openvla_oft_l1_proxy`, `marc_full`, `marc_no_disagreement_gate_ablation`, and `static_l1_mixture_baseline` all disk-reload, preserve initial base passthrough, and have validation action validity `1.0`.

Checkpoint root: `runs\marc_vla_checkpoints\marc_a020_gate_mlp`. MARC full validation delta L2 p95 is `0.010693175718188286`; OpenVLA-OFT-style L1 proxy p95 is `0.2307613492012024`; no-disagreement-gate p95 is `0.12246084958314896`; static L1 mixture p95 is `0.07999999821186066`. Full-versus-L1 mean L2 is `0.08430124074220657`, full-versus-no-gate is `0.04372206702828407`, and full-versus-static is `0.032826922833919525`.

The MARC Stage A matched manifest is frozen in `reports/marc_vla/stage_a_manifest.json` with canonical payload hash `3383E377CEDD2B44E7730AAD3617E64838786E7094B9CF60D39F9679DE97D74E`. It contains exactly five policies (`frozen_smolvla`, `openvla_oft_l1_proxy`, `marc_full`, `marc_no_disagreement_gate_ablation`, `static_l1_mixture_baseline`), five evenly spaced official tasks, fresh reset seeds `20261209` and `20261210`, `10` paired cases per policy, and `50` total planned episodes. `openvla_oft_l1_proxy` remains labeled as a faithful transparent local proxy, not an official OpenVLA-OFT reproduction.

MARC Stage A policy preflight passed as `MARC_STAGE_A_PREFLIGHT_PASS_READY_FOR_OFFICIAL_ROLLOUT` in `reports/marc_vla/stage_a_preflight.json`: `5` policies loaded through the official SmolVLA/LIBERO path, `4` checkpoint identities checksum-verified, CUDA checks passed, no accidental checkpoint reuse was detected, and finite 7D MARC actions were produced. No rollout result, training, or confirmatory-test tuning happened during preflight.

The official MARC Stage A rollout completed from `runs/marc_vla_stage_a/20260714T171356Z` with exit code `0`, `50 / 50` episodes, zero exceptions, and no confirmatory-test tuning. The result is saved in `reports/marc_vla/stage_a_result.json` and summarized in `reports/marc_vla/stage_a_result.md`.

Stage A decision: `MARC_STAGE_A_CATASTROPHIC_KILL_ZERO_VS_STRONG_BASELINE`. Frozen SmolVLA reached `8 / 10`, OpenVLA-OFT-style L1 proxy reached `0 / 10`, MARC full reached `0 / 10`, no-disagreement-gate ablation reached `7 / 10`, and static L1 mixture reached `7 / 10`. MARC full-minus-Base paired delta was `-0.8`, full-minus-no-gate was `-0.7`, and full-minus-static was `-0.7`.

Final MARC decision: valid current-formulation kill. MARC full was catastrophically worse than Base and dominated by both the key ablation and simple static-mixture baseline. Do not rescue MARC by retuning `marc_a020_gate_mlp`, changing thresholds, changing policies, changing task/reset identities, or reinterpreting Stage A outcomes.

## Epoch 4 Cycle 9

Exactly three post-MARC candidates were generated and scored in `reports/epoch_4_cycle_9_candidate_generation.md` after building the prior mechanism map in `reports/epoch_4_cycle_9_prior_mechanism_map.md`. MARC remains archived and may not be rescued.

`PESA-VLA` is selected as a PriorVLA, LoRA-SP, and VLA-GSE anchored prior-expert spectral adaptation method for frozen SmolVLA 7D policies. Proposal hash: `B05B1ACF7CD3514365B418E25C7E995604FCA8C117CDC0F3384F1046BAF26B63`.

The selected first comparison is frozen at the design level to five policies: Base, a PriorVLA-style proxy, PESA full, a no-spectral/no-prior-query ablation, and one strongest simple standard-LoRA or clean-retention adaptation baseline.

The Researcher A proposal is frozen in `reports/pesa_vla/researcher_proposal.md`.

Reviewer B attack is complete in `reports/pesa_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. The attack requires the closest-prior proxy to remain transparent, forbids deterministic-action KL, and makes label health, mechanism observability, bounded deltas, identity preservation, and a strong simple killer mandatory before rollout.

Researcher A rebuttal is complete in `reports/pesa_vla/researcher_rebuttal.md` with decision `PESA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. The rebuttal accepts the narrow claim and keeps the PriorVLA-style proxy, no-spectral/no-prior-query ablation, and standard-LoRA or clean-retention simple killer live.

The PESA mathematical mechanism audit is frozen in `reports/pesa_vla/mathematical_mechanism_audit.md` with decision `PESA_MATHEMATICAL_AUDIT_PREREGISTERED`. The audit defines variables, shapes, spectral-energy masking, objectives, gradient paths, small-batch scale checks, required ablations, and the no deterministic-action KL rule.

The PESA preregistration and prototype protocol are frozen in `reports/pesa_vla/preregistration.md` and `reports/pesa_vla/prototype_protocol.md`. Stage 0 must pass the frozen label, split, headroom, spectral activation, gradient, action-validity, action-distinction, and identity-preservation gates before validation search or training.

PESA Stage 0 completed without closed-loop rollout, training, manifest freeze, or confirmatory-test tuning. The development audit is saved in `reports/pesa_vla/development_audit.json` and summarized in `reports/pesa_vla/development_audit.md`.

Final PESA Stage 0 decision: `DESIGN_FAILURE`. The only hard stop was query observability: validation query-probe accuracy `0.5225`, majority `0.6`, margin `-0.07750000000000001`, below the frozen `+0.02` requirement. Other Stage 0 checks passed or remained healthy, including split integrity, label balance, standard LoRA headroom, noncollapsed spectral activation, action distinctions, finite gradients, initial Base equality, and Base action validity.

This is a pre-rollout design failure, not a closed-loop kill. Do not rescue PESA by changing query-label construction, thresholds, features, or criteria.

Current PESA disposition: `PESA_STAGE_0_STOP_DESIGN_FAILURE`. This remains a pre-rollout design stop, not a closed-loop kill.

## Epoch 4 Cycle 10

Epoch 4 Cycle 10 generated exactly three post-PESA candidates in `reports/epoch_4_cycle_10_candidate_generation.md` after the prior map in `reports/epoch_4_cycle_10_prior_mechanism_map.md`, and selected `EAC-VLA`, Entropy-Calibrated Adaptive Chunking for frozen SmolVLA.

EAC is anchored to Adaptive Action Chunking, with AR-VLA and AC2-VLA as secondary temporal/action-context priors. The method preserves frozen SmolVLA weights and emitted 7D action values, and changes only action-queue commitment length from deployment-observable uncertainty and queue-boundary risk.

The design-level five-policy comparison is Base fixed queue, AAC entropy-only proxy, EAC full, no-calibration/no-hysteresis ablation, and fixed short-replan simple killer. No rollout, training, validation search, or confirmatory-test tuning has happened for EAC.

The EAC Researcher A proposal is frozen in `reports/eac_vla/researcher_proposal.md` with proposal hash `A89ED48AE9FD4D26A8DA9E3E987FACDBBD9F861D070AE135372A092A44581E4E`.

Reviewer B attack is complete in `reports/eac_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. The attack narrows novelty against AAC, forces `aac_entropy_proxy` and `fixed_short_replan_baseline` to remain live, requires uncertainty/dispersion validity before rollout, and treats action-value modification as implementation failure.

Researcher A rebuttal is complete in `reports/eac_vla/researcher_rebuttal.md` with decision `EAC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. The rebuttal accepts narrow AAC-extension novelty, exact action-value passthrough, live AAC proxy and fixed-replan killer baselines, uncertainty/dispersion terminology, and Stage 0 hard stops before rollout.

The EAC mathematical mechanism audit is frozen in `reports/eac_vla/mathematical_mechanism_audit.md` with decision `EAC_MATHEMATICAL_AUDIT_PREREGISTERED`. It defines the `50 x 7` chunk variables, dispersion/entropy rules, queue-risk formula, commitment map, action-value equality gate, validation-search score, required ablation, and Stage 0 hard stops.

The EAC preregistration and prototype protocol are frozen in `reports/eac_vla/preregistration.md` and `reports/eac_vla/prototype_protocol.md`.

EAC Stage 0 completed without training, validation search, closed-loop rollout, or confirmatory-test tuning. The audit is saved in `reports/eac_vla/stage_0_audit.json` and summarized in `reports/eac_vla/stage_0_audit.md`.

Final EAC Stage 0 decision: `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`. The audit used `2000` validation records and `400` unique validation frames, reserved `6000` confirmatory records unused, found zero validation/test frame or sample overlap, confirmed queue helpers and the canonical `50 x 7` chunk shape, and found noncollapsed first-two chunk dispersion with p95 `0.0007983036317792467` and nonzero fraction `1.0`. The preregistered commitment map was noncollapsed (`2`: `136`, `8`: `132`, `50`: `132`), max commitment share was `0.34`, and first-action passthrough max error was `5.07000000038449e-07`. There were no hard stops.

The canonical artifact stores first-two chunk previews and chunk hashes, not all `50` postprocessed actions, so the runtime full-chunk equality and queue-prefix execution check was run before validation search.

EAC runtime queue check completed without training, validation search, closed-loop rollout, or confirmatory-test tuning. The check is saved in `reports/eac_vla/runtime_queue_check.json` and summarized in `reports/eac_vla/runtime_queue_check.md`. It loaded frozen SmolVLA on `NVIDIA GeForce RTX 5080`, produced a full postprocessed chunk shape `[50, 7]`, verified `select_action` matched `chunk[0]` with max absolute diff `0.0`, observed the official queue length change from `0` before selection to `49` afterward, and verified every commitment prefix in `{1, 2, 4, 8, 16, 50}` preserved action values exactly with max prefix and queue-pop diffs `0.0`.

EAC bounded validation search completed with exactly six configurations and no confirmatory records used for tuning. The search is saved in `reports/eac_vla/validation_search.json`, summarized in `reports/eac_vla/validation_search.md`, and the selected frozen config is `reports/eac_vla/selected_config.json`.

Final EAC validation decision: `EAC_VALIDATION_SEARCH_SELECT_CONFIG_STAGE_A_MANIFEST_READY`. The selected config is `eac_q33_aggressive_1_4_50`, with validation score `0.7530415186081504`, commitment counts `1:132`, `4:136`, `50:132`, policy-calls-per-step proxy `0.4216`, oscillation fraction `0.6388888888888888`, risk-exposure-reduction proxy `0.9032794643799159`, mechanism activation `0.6599999999999999`, clean action-value passthrough `1.0`, and runtime action validity `1.0`.

The EAC Stage A matched manifest is frozen in `reports/eac_vla/stage_a_manifest.json` with canonical payload hash `63E96D0629F3D34E4801EB1084D094CB287EC4F2F2FCD96373981787EDA9954C`. It contains exactly five policies (`frozen_smolvla_fixed_queue`, `aac_entropy_proxy`, `eac_full`, `eac_no_calibration_no_hysteresis_ablation`, `fixed_short_replan_baseline`), fresh reset seeds `20261211` and `20261212`, `10` paired cases per policy, and `50` total planned episodes. `aac_entropy_proxy` remains a faithful transparent local proxy, not an official AAC reproduction.

EAC Stage A policy preflight passed in `reports/eac_vla/stage_a_preflight.json` as `EAC_STAGE_A_PREFLIGHT_PASS_RUNNER_IMPLEMENTATION_PENDING`: `5` scheduler identities were checked, `0` checkpoint policies were required, CUDA ran on `NVIDIA GeForce RTX 5080`, the policy output shape was `[50, 7]`, all policy prefixes preserved action values exactly, and no accidental checkpoint reuse was possible. No rollout, training, validation search, or confirmatory-test tuning happened during preflight.

EAC Stage A runner validation passed in `reports/eac_vla/stage_a_runner_validation.json` as `EAC_STAGE_A_RUNNER_VALIDATED_READY_FOR_ROLLOUT`. The validated runner reconstructs the frozen validation-only EAC thresholds, uses `2` runtime samples for dynamic schedulers, preserves all policy prefixes without action-value modification, and keeps rollout/training/confirmatory-test tuning at `False`.

EAC Stage A ran detached from `runs/eac_vla_stage_a/20260714T194025Z` and completed `50 / 50` episodes with zero exceptions. The result is saved in `reports/eac_vla/stage_a_result.json` and summarized in `reports/eac_vla/stage_a_result.md`.

Stage A decision: `EAC_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`. EAC full reached `8 / 10`, Base fixed queue reached `7 / 10`, AAC entropy proxy reached `9 / 10`, no-calibration ablation reached `7 / 10`, and fixed short-replan reached `7 / 10`. EAC full preserved action values, activated the scheduler with commitment counts `{'1': 150, '4': 25, '50': 33}`, and did not satisfy any valid Stage A kill criterion.

The EAC Stage B matched manifest is frozen in `reports/eac_vla/stage_b_manifest.json` with canonical payload hash `31F7590D81D95AECE9D7D1E8D6A2332364D5A9B36F6A913F9634D30D2C27B24D`. It uses all `20` official tasks, fresh reset seeds `20261213` and `20261214`, `40` paired cases per policy, and `200` total planned episodes. The five policy identities remain unchanged from Stage A.

EAC Stage B completed from the detached run `runs/eac_vla_stage_b/20260714T202334Z` with wrapper exit code `0`, `200 / 200` official LIBERO episodes, zero exceptions, and no confirmatory-test tuning. The result is saved in `reports/eac_vla/stage_b_result.json`, summarized in `reports/eac_vla/stage_b_result.md`, and checkpointed in `reports/eac_vla/stage_b_partial_result.json`.

Stage B decision: `EAC_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD`. Frozen Base fixed queue reached `30 / 40`, AAC entropy proxy reached `30 / 40`, EAC full reached `29 / 40`, the no-calibration/no-hysteresis ablation reached `30 / 40`, and fixed short-replan reached `29 / 40`. EAC full preserved action values and activated the scheduler with commitment counts `{'1': 807, '4': 199, '50': 148}`.

EAC full-minus-Base paired delta was `-0.025` with CI `[-0.175, 0.125]`; full-minus-AAC proxy was `-0.025` with CI `[-0.15, 0.1]`; full-minus-ablation was `-0.025` with CI `[-0.175, 0.125]`; and full-minus-fixed-short-replan was `0.0` with CI `[-0.15, 0.15]`.

Final EAC decision: valid current-formulation kill. Do not rescue EAC by retuning `eac_q33_aggressive_1_4_50`, changing thresholds, changing tasks or resets, changing the five-policy list, reinterpreting partial results, or applying any post-hoc expansion.

## Epoch 4 Cycle 11

Epoch 4 Cycle 11 generated exactly three post-EAC candidates in `reports/epoch_4_cycle_11_candidate_generation.md` after the prior map in `reports/epoch_4_cycle_11_prior_mechanism_map.md`, and selected `G3P-VLA`, Grounded 3D Point Injection for frozen SmolVLA.

G3P is anchored to Direct Action-Head Injection of A Grounded 3D Point, with RoboPoint, RoboGround, and AffordanceVLA as secondary spatial-grounding priors. Its first design-level comparison is Base, closest-prior 3D-point proxy, G3P full, no-3D/no-injection ablation, and one simple 2D/phase/nearest-object heuristic.

The G3P-VLA Researcher A proposal is frozen in `reports/g3p_vla/researcher_proposal.md` with proposal hash `BEE3822D8F54EFBD09C1CA47A9BF126EBE694B7B6219002FF770C5794ED7AA71`. Reviewer B attack is complete in `reports/g3p_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. Researcher A rebuttal is complete in `reports/g3p_vla/researcher_rebuttal.md` with decision `G3P_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. The mathematical mechanism audit is frozen in `reports/g3p_vla/mathematical_mechanism_audit.md` with decision `G3P_MATHEMATICAL_AUDIT_PREREGISTERED`. The preregistration and prototype protocol are frozen in `reports/g3p_vla/preregistration.md` and `reports/g3p_vla/prototype_protocol.md`.

G3P Stage 0 stopped as `DATA_OR_SUPERVISION_FAILURE` in `reports/g3p_vla/development_audit.json`: the material point label collapsed with train fraction `0.9982142857142857` and validation fraction `1.0`. No training, validation search, rollout, or confirmatory-test tuning happened.

Epoch 4 Cycle 12 generated exactly three candidates in `reports/epoch_4_cycle_12_candidate_generation.md` after the prior mechanism map in `reports/epoch_4_cycle_12_prior_mechanism_map.md` and selected `CALA-VLA`.

The CALA-VLA Researcher A proposal is frozen in `reports/cala_vla/researcher_proposal.md` with proposal hash `5B3933C9C0FD5AE5F07FDB0CEC447B48040238FB6D872D97E545E3D93E257E76`.

Reviewer B attack is complete in `reports/cala_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`.

Researcher A rebuttal is complete in `reports/cala_vla/researcher_rebuttal.md` with decision `CALA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`.

The mathematical mechanism audit is frozen in `reports/cala_vla/mathematical_mechanism_audit.md` with decision `CALA_MATHEMATICAL_AUDIT_PREREGISTERED`.

The preregistration and prototype protocol are frozen in `reports/cala_vla/preregistration.md` and `reports/cala_vla/prototype_protocol.md`.

CALA Stage 0 stopped as `DESIGN_FAILURE` in `reports/cala_vla/development_audit.json`: the latent predictability margin was `-0.01171824382857035`, with `action_history_only` beating the full deployment-observable probe. Source legality, split health, label variance, headroom, gradients, Base action validity, and identity-preserving zero delta passed. No training, validation search, rollout, or confirmatory-test tuning happened.

Epoch 4 Cycle 13 generated exactly three candidates in `reports/epoch_4_cycle_13_candidate_generation.md` after the prior mechanism map in `reports/epoch_4_cycle_13_prior_mechanism_map.md` and selected `RAR-VLA`.

The RAR-VLA Researcher A proposal is frozen in `reports/rar_vla/researcher_proposal.md` with proposal hash `723C16C3885A974E2CA12D90BC36267FA6E86827AC9D2A1E0E0E475E16FB0E56`.

Reviewer B attack is complete in `reports/rar_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`.

Researcher A rebuttal is complete in `reports/rar_vla/researcher_rebuttal.md` with decision `RAR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`.

The mathematical mechanism audit is frozen in `reports/rar_vla/mathematical_mechanism_audit.md` with decision `RAR_MATHEMATICAL_AUDIT_PREREGISTERED`.

The preregistration and prototype protocol are frozen in `reports/rar_vla/preregistration.md` and `reports/rar_vla/prototype_protocol.md`.

RAR Stage 0 stopped as `DESIGN_FAILURE` in `reports/rar_vla/development_audit.json`: the residual predictability margin was `-0.03837609884238533`, with `zero_residual` beating the full legal causal probe. Source legality, split health, residual headroom, gradients, Base action validity, and identity-preserving zero delta passed. No training, validation search, rollout, or confirmatory-test tuning happened.

Epoch 4 Cycle 14 generated exactly three candidates in `reports/epoch_4_cycle_14_candidate_generation.md` after the prior mechanism map in `reports/epoch_4_cycle_14_prior_mechanism_map.md` and selected `COVI-VLA`.

COVI is anchored to LIBERO-Occ / Viewpoint Imagination, with CamVLA and STRONG-VLA as secondary priors. The selected first comparison is frozen at the design level to five policies: Base under occlusion, VIM-style proxy, COVI full, no-imagined-view ablation, and `random_cutout_clean_retention_baseline`.

The COVI-VLA Researcher A proposal is frozen in `reports/covi_vla/researcher_proposal.md` with proposal hash `338430D2C6CF1D82410C036D79102ED3F38B2367BB35B9AE2811161698A3E621`.

Reviewer B attack is complete in `reports/covi_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. The novelty is narrowed to a frozen-SmolVLA identity-preserving complementary-feature adapter for scene-induced occlusion; the VIM proxy must remain transparent, direct two-camera fusion diagnostics are required, and `random_cutout_clean_retention_baseline` remains live.

Researcher A rebuttal is complete in `reports/covi_vla/researcher_rebuttal.md` with decision `COVI_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. Researcher A accepted narrowed novelty, transparent VIM proxy status, direct two-camera fusion diagnostics, the random-cutout simple killer, physical occlusion validation, identity-preserving integration, and no privileged inference.

The COVI mathematical mechanism audit is frozen in `reports/covi_vla/mathematical_mechanism_audit.md` with decision `COVI_MATHEMATICAL_AUDIT_PREREGISTERED`. It freezes variables and tensor shapes, the legal source gate, complementary-target construction, adapter/gate formulas, objective terms, gradient checks, direct two-camera fusion diagnostics, random-cutout simple killer, transparent VIM proxy status, and the six-configuration validation budget. No implementation, validation search, training, manifest freeze, or rollout has happened.

The COVI preregistration and prototype protocol are frozen in `reports/covi_vla/preregistration.md` and `reports/covi_vla/prototype_protocol.md`. They freeze the measured `[64, 960]` visual-token hook, `600 / 600 / 400 / 1200` fit/one-check/validation/confirmatory record partitions, one fixed Stage 0 configuration, one unresolved-result check, and episode-cluster bootstrap false-negative safeguard. Synthetic Stage 0 occlusion is a development proxy only and cannot establish the physical-occlusion claim.

Previous decision: `COVI_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`. COVI Stage 0 then completed under its frozen protocol. The repaired admissible result is `IMPLEMENTATION_OR_DATA_FAILURE`: the full weighted objective-gradient ratio was `1345.9529990435792:1` against a frozen `100:1` maximum, only two objectives had nonzero pretraining gradients, and output validity was `0.2`. The no-imagined-view target also had no preregistered headroom. This is `COVI_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE_NO_SCIENTIFIC_KILL`; the one-check set stayed sealed and no validation search or rollout ran.

Epoch 4 Cycle 15 generated exactly three prior-anchored candidates in `reports/epoch_4_cycle_15_candidate_generation.md` after the source audit in `reports/epoch_4_cycle_15_prior_mechanism_map.md`. It selected `LIFT-VLA`, Language-Induced Flow Transport, with score `90 / 100`.

LIFT is a frozen, inference-only cross-domain mechanism transfer. It applies conditional-minus-unconditional language guidance at every SmolVLA action-flow step and compares against CAG final-action mixing under the same two-branch budget. The first comparison contains exactly Base, transparent training-free CAG, LIFT full, and last-step-only LIFT. Standard LoRA and a fifth policy are omitted because they do not test the claimed mechanism.

The Researcher A proposal is frozen in `reports/lift_vla/researcher_proposal.md` with hash `3D263AA6FF73B342523D85AD4854145AF4D79DE2B90C6119F417D37A8B08F55F`.

Reviewer B attack is complete in `reports/lift_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. Essential evidence now includes narrow VLA-flow novelty, a scoreable feasible counterfactual manifest, native-flow-space same-noise CAG, a matched-compute last-step ablation, practical-equivalence thresholds, Base-and-CAG headroom, and one-chunk memory/latency feasibility.

Researcher A accepted every constraint in `reports/lift_vla/researcher_rebuttal.md`.

The mathematical audit, preregistration, and prototype protocol are frozen in `reports/lift_vla/mathematical_mechanism_audit.md`, `reports/lift_vla/preregistration.md`, and `reports/lift_vla/prototype_protocol.md`. They freeze native `[1,50,32]` flow tensors, ten steps, native-space same-noise CAG, `20` evaluations for Prior/Ours/ablation, practical-equivalence threshold construction, target-task partitions `[0-3] / [4-6] / [7-9]`, three guidance scales, and zero confirmatory policy decodes in Stage 0.

LIFT Stage 0 is implemented and adjudicated in `reports/lift_vla/stage_0_adjudication.md` as `LIFT_COMPUTE_INFEASIBLE`. The same-scene target-BDDL manifest retained `20 / 20` valid rows, exact identity passed at `0.0`, mechanism activation and practical separation passed, peak CUDA allocation was `0.9200425148010254 GiB`, and LIFT latency was `2.013133036365988` times Base. Executed-action bound validity was only `0.8023809523809524` against the frozen `1.0` requirement. No clipping rescue, headroom rollout, validation search, training, Stage A, or confirmatory evaluation ran; confirmatory policy observations and actions remained zero.

Archived Cycle 15 decision: `LIFT_COMPUTE_INFEASIBLE`. Cycle 15 is closed without rescue. The campaign then advanced through Cycle 16 candidate generation under the active governance.

## Epoch 4 Cycle 16

Cycle 16 generated exactly three candidates and selected `IARC-VLA` with score
`95 / 100`. IARC is a narrow STRONG-VLA plus GEM cross-paper synthesis: it
projects the actual zero-momentum, zero-decay Stage II SGD step against a paired
perturbation-replay SmolVLA action gradient only when clean and robust gradients
conflict. Rank-4 LoRA is fixed low-compute infrastructure, not the method claim.

The frozen proposal hash is
`A1B0CF8BCBCF6A88F27B31EF5E38BAF408A3E62BB34206A1AC9F051EA6B57408`.
Reviewer B attack and Researcher A rebuttal are complete. The mathematical
audit, preregistration, and executable Stage 0A protocol are frozen under
`reports/iarc_vla/`.

The five policies are Base, transparent STRONG proxy, IARC full, unprojected
joint replay, and standard clean-only LoRA. Stage 0A must use `40 / 40 / 1200`
fit-audit/validation/confirmatory source boundaries with zero confirmatory
decode, `20` micro-fit steps, and `40` same-noise/time gradient pairs before any
headroom rollout or validation search.

The Windows Efficiency Mode interval and durable-worker audit are recorded in
`reports/resource_contention_intervals.json`. No worker was alive; EAC Stage B
was already complete and accepted without rerun after `200 / 200`, zero-error,
zero-duplicate, synchronous manifest validation. Overlapping or
overlap-unknown efficiency metrics are quarantined.

IARC Stage 0A completed normally: `40 / 40` gradient pairs, `40 / 40`
validation rows, zero exceptions, zero duplicate/manifest mismatches, and zero
confirmatory decodes or actions. The mechanism activated on `18 / 40` rows
across all four families, with projection `18 / 18`, agreeing rows unchanged
`22 / 22`, exact identity/reload, and unchanged Base hash.

The frozen dataset-range action-validity gate failed at `12 / 40 = 0.30`
against `1.0`. The result is adjudicated as
`IARC_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`, not a scientific kill. Clipping,
one-check, Stage 0B, and validation search are forbidden.

## Epoch 4 Cycle 17

Cycle 17 generated exactly three candidates and selected `FAMR-VLA` with
`93 / 100`. RETAIN is the closest positive prior and Fisher-weighted model
merging is the secondary mechanism prior. FAMR fits bounded checkpoint
task-vector coefficients from groupwise postprocessed action responses while
limiting original-task action drift; rank-4 LoRA is only the local low-compute
endpoint.

Proposal hash:
`96E067FFFC48D5EF9986E35E5336D679EA841BFD1F06D5E5AD4F28B5B551FD69`.
The proposal, Reviewer B attack, rebuttal, mathematical audit,
preregistration, and executable protocol are frozen under
`reports/famr_vla/`.

FAMR Stage 0A completed under the frozen protocol and is adjudicated as
`FAMR_STAGE_0A_PASS_ENDPOINT_TRAINING_ALLOWED`. All `24` fixed discovery rows
loaded with `150 / 150` source-terminal successes, zero task overlap, zero
duplicate episode or frame keys, and zero validation or test decode. The
rank-4 endpoint completed `20 / 20` optimizer steps with finite nonzero
gradients and reduced fixed-subset loss from `0.7321685557253659` to
`0.6487047945459684`, a relative reduction of `0.11399528227036353`.

Zero-effect identity, all-zero/all-one coefficient identity, Base hash
retention, and disk reload all passed at zero output error. The 74 trainable
tensors covered every frozen coarse and fine group exactly once; effective
LoRA scaling relative error was at most `2.052311574965452e-07`. Peak CUDA
allocation was `1.0808053016662598 GiB`, exceptions and duplicate keys were
zero, and confirmatory observations/actions remained `0 / 0`.

The frozen endpoint completed `300 / 300` optimizer steps and `2400 / 2400`
discovery microbatches with zero exceptions and duplicates. Fit, gradient,
action-effect, Base-hash, checkpoint-reload, manifest, and memory checks
passed. Base-relative action validity failed: outside-`[-1,1]` frequency was
`0.1130952380952381` versus limit `0.08738095238095238`, and p99 exceedance
was `0.09376012921333322` versus limit `0.04096377015113834`.

The endpoint is closed unchanged as
`FAMR_ENDPOINT_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`. This is an
implementation/optimization failure, not a scientific kill.

Cycle 18 generated exactly three candidates and selected `PCAV-VLA` with
`95 / 100`, anchored to TACO and extended with ProgressVLA's
progress-consequence mechanism. Proposal hash:
`E8B23C755C6D4E450FD193101CC0B15F88AAFE20E137A0F86830ED6D421E12AA`.
All pre-Stage 0A research artifacts are frozen under `reports/pcav_vla/`.

PCAV Stage 0A completed `96 / 96` rows with zero final exceptions, duplicates,
or manifest mismatches. The first 24 rows were preserved exactly and only 72
missing keys were added on resume. Candidate generation was noncollapsed and
Base-safe, but only `7 / 96` rows met the 5% material-improvement threshold;
median reduction over improvable rows was `0.0166833`.

The frozen decision is `PCAV_STAGE_0A_NO_USABLE_HEADROOM`. Stage 0B is
forbidden. Current cycle is `19`; current stage is
`epoch_4_cycle_19_candidate_search_pending`.
