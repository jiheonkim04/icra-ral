# Autonomous RA-L Decision

Date: 2026-07-16 KST

Current decision:
`LCG_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`

This is not a terminal state under the active governance.

Active governance: `reports/current_research_governance.md`

## Epoch 4 Cycle 32 Selection

Cycle 32 completed the primary-source mechanism map in
`reports/epoch_4_cycle_32_prior_mechanism_map.md` and generated exactly three
candidates in `reports/epoch_4_cycle_32_candidate_generation.md`. S2C remains
preserved unchanged as `S2C_STAGE_0_DATA_OR_SUPERVISION_FAILURE`.

`LCG-VLA`, Language-Contrastive Guidance for Base-preserving SmolVLA actions,
is selected at `93 / 100`. Its closest positive prior is Counterfactual Action
Guidance, anchored to `https://arxiv.org/abs/2602.17659`, which reports
improved LIBERO-CF language-following and task success plus real-world
counterfactual failure reductions.

LCG compares frozen SmolVLA Base action chunks under the original instruction
and a legal language-null or counterfactual-language branch, then learns an
identity-initialized action-cell gate that permits bounded edits only where
language contrast predicts vision-shortcut risk. LoRA is only implementation
infrastructure.

The first serious comparison is Base, `counterfactual_action_guidance_proxy`,
`lcg_full`, `lcg_no_language_contrast_ablation`, and matched `standard_lora`.

Current cycle: `32`. Current stage:
`epoch_4_cycle_32_lcg_prototype_protocol_pending`.

The LCG-VLA Researcher A proposal is frozen in
`reports/lcg_vla/researcher_proposal.md` with SHA-256
`F0D980AA0760F143D781C723DB632BC324C1E18F390D9C33C5DA94F3A897D11E` and
decision `LCG_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`. No LCG implementation,
training, validation search, rollout, simulator access, or confirmatory-test
tuning has happened.

Reviewer B attack is complete in `reports/lcg_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. It preserves
CAG as closest prior and policy 2, rejects treating `B_t - N_t` as a residual
target by itself, requires null-branch proxy validation, and requires
contrast/residual/mask noncollapse before progression.

Researcher A rebuttal is complete in
`reports/lcg_vla/researcher_rebuttal.md` with decision
`LCG_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. It accepts all Reviewer B
conditions and sends LCG to mathematical mechanism audit before any
preregistration, prototype protocol, implementation, validation search,
training, rollout, or confirmatory-test access.

The mathematical mechanism audit is frozen in
`reports/lcg_vla/mathematical_mechanism_audit.md` with decision
`LCG_MATHEMATICAL_AUDIT_PREREGISTERED`. It fixes the language-null branch,
contrast mask, CAG proxy, objective, gradient checks, Stage 0 stop classes, and
no deterministic-action KL.

Preregistration is frozen in `reports/lcg_vla/preregistration.md` with
decision `LCG_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`. It freezes
development partitions, Stage 0 artifacts, pass/stop gates, worker resume keys,
and bounded validation search limits.

## Epoch 4 Cycle 31 Selection

Cycle 31 generated exactly three candidates in
`reports/epoch_4_cycle_31_candidate_generation.md` after the primary-source
map in `reports/epoch_4_cycle_31_prior_mechanism_map.md`. URF remains
preserved unchanged as `URF_STAGE_0_NO_USABLE_HEADROOM`.

`S2C-VLA`, Seam-Supervised Chunk Consistency for Base-preserving SmolVLA
execution, is selected at `95 / 100`. Its closest positive prior is ChunkFlow,
anchored to `https://arxiv.org/html/2607.12992v1` and project page
`https://cytoderm-ai.github.io/chunkflow`, which reports `93.4%` LIBERO
long-horizon success with improved boundary jump, high-frequency energy,
smoothness metrics, and low-latency inference.

S2C learns a Base-preserving overlap edit mask and tail-anchored bridge for
SmolVLA action-chunk boundary consistency. LoRA is only implementation
infrastructure.

The first serious comparison is Base, `chunkflow_overlap_proxy` or official
ChunkFlow if installed, S2C full, no-learned-overlap-mask ablation, and matched
standard LoRA.

The S2C-VLA Researcher A proposal is frozen in
`reports/s2c_vla/researcher_proposal.md` with SHA-256
`399A3960F9FF9AFA8EDA7C3F743A95C3FD4DC711644C2398630F1E68486DC5B3` and
decision `S2C_PROPOSAL_FROZEN_REVIEWER_ATTACK_COMPLETED`.

Reviewer B attack is complete in `reports/s2c_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. It accepts S2C
only as a frozen-SmolVLA, Base-preserving learned overlap edit layer; keeps
ChunkFlow as policy 2; keeps SEAM as a secondary prior; requires previous-tail
inference to use only executed or committed Base/S2C tail; and preserves
standard LoRA as the simple control. No S2C training, validation search,
rollout, simulator access, or confirmatory-test tuning has happened.

Researcher A rebuttal is complete in
`reports/s2c_vla/researcher_rebuttal.md` with decision
`S2C_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. It accepts all Reviewer B
conditions, defines deployment previous-tail construction, preserves ChunkFlow
as the closest prior, and sends S2C to mathematical audit before any
implementation or validation search.

The mathematical mechanism audit is frozen in
`reports/s2c_vla/mathematical_mechanism_audit.md` with decision
`S2C_MATHEMATICAL_AUDIT_PREREGISTERED`. It fixes `H=50`, stride `10`,
overlap `K=10`, the deterministic tail-anchored bridge target, the learned
effective edit mask, action-group caps, objective coefficients, no
deterministic-action KL, and Stage 0 pass/stop gates.

Preregistration is frozen in `reports/s2c_vla/preregistration.md` with
decision `S2C_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`. It freezes
development partitions, Stage 0 artifacts, pass/stop gates, bounded validation
budget, worker resume keys, and the five-policy prior-first comparison.

Prototype protocol is frozen in `reports/s2c_vla/prototype_protocol.md` with
decision `S2C_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING`. It
fixes helper, runner, focused tests, required Stage 0 artifacts, serializer
preflight, and worker-safety requirements.

S2C Stage 0 implementation is validated as
`S2C_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY`: WSL py_compile passed,
focused S2C tests reported `7 passed`, and
`reports/s2c_vla/stage_0_serializer_preflight.json` has matching fixture and
reproduced hashes.

S2C Stage 0 completed as `S2C_STAGE_0_DATA_OR_SUPERVISION_FAILURE` in
`reports/s2c_vla/stage_0_result.json`: `885 / 885` model rows, exception
count `0`, duplicate / missing / extra / split-overlap keys all `0`, and exact
key-set equality. The development audit found `177` adjacent pairs, but task
coverage was insufficient for the frozen validation gate; secondary gates also
showed no Base boundary headroom (`mean=0.001199425821980814`,
`p75=0.0008132225130898095`) and failed mask/gripper criteria. This is not a
closed-loop scientific kill, and S2C repair/rescue is disallowed.

Cycle 32 candidate search has completed and selected LCG-VLA for proposal
freeze.

## Epoch 4 Cycle 30 Selection

Cycle 30 generated exactly three candidates in
`reports/epoch_4_cycle_30_candidate_generation.md` after the primary-source
map in `reports/epoch_4_cycle_30_prior_mechanism_map.md`. CCIF remains
preserved unchanged as `CCIF_STAGE_0_DESIGN_FAILURE`.

`URF-VLA`, Uncertainty-Routed Residual Flow for Base-preserving SmolVLA
chunks, is selected at `92 / 100`. Its closest positive prior is SUREFlow,
anchored to `https://arxiv.org/abs/2607.10504` and official repository
`https://github.com/tanvirnwu/SUREFlow`, which reports `92.5%` average LIBERO
success and LIBERO-PRO robustness with a 179M uncertainty-aware residual-flow
VLA.

URF predicts a heteroscedastic residual-flow field around the frozen SmolVLA
Base action chunk and routes bounded residual transport only where predicted
residual uncertainty supports intervention. LoRA is only implementation
infrastructure.

The first serious comparison is Base, `sureflow_uncertainty_residual_proxy` or
official `sureflow` if installed, URF full, no-uncertainty-route ablation, and
matched standard LoRA.

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
`URF_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY` with
`tca_map/smolvla/urf_vla.py`, `scripts/run_urf_vla_stage0.py`, and
`tests/test_urf_vla.py`. WSL py_compile passed, focused URF tests reported
`8 passed`, and serializer preflight wrote
`reports/urf_vla/stage_0_serializer_preflight.json` with matching fixture and
reproduced hashes
`799BB904F82C96473A08773159E4C9E0BCA7AA8701FF86D95F727A730D0E431F`.
This is not a Stage 0 experimental result; no URF training, validation search,
rollout, simulator access, or confirmatory-test tuning has happened.

URF Stage 0 then completed as `URF_STAGE_0_NO_USABLE_HEADROOM` in
`reports/urf_vla/stage_0_result.json`: `5120 / 5120` model rows, exception
count `0`, duplicate / missing / extra / split-overlap keys all `0`, and exact
key-set equality. Bounded validation is not allowed because Base residual
headroom was below the frozen `0.005` normalized Huber gate
(`0.0033407550543043956`) and the heteroscedastic residual proxy did not beat
homoscedastic or task/phase baselines. This is a development-only no-headroom
stop, not a closed-loop scientific kill; URF rescue is forbidden.

Current cycle: `31`. Current stage:
`epoch_4_cycle_31_candidate_search_pending`.

## Epoch 4 Cycle 29 Selection

Cycle 29 generated exactly three candidates in
`reports/epoch_4_cycle_29_candidate_generation.md` after the primary-source
map in `reports/epoch_4_cycle_29_prior_mechanism_map.md`. TSC remains preserved
unchanged as `TSC_STAGE_0_NO_USABLE_HEADROOM`.

`CCIF-VLA`, Continuous Coarse Intent Field for base-preserving VLA chunks, is
selected at `92 / 100`. Its closest positive prior is Coarse-to-Control,
anchored to `https://arxiv.org/abs/2606.07107`, which reports `97.9%` average
LIBERO success.

CCIF predicts a deployment-observable continuous coarse motor-intent field from
current inputs and the Base decoded chunk, then conditions a bounded
identity-preserving residual action field on that intent. LoRA is only
implementation infrastructure.

The first serious comparison is exactly Base,
`coarse_to_control_continuous_proxy`, CCIF full, no-coarse-intent ablation, and
matched standard LoRA.

The CCIF-VLA Researcher A proposal is frozen in
`reports/ccif_vla/researcher_proposal.md` with SHA-256
`2AFC40F050FD7F0D28507344358CBCB70BF27CC901C57474A501D3EB87E7FAA1` and
decision `CCIF_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`. No CCIF training,
validation search, rollout, simulator access, or confirmatory-test tuning has
happened.

Reviewer B attack is complete in `reports/ccif_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`; Researcher A
rebuttal is complete in `reports/ccif_vla/researcher_rebuttal.md` with
decision `CCIF_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`.

The CCIF mathematical mechanism audit is frozen in
`reports/ccif_vla/mathematical_mechanism_audit.md` with decision
`CCIF_MATHEMATICAL_AUDIT_PREREGISTERED`.

The CCIF preregistration is frozen in `reports/ccif_vla/preregistration.md`
with decision `CCIF_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`.

The executable CCIF prototype protocol is frozen in
`reports/ccif_vla/prototype_protocol.md` with decision
`CCIF_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`.

CCIF Stage 0 implementation was validated as
`CCIF_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY` with
`tca_map/smolvla/ccif_vla.py`, `scripts/run_ccif_vla_stage0.py`, and
`tests/test_ccif_vla.py`. WSL py_compile passed, focused CCIF tests reported
`9 passed`, and serializer preflight wrote
`reports/ccif_vla/stage_0_serializer_preflight.json`. This is not a Stage 0
experimental result; no training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened.

CCIF Stage 0 then completed as `CCIF_STAGE_0_DESIGN_FAILURE` in
`reports/ccif_vla/stage_0_result.json`: `4480 / 4480` model rows, duplicate /
missing / extra keys all `0`, final exception count `0`, and two repaired
resume blocker exceptions recorded separately. Bounded validation is not
allowed because the deployment intent probe did not beat task/phase mean and
endpoint-only diagnostics explained the signal. This is a development-only
design failure, not a closed-loop scientific kill.

## Epoch 4 Cycle 28 Selection

Cycle 28 generated exactly three candidates in
`reports/epoch_4_cycle_28_candidate_generation.md` after the primary-source
map in `reports/epoch_4_cycle_28_prior_mechanism_map.md`. CFR remains preserved
unchanged as `CFR_STAGE_0_NO_USABLE_HEADROOM`.

`TSC-VLA`, Temporal-Spatial masked action completion for continuous VLA chunks,
is selected at `91 / 100`. Its closest positive prior is TS-Mask VLA,
anchored to `https://arxiv.org/abs/2607.09818`.

TSC predicts a deployment-observable sparse time-dimension action-cell error
mask over the Base decoded `[50,7]` SmolVLA chunk and runs continuous masked
completion that changes only selected cells while clamping all unselected cells
exactly to Base. LoRA is only implementation infrastructure.

The first serious comparison is exactly Base,
`ts_mask_continuous_proxy` or official `ts_mask_vla` if installed, TSC full,
no-targeted-mask ablation, and matched standard LoRA. No TSC training,
validation search, rollout, simulator access, or confirmatory-test tuning has
happened.

The TSC-VLA Researcher A proposal is frozen in
`reports/tsc_vla/researcher_proposal.md` with SHA-256
`0DF143D2D8773D7ABF4FC76AB7CC083FE7EE65DF84EA06631E67C2445F6DC941`.

Reviewer B attack is complete in `reports/tsc_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`; Researcher A
rebuttal is complete in `reports/tsc_vla/researcher_rebuttal.md` with decision
`TSC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`.

The TSC mathematical mechanism audit is frozen in
`reports/tsc_vla/mathematical_mechanism_audit.md` with decision
`TSC_MATHEMATICAL_AUDIT_PREREGISTERED`.

The TSC preregistration is frozen in `reports/tsc_vla/preregistration.md` with
decision `TSC_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`.

The executable TSC prototype protocol is frozen in
`reports/tsc_vla/prototype_protocol.md` with decision
`TSC_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`.

The TSC Stage 0 runner is implemented in `scripts/run_tsc_vla_stage0.py` with
helper module `tca_map/smolvla/tsc_vla.py`; focused TSC tests pass
(`8 passed`) and `reports/tsc_vla/stage_0_serializer_preflight.json` is
persisted.

TSC Stage 0 completed `640 / 640` rows with exception count `0`, exit code `0`,
and decision `TSC_STAGE_0_NO_USABLE_HEADROOM`. Manifest/partial duplicate,
missing, extra, and split-overlap counts are all `0`, with exact key-set
equality. This is a development no-headroom stop, not a closed-loop scientific
kill, and TSC rescue is forbidden.

Current cycle: `29`. Current stage:
`epoch_4_cycle_29_candidate_search_pending`.

## Epoch 4 Cycle 27 Selection

Cycle 27 generated exactly three candidates in
`reports/epoch_4_cycle_27_candidate_generation.md` after the primary-source
map in `reports/epoch_4_cycle_27_prior_mechanism_map.md`.

`CFR-VLA`, Continuous Full-Chunk Refinement for VLA action-flow decoding, is
selected at `92 / 100`. Its closest positive prior is DFM-VLA, anchored to
`https://arxiv.org/html/2603.26320v1` and project page
`https://chris1220313648.github.io/DFM-VLA/`.

CFR's single mechanism is a bounded continuous residual velocity/refinement
field over the full `[50,7]` SmolVLA action chunk, applied iteratively from a
Base decoded chunk before execution. LoRA is only the implementation
infrastructure, not the scientific method.

The first serious comparison is exactly Base,
`dfm_vla_continuous_refinement_proxy` or official `dfm_vla` if installed, CFR
full, no-iterative-refinement ablation, and matched standard LoRA. The CFR
proposal is frozen in `reports/cfr_vla/researcher_proposal.md` with SHA-256
`9E2FC510B2D97C869F18BE6C5B339CE034DD98223802078358320AA8BEF3D0AE`. No CFR
training, validation search, rollout, simulator access, or confirmatory-test
tuning has happened.

Reviewer B attack is complete in `reports/cfr_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. It conditionally
passes CFR only if Researcher A accepts the narrowed novelty, DFM proxy,
no-iterative ablation, standard-LoRA killer baseline, official action-validity
semantics, residual/headroom gates, mathematical audit, and no-privileged-input
conditions.

Researcher A rebuttal is complete in `reports/cfr_vla/researcher_rebuttal.md`
with decision `CFR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`; all Reviewer B
conditions are accepted before mathematical audit.

The CFR mathematical mechanism audit is frozen in
`reports/cfr_vla/mathematical_mechanism_audit.md` with decision
`CFR_MATHEMATICAL_AUDIT_PREREGISTERED`.

The CFR preregistration is frozen in `reports/cfr_vla/preregistration.md` with
decision `CFR_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`.

The executable prototype protocol is frozen in
`reports/cfr_vla/prototype_protocol.md` with decision
`CFR_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`.

The CFR Stage 0 runner is implemented in `scripts/run_cfr_vla_stage0.py` with
helper module `tca_map/smolvla/cfr_vla.py`; focused CFR tests pass (`8` tests).
No CFR training, validation search, rollout, simulator access, or
confirmatory-test tuning has happened.

CFR Stage 0 completed from Linux worker PID `310` with exit code `0`, `640 /
640` rows, exception count `0`, duplicate/missing/extra/split-overlap counts
all `0`, and key sets equal `true`. The final decision is
`CFR_STAGE_0_NO_USABLE_HEADROOM`, driven by negative residual-probe gain
(`-6.04941221711208 / -0.11968147462337628`) and negative CFR-minus-DFM
headroom (`-6.068176722319228 / -0.11975307303185317`). This is a
development-only no-headroom stop, not a closed-loop scientific kill.

Current cycle: `28`. Current stage:
`epoch_4_cycle_28_candidate_search_pending`.

## Epoch 4 Cycle 26 Selection

Cycle 26 generated exactly three candidates in
`reports/epoch_4_cycle_26_candidate_generation.md` after the primary-source
map in `reports/epoch_4_cycle_26_prior_mechanism_map.md`.

`AMP-VLA`, Action-Manifold Projection for VLA action-flow adaptation, is
selected at `95 / 100`. Its closest positive prior is ABot-M0, anchored to
`https://arxiv.org/abs/2602.11236` and the official repository
`https://github.com/amap-cvlab/ABot-Manipulation`.

AMP's single mechanism is a discovery-only low-dimensional action manifold
over LIBERO action chunks, used to constrain a SmolVLA adapter through an
identity-preserving projection or bounded gated residual. LoRA is only the
implementation infrastructure for that residual/gate path, not the scientific
method.

The first serious comparison is exactly Base, transparent ABot-M0 action
manifold proxy, AMP full, no-manifold-projection ablation, and matched
standard LoRA. No AMP training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened.

The AMP-VLA Researcher A proposal is frozen in
`reports/amp_vla/researcher_proposal.md` with SHA-256
`67ACC693C706B76BC9FB84F9E59BA3DF9C0463A0BAFABE539312D0E232DFE9A4`
and decision `AMP_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`.
Reviewer B attack is complete in `reports/amp_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. The required
conditions keep the ABot-M0 proxy, no-projection ablation, matched standard
LoRA, clipping/bound-only diagnostics, manifold-health gates, identity/reload
checks, and mathematical objective audit live.
Researcher A rebuttal is complete in `reports/amp_vla/researcher_rebuttal.md`
with decision `AMP_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`.
The mathematical mechanism audit is frozen in
`reports/amp_vla/mathematical_mechanism_audit.md` with decision
`AMP_MATHEMATICAL_AUDIT_PREREGISTERED`. It forbids KL between deterministic
actions or SmolVLA flow vectors and requires projection-vs-clipping diagnostics.
Preregistration is frozen in `reports/amp_vla/preregistration.md` with decision
`AMP_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`.
The executable prototype protocol is frozen in
`reports/amp_vla/prototype_protocol.md` with decision
`AMP_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`.

AMP Stage 0 completed in `reports/amp_vla/stage_0_result.json`: `1280 /
1280` development rows, exception count `0`, duplicate/missing/extra/split
overlap counts all `0`, and final decision
`AMP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`. Bounded validation is not
allowed. This is not a closed-loop scientific kill; it is a frozen Stage 0
implementation/optimization stop driven by failed postprocessed action-validity
(`base_action_in_bounds = false`) plus negative coordinate/headroom probes.
Current cycle: `27`. Current stage:
`epoch_4_cycle_27_candidate_search_pending`.

Cycle 25 generated exactly three candidates after a current primary-source
anchor pass. `RAP-VLA`, Retrieval-Anchored Prior residualization for VLA action
flows, is selected at `94 / 100`. Its closest positive prior is OptimusVLA,
anchored to `https://arxiv.org/abs/2602.20200` and the official repository
`https://github.com/iLearn-Lab/CVPR26-OptimusVLA`.

RAP's single mechanism is retrieved legal action anchors plus bounded
residualized action-flow learning. LoRA is only identity-preserving
implementation infrastructure. The first serious comparison is exactly Base,
transparent OptimusVLA memory prior proxy, RAP full, anchor-only/no-residual
ablation, and matched standard LoRA. The Researcher A proposal is frozen in
`reports/rap_vla/researcher_proposal.md` with SHA-256
`E9C3672544E486E4D5BAA883917F8429DB0FB36982F3F5944AC26A85783D1008`. No RAP
training, validation search, rollout, simulator access, or confirmatory-test
tuning has happened. Reviewer B attack is complete in
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

RAP Stage 0 completed under the frozen protocol. Worker PID `287` completed
`640 / 640` planned development rows with exception count `0`, exact
manifest/partial key equality, duplicate partial keys `0`, missing keys `0`,
extra keys `0`, and split-overlap keys `0`. The OptimusVLA comparison status
is fixed as `optimusvla_memory_prior_proxy`.

The fixed decision is
`RAP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`, not a scientific kill.
No training, validation search, rollout, simulator access, or
confirmatory-test tuning occurred. The hard blocking gate is postprocessed
action validity (`action_validity_ok = false`, `base_action_in_bounds =
false`). Retrieval-anchor headroom was positive (`0.23865551292280293`
relative MSE improvement), but the residual probe failed
(`-3.830674623085068` relative improvement and `-0.07385385182729762`
absolute Huber improvement). Bounded validation, RAP rerun, repair, rescue,
clipping, threshold changes, and reinterpretation are forbidden. RAP remains
closed while Cycle 26 proceeds with AMP-VLA.

Epoch 4 Cycle 19 selected SPARC-VLA from exactly three candidates. Its final
Stage 0A smoke completed `2 / 2` rows with zero exceptions and no duplicate,
missing, or extra indices after the one allowed implementation repair. The
hook and identity gates passed, but both synthetic action rows failed the
frozen Base-relative range-safety gate. No labeled fit, validation, rollout,
or confirmatory evaluation occurred.

This is an implementation/prototype action-validity failure, not a scientific
kill. Do not run Stage 0B or rescue SPARC. Historical SPARC decision:
`SPARC_STAGE_0A_IMPLEMENTATION_OR_PROTOTYPE_ACTION_VALIDITY_FAILURE_NO_SCIENTIFIC_KILL`.

Cycle 20 selected NICE-VLA from exactly three candidates and froze the full
pre-implementation protocol package. Stage 0B1 closed as a collapsed-contrast
data failure with no confirmatory access.

Cycle 21 generated exactly three candidates and selected HEST-VLA at
`93 / 100`, anchored to Spline Policy. Proposal hash:
`E56B4717BDF949E1A4371457058DFC662E0D79C70D9E2FBEF35A5415FD0F0527`.
Stage 0A completed all `160 / 160` action windows with zero exceptions and
exact artifact integrity. The frozen all-variant support gate failed because
one validation Base row and HEST's required whole-Base fallback were outside
discovery-defined support. This is a pre-rollout implementation/prototype
support failure, not a scientific kill. Stage 0B and HEST rescue are forbidden.
Current cycle: `24`; current stage:
`epoch_4_cycle_24_candidate_search_pending`.

Cycle 22 selected HASTE-VLA from exactly three candidates at `95 / 100`,
anchored to StaKe. Proposal hash:
`5415BC1533A24EC55CC511DDEB014BB11D9C19F603C59D1F1D3E151E15B930A6`.
The protocol package and pushed runner commit `3dd76f0` are frozen. Stage 0A
PID `295` exited `1` before manifest persistence on NumPy JSON serialization;
no partial, model row, or scientific evidence exists. Do not repair, rerun, or
rescue HASTE. Continue to Cycle 23.

Cycle 23 selected KITE-VLA from exactly three candidates at `96 / 100`,
anchored to GeoPredict. Proposal SHA-256:
`FA00DE56D14E4C69388BE1642F7D52153841D58E77FD5A3F5C68B6C624A152B8`.
The full protocol package is frozen. Runner commit `62dbb75` passed serializer,
data/operator, real-checkpoint gradient, and identity/reload validation. Only
the foreground serializer preflight and frozen Stage 0A execution are
authorized.

KITE Stage 0A completed `128 / 128` rows through a missing-key-only resume from
115 rows. Final key and cache integrity pass, while one atomic persistence
exception remains recorded. All 128 reconstructed rows independently failed
the frozen raw action bound, with maximum absolute value
`1.1056011915206909`. The decision is
`KITE_STAGE_0A_IMPLEMENTATION_FAILURE`, not a scientific kill. Stage 0B,
rerun, and KITE rescue are forbidden. Continue to Cycle 24.

The prior fixed-cycle terminal stop is procedurally invalid under the current Goal. Epoch 1 is corrected as a completed related-method set that requires an Epoch 2 pivot.

Corrected adjudication:

- Cycle 1 `DICD-VLA`: `UNDERPOWERED_STAGE_A_NON_GO_ARCHIVED`
- Cycle 2 `FEDO-VLA`: `VALID_CURRENT_FORMULATION_KILL`
- Cycle 3 `GCAP-VLA`: `UNDERPOWERED_TARGET_AXIS_NON_GO_ARCHIVED`

Epoch 2 Cycle 1 `PTC-VLA` is archived as `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`: full PTC reached `0 / 10` versus frozen SmolVLA `3 / 10`, with zero exceptions and active transition mechanism.

Epoch 2 Cycle 2 `SACF-VLA` is archived as `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`: full SACF reached `0 / 10` versus frozen SmolVLA `7 / 10`, with zero exceptions and active semantic mechanism.

Epoch 2 Cycle 3 `OCFN-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: expanded Stage B completed `80` paired episodes per key policy with zero exceptions and active mechanism. OCFN full reached `26 / 80`, zero-noise SmolVLA reached `27 / 80`, and the paired upper confidence bound for full minus zero-noise was `0.0625`.

Epoch 3 Cycle 1 `CBFD-VLA` is archived as `STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE`: full CBFD reached `0 / 10` while frozen SmolVLA reached `7 / 10`, with zero exceptions and active mechanism.

Epoch 3 Cycle 2 `SCVC-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: full SCVC reached `11 / 40`, shifted frozen SmolVLA reached `20 / 40`, and the paired bootstrap CI versus shifted frozen was `[-0.425, -0.025]`.

Epoch 3 Cycle 3 `PSE-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: expanded Stage B completed `400 / 400` rows with zero exceptions, full PSE reached `50 / 80`, bright-single reached `51 / 80`, and the paired CI versus bright-single was `[-0.1000, 0.0750]`.

Epoch 4 Cycle 1 `RCV-VLA` is archived as `STAGE_2B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: Stage 2B completed `200 / 200` episodes with zero exceptions, full RCV reached `20 / 40`, no-context ablation reached `24 / 40`, and stateless first-action reached `24 / 40`.

Epoch 4 Cycle 2 `CAVM-VLA` is archived as `STAGE_2B_EXPANDED_NON_GO_NO_THIRD_EXPANSION`: the expanded result completed `290 / 290` rows with zero exceptions, full CAVM reached `24 / 58`, nearest-success replay reached `23 / 58`, frozen SmolVLA reached `22 / 58`, success-only memory proxy reached `20 / 58`, and no-contrast ablation reached `21 / 58`.

Epoch 4 Cycle 3 selected and preregistered `FANG-VLA`. Proposal hash: `6837DBA2A1307F7C9938FA9F5463ED483907AF3C168F1C0514F6E281804E859B`.

The development audit passed and the calibrated validation search selected `fang_c01`. The uncalibrated gate failure is preserved as a negative validation result. Stage A completed `50 / 50` episodes with all five policies tied at `3 / 10`.

Stage B completed `200 / 200` episodes with zero exceptions. Full FANG reached `11 / 40`, while frozen SmolVLA reached `16 / 40`, AFIL local proxy reached `15 / 40`, nearest-success replay reached `14 / 40`, and the no-failure ablation also reached `11 / 40`. Full-minus-base paired delta was `-0.125` with CI `[-0.250, 0.000]`; full was exactly tied with the key ablation.

Final FANG decision: `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT`. Do not rescue this formulation.

Epoch 4 Cycle 4 selected and preregistered `EvoState-VLA`. Proposal hash: `A44ED68CC8E1F296DB8B0B3E16FF84D7D5BBE684EAF63EAE29E7CC91DCFD93C9`.

Stage 0 stopped before rollout as `AUDIT_STOP_DESIGN_FAILURE`: the full transition model improved only `0.024689` over an actionless model, below the preregistered `0.05` threshold.

Epoch 4 Cycle 5 selected and preregistered `RAC-VLA`, a Reflective VLA-anchored frozen-policy action-consequence calibration method. Proposal hash: `71ABA93E37FC725C1A2E5EAE6E1461BC77AACDAFF9B0711C37F17D5C0AB0902F`.

RAC Stage 0 passed without rollout: full action-consequence validation accuracy `0.585745` beat action-only `0.368496` and no-consequence `0.374483`, with margin `0.211262`; clean action delta p95 was `0.0`. The six-config validation search selected `rac_h4_a0.05` with score `0.508926`.

Stage A completed `50 / 50` episodes with zero exceptions. RAC full reached `0 / 10`, frozen shifted Base reached `0 / 10`, the no-consequence ablation reached `0 / 10`, the Reflective-history proxy reached `1 / 10`, and the online diagonal inverse-gain baseline reached `1 / 10`. This was `STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`, not a valid Stage A kill.

Stage B completed `200 / 200` episodes with zero exceptions and a valid shared manifest. RAC full reached `1 / 40`, shifted Base reached `1 / 40`, Reflective-history proxy reached `1 / 40`, no-consequence ablation reached `2 / 40`, and online diagonal inverse-gain reached `2 / 40`. Full-minus-ablation paired delta was `-0.025` with CI `[-0.125, 0.050]`; full-minus-simple-baseline paired delta was also `-0.025` with CI `[-0.125, 0.050]`.

Final RAC decision: `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT`. Do not rescue or retune RAC.

The post-RAC governance update is installed and active. It requires future methods to maximize the probability of an honest paper-worthy positive result through stronger positive-prior-anchored design, usable-headroom audits, data/supervision health gates, identity-preserving integration, bounded validation search, mathematical objective engineering, mechanism smoke, and frozen confirmatory tests.

Epoch 4 Cycle 6 generated exactly three post-RAC candidates and selected `MTF-VLA`. Proposal hash: `11DC94A2B75CD8605577AB044E5743DFDA4131A4FA7F6C6A7390519B9F995B31`.

MTF-VLA is a FrameSkip and StructVLA anchored milestone-transition data-supervision method for identity-preserving SmolVLA adapter training. The first comparison is frozen to Base, FrameSkip proxy, MTF full, no-retention ablation, and uniform retained-ratio LoRA.

MTF Stage 0 development audit passed without training or closed-loop rollout using the official stable train/val/test prediction artifact: `1600` development records, duplicate sample keys `0`, duplicate frame keys `0`, high-low score gap `0.585702`, gripper-transition fraction `0.341875`, and adapter-init action delta p95 `0.0`.

The bounded six-config validation search selected `mtf_r20_ret100`: retained high-frame ratio `0.20`, retention coefficient `1.00`, validation score `0.643663`, `176` high train frames, and `391` base-retention train frames. The selected config and training manifest are frozen under `reports/mtf_vla/`.

Next action: train disk-reloadable selected-config adapter checkpoints for MTF full, no-retention ablation, FrameSkip proxy, and uniform retained-ratio LoRA before any Stage A rollout.


The MTF adapter-training runner is now implemented and dry-run validated. The frozen selected manifest produces four trainable jobs: MTF full `567` events (`176` milestone + `391` retention), no-retention ablation `176`, FrameSkip proxy `176`, and uniform retained-ratio LoRA `240`, with zero train/validation/test frame overlap. This is not adapter training yet; it is the validated checkpoint-production contract.

Next action: run the MTF adapter trainer to produce and disk-reload verify all four selected-config checkpoints before any Stage A rollout.

MTF adapter training is now complete for all four trainable Stage A policies after repairing the development-only FrameSkip proxy collapse. The checkpoints are saved under `runs/mtf_vla_checkpoints/mtf_r20_ret100`, disk-reloaded successfully, and summarized in `reports/mtf_vla/adapter_checkpoint_manifest.json`. Validation action L2 means were `0.082590885` for MTF full, `0.082867367` for no-retention, `0.082553130` for the corrected FrameSkip proxy, and `0.082396918` for uniform retained-ratio LoRA. No rollout or confirmatory-test tuning occurred.

The MTF Stage A manifest is frozen in `reports/mtf_vla/stage_a_manifest.json` and has now completed as `reports/mtf_vla/stage_a_result.json`. It used exactly `frozen_smolvla`, `frameskip_proxy_lora`, `uniform_retained_ratio_lora`, `mtf_no_retention_ablation`, and `mtf_full`; `frameskip_proxy_lora` is a faithful local proxy rather than an official FrameSkip reproduction. Stage A completed `50 / 50` official LIBERO episodes with zero exceptions. Frozen SmolVLA, FrameSkip proxy, and uniform retained-ratio LoRA each reached `8 / 10`; no-retention and MTF full each reached `7 / 10`. The frozen decision is `MTF_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`, so Stage B is required.

The MTF Stage B manifest `reports/mtf_vla/stage_b_manifest.json` completed as `reports/mtf_vla/stage_b_result.json`: `200 / 200` official LIBERO episodes, zero exceptions, all `20` official tasks, reset seeds `20261203` and `20261204`, and the unchanged five-policy comparison. Frozen SmolVLA reached `28 / 40`, the FrameSkip proxy reached `27 / 40`, uniform retained-ratio LoRA reached `29 / 40`, the no-retention ablation reached `32 / 40`, and MTF full reached `26 / 40`.

Final MTF decision: `MTF_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD`. Full-minus-no-retention paired delta was `-0.15` with CI `[-0.275, -0.025]`, so the simpler ablation explains or exceeds the full method. Do not rescue or retune MTF.

Epoch 4 Cycle 7 generated exactly three post-MTF candidates in `reports/epoch_4_cycle_7_candidate_generation.md` and selected `DAGR-VLA`, a DAM-VLA anchored dynamic arm/gripper routing method. Proposal hash: `BDE0EC67ACE8EC457CE6495D723EE476064F3D80946151326B11F0B5A1AFEF89`.

Reviewer B attack completed with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`: DAGR is not killed before implementation, but novelty is narrowed to frozen SmolVLA identity-preserving route-gated residual adaptation, `dam_static_component_proxy` must remain a transparent local proxy, and Stage 0 must reject collapsed or unobservable route supervision before rollout.

Researcher A rebuttal completed with decision `DAGR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. The DAGR mathematical audit, preregistration, and prototype protocol are now frozen under `reports/dagr_vla/`.

DAGR Stage 0 passed as `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`: `1600` development records, zero duplicate sample/frame keys, zero train/validation/test overlap, validation any-route fraction `0.865`, route-probe margins `0.0375`, `0.0725`, and `0.26`, and no hard stops.

The bounded six-config validation search selected `dagr_a020_route_mlp` as `VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING`: residual alpha `0.20`, route architecture `mlp`, validation score `0.8571740870493018`, delta L2 p95 `0.008609326556324959`, clean delta L2 p95 `0.00672802422195673`, and action validity `1.0`.

DAGR policy identity training completed as `DAGR_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY`. The checkpoint root is `runs/dagr_vla_checkpoints/dagr_a020_route_mlp`; `dagr_full`, `dam_static_component_proxy`, and `dagr_no_dynamic_route_ablation` all disk-reload and keep validation action validity `1.0`, while `gripper_transition_heuristic` is a saved nontrainable identity.

The DAGR Stage A manifest is frozen as `DAGR_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT`: `50` planned episodes, reset seeds `20261205` and `20261206`, canonical hash `8379E47D3C3C73E21ADDD285491750E7406B8389578C0003278E5E187EA27E7B`, and the unchanged five-policy comparison.

DAGR Stage A policy preflight passed as `DAGR_STAGE_A_PREFLIGHT_PASS_READY_FOR_OFFICIAL_ROLLOUT`: `5` policies loaded through the official SmolVLA/LIBERO path, `4` checkpoint identities checksum-verified, CUDA checks passed, no accidental checkpoint reuse was detected, and finite 7D action wrappers were produced. At preflight time, no DAGR closed-loop rollout or confirmatory-test tuning had happened.

DAGR Stage A completed as `DAGR_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`: `50 / 50` official LIBERO episodes, zero exceptions, frozen SmolVLA `8 / 10`, gripper-transition heuristic `7 / 10`, DAGR full `6 / 10`, no-dynamic-route ablation `5 / 10`, and DAM static proxy `2 / 10`. This is not a valid Stage A kill; freeze the DAGR Stage B matched manifest next without retuning.

The DAGR Stage B manifest froze all `20` official tasks, reset seeds `20261207` and `20261208`, `40` paired cases per policy, `200` total episodes, canonical hash `2A14FA11271EC8FAD9BD91A1251952E9039A5BD297105BEBB78E27EFC4470A3B`, and the unchanged five-policy comparison.

DAGR Stage B completed `200 / 200` official LIBERO episodes with zero exceptions and no confirmatory-test tuning. Frozen SmolVLA reached `28 / 40`; the DAM-style static component proxy reached `5 / 40`; DAGR full reached `18 / 40`; the no-dynamic-route ablation reached `16 / 40`; and the gripper-transition heuristic reached `24 / 40`. Full-minus-Base paired delta was `-0.25` with CI `[-0.4, -0.1]`; full-minus-gripper paired delta was `-0.15` with CI `[-0.3, 0.0]`.

Final DAGR decision: `DAGR_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD`. This is a valid current-formulation kill because the simple gripper-transition heuristic and Base explain or exceed the full method under the frozen protocol. Do not rescue DAGR by retuning `dagr_a020_route_mlp`, changing route thresholds, changing task/reset identities, changing the policy list, or reinterpreting partial results.

Epoch 4 Cycle 8 generated exactly three post-DAGR candidates in `reports/epoch_4_cycle_8_candidate_generation.md` and selected `MARC-VLA`, Median-Anchored Regression Correction for frozen SmolVLA flow actions. Proposal hash: `D1F910465D4E415C996B3F8C7CE2B2CF47339EA94D697B06A9DCED49AC1E585A`.

Reviewer B attack completed with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`; Researcher A rebuttal completed with decision `MARC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. MARC is narrowed to a frozen-SmOLVLA, identity-preserving median-anchor correction of the OpenVLA-OFT continuous-action prior, not a broad claim that L1 continuous actions are novel.

MARC Stage 0 passed as `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`: `1600` development records, `1200` train records, `400` validation records, `1200` reserved test records not used, zero duplicate sample/frame keys, zero split overlap, train disagreement positive fraction `0.4`, validation disagreement positive fraction `0.44`, and gate-probe margin `0.0475`.

The bounded six-config validation search selected `marc_a020_gate_mlp` as `VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING`: correction alpha `0.20`, gate architecture `mlp`, validation score `0.5457964262366295`, gate accuracy margin `0.0525`, gate predicted-positive fraction `0.3325`, delta L2 p95 `0.011818917468190193`, clean delta L2 p95 `0.010853752493858337`, and action validity `1.0`. Linear configs were stopped for collapsed gates.

MARC full differs from the L1 proxy on validation (`0.007010325323790312` mean L2), but full-versus-static mixture is small (`0.0019475044682621956` mean L2). The static L1 mixture therefore remains the important simple reviewer-killer in the frozen five-policy comparison.

MARC policy identity training completed as `MARC_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY`. Checkpoints are saved under `runs\marc_vla_checkpoints\marc_a020_gate_mlp`; all four trainable identities disk-reload, keep validation action validity `1.0`, and preserve initial base passthrough with initial delta p95 `0.0`. MARC full delta L2 p95 is `0.010693175718188286`, while the L1 proxy and static mixture have p95 values `0.2307613492012024` and `0.07999999821186066`.

The disk-reloaded policy identities are action-distinct: full-versus-L1 mean L2 is `0.08430124074220657`, full-versus-no-gate is `0.04372206702828407`, and full-versus-static mixture is `0.032826922833919525`. No closed-loop rollout or confirmatory-test tuning happened during policy identity training.

The MARC Stage A manifest is now frozen as `MARC_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT`: `50` planned episodes, reset seeds `20261209` and `20261210`, canonical hash `3383E377CEDD2B44E7730AAD3617E64838786E7094B9CF60D39F9679DE97D74E`, and the unchanged five-policy comparison. `openvla_oft_l1_proxy` is explicitly a faithful transparent local proxy, not an official OpenVLA-OFT reproduction. No closed-loop rollout or confirmatory-test tuning happened during manifest freeze.

MARC Stage A preflight passed as `MARC_STAGE_A_PREFLIGHT_PASS_READY_FOR_OFFICIAL_ROLLOUT`: `5` policies loaded through the official SmolVLA/LIBERO path, `4` checkpoint identities checksum-verified, CUDA checks passed, no accidental checkpoint reuse was detected, and finite 7D MARC actions were produced.

MARC Stage A completed as `MARC_STAGE_A_CATASTROPHIC_KILL_ZERO_VS_STRONG_BASELINE`: `50 / 50` official LIBERO episodes from `runs/marc_vla_stage_a/20260714T171356Z`, zero exceptions, frozen SmolVLA `8 / 10`, OpenVLA-OFT-style L1 proxy `0 / 10`, MARC full `0 / 10`, no-disagreement-gate ablation `7 / 10`, and static L1 mixture `7 / 10`. Full-minus-Base paired delta was `-0.8`; full-minus-no-gate was `-0.7`; full-minus-static was `-0.7`.

Final MARC decision: valid current-formulation kill. Do not rescue MARC by retuning checkpoints, changing thresholds, changing policies, changing task/reset identities, or reinterpreting Stage A outcomes.

Epoch 4 Cycle 9 generated exactly three post-MARC candidates in `reports/epoch_4_cycle_9_candidate_generation.md` and selected `PESA-VLA`, Prior-Expert Spectral Adaptation for frozen SmolVLA 7D policies. PESA is anchored to PriorVLA, LoRA-SP, and VLA-GSE, with a design-level comparison against Base, a PriorVLA-style proxy, PESA full, a no-spectral/no-prior-query ablation, and one strongest simple standard-LoRA or clean-retention adaptation baseline.

The PESA Researcher A proposal is frozen in `reports/pesa_vla/researcher_proposal.md` with proposal hash `B05B1ACF7CD3514365B418E25C7E995604FCA8C117CDC0F3384F1046BAF26B63`.

Reviewer B attack is complete in `reports/pesa_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. PESA is not killed before implementation, but novelty is narrow and must survive the frozen five-policy comparison against Base, PriorVLA-style proxy, PESA full, no-spectral/no-prior-query ablation, and one strongest simple standard-LoRA or clean-retention adaptation baseline.

Researcher A rebuttal is complete in `reports/pesa_vla/researcher_rebuttal.md` with decision `PESA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. The rebuttal accepts the narrow claim, preserves the simple killer and closest-prior proxy, and requires mathematical objective/gradient audit before implementation.

The PESA mathematical mechanism audit is frozen in `reports/pesa_vla/mathematical_mechanism_audit.md` with decision `PESA_MATHEMATICAL_AUDIT_PREREGISTERED`. It explicitly forbids KL between deterministic 7D actions and requires Base-passthrough, bounded deltas, spectral activation, gradient, label-health, and clean-retention audits before rollout.

The PESA preregistration and prototype protocol are frozen in `reports/pesa_vla/preregistration.md` and `reports/pesa_vla/prototype_protocol.md`. The first serious comparison is fixed to exactly five policies and the validation search is capped at six named configurations.

PESA Stage 0 completed without rollout, training, manifest freeze, or confirmatory-test tuning. The development audit is saved in `reports/pesa_vla/development_audit.json`.

Final PESA Stage 0 decision: `DESIGN_FAILURE`. The hard stop was query-probe validation margin `-0.07750000000000001`, below the frozen `+0.02` requirement. Do not rescue PESA by retuning labels, thresholds, features, or criteria.

Current PESA disposition: `PESA_STAGE_0_STOP_DESIGN_FAILURE`. This remains a pre-rollout design stop, not a closed-loop kill.

Epoch 4 Cycle 10 generated exactly three post-PESA candidates in `reports/epoch_4_cycle_10_candidate_generation.md` after the prior map in `reports/epoch_4_cycle_10_prior_mechanism_map.md`, and selected `EAC-VLA`, Entropy-Calibrated Adaptive Chunking for frozen SmolVLA.

EAC is anchored to Adaptive Action Chunking. It preserves frozen SmolVLA weights and emitted 7D action values, changing only action-queue commitment length from deployment-observable uncertainty and queue-boundary risk. The frozen design-level five-policy comparison is Base fixed queue, AAC entropy-only proxy, EAC full, no-calibration/no-hysteresis ablation, and fixed short-replan simple killer.

The EAC Researcher A proposal is frozen in `reports/eac_vla/researcher_proposal.md` with proposal hash `A89ED48AE9FD4D26A8DA9E3E987FACDBBD9F861D070AE135372A092A44581E4E`.

Reviewer B attack is complete in `reports/eac_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. It requires Researcher A to accept narrow AAC-extension novelty, keep the AAC proxy and fixed short-replan simple killer live, audit uncertainty/dispersion validity, and treat action-value modification as implementation failure.

Researcher A rebuttal is complete in `reports/eac_vla/researcher_rebuttal.md` with decision `EAC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. It accepts the review constraints and passes only to mathematical mechanism audit, not implementation.

The EAC mathematical mechanism audit is frozen in `reports/eac_vla/mathematical_mechanism_audit.md` with decision `EAC_MATHEMATICAL_AUDIT_PREREGISTERED`. It defines exact variables, shapes, dispersion/entropy rules, action-value passthrough, validation search limits, required ablation, and Stage 0 hard stops.

The EAC preregistration and prototype protocol are frozen in `reports/eac_vla/preregistration.md` and `reports/eac_vla/prototype_protocol.md`.

EAC Stage 0 completed without training, validation search, closed-loop rollout, or confirmatory-test tuning. The audit is saved in `reports/eac_vla/stage_0_audit.json` and passed as `AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH`: `2000` validation records, `400` validation frames, `6000` reserved confirmatory records untouched, zero validation/test overlap, first-two dispersion p95 `0.0007983036317792467`, commitment counts `2:136`, `8:132`, `50:132`, and passthrough max error `5.07000000038449e-07`.

Because the canonical artifact stores first-two previews rather than all `50` postprocessed chunk actions, the runtime full-chunk equality and queue-prefix execution check was run before validation search.

EAC runtime queue check completed without training, validation search, closed-loop rollout, or confirmatory-test tuning. It loaded frozen SmolVLA on `NVIDIA GeForce RTX 5080`, produced a full postprocessed chunk shape `[50, 7]`, verified `select_action` matched `chunk[0]` with max absolute diff `0.0`, observed queue length `0 -> 49`, and verified every commitment prefix in `{1, 2, 4, 8, 16, 50}` preserved action values exactly.

EAC bounded validation search completed with exactly six configurations and no confirmatory records used for tuning. The selected frozen config is `eac_q33_aggressive_1_4_50`, with validation score `0.7530415186081504`, commitment counts `1:132`, `4:136`, `50:132`, policy-calls-per-step proxy `0.4216`, and risk-exposure-reduction proxy `0.9032794643799159`.

The EAC Stage A matched manifest is frozen in `reports/eac_vla/stage_a_manifest.json` with canonical payload hash `63E96D0629F3D34E4801EB1084D094CB287EC4F2F2FCD96373981787EDA9954C`, reset seeds `20261211` and `20261212`, `10` paired cases per policy, and `50` total planned episodes. EAC Stage A policy preflight passed in `reports/eac_vla/stage_a_preflight.json`: CUDA was available on `NVIDIA GeForce RTX 5080`, output shape was `[50, 7]`, and all policy prefixes preserved action values exactly.

EAC Stage A runner validation passed in `reports/eac_vla/stage_a_runner_validation.json`: the runner preserves action values, reconstructs frozen validation-only thresholds, and authorizes the frozen Stage A rollout without training or confirmatory-test tuning.

EAC Stage A completed `50 / 50` episodes with zero exceptions. EAC full reached `8 / 10`; Base fixed queue, no-calibration ablation, and fixed short-replan each reached `7 / 10`; AAC entropy proxy reached `9 / 10`.

The EAC Stage B matched manifest is frozen in `reports/eac_vla/stage_b_manifest.json` with canonical payload hash `31F7590D81D95AECE9D7D1E8D6A2332364D5A9B36F6A913F9634D30D2C27B24D`, all `20` official tasks, fresh reset seeds `20261213` and `20261214`, `40` paired cases per policy, and `200` planned episodes.

EAC Stage B completed from the detached run `runs/eac_vla_stage_b/20260714T202334Z` with wrapper exit code `0`, `200 / 200` official LIBERO episodes, zero exceptions, and no confirmatory-test tuning. The final result is saved in `reports/eac_vla/stage_b_result.json` and summarized in `reports/eac_vla/stage_b_result.md`.

Stage B decision: `EAC_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD`. Frozen Base fixed queue reached `30 / 40`, AAC entropy proxy reached `30 / 40`, EAC full reached `29 / 40`, the no-calibration/no-hysteresis ablation reached `30 / 40`, and fixed short-replan reached `29 / 40`. EAC preserved action values, kept finite valid `[50, 7]` action chunks, and activated the scheduler with commitment counts `{'1': 807, '4': 199, '50': 148}`.

EAC full-minus-Base paired delta was `-0.025` with CI `[-0.175, 0.125]`; full-minus-AAC proxy was `-0.025` with CI `[-0.15, 0.1]`; full-minus-ablation was `-0.025` with CI `[-0.175, 0.125]`; and full-minus-fixed-short-replan was `0.0` with CI `[-0.15, 0.15]`.

Final EAC decision: valid current-formulation kill. Do not rescue EAC by retuning `eac_q33_aggressive_1_4_50`, changing thresholds, changing tasks or resets, changing the five-policy list, reinterpreting partial results, or applying any post-hoc expansion.

Epoch 4 Cycle 11 generated exactly three post-EAC candidates in `reports/epoch_4_cycle_11_candidate_generation.md` after the prior map in `reports/epoch_4_cycle_11_prior_mechanism_map.md`, and selected `G3P-VLA`, Grounded 3D Point Injection for frozen SmolVLA.

G3P is anchored to Direct Action-Head Injection of A Grounded 3D Point, with RoboPoint, RoboGround, and AffordanceVLA as secondary spatial-grounding priors. The selected design changes the mechanism axis from queue scheduling to source-gated gripper-relative spatial grounding at the action interface. It must use only deployment-observable RGB, proprioception, language, and Base features at inference; oracle object state may be used only for discovery/validation diagnostics and training labels, never as hidden confirmatory-test input.

The design-level five-policy comparison is Base, a closest-prior 3D-point proxy, G3P full, no-3D/no-injection ablation, and one simple 2D/phase/nearest-object heuristic.

The G3P-VLA Researcher A proposal is frozen in `reports/g3p_vla/researcher_proposal.md` with proposal hash `BEE3822D8F54EFBD09C1CA47A9BF126EBE694B7B6219002FF770C5794ED7AA71`. Reviewer B attack is complete in `reports/g3p_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. Researcher A rebuttal is complete in `reports/g3p_vla/researcher_rebuttal.md` with decision `G3P_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. The mathematical mechanism audit is frozen in `reports/g3p_vla/mathematical_mechanism_audit.md` with decision `G3P_MATHEMATICAL_AUDIT_PREREGISTERED`. The preregistration and prototype protocol are frozen in `reports/g3p_vla/preregistration.md` and `reports/g3p_vla/prototype_protocol.md`.

G3P Stage 0 completed without training, validation search, rollout, or confirmatory-test tuning. The development audit is saved in `reports/g3p_vla/development_audit.json` and summarized in `reports/g3p_vla/development_audit.md`.

Final G3P Stage 0 decision: `DATA_OR_SUPERVISION_FAILURE`. The material point label collapsed with train material fraction `0.9982142857142857` and validation material fraction `1.0` under the frozen Stage 0 gate. Do not rescue G3P by changing labels or thresholds.

Epoch 4 Cycle 12 generated exactly three post-G3P candidates in `reports/epoch_4_cycle_12_candidate_generation.md` after the prior map in `reports/epoch_4_cycle_12_prior_mechanism_map.md`, and selected `CALA-VLA`, Context-Gated Action-Latent Adapter for frozen SmolVLA.

CALA is anchored to CAC-VLA, with VLS and World Pilot as secondary action-interface priors. The selected design changes the mechanism axis from source-gated point labels to action-structured latent conditioning. Future 7D action segments may be used only as discovery/validation supervision; inference must use only deployment-observable current RGB, proprioception, language, and Base features.

The design-level five-policy comparison is Base, a CAC-style latent-action proxy, CALA full, no-context-gate ablation, and one simple task-mean latent-action baseline.

The CALA-VLA Researcher A proposal is frozen in `reports/cala_vla/researcher_proposal.md` with proposal hash `5B3933C9C0FD5AE5F07FDB0CEC447B48040238FB6D872D97E545E3D93E257E76`.

Reviewer B attack is complete in `reports/cala_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. CALA is not killed before implementation, but novelty is narrowed to frozen-SmolVLA identity-preserving CAC-style latent-action adaptation. The CAC proxy, future-action leakage gate, no broad latent-action novelty claim, and task-mean simple baseline must remain live.

Researcher A rebuttal is complete in `reports/cala_vla/researcher_rebuttal.md` with decision `CALA_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. It accepts the narrowed novelty, source-fidelity, future-action leakage, task-mean baseline, matched ablation, and identity-preserving integration constraints.

The CALA mathematical mechanism audit is frozen in `reports/cala_vla/mathematical_mechanism_audit.md` with decision `CALA_MATHEMATICAL_AUDIT_PREREGISTERED`. It defines the latent-action encoder, source legality, context gate, identity-preserving residual, objective terms, gradient paths, small-batch magnitude audit, bounded validation search, first five-policy comparison, and no deterministic-action KL rule.

The CALA preregistration and prototype protocol are frozen in `reports/cala_vla/preregistration.md` and `reports/cala_vla/prototype_protocol.md`. Stage 0 must run before validation search, training, manifest freeze, or rollout.

CALA Stage 0 is complete in `reports/cala_vla/development_audit.json` and `reports/cala_vla/development_audit.md` with final decision `DESIGN_FAILURE`. This is a pre-rollout development stop, not a closed-loop scientific kill: source legality passed, future action segments and latent labels were not used at inference, split duplicates were `0`, latent variance was healthy, Base action validity was `1.0`, initial action delta p95 was `0.0`, and diagnostic action headroom was `0.08630366897708504`. The hard stop was latent predictability: the deployment-observable full probe had margin `-0.01171824382857035` because `action_history_only` was the strongest trivial baseline (`3.1439661695829484` RMSE) and beat the full probe (`3.198806582620636` RMSE).

Do not rescue CALA by changing latent labels, prediction features, thresholds, validation configs, or the source gate. Validation search, training, Stage A manifest freeze, and rollout are disallowed for this CALA formulation.

Epoch 4 Cycle 13 generated exactly three post-CALA candidates in `reports/epoch_4_cycle_13_candidate_generation.md` after the prior map in `reports/epoch_4_cycle_13_prior_mechanism_map.md`, and selected `RAR-VLA`, Re-Anchored Autoregressive Residuals for frozen SmolVLA.

RAR is anchored to AR-VLA, with ReactVLA and DSWAM as secondary action-generation priors. The design changes the mechanism axis from future-action latent prediction to causal action memory: it uses current Base action chunks, proprioception, task identity, and previous emitted actions, with a re-anchored zero-initialized residual gate. The frozen first comparison is Base, an AR-VLA re-anchored expert proxy, RAR full, no-reanchor-memory ablation, and `ema_action_history_baseline`.

The RAR-VLA Researcher A proposal is frozen in `reports/rar_vla/researcher_proposal.md` with proposal hash `723C16C3885A974E2CA12D90BC36267FA6E86827AC9D2A1E0E0E475E16FB0E56`.

Reviewer B attack is complete in `reports/rar_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. RAR is not killed before implementation, but novelty is narrowed to a frozen-SmolVLA identity-preserving AR-style residual memory adapter. REMAC/TAS distinctions, transparent AR proxy status, inter-chunk and intra-chunk Stage 0 diagnostics, and the `ema_action_history_baseline` simple killer must remain live.

Researcher A rebuttal is complete in `reports/rar_vla/researcher_rebuttal.md` with decision `RAR_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. It accepts the narrowed novelty, REMAC/TAS distinction, transparent AR proxy status, EMA/action-history killer baseline, source legality, and identity-preserving integration constraints.

The RAR mathematical mechanism audit is frozen in `reports/rar_vla/mathematical_mechanism_audit.md` with decision `RAR_MATHEMATICAL_AUDIT_PREREGISTERED`. It defines legal causal memory, residual/gate formulas, objective terms, gradient paths, no deterministic-action KL, EMA/history baselines, REMAC/TAS distinctions, bounded validation search, and the first five-policy comparison.

The RAR preregistration and prototype protocol are frozen in `reports/rar_vla/preregistration.md` and `reports/rar_vla/prototype_protocol.md`. Stage 0 must run before validation search, training, manifest freeze, or rollout.

RAR Stage 0 is complete in `reports/rar_vla/development_audit.json` and `reports/rar_vla/development_audit.md` with final decision `DESIGN_FAILURE`. This is a pre-rollout development stop, not a closed-loop scientific kill: source legality passed, future actions and CALA latents were not used at inference, split duplicates were `0`, residual headroom was `0.08630366897708504`, gradients were finite/nonzero, Base action validity was `1.0`, and initial action delta p95 was `0.0`. The hard stop was residual predictability: the legal full probe had margin `-0.03837609884238533` because `zero_residual` was the strongest trivial baseline (`0.16559729909097304` RMSE) and beat the full probe (`0.1719540079557317` RMSE).

Do not rescue RAR by changing history features, residual labels, thresholds, validation configs, or source gates. Validation search, training, Stage A manifest freeze, and rollout are disallowed for this RAR formulation.

Epoch 4 Cycle 14 generated exactly three post-RAR candidates in `reports/epoch_4_cycle_14_candidate_generation.md` after the prior mechanism map in `reports/epoch_4_cycle_14_prior_mechanism_map.md`, and selected `COVI-VLA`, Complementary Occlusion View Imagination for frozen SmolVLA.

COVI is anchored to LIBERO-Occ / Viewpoint Imagination, with CamVLA and STRONG-VLA as secondary priors. The design changes the mechanism axis from causal action-memory residuals to scene-induced partial observability: it uses legal current observations, proprioception, task/language input, Base action chunks, and an internally predicted complementary-view representation, with an identity-preserving visual adapter gate. The frozen first comparison is Base under occlusion, a VIM-style transparent proxy, COVI full, no-imagined-view ablation, and `random_cutout_clean_retention_baseline`.

The COVI-VLA Researcher A proposal is frozen in `reports/covi_vla/researcher_proposal.md` with proposal hash `338430D2C6CF1D82410C036D79102ED3F38B2367BB35B9AE2811161698A3E621`.

Reviewer B attack is complete in `reports/covi_vla/reviewer_attack.md` with decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. It requires narrowed novelty, transparent VIM proxy status, direct two-camera fusion diagnostics, physical occlusion validation, and the live random-cutout simple killer.

Researcher A rebuttal is complete in `reports/covi_vla/researcher_rebuttal.md` with decision `COVI_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. The rebuttal accepts all Reviewer B constraints.

The COVI mathematical mechanism audit is frozen in `reports/covi_vla/mathematical_mechanism_audit.md` with decision `COVI_MATHEMATICAL_AUDIT_PREREGISTERED`. The audit keeps the narrowed feature-adapter claim, legal source gate, direct two-camera diagnostic, random-cutout simple killer, physical occlusion requirement, identity-preserving integration, bounded validation search, and no deterministic-action KL constraint live.

The COVI preregistration and prototype protocol are frozen in `reports/covi_vla/preregistration.md` and `reports/covi_vla/prototype_protocol.md` under `APPROVE_WITH_FIXED_EMPIRICAL_RISKS`. The measured official hook is `[64, 960]` visual tokens per stream, and the Stage 0 split is `600` fit, `600` sealed one-check, `400` validation, and `1200` untouched confirmatory records.

Previous stage: `epoch_4_cycle_14_covi_stage_0_implementation_pending`. COVI Stage 0 is preserved and adjudicated as `COVI_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE_NO_SCIENTIFIC_KILL`; no one-check, validation search, rollout, or confirmatory-test tuning occurred.

Epoch 4 Cycle 15 generated exactly three candidates and selected `LIFT-VLA` with score `90 / 100`. LIFT narrowly transfers pathwise classifier-free guidance to SmolVLA's continuous action flow and tests it against CAG final-action mixing under a matched two-branch inference budget.

The frozen four-policy comparison is Base, transparent training-free CAG, LIFT full, and last-step-only LIFT. No standard-LoRA or fifth-policy control is included because LIFT is inference-only, the backbone remains frozen, and those controls do not test the claimed mechanism.

The proposal is frozen at `reports/lift_vla/researcher_proposal.md` with hash `3D263AA6FF73B342523D85AD4854145AF4D79DE2B90C6119F417D37A8B08F55F`.

Reviewer B returned `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. Researcher A accepted the narrow novelty boundary, feasible counterfactual benchmark gate, native-flow same-noise CAG, matched-compute ablation, practical-equivalence threshold, headroom, and compute gates.

The frozen Stage 0 completed and is adjudicated in `reports/lift_vla/stage_0_adjudication.md`. Manifest, shape, identity, activation, field-count, separation, memory, and latency gates passed, but executed-action bound validity was `0.8023809523809524` against the frozen `1.0` requirement. The final decision is `LIFT_COMPUTE_INFEASIBLE`; clipping, scale changes, headroom rollout, and validation search are forbidden. Confirmatory policy observations and actions remained zero. The campaign then advanced to Epoch 4 Cycle 16 candidate generation.

Epoch 4 Cycle 16 generated exactly three candidates and selected `IARC-VLA`
with `95 / 100`. Its scientific method is actual-step projected SGD during
clean refinement against a paired perturbation-replay SmolVLA action gradient;
rank-4 LoRA is implementation infrastructure.

Proposal, review, rebuttal, mathematical audit, preregistration, and executable
protocol are frozen under `reports/iarc_vla/` with proposal hash
`A1B0CF8BCBCF6A88F27B31EF5E38BAF408A3E62BB34206A1AC9F051EA6B57408`.
IARC Stage 0A completed `40 / 40` gradient pairs and `40 / 40` validation rows
with zero exceptions and zero duplicate/manifest mismatches. It produced
`18 / 40` conflicts across all four families, passed projection `18 / 18`,
kept agreeing rows unchanged `22 / 22`, preserved Base weights, and disk-
reloaded exactly.

Dataset-range action validity was `12 / 40 = 0.30`, below frozen `1.0`. The
decision is `IARC_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`, not a scientific
kill. No clipping, one-check, Stage 0B, or validation search is allowed.
Confirmatory decode/action counts remain zero.

Cycle 17 generated exactly three candidates and selected `FAMR-VLA`,
Function-Aware Model Retention, with `93 / 100`. RETAIN is the closest positive
prior and Fisher-weighted model merging is the secondary mechanism prior.
FAMR fits bounded checkpoint task-vector coefficients from groupwise
postprocessed action responses while limiting original-task action drift. The
rank-4 LoRA endpoint is only the local low-compute parameterization.

Proposal hash:
`96E067FFFC48D5EF9986E35E5336D679EA841BFD1F06D5E5AD4F28B5B551FD69`.
The complete proposal/review/rebuttal/math/preregistration/protocol package is
frozen under `reports/famr_vla/`.

FAMR Stage 0A completed as
`FAMR_STAGE_0A_PASS_ENDPOINT_TRAINING_ALLOWED`. Source, split, identity,
gradient, fixed-subset fit, parameter-group, scaling, Base-hash, checkpoint,
and memory gates passed. The 20-step loss reduction was
`0.11399528227036353`; exceptions, duplicate keys, and confirmatory
observations/actions were zero.

The frozen endpoint completed `300 / 300` optimizer steps and `2400 / 2400`
discovery microbatches. Fit, gradients, action effect, Base hash, reload,
manifest, and memory passed. Base-relative action validity failed, so the
result is `FAMR_ENDPOINT_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`, an
implementation/optimization failure rather than a scientific kill. No
headroom, validation, rollout, or confirmatory evaluation ran.

FAMR remains closed without rescue. Cycle 18 generated exactly three
candidates and selected `PCAV-VLA` with `95 / 100`. The closest prior is TACO;
the extension uses action-conditioned task progress motivated by ProgressVLA.
Proposal hash:
`E8B23C755C6D4E450FD193101CC0B15F88AAFE20E137A0F86830ED6D421E12AA`.

PCAV Stage 0A completed `96 / 96` discovery rows with zero final exceptions,
duplicates, or manifest mismatches. The resume preserved all 24 initial rows
and added exactly 72 missing keys. Base identity, reload, source, partition,
mapping, and checkpoint-hash checks passed; confirmatory counts remained zero.

Only `7 / 96` rows met the frozen 5% material oracle-improvement threshold,
and median reduction over improvable rows was `0.0166833`. Final decision:
`PCAV_STAGE_0A_NO_USABLE_HEADROOM`. Stage 0B is forbidden.

Cycle 24 generated exactly three candidates in
`reports/epoch_4_cycle_24_candidate_generation.md` after the prior map in
`reports/epoch_4_cycle_24_prior_mechanism_map.md`, and selected `VDR-VLA`,
Visuomotor Dynamic Residual alignment, with `92 / 100`. The closest positive
prior is FutureVLA. Proposal hash:
`0229EBC15901F4FE1EDD3839AB6B984AFA3E0E99836B5C88CF21F2C7DE2B3E72`.

VDR subtracts a discovery-fitted actionless static future-feature predictor
and supervises generated-action-conditioned prediction of the remaining
dynamic visual residual. The frozen first comparison is Base, transparent
FutureVLA proxy, VDR full, no-action-residual ablation, and standard LoRA.

The VDR proposal/review/rebuttal/math/preregistration/protocol package is
frozen under `reports/vdr_vla/`. VDR Stage 0A worker PID `411` completed
`1536 / 1536` development rows with runner exception count `0`, exact
manifest/partial key equality, duplicate partial keys `0`, missing keys `0`,
extra keys `0`, and split-overlap keys `0`. Attempt 1 is preserved as a
pre-manifest preflight/self-worker launch wrapper blocker with completed rows
`0`.

The fixed decision is
`VDR_STAGE_0A_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`, not a scientific kill.
The blocking development gates include action-validity input `False`, static
predictor relative improvement `-2.727311064830038`, action-residual
relative / absolute improvement
`1.2785489495615547e-05 / 5.777584853650097e-06`, and FutureVLA-proxy
relative / absolute gap `-0.08671267131320196 / -0.17766005523582384`. No
training, validation search, rollout, simulator access, confirmatory-test
tuning, or KITE rescue occurred. Stage 0B, VDR rerun, repair, rescue,
threshold changes, clipping, and reinterpretation are forbidden. The campaign
advanced to Epoch 4 Cycle 25 candidate generation without VDR repair or
rescue.

The Windows Efficiency Mode intervals are recorded in
`reports/resource_contention_intervals.json`; overlap-unknown efficiency
metrics are not final paper evidence. PCAV PID `371` completed with exit `0`
after both intervals had ended. The existing EAC Stage B result remains
accepted after its synchronous duplicate/manifest audit.
