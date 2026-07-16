# Next Actions

## 2026-07-16 Epoch 4 Cycle 35 Current Action

Active governance: `reports/current_research_governance.md`

Current decision:
`MHS_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY`

Immediate next action: run worker-safety checks, then launch or resume the
frozen MHS Stage 0 development audit without duplicating completed rows.

Cycle 35 completed the primary-source prior mechanism map in
`reports/epoch_4_cycle_35_prior_mechanism_map.md` and generated exactly three
candidates in `reports/epoch_4_cycle_35_candidate_generation.md`.

`MHS-VLA`, Mamba History State for Base-preserving SmolVLA, is selected at
`95 / 100`. Its closest prior is MTIL (`https://arxiv.org/abs/2505.12410`,
`https://arxiv.org/html/2505.12410v3`, and
`https://github.com/yulinzhouZYL/MTIL`). The first serious comparison is
`smolvla_base`, `mtil_history_state_proxy`, `mhs_full`,
`mhs_no_history_state_ablation`, and `standard_lora`.

LoRA may only parameterize the history encoder or residual head; it is not the
scientific mechanism.

The MHS-VLA Researcher A proposal is frozen in
`reports/mhs_vla/researcher_proposal.md` with SHA-256
`BBDF67AE3EC4BD9D025707A8BB3A5008BAB5EB5C691D02D44516157802A87BF3` and
decision `MHS_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`.

Reviewer B attack is complete in `reports/mhs_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. It requires MTIL
or a transparent MTIL proxy as policy 2, a no-history-state ablation, standard
LoRA, noncollapsed history labels, history-over-current-frame predictability,
exact Base passthrough, bounded action deltas, clean retention, and no
privileged inference input.

Researcher A rebuttal is complete in
`reports/mhs_vla/researcher_rebuttal.md` with decision
`MHS_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`; all Reviewer B conditions are
accepted.

The MHS mathematical mechanism audit is frozen in
`reports/mhs_vla/mathematical_mechanism_audit.md` with decision
`MHS_MATHEMATICAL_AUDIT_PREREGISTERED`.

The MHS preregistration is frozen in `reports/mhs_vla/preregistration.md` with
decision `MHS_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`.

The MHS executable prototype protocol is frozen in
`reports/mhs_vla/prototype_protocol.md` with decision
`MHS_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING`.

MHS Stage 0 implementation validation is complete with decision
`MHS_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY`; the helper, runner,
focused tests, and serializer preflight are implemented.

BRID remains closed unchanged as `BRID_STAGE_0_NO_RESIDUAL_HEADROOM`.

## 2026-07-16 Epoch 4 Cycle 34 Prior Action

Cycle 34 completed the primary-source prior mechanism map in
`reports/epoch_4_cycle_34_prior_mechanism_map.md` and generated exactly three
candidates in `reports/epoch_4_cycle_34_candidate_generation.md`.

`BRID-VLA`, Base-Residual Implicit Diffusion for SmolVLA action chunks, is
selected at `94 / 100`. Its closest prior is Diffusion Policy
(`https://diffusion-policy.cs.columbia.edu/` and
`https://github.com/real-stanford/diffusion_policy`). The first serious
comparison is `smolvla_base`, `diffusion_policy_action_chunk_proxy`,
`brid_full`, `brid_no_base_residual_ablation`, and `standard_lora`.

LoRA may only parameterize the residual score network; it is not the
scientific mechanism.

The BRID-VLA Researcher A proposal is frozen in
`reports/brid_vla/researcher_proposal.md` with SHA-256
`2D4769CF126DF0580029486F7D64EF3C09D435571589F87C569F60A71CBC5CA2` and
decision `BRID_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`. No BRID
implementation, training, validation search, rollout, or confirmatory-test
access has happened.

Reviewer B attack is complete in `reports/brid_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. The rebuttal
must accept or reject the Diffusion Policy closest-prior boundary, transparent
raw action-chunk diffusion proxy, no-Base-residual ablation, matched standard
LoRA, residual/score noncollapse gates, exact Base passthrough, clean
retention, and no deterministic-action KL.

Researcher A rebuttal is complete in `reports/brid_vla/researcher_rebuttal.md`
with decision `BRID_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`; all Reviewer B
conditions are accepted.

The BRID mathematical mechanism audit is frozen in
`reports/brid_vla/mathematical_mechanism_audit.md` with decision
`BRID_MATHEMATICAL_AUDIT_PREREGISTERED`.

BRID preregistration is frozen in `reports/brid_vla/preregistration.md` with
decision `BRID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`.

BRID prototype protocol is frozen in `reports/brid_vla/prototype_protocol.md`
with decision
`BRID_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING`. Stage 0
implementation must use `tca_map/smolvla/brid_vla.py`,
`scripts/run_brid_vla_stage0.py`, and `tests/test_brid_vla.py`.

BRID Stage 0 completed in `reports/brid_vla/stage_0_result.json` with decision
`BRID_STAGE_0_NO_RESIDUAL_HEADROOM`. It completed `46080 / 46080` model rows
with exception count `0`, duplicate/missing/extra/split-overlap counts all
`0`, and `key_sets_equal = true`. Do not rescue this BRID formulation; continue
to Cycle 35 candidate generation.

Cycle 33 generated exactly three candidates in
`reports/epoch_4_cycle_33_candidate_generation.md` after the primary-source
mechanism map in `reports/epoch_4_cycle_33_prior_mechanism_map.md`.

`AFID-VLA`, Action-Factor Instruction Densification for Base-preserving
SmolVLA, is selected at `90 / 100`. Its closest prior is FineVLA
(`https://arxiv.org/html/2605.27284v1`). The first serious comparison is
`smolvla_base`, `finevla_action_factor_proxy`, `afid_full`,
`afid_no_factor_ablation`, and `standard_lora`.

LoRA may only parameterize the action-factor predictor/gate; it is not the
scientific mechanism.

The AFID-VLA Researcher A proposal is frozen in
`reports/afid_vla/researcher_proposal.md` with SHA-256
`B5D1EE12FF2D0280511452DA7FE55295740FD9942A8BE293F444C8EB157062BC`.

No AFID implementation, training, validation search, rollout, or
confirmatory-test access has happened. LCG repair/rescue remains disallowed.

Reviewer B attack is complete in `reports/afid_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. The rebuttal
must accept or reject the FineVLA prior boundary, fair proxy requirement,
frozen factor extraction, noncollapsed factor/mask health, factor observability
gate, no-factor and standard-LoRA controls, exact Base passthrough, and no
deterministic-action KL.

Researcher A rebuttal is complete in `reports/afid_vla/researcher_rebuttal.md`
with decision `AFID_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`; all Reviewer B
conditions are accepted.

The AFID mathematical mechanism audit is frozen in
`reports/afid_vla/mathematical_mechanism_audit.md` with decision
`AFID_MATHEMATICAL_AUDIT_PREREGISTERED`.

AFID preregistration is frozen in `reports/afid_vla/preregistration.md` with
decision `AFID_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`.

AFID prototype protocol is frozen in
`reports/afid_vla/prototype_protocol.md` with decision
`AFID_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING`.

AFID Stage 0 implementation validation is complete with decision
`AFID_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY`. The validated serializer
preflight is `reports/afid_vla/stage_0_serializer_preflight.json`.

AFID Stage 0 completed in `reports/afid_vla/stage_0_result.json` and is
adjudicated in `reports/afid_vla/stage_0_adjudication.md` with decision
`AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE`. It completed
`5120 / 5120` rows with zero exceptions and zero duplicate/missing/extra keys.
AFID repair/rescue is disallowed for this formulation.

Cycle 32 generated exactly three candidates in
`reports/epoch_4_cycle_32_candidate_generation.md` after the primary-source
mechanism map in `reports/epoch_4_cycle_32_prior_mechanism_map.md`.

`LCG-VLA`, Language-Contrastive Guidance for Base-preserving SmolVLA actions,
is selected at `93 / 100`. Its closest positive prior is Counterfactual Action
Guidance (`https://arxiv.org/abs/2602.17659`). The frozen design-level first
comparison is `smolvla_base`, `counterfactual_action_guidance_proxy`,
`lcg_full`, `lcg_no_language_contrast_ablation`, and `standard_lora`. LoRA may
only parameterize the language-contrast gate; it is not the scientific
mechanism.

The LCG-VLA Researcher A proposal is frozen in
`reports/lcg_vla/researcher_proposal.md` with SHA-256
`F0D980AA0760F143D781C723DB632BC324C1E18F390D9C33C5DA94F3A897D11E` and
decision `LCG_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`.

Reviewer B attack is complete in `reports/lcg_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. The rebuttal
must accept CAG as policy 2, null-branch proxy limits, the narrowed novelty
boundary, contrast/residual/mask noncollapse gates, standard LoRA, no
deterministic-action KL, and closure of S2C plus previous methods.

Researcher A rebuttal is complete in
`reports/lcg_vla/researcher_rebuttal.md` with decision
`LCG_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`.

The mathematical mechanism audit is frozen in
`reports/lcg_vla/mathematical_mechanism_audit.md` with decision
`LCG_MATHEMATICAL_AUDIT_PREREGISTERED`.

Preregistration is frozen in `reports/lcg_vla/preregistration.md` with decision
`LCG_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`.

Prototype protocol is frozen in `reports/lcg_vla/prototype_protocol.md` with
decision `LCG_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_IMPLEMENTATION_PENDING`.

LCG Stage 0 implementation is validated as
`LCG_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY`: WSL py_compile passed,
focused LCG tests reported `6 passed`, and
`reports/lcg_vla/stage_0_serializer_preflight.json` has matching fixture and
reproduced hashes.

LCG Stage 0 completed as `LCG_STAGE_0_DESIGN_FAILURE` in
`reports/lcg_vla/stage_0_result.json`: `5120 / 5120` model rows, exception
count `0`, duplicate / missing / extra / split-overlap keys all `0`, exact
key-set equality, near-everywhere gate activation (`0.99978125`), and
`lora_explains=true`. This is development-only, not a closed-loop scientific
kill.

Cycle 31 generated exactly three candidates and selected `S2C-VLA`,
Seam-Supervised Chunk Consistency for Base-preserving SmolVLA execution, at
`95 / 100`. Its closest positive prior is ChunkFlow
(`https://arxiv.org/html/2607.12992v1`, project page
`https://cytoderm-ai.github.io/chunkflow`).

The frozen design-level first comparison is Base, `chunkflow_overlap_proxy` or
official ChunkFlow if installed, `s2c_full`,
`s2c_no_learned_overlap_mask_ablation`, and `standard_lora`. The S2C-VLA
Researcher A proposal is frozen in `reports/s2c_vla/researcher_proposal.md`
with SHA-256
`399A3960F9FF9AFA8EDA7C3F743A95C3FD4DC711644C2398630F1E68486DC5B3`. No S2C
training, validation search, rollout, simulator access, or confirmatory-test
tuning has happened.

Reviewer B attack is complete in `reports/s2c_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. The rebuttal
must accept the narrowed frozen-SmolVLA edit-layer novelty boundary, ChunkFlow
as policy 2, SEAM as secondary prior, no expert future-tail inference, gripper
and legitimate-discontinuity protection, standard LoRA control, no
deterministic-action KL, and no rescue of URF or previous closed methods.

Researcher A rebuttal is complete in
`reports/s2c_vla/researcher_rebuttal.md` with decision
`S2C_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. It accepts those conditions and
defines the deployment previous-tail rule, Stage 0 headroom gate, gripper-event
protection, and bounded decoding resume keys.

The mathematical mechanism audit is frozen in
`reports/s2c_vla/mathematical_mechanism_audit.md` with decision
`S2C_MATHEMATICAL_AUDIT_PREREGISTERED`. It fixes `H=50`, stride `10`, overlap
`K=10`, deterministic bridge target, learned effective edit mask, group caps,
objective coefficients, Stage 0 gates, and no deterministic-action KL.

Preregistration is frozen in `reports/s2c_vla/preregistration.md` with
decision `S2C_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`. It freezes
development tasks, discovery/validation demo IDs, Stage 0 artifacts and gates,
bounded validation search limits, and worker resume keys.

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
count `0`, duplicate / missing / extra / split-overlap keys all `0`, exact
key-set equality, `177` adjacent pairs, insufficient task coverage, no Base
boundary headroom, and failed mask/gripper criteria.

Cycle 30 generated exactly three candidates and selected `URF-VLA`,
Uncertainty-Routed Residual Flow for Base-preserving SmolVLA chunks, at
`92 / 100`. Its closest positive prior is SUREFlow
(`https://arxiv.org/abs/2607.10504`, official repository
`https://github.com/tanvirnwu/SUREFlow`).

The frozen design-level first comparison is Base,
`sureflow_uncertainty_residual_proxy` or official `sureflow` if installed,
`urf_full`, `urf_no_uncertainty_route_ablation`, and `standard_lora`. The
Researcher A proposal is frozen in `reports/urf_vla/researcher_proposal.md`
with SHA-256
`E78829E736C3F22451E72574092221904ACBE4C4BE0BDA7FA046832DABED3532`.

Reviewer B attack is complete in `reports/urf_vla/reviewer_attack.md` with
decision `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`. Researcher A
rebuttal is complete in `reports/urf_vla/researcher_rebuttal.md` with decision
`URF_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`. The mathematical mechanism audit is
frozen in `reports/urf_vla/mathematical_mechanism_audit.md` with decision
`URF_MATHEMATICAL_AUDIT_PREREGISTERED`. Preregistration is frozen in
`reports/urf_vla/preregistration.md` with decision
`URF_PREREGISTRATION_FROZEN_PROTOTYPE_PROTOCOL_PENDING`. Prototype protocol is
frozen in `reports/urf_vla/prototype_protocol.md` with decision
`URF_PROTOTYPE_PROTOCOL_FROZEN_STAGE_0_PENDING`. URF Stage 0 implementation is
validated as `URF_STAGE_0_IMPLEMENTATION_VALIDATED_STAGE_0_READY`: WSL
py_compile passed, focused URF tests reported `8 passed`, and
`reports/urf_vla/stage_0_serializer_preflight.json` has matching fixture and
reproduced hashes. No URF training, validation search, rollout, simulator
access, or confirmatory-test tuning has happened.

URF Stage 0 completed as `URF_STAGE_0_NO_USABLE_HEADROOM` with `5120 / 5120`
model rows, exception count `0`, duplicate / missing / extra / split-overlap
keys all `0`, and exact key-set equality. Bounded validation is not allowed:
Base residual headroom was below the frozen `0.005` normalized Huber gate
(`0.0033407550543043956`) and the heteroscedastic residual proxy did not beat
homoscedastic or task/phase baselines. This is a development-only no-headroom
stop, not a closed-loop scientific kill.

Do not repair, rerun, or rescue `VDR-VLA`; do not change its thresholds,
horizons, residual construction, memory construction, or action-validity
interpretation.

Cycle 24 generated exactly three candidates and selected VDR at `92 / 100`
with proposal hash
`0229EBC15901F4FE1EDD3839AB6B984AFA3E0E99836B5C88CF21F2C7DE2B3E72`. VDR
Stage 0A worker PID `411` completed `1536 / 1536` development rows with
runner exception count `0`, exact manifest/partial key equality, duplicate
partial keys `0`, missing keys `0`, extra keys `0`, and split-overlap keys
`0`. The fixed decision is
`VDR_STAGE_0A_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`, not a scientific kill.
Stage 0B is forbidden.

Do not repair, rerun, or rescue KITE. Its Stage 0A completed `128 / 128`
through a valid missing-key-only resume but retained one persistence exception
and failed the frozen action bound on all 128 reconstructed rows. Stage 0B is
forbidden.

Do not change VDR horizons `{4,12}`, projection dimension `32`, ridge `1e-4`,
four task sources, demo splits, dynamic-residual construction, headroom bars,
rank-4 identity path, or failure taxonomy. Do not reinterpret the launcher
exit-code formatting defect or non-row heartbeat-thread stderr as a reason to
rerun VDR.

HASTE Stage 0A PID `295` exited `1` before manifest persistence on a canonical
JSON serialization defect. It created no partial, model row, feature cache,
adapter, or closed-loop evidence. Do not repair, resume, rerun, reinterpret, or
rescue HASTE; Stage 0B is forbidden.

HEST Stage 0A completed `160 / 160` windows with zero exceptions and exact
artifact integrity. It failed only the frozen all-variant support gate: one
validation Base row and its HEST whole-Base fallback were outside
discovery-defined support. This is a pre-rollout implementation/prototype
support failure, not a scientific kill.

Do not widen support, clip actions, alter fallback, change sources or
thresholds, rerun HEST Stage 0A, or authorize Stage 0B. Do not rescue NICE or
HEST.

Do not train a HASTE adapter or run the simulator in Stage 0A. Do not change
the event threshold, horizons, target coordinates, four tasks, demo splits,
probe baselines, headroom bars, or failure taxonomy.

## Epoch 4 Cycle 19 Historical Action

COVI, IARC, and FAMR remain preserved under their fixed protocols. FAMR's
endpoint completed `300 / 300` optimizer steps and `2400 / 2400` discovery
microbatches, then stopped on its frozen Base-relative action-validity gate as
an implementation/optimization failure rather than a scientific kill.

Cycle 18 PCAV Stage 0A completed `96 / 96` rows. Only `7 / 96` rows met the
frozen 5% material oracle-improvement threshold and median reduction over
improvable rows was `0.0166833`, so it closed as
`PCAV_STAGE_0A_NO_USABLE_HEADROOM`. Do not load an adapted candidate generator,
change noise, or continue to PCAV Stage 0B. Such a redesign belongs to a new
method cycle. Do not rescue FAMR or PCAV.

Before any long WSL launch, inspect state, newest PID/heartbeat/status/partial/
result/log/exit files, worker liveness, JSON parseability, completed/planned
counts, exceptions, and duplicate/manifest keys. Do not duplicate a live or
completed run. Both Windows Efficiency Mode intervals are recorded in
`reports/resource_contention_intervals.json`; overlap-unknown timing and
resource metrics are excluded from final paper evidence.

## 2026-07-13 Governance V2 Current Action

Active governance: `reports/current_research_governance.md`

Current decision: `EPOCH_4_CYCLE_1_RCV_KILLED_CONTINUE_CYCLE_2`

Immediate next action: start Epoch 4 Cycle 2 autonomous research. Generate exactly three candidates under the post-PSE problem-first and external-prior-early quality gate, select exactly one, freeze/hash the Researcher proposal, run Reviewer B novelty and mathematical mechanism attack, then implement unless exact duplication, trivial equivalence, mathematical invalidity, or hard infeasibility is proven.

Do not stop after governance migration, method failure, three historical method failures, or prototype GO unless an allowed final state in `reports/current_research_governance.md` is reached. Epoch 4 Cycle 2 must not be a cosmetic variant of CBFD-VLA, SCVC-VLA, PSE-VLA, or RCV-VLA. In particular, do not rescue RCV by threshold retuning, a renamed verifier, or another receding-chunk replanning ablation; change at least two core dimensions relative to RCV's frozen-policy disagreement verifier, inference-time replanning intervention, and efficiency-versus-stateless claim.

Date: 2026-07-10 KST

Current decision: `PROTOCOL_DRIFT_FOUND`

## Immediate Rule

Do not run official rollout.

The regenerated persisted LoRA checkpoints are complete and deterministic under fixed-seed repeated disk evaluation, but they are not accepted as the canonical rollout baseline set because the historical in-memory evaluation protocol differs from the regenerated persisted-reload protocol and because the saved regenerated artifact metrics differ from fixed-seed disk re-evaluation metrics.

## Evidence To Preserve

- historical result: `reports/official_smolvla_lora_seed_repro_result.json`
- regenerated result: `reports/official_smolvla_lora_checkpoint_regen_result.json`
- checkpoint manifest: `reports/official_smolvla_lora_checkpoint_manifest.json`
- drift audit: `reports/official_smolvla_lora_drift_audit.md`
- config diff: `reports/official_smolvla_old_vs_regen_config_diff.md`
- artifact alignment: `reports/official_smolvla_artifact_alignment_audit.md`
- deterministic eval: `reports/official_smolvla_eval_determinism_check.md`
- training identity status: `reports/official_smolvla_training_determinism_status.md`
- canonical proposal: `reports/official_smolvla_canonical_checkpoint_proposal.md`
- drift decision: `reports/official_smolvla_lora_drift_decision.md`

## Required Before Canonicalization

1. Decide how to handle the PEFT protocol difference:
   - old path: in-memory policy evaluation, no adapter save/reload, no assigned `wrap_with_peft` return;
   - regenerated path: assigned PEFT wrapper, adapter save, `PeftModel.from_pretrained` reload, disk evaluation.
2. Decide and pin the evaluation RNG-state policy, since fixed-seed disk re-evaluation is repeatable but does not exactly reproduce the saved regenerated artifact metrics.
3. Do not claim the regenerated persisted checkpoints are identical to the historical ephemeral runs.
4. Preserve old metrics as historical.
5. If re-baselining is approved, adopt the persisted checkpoint metrics explicitly as a new canonical baseline and update the reproducibility lock with checkpoint hashes and evaluation seed policy.
6. If the protocol difference is treated as a bug, fix the protocol first and run a new explicitly authorized bounded check.

## Still Forbidden

- no closed-loop rollout
- no simulator dependency installation
- no OpenVLA-OFT
- no FCAR revival
- no method design
- no full benchmark
- no asset downloads
- no static-alpha tuning on test
- no favorable-seed-only reporting
- no silent historical metric replacement

## Exact Next Step

Create a no-rollout protocol-adjudication branch that either fixes the PEFT in-memory vs persisted-reload mismatch and pins evaluation RNG state, or explicitly records a re-baselining decision before any canonical rollout baseline is used.

## 2026-07-10 Canonical Next Action

Current decision: `NEEDS_WSL_OR_LINUX_OFFICIAL_ROLLOUT`

Move the same canonical artifacts/checkpoints into the verified WSL/Linux LeRobot LIBERO environment, install only official `lerobot[libero]` dependencies, then run the official smoke before any bounded pilot. Do not retrain, select a LoRA seed from rollout outcomes, revive FCAR, or use the old custom LIBERO_7D route.

## 2026-07-10 WSL Official Rollout Next Action

Current decision: `OFFICIAL_ROLLOUT_BASELINE_READY`

Exact next step: run a larger predeclared official baseline rollout/failure-mining pass with frozen base and all three LoRA seeds, using official videos for failed episodes. Keep static mixes skipped at alpha `0.0`, keep all seeds reported, and do not select a winning LoRA seed or design a new method from the 48-episode pilot alone.

## 2026-07-11 Closed-Loop Scaleup Next Action

Current decision: `OFFLINE_ONLINE_MISMATCH_CONFIRMED`

Exact next step: run a bounded visual review pass on the already identified repeated failures, with official video capture enabled and no policy changes.

Start with:

1. `libero_10/task_4/seed_20260713`
2. `libero_10/task_4/seed_20260715`
3. `libero_spatial/task_4/seed_20260712`
4. `libero_spatial/task_4/seed_20260713`
5. `libero_spatial/task_4/seed_20260714`

Rules for the next pass:

- do not retrain any policy
- do not select seed `11` as best after outcomes
- do not run static-mix duplicates
- do not revive FCAR
- do not design routing, retention, prior, correction, or chunking methods yet
- do not use old custom `LIBERO_7D` or exact-init replay routes
- keep frozen_base and all LoRA seeds paired on identical task/reset cases
- stop at video/phase annotation unless a repeated mechanism is visually supported

The novelty/method-design gate can only reopen if the bounded review converts the current `ambiguous_or_unclassified` failures into a repeated, success-critical, mechanism-linked phase failure that survives frozen-base, LoRA-seed, task, and reset explanations.

## 2026-07-11 Closed-Loop Visual Gate Next Action

Current decision: `NO_SAFE_RA_L_METHOD_YET`

Do not implement a method from the current evidence.

The bounded video review found real visible failures, but not a safe RA-L method route:

- `libero_spatial/task_4` shows a drawer/bowl stable-grasp extraction failure on only two independent rerun-failure reset seeds.
- `libero_10/task_4` shows a different multi-object long-horizon failure.
- `8/24` same-identity reruns changed success status, so original failure identity is not stable enough for a narrow causal method claim.
- recent work kills generic confidence, verification, correction, adaptive chunking, progress/recovery, failure-negative learning, and adapter-routing routes.

Allowed next actions:

1. Archive this method gate as a no-implementation result.
2. If reopening later, collect new bounded visual evidence only for a predeclared mechanism and stop once either three independent reset seeds or two tasks are verified.
3. Before any implementation, predeclare a second-backbone plan, a second-benchmark plan, and simple baseline kill tests.

Still forbidden:

- no LoRA training as a method contribution
- no best-seed selection
- no FCAR revival
- no generic correction/replanning/chunking/progress method
- no full sweep just to rescue the gate
- no paper claim from SmolVLA-only evidence

## 2026-07-11 Cross-Model Gate Next Action

Current decision: `SECOND_BACKBONE_OR_BENCHMARK_BLOCKED`

Do not implement a method.

The selected second backbone is `OpenVLA-OFT` with checkpoint `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`. The selected second benchmark is `LIBERO-PRO`.

Immediate next action:

1. Ask for explicit approval before downloading the `14.845` GiB OpenVLA-OFT checkpoint.
2. Choose the hardware path:
   - preferred: lab GPU path with 24GB+ VRAM per inference process;
   - local RTX 5080 16GB only with a predeclared offload/quantization risk note, because official full-precision inference is not proven.
3. After approval, run only the frozen protocol in `reports/cross_model_failure_manifest.json`.

Still forbidden:

- no SmolVLA retraining
- no OpenVLA-OFT fine-tuning
- no FCAR revival
- no method implementation
- no full benchmark
- no generic retry/progress/verification/replanning/chunking method
- no claim that either mechanism generalizes before OpenVLA-OFT and LIBERO-PRO evidence exists

## 2026-07-11 Quantized OpenVLA-OFT Gate Next Action

Current decision: `FAILURE_NOT_REPRODUCED_IN_SECOND_ARCHITECTURE`

Do not implement a method and do not proceed to LIBERO-PRO from the current evidence.

Why:

- quantized OpenVLA-OFT INT4 succeeded on all exact hard-slice and matched-control episodes (`20/20`)
- SmolVLA frozen-base still failed on the exact hard slices (`libero_spatial/task_4 = 1/5`, `libero_10/task_4 = 1/5`)
- visual comparison therefore does not support a shared cross-backbone mechanism
- INT4 is quantized, so this is not a full-precision OpenVLA-OFT claim

Allowed next actions:

1. Archive the result as `FAILURE_NOT_REPRODUCED_IN_SECOND_ARCHITECTURE`.
2. If continuing later, predeclare a new second-backbone or full-precision hardware run before any method design.
3. Keep LIBERO-PRO blocked unless a future decision is one of the confirmed cross-backbone failure decisions.

Still forbidden: no FCAR revival, no LoRA training, no OpenVLA-OFT fine-tuning, no generic correction/chunking/progress method, no LIBERO-PRO run from this decision.

## 2026-07-11 Paper-First VLA Method Design Next Action

Current decision: `READY_TO_IMPLEMENT_PRIMARY_VLA_METHOD`

Immediate next action: implement the first bounded `ECHO-VLA` prototype only.

Allowed first implementation scope:

1. Use official SmolVLA-LIBERO as the only first backbone.
2. Use the four predeclared predicate-diversity tasks:
   - `libero_spatial/task_0`
   - `libero_object/task_4`
   - `libero_goal/task_0`
   - `libero_10/task_0`
3. Build a small effect-label dataset from official demonstrations and training-time BDDL/simulator predicate labels.
4. Train only lightweight visual predicate/effect heads and the ECHO counterfactual ranking objective.
5. Compare against frozen SmolVLA, heuristic effect/progress, progress/value head, Pre-VLA-style validity/advantage head, and no-counterfactual ECHO.

Kill before scaling if ECHO full fails to beat the strongest simple baseline by at least `5` absolute task-balanced success points or fails to beat its no-counterfactual ablation.

Still forbidden before the first gate passes:

- no OpenVLA-OFT INT4 validation,
- no full benchmark,
- no LIBERO-PRO,
- no broad failure mining,
- no generic confidence/verification/progress/replanning/chunking method,
- no LoRA, SmolVLA, or quantization as novelty.

## 2026-07-11 ECHO-VLA First Prototype Next Action

Current decision: `NO_ECHO_CANDIDATE_HEADROOM`

Immediate next action: stop the current ECHO implementation path.

Why:

- the focused novelty gate passed, but only for the strict same-state intervention effect-mediator claim;
- the bounded same-state candidate-headroom gate generated `4` intervention groups and `16` candidate records;
- all same-state identity proofs passed;
- oracle realized-effect selection achieved `0.0` percentage-point improvement over the default candidate;
- `0.0` of default-failure states contained a successful or materially better candidate;
- therefore training lightweight ECHO heads would only train a selector over a candidate set with no recoverable headroom.

Allowed next actions:

1. Archive this as a no-headroom kill for the current ECHO candidate generator.
2. If reopening ECHO, design a new candidate generator or longer-horizon state selection protocol first, then freeze a new headroom gate before training.
3. Preserve the no-privileged-inference and same-state-intervention tests.

Still forbidden:

- no ECHO head training on the current no-headroom data;
- no OpenVLA-OFT validation;
- no full benchmark;
- no claim that a local Pre-VLA-style proxy was officially reproduced;
- no causality claim from ordinary demonstration transitions.

## After ECHO Final Candidate Headroom Gate - 2026-07-11

- decision: `NO_ECHO_HEADROOM_CONFIRMED`
- next: `Archive ECHO and return to the paper-first candidate portfolio.`

## After Implementation V2 Empirical Postmortem - 2026-07-12

- decision: `PROTOTYPE_EVIDENCE_INSUFFICIENT_FOR_TERMINAL_CLAIM`
- next: `Do not start another autonomous campaign from the implementation-v2 terminal claim.`

Allowed only if explicitly reopened later:

1. A bounded CensorCredit repair that first proves censored and uncensored labels differ on held-out generated intervention records, then repeats an adequately powered evaluation.
2. A genuinely distinct policy-distribution training method using intervention-generated sequence-level supervision, after targeted novelty review against SDP, TORL-VLA, ConRFT, VLA-Corrector, and OpenVLA-OFT.

Still forbidden:

- no PhaseBarrier threshold tuning as a rescue;
- no CensorCredit hold-strength tuning as a rescue;
- no rollout rerun inside this postmortem;
- no claim that `TWO_IMPLEMENTED_METHODS_KILLED` is a review-resistant scientific terminal result;
- no final method promotion from the current evidence alone.

## After PhaseBarrier Bounded Adjudication - 2026-07-12

- decision: `PHASEBARRIER_COMPONENT_NOT_USEFUL`
- next: `Archive PhaseBarrier; do not rescue or tune it.`

Evidence:

- valid bounded result reused original saved PhaseBarrier weights;
- full PhaseBarrier completed `20/20` held-out episodes and changed actions in every episode;
- full PhaseBarrier success was `0/20`;
- no-phase ablation success was `9/20`;
- frozen SmolVLA success was `8/20`.

Allowed later action:

1. Inspect CensorCredit's documented implementation failure only if explicitly reopened.
2. A CensorCredit repair is allowed only if it first demonstrates that censored and uncensored labels differ and that the intended component changes actions.

Still forbidden:

- no PhaseBarrier threshold tuning;
- no new PhaseBarrier repeat;
- no PhaseBarrier redesign under another name;
- no CensorCredit repair inside the PhaseBarrier adjudication branch.

## After CensorCredit One-Repair Gate and Final Method - 2026-07-12

- decision: `NO_VALID_CENSORCREDIT_REPAIR_FINAL_METHOD_KILLED`
- next: `Stop this autonomous method chain unless new intervention/correction data or genuinely new primary-source gap appears.`

What happened:

- CensorCredit was diagnosed exactly as `LABEL_OR_DATA_FAILURE`.
- No repair was allowed because the objective only allowed concrete implementation or optimization bugs.
- The final distinct candidate, `ISAC-VLA`, was killed before implementation due SDP/TORL-VLA/ConRFT overlap and unavailable paired intervention/correction chunk data.

Allowed reopen conditions:

1. Add real paired negative-policy and corrective-action chunk data from human, robot, or validated intervention simulator.
2. Identify a new primary-source gap not equivalent to action-chunk correction learning, intervention-censored VLA refinement, contact barriers, candidate selection, or post-hoc temporal wrappers.
3. Explicitly ask for a non-paper engineering prototype that is allowed to be incremental rather than RA-L novel.

Still forbidden:

- no CensorCredit relabeling as a repair;
- no CensorCredit hold-strength or threshold tuning;
- no final-method implementation using synthetic local labels as a substitute for intervention chunks;
- no PhaseBarrier rescue or rename.

## After Autonomous Dual-Review RA-L Campaign - 2026-07-11

- decision: `NO_METHOD_AFTER_3_VALID_CYCLES`
- next: `Do not start another generic VLA method cycle from the current evidence.`

Allowed reopen conditions:

- reproduce an official action-representation baseline and find a residual not solved by mean-action, MLP, or the official baseline;
- find a new matched exact-state cross-backbone failure affecting both SmolVLA and Quantized OpenVLA-OFT INT4;
- add new physical robot intervention, tactile/force, or larger-GPU resources;
- identify a new primary-source gap not already occupied by action-conditioning, correction, verification, progress, chunking, contact-adaptation, or prior-preservation papers.

Still forbidden:

- ECHO rescue without a new predeclared headroom gate;
- generic confidence, verification, correction, adaptive chunking, progress, failure-negative, adapter-routing, or LoRA-as-novelty claims;
- main-branch pollution from abandoned method implementations.

## After Autonomous RA-L Research Implementation V2 - 2026-07-11

- decision: `TWO_IMPLEMENTED_METHODS_KILLED`
- next: `Stop autonomous no-method campaign unless the user explicitly reopens a stronger repeat or a new mechanism.`

Reason:

- `PhaseBarrier-VLA` was implemented, trained, evaluated closed-loop, and killed.
- `CensorCredit-VLA` was implemented, trained, evaluated closed-loop, and killed because its key uncensored ablation matched the full method.

Allowed reopen:

- one explicitly requested repeat of CensorCredit on more held-out resets, because it showed a weak positive signal over frozen/simple baselines but failed the ablation gate;
- a genuinely new mechanism that differs from both physical feasibility projection and temporal credit/action-history blending.

## After ECHO Final Candidate Headroom Gate - 2026-07-11

- decision: `NO_ECHO_HEADROOM_CONFIRMED`
- next: `Archive ECHO and return to the paper-first candidate portfolio.`
