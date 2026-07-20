# Stability-Qualified Completion Paperability Contract

Frozen: 2026-07-20T16:38:00+09:00, before any Epoch 7 task-persistence outcome.

Status: `PRIMARY_PROBLEM_VERIFICATION_ONLY`. No policy rollout, Ours design, training, or paper generation is authorized.

## Archetype and claim

Archetype: concise RA-L benchmark/evaluation paper.

One-sentence claim: native first-hit task success can certify transient manipulation states, while a controller-neutral post-policy dwell distinguishes durable completion and can change the measured reliability and ranking of otherwise identical VLA rollouts.

Minimum reviewer-defensible novelty requires all of the following:

1. an explicit, reproducible post-policy completion contract rather than another safety or trajectory-quality score;
2. unchanged native success reported alongside persistence curves and a frozen dwell endpoint;
3. a legal recoverability oracle showing that at least some first-hit failures become durable after task-relevant follow-through;
4. multi-policy, multi-suite, multi-mechanism evidence that persistent completion changes an evaluation conclusion rather than merely flagging one malformed predicate;
5. public task/controller mappings and per-rollout artifacts.

The paper may not claim that stable manipulation, temporal logic, termination detection, or distributional evaluation is itself novel. SafeVLA-Bench, vla-eval, VLA-SCT, PhAIL, and Beyond Binary Success own adjacent components.

## Frozen problem gate

The authoritative machine-readable protocol is `problem_verification_protocol.json`. The gate uses all ten LIBERO-Goal tasks and official demonstration HDF5 files. For each task, demonstrations are tried in numeric order using standard exact-init replay; the lowest-index demonstration that reaches official native success is selected without inspecting either persistence branch.

At the selected replay's first `OffScreenRenderEnv.check_success()`:

- `immediate_neutral_hold` replaces all future actions with `[0,0,0,0,0,0,g]` for 30 simulator steps, where `g` is +1 for a nonnegative last executed gripper command and -1 otherwise;
- `expert_suffix_then_hold` continues the exact unused actions from the same demonstration through the recorded end, then applies the identical 30-step neutral hold;
- `last_action_repeat` is a secondary sensitivity control and cannot define the primary result.

Native success semantics are unchanged. Persistent success requires the native predicate to remain true at every one of the 30 primary hold steps. The full Boolean persistence trace, first failure index, final state, and exceptions are logged.

## Problem and headroom gates

Execution validity requires ten completed task rows, deterministic cold-repeat equality on two frozen tasks, finite legal actions, one live environment, no policy load, and zero exceptions.

Base/predicate coverage requires native success on at least eight tasks spanning placement, containment/insertion, planar push, and articulation.

A practically meaningful problem requires:

- at least three tasks with native first-hit success but failed immediate persistence;
- those disagreements span at least two task mechanisms;
- the disagreement rate among native-success tasks is at least 20%;
- no single task-specific predicate implementation explains more than two-thirds of disagreements.

Legal headroom requires at least two disagreement tasks across two mechanisms to become persistent in `expert_suffix_then_hold`. The expert suffix is a feasibility/recoverability oracle, never policy evidence.

There is no bounded outcome-driven task expansion. Failure of any validity, coverage, problem, or headroom gate rotates the thesis.

## Closest-overlap falsification

Rotate if the endpoint reduces to SafeVLA-Bench's existing stability/safety specifications, vla-eval already implements the same post-success dwell semantics, or VLA-SCT evaluates durability after its stop decision. Also rotate if every disagreement is a single LIBERO predicate defect, neutral action semantics are not task-independent, or a multi-policy comparison would add no scientific conclusion beyond a simulator bug report.

Mandatory controls after a problem pass are native first-hit success, immediate neutral hold, expert-suffix neutral hold, last-action repeat, dwell-length sensitivity, and task-family-specific state traces. SafeVLA-Bench-aligned stable-object diagnostics are required to test whether safety metrics fully explain the endpoint.

## Paper-level evidence

Paper-GO requires at least two competent VLA policy families, LIBERO-Goal plus LIBERO-Long or another compatible suite, four manipulation mechanisms, multiple fixed reset identities, paired native-versus-persistent uncertainty, a changed policy reliability or ranking conclusion, and reproducible artifacts. Native success must be retained and reported rather than replaced.

Expected resource envelope: the expert gate uses one 64x64 simulator, no model, under 12 GiB host memory, no CUDA requirement, under 100 MiB artifacts, and under one hour. Full evidence uses one model and one simulator at a time, below 24 GiB host RAM, 16 GiB VRAM, 15 GiB new storage, and 18 GPU-hours.

Predicted main table columns: policy, suite, mechanism, native success, persistent success at 10/20/30 hold steps, native-success/persistent-failure count, suffix recoverability, paired interval, ranking, dwell overhead, and SafeVLA-aligned explanation category.

## Decisions

- Problem pass: `PROBLEM_VERIFIED_STRONG_COMPARATOR_RESIDUAL` only when every execution, coverage, disagreement, diversity, attribution, and headroom gate passes.
- Kill/rotate: `NO_REPEATABLE_GAP`, `NO_LEGAL_HEADROOM`, `EVALUATION_INVALID`, `BASE_NOT_COMPETENT`, `PRIOR_SATURATES_PROBLEM`, or `RESOURCE_OR_ARTIFACT_BLOCKED` under the frozen definitions.
- Narrow: claim language may narrow from ranking to reliability only if all problem/headroom gates pass and a later multi-policy result changes a practically meaningful reliability conclusion without changing rank.
- Paper-GO: `BENCHMARK_PAPER_CANDIDATE_GO` only after the complete multi-policy/generalization/control/statistics package passes.
