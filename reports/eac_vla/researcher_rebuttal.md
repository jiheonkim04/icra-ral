# EAC-VLA Researcher A Rebuttal

Date: 2026-07-15 KST

Responds to: `reports/eac_vla/reviewer_attack.md`

Reviewed proposal hash: `A89ED48AE9FD4D26A8DA9E3E987FACDBBD9F861D070AE135372A092A44581E4E`

Decision: `EAC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

## Response Summary

Researcher A accepts Reviewer B's narrowing. EAC-VLA will not claim generic adaptive chunking novelty. The method is a local, fixed-protocol AAC extension for frozen SmolVLA's official `50 x 7` action queue, with exact action-value passthrough and a mandatory comparison to both an AAC proxy and fixed short replanning.

No implementation, Stage 0, validation search, manifest freeze, or rollout may happen until the mathematical audit and preregistration encode the constraints below.

## Accepted Novelty Boundary

Accepted:

- AAC is the closest prior and already owns the broad adaptive-chunking idea.
- EAC's local novelty is limited to SmolVLA-specific uncertainty-source validation, queue-boundary risk, hysteresis/retention against mode-jumping, and exact 7D action-value passthrough.
- If `aac_entropy_proxy` matches or beats EAC under the frozen comparison, the local contribution is explained by the closest prior.
- If `fixed_short_replan_baseline` matches or beats EAC, the method is explained by a simple reviewer-killer.

Required wording for all later reports:

`aac_entropy_proxy` is a faithful transparent local proxy, not an official AAC reproduction.

## RCV And Fixed-Replan Boundary

Accepted:

- EAC must not become a replay of RCV or a hidden no-context queue flush.
- The fixed short-replan baseline remains the single mandatory simple killer.
- A selected EAC map that collapses to one fixed cadence is exact trivial equivalence and must stop before rollout.

Required metrics:

- commitment-length histogram;
- queue flush rate;
- action chunks generated;
- policy calls per step;
- latency per step and per episode;
- paired success deltas;
- smoothness/jump statistics;
- whether high-uncertainty states actually receive shorter commitments than low-uncertainty states.

## Entropy And Dispersion Language

Accepted:

- SmolVLA flow vectors are not probability distributions.
- Deterministic 7D actions do not have KL or entropy by default.
- Unless Stage 0 defines a valid normalized distribution and estimator, the method will use the term `dispersion proxy`, not entropy.

The mathematical audit must specify one of two legal paths:

1. valid entropy:
   - define probability support;
   - define normalization;
   - define estimator;
   - justify why samples are drawn from the predictive distribution;
   - report estimator stability;
2. dispersion proxy:
   - define repeated-sample variance or pairwise chunk distance;
   - explicitly state it is not calibrated entropy;
   - test noncollapse and task/phase variation.

EAC may proceed with a dispersion proxy only as `EAC dispersion-calibrated chunking`; it must keep the AAC proxy separate and honestly labeled.

## Compute And Latency Boundary

Accepted:

- EAC may change policy-call count and latency.
- Success without compute reporting is insufficient.
- A method that improves success only by excessive policy calls may be nonviable even if scientifically informative.

Required reports:

- policy calls per step;
- action chunks generated per episode;
- wall-clock latency;
- VRAM;
- exact commitment map;
- expected call budget for Base, AAC proxy, EAC full, ablation, and fixed short replan.

## Action-Value Passthrough

Accepted:

- EAC's identity-preserving claim requires exact equality of postprocessed 7D action values before scheduling.
- Any smoothing, averaging, rescaling, clipping, learned residual, low-pass filtering, action-stat mapping, or action-value replacement is outside this proposal.
- If action values differ from Base before scheduling, the outcome is `IMPLEMENTATION_FAILURE`.

Required Stage 0 equality test:

- On a development batch, compare the frozen Base postprocessed chunk and the EAC pre-scheduling chunk elementwise.
- Required maximum absolute difference: `0.0` up to serialization/device roundoff defined in the audit.
- Report chunk shape, dtype, finite check, and postprocessor path.

## Hysteresis Boundary

Accepted:

- Hysteresis cannot become the whole method.
- The key ablation must remove the calibration/hysteresis component while preserving comparable queue mechanics.
- Stage 0/validation must report whether the full method's commitment choices vary with uncertainty/dispersion rather than mapping nearly every state to one length.

If the selected map is constant or nearly constant, stop as `DESIGN_FAILURE` or exact trivial equivalence before rollout.

## Validation Search Boundary

Accepted:

- Maximum six total configurations.
- Discovery/validation identities only.
- Freeze threshold, commitment map, and decision rules before Stage A.
- Save all tried configurations and negative results.
- No confirmatory test retuning.

The mathematical audit/preregistration must name the search factors and selection score before any validation run.

## Required Five-Policy Comparison

The first serious comparison remains exactly:

1. `frozen_smolvla_fixed_queue`
2. `aac_entropy_proxy`
3. `eac_full`
4. `eac_no_calibration_no_hysteresis_ablation`
5. `fixed_short_replan_baseline`

No extra simple baselines will be added before this comparison unless the mathematical audit identifies a concrete equivalence issue that could change the decision and is cheaper than proceeding.

## Stage 0 Stop Rules Accepted

EAC stops before rollout for:

- queue surface unavailable;
- uncertainty/dispersion collapsed or nonfinite;
- commitment map effectively constant;
- equality to Base, AAC proxy, fixed short replan, or no-calibration ablation;
- action-value mismatch before scheduling;
- catastrophic latency overhead;
- split/identity leakage;
- any hidden use of confirmatory identities or outcomes.

These are pre-rollout design or implementation failures, not closed-loop scientific kills.

## Rebuttal Decision

`EAC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

The method may proceed to mathematical mechanism audit because the reviewer constraints are accepted and the remaining risks are auditable before implementation or rollout. The audit must bind exact variables, shapes, uncertainty statistic, queue rule, equality checks, validation search budget, required ablation, and failure classifications.
