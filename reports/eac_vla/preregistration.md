# EAC-VLA Preregistration

Date: 2026-07-15 KST

Method: `EAC-VLA`, Entropy-Calibrated Adaptive Chunking for frozen SmolVLA.

Proposal hash: `A89ED48AE9FD4D26A8DA9E3E987FACDBBD9F861D070AE135372A092A44581E4E`

Mathematical audit: `reports/eac_vla/mathematical_mechanism_audit.md`

Preregistered decision: `EAC_PREREGISTRATION_FROZEN_STAGE_0_PENDING`

## Evidence Partitions

DISCOVERY:

- prior-cycle failures and source inspection;
- current official SmolVLA queue/chunk semantics;
- local mechanism feasibility hypotheses.

VALIDATION:

- Stage 0 source, queue, dispersion, action-value equality, latency, and split audits;
- bounded validation search over at most six configurations if Stage 0 passes;
- any validation rollout or proxy explicitly named before it is run.

CONFIRMATORY_TEST:

- Stage A/B official LIBERO paired manifests only after Stage 0, validation search, selected configuration, baselines, ablation, metrics, thresholds, task/reset identities, and policy identities are frozen.

Confirmatory outcomes may not tune the same method.

## Frozen Method Definition

EAC keeps the frozen SmolVLA policy and official postprocessed 7D actions unchanged. It computes a queue-risk statistic from deployment-observable action-chunk statistics and selects how many actions from the current `50 x 7` chunk to execute before refreshing observation.

No training is part of the default EAC method. If a learned scalar calibrator is later proposed, it must fit inside the six-configuration validation budget and may update only calibrator parameters, never SmolVLA.

## Stage 0 Development Audit

Stage 0 must run before validation search, implementation scale-up, Stage A manifest freeze, or rollout.

Required checks:

1. Queue surface:
   - frozen Base postprocessed chunk shape `[50, 7]`;
   - queue length observable or controllable;
   - prefix execution or queue flush can be implemented without changing action values.

2. Dispersion or entropy source:
   - repeated/stochastic chunk generation is legal if used;
   - statistic is finite and noncollapsed;
   - task and phase variation exists;
   - no target action, reward, success, future observation, reset identity, or confirmatory outcome is used.

3. Action-value passthrough:
   - `max_abs(EAC_pre_scheduling_chunk - Base_chunk) <= 1e-7`, unless a stricter exact-equality implementation is available;
   - postprocessor path unchanged;
   - action dimension and bounds valid.

4. Commitment map:
   - candidate commitment lengths in `{1, 2, 4, 8, 16, 50}`;
   - selected or candidate map not collapsed;
   - high-risk states choose shorter commitments more often than low-risk states;
   - no more than `90%` of decisions map to one commitment length unless the method stops.

5. Resource and validity:
   - policy calls per step;
   - chunk generation count;
   - latency;
   - finite action checks;
   - zero discovery/validation/confirmatory identity overlap.

Stage 0 hard stops:

- `DESIGN_FAILURE`: uncertainty/dispersion collapsed, commitment map constant, equivalence to Base/AAC proxy/fixed short replan, or no usable queue surface.
- `DATA_OR_SUPERVISION_FAILURE`: split/source/identity health fails.
- `IMPLEMENTATION_FAILURE`: queue control changes action values, corrupts action semantics, breaks resume, or changes official postprocessing.
- `NO_HEADROOM`: diagnostic oracle or proxy shows queue scheduling cannot plausibly affect the claimed condition.

Stage 0 is not a closed-loop scientific result.

## Bounded Validation Search

Allowed only if Stage 0 passes.

Budget:

- at most six total configurations;
- at most three values for one risk threshold;
- at most two commitment/hysteresis maps;
- no combinatorial expansion beyond six;
- at most two lightweight seeds only if a scalar calibrator is introduced.

Selection score:

`S = s_proxy + s_clean + s_active + s_valid - p_latency - p_oscillation`

Preregistered component intent:

- `s_proxy`: validation success proxy or preregistered small validation rollout outcome;
- `s_clean`: action-value passthrough, action-bound validity, and clean behavior;
- `s_active`: noncollapsed commitment choices and risk-commitment monotonicity;
- `s_valid`: finite outputs and queue-control validity;
- `p_latency`: extra policy-call and wall-clock penalty;
- `p_oscillation`: rapid commitment-switching penalty.

Exact weights must be frozen in the Stage 0/validation implementation report before the search is executed. Do not select purely by offline action L2.

## Frozen First Serious Comparison

Exactly five policies:

1. `frozen_smolvla_fixed_queue`
2. `aac_entropy_proxy`
3. `eac_full`
4. `eac_no_calibration_no_hysteresis_ablation`
5. `fixed_short_replan_baseline`

`aac_entropy_proxy` is a faithful transparent local proxy, not an official AAC reproduction unless exact official equivalence is independently established.

## Stage A

Stage A uses approximately ten paired episodes per policy with a matched task/reset manifest frozen before rollout.

Purpose:

- detect catastrophic harm;
- detect exact trivial equivalence;
- detect AAC proxy or fixed-replan dominance;
- verify mechanism activation and action-value preservation during real rollout.

Stage A may permanently kill only for:

- mechanism invalidity;
- no headroom;
- catastrophic degradation under current governance;
- clear prior, ablation, or simple-baseline dominance;
- exact trivial equivalence.

Small negative, tie, or unresolved differences advance to Stage B.

## Stage B

Stage B uses at least forty paired episodes per key policy.

Report:

- task-balanced official closed-loop success;
- paired wins/losses/ties;
- bootstrap confidence intervals;
- per-task breakdown;
- commitment-length distribution;
- queue flush rate;
- policy calls per step;
- latency and VRAM;
- action validity;
- smoothness and boundary-jump statistics.

Allow one expansion to eighty only if Stage B is genuinely unresolved under current governance.

## Prototype GO Criteria

EAC becomes a serious paper candidate only if:

- EAC beats frozen Base;
- EAC beats the AAC proxy on the matched claim axis;
- EAC beats the no-calibration/no-hysteresis ablation;
- fixed short replanning does not explain the gain;
- action values are preserved before scheduling;
- clean behavior and action validity are retained;
- mechanism evidence supports uncertainty-calibrated commitment rather than fixed cadence;
- compute/latency overhead is acceptable.

## Non-Rescue Rules

Do not rescue EAC after a valid kill by:

- changing the dispersion/entropy statistic;
- changing thresholds or commitment maps;
- changing policy identities;
- changing tasks or reset identities;
- adding hyperparameter variants beyond budget;
- reinterpreting fixed short-replan or AAC proxy dominance;
- using confirmatory outcomes to retune the method.

Major redesign after confirmatory test becomes a new method cycle.
