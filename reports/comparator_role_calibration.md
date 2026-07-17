# Comparator-Role Calibration

Date: 2026-07-18 KST

This is an active clarification for future unfrozen stages of the official-prior-first campaign. It is not a new epoch, not a restart, and not a retroactive conversion of any frozen non-GO into GO.

## Current-State Inspection

- Branch: `codex/epoch5-official-prior-first`
- HEAD at inspection: `05a0537db811b3684eb06568ea39c350ee47353e`
- Current stage: Epoch 5 official-prior-first residual search after the task75 second-prior gate.
- Active Ours method: none.
- Active training: none.
- Active preregistered worker: none found. The post-task75 X-VLA `libero_spatial` identity `20260725` scan is complete, with exit code `0` and heartbeat/finish time `2026-07-18T02:24:11+09:00`.
- Frozen task75 decision: `TASK75_SECOND_PRIOR_INFRASTRUCTURE_BLOCKED`; do not convert to policy failure or Ours authorization.

If a future preregistered experiment is already running under a frozen protocol, finish it under that protocol. If the frozen protocol explicitly requires a universal beat-all scalar condition, report both `FROZEN_PROTOCOL_DECISION` and `CALIBRATED_SCIENTIFIC_INTERPRETATION`. Do not rewrite the historical decision rule after observing results.

## Binding Clarification

The campaign still requires official-prior-first research, matched local evaluation, preregistration, held-out confirmatory tests, no cherry-picking, no test-set tuning, and honest negative reporting.

The clarification is narrower: Base, closest Prior, key Ablation, and simple Control are not interchangeable entries in a single global max-score threshold. Each comparator blocks only the claim it was included to test.

| Comparator role | Scientific question | Claim metric | Blocking condition | Nonblocking tradeoffs |
| --- | --- | --- | --- | --- |
| Base | Does Ours improve the backbone on the prespecified claim axis? | Primary claim metric, plus clean retention when relevant | No meaningful positive effect and no paired evidence of direction; or clean degradation exceeds frozen margin without a declared tradeoff claim | Clean success can be slightly lower if inside a frozen retention margin and target-condition gain is substantial |
| Closest Prior | Does Ours advance beyond the closest existing method under a matched local protocol? | Same claim axis as the prior comparison, or a prespecified Pareto axis | Ours is lower on the claim metric and offers no meaningful efficiency, robustness, data, supervision, or generalization advantage | Comparable success can be acceptable with lower latency, lower compute, less supervision, broader generalization, or stronger robustness |
| Key Ablation | Is the claimed component responsible for the effect? | Mechanism-relevant metric, claim success, robustness, generalization, clean-retention tradeoff, or efficiency-adjusted performance | Ablation matches or exceeds full on the primary claim, mechanism metric, and any claimed efficiency/generalization benefit | Ablation can be slightly higher on clean success when the component is designed for robustness and the robustness/mechanism effect is prespecified |
| Simple Control | Can a trivial explanation account for the claimed gain? | The strongest plausible simple explanation under matched cost/data/setup | Control explains substantially all gain with equal or better claim performance, equal/lower cost, no clean/generalization loss, and no missing mechanism capability | A control winning one isolated task does not block if it loses the matched aggregate, second condition, consistency, efficiency-adjusted result, or mechanism capability |
| Standard LoRA, when included | Would generic adaptation with comparable data and compute produce the same effect? | Matched setup with same base, data, split, budget, steps, checkpoint rule, and evaluation manifest | Generic LoRA explains the claimed gain under matched setup | LoRA remains implementation infrastructure unless the paper explicitly claims an adaptation algorithm |

## Stage Interpretation

- Stage 0 validates data, supervision, observability, gradients, capacity, identity preservation, and mechanism activity. It is not a final beat-all task-success gate.
- Stage A detects catastrophic degradation, verifies real closed-loop mechanism activity, and estimates direction. One- or two-episode gaps are not permanent decisions unless a frozen catastrophic rule explicitly applies.
- Stage B evaluates the primary claim on a matched paired protocol. It must report paired outcomes, effect size, confidence interval, clean retention, mechanism evidence, and comparator-specific answers.
- Paper scale-up requires the full package: Base improvement on the claim axis, prior advance or Pareto advantage, ablation evidence, strongest simple explanation not sufficient, second backbone/condition/benchmark evidence, statistics, efficiency, and reproducibility.

## Decision Output For Serious Ours Results

For every serious Ours result, include a comparator-role table:

| Comparator | Scientific question | Matched result | Uncertainty | Does it block the claim? | Reason |
| --- | --- | --- | --- | --- | --- |

Then report:

- `BASE_CLAIM_STATUS`
- `PRIOR_ADVANCE_STATUS`
- `ABLATION_COMPONENT_STATUS`
- `SIMPLE_EXPLANATION_STATUS`
- `CLEAN_RETENTION_STATUS`
- `GENERALIZATION_STATUS`
- `OVERALL_PAPER_CANDIDATE_STATUS`

Allowed overall statuses:

1. `PAPER_CANDIDATE_GO`
2. `PROMISING_NEEDS_CONFIRMATION`
3. `CLAIM_NARROWING_REQUIRED`
4. `UNDERPOWERED_ONE_EXPANSION_ALLOWED`
5. `PRIOR_ADVANCE_NOT_ESTABLISHED`
6. `KEY_COMPONENT_NOT_SUPPORTED`
7. `SIMPLE_CONTROL_EXPLAINS_GAIN`
8. `CLEAN_RETENTION_FAILURE`
9. `VALID_METHOD_KILL`
10. `IMPLEMENTATION_DATA_OR_RESOURCE_FAILURE`

## Current Application

The task75 second-prior gate and the post-task75 X-VLA spatial scan are prior diagnostics, not Ours results. This calibration therefore does not authorize task75 method design, training, LoRA/QLoRA diagnostics, or reopening BR-XVLA/MPR-XVLA/PRC-XVLA/CR-LightVLA/ATCD/MCI/CSPR.

Apply this clarification to the next unfrozen Ours protocol if official-prior-first residual gates eventually authorize candidate generation.
