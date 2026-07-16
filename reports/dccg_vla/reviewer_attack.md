# DCCG-VLA Reviewer B Attack

Date: 2026-07-16 KST

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Reviewed proposal: `reports/dccg_vla/researcher_proposal.md`

Proposal hash:
`AE5DBB13F0B4C19E3DD8BD054433DCFBCC301F4C4293D7B98883D76CA4A1390E`

## Summary Judgment

DCCG is allowed to proceed only as a narrowly defined extension of ACG:
demonstration-calibrated action-coherence guidance for flow-based SmolVLA.

The proposal is plausible and prior-anchored, but it sits near several crowded
axes: ACG, generic smoothing, temporal ensembling, chunk-consistency methods,
Legato-style smoothness, and earlier campaign spline/seam methods. The rebuttal
must accept the constraints below before mathematical audit or implementation.

## Major Risks

### 1. Novelty Can Collapse Into ACG Or Smoothing

ACG already claims action-coherence guidance for flow-based VLA models. DCCG is
novel only if the data-calibrated coherence energy is the causal technical
object, not just a retuned guidance scale, a smoothed action sequence, or a
renamed ACG perturbation.

Required condition:

- ACG remains policy 2 in the first serious comparison.
- `action_smoothing_simple_killer` remains policy 5.
- DCCG must report a mechanism metric showing it changes chunks differently
  from both ACG and smoothing.
- Any gain explained by policy 2 or policy 5 kills the DCCG claim.

### 2. Demonstration Phase Bins Risk Privileged Deployment Leakage

The proposal mentions normalized chunk index within demonstrations. That value
is available when fitting statistics but not during deployment unless inferred
from legal inputs.

Required condition:

- Deployment bin selection may use only current generated action features,
  ordinary instruction/task-family information, queue/chunk index, and
  nonprivileged action history.
- Demonstration time index may be used only to stratify training diagnostics,
  not as an inference input.
- The mathematical audit must define the exact deployment bin function.

### 3. P95, Median, And IQR Are Not Automatically Differentiable Guidance

The proposal uses robust statistics and percentile-like features. That is fine
for diagnostics, but flow guidance needs a well-defined gradient path.

Required condition:

- The mathematical audit must specify differentiable or subgradient-safe
  implementations of all guided features.
- Stage 0A must prove finite nonzero gradients on real action chunks.
- If a feature is nondifferentiable, it may gate or score but may not supply
  unverified action gradients.

### 4. Gripper Events Are Easy To Destroy

Generic coherence penalties often smooth away sparse gripper transitions, which
can improve a coherence score while damaging manipulation.

Required condition:

- Gripper transition preservation is a hard pre-rollout gate.
- The simple smoothing baseline must include a gripper-event-preserving
  variant if that is the strongest reviewer-killer.
- DCCG must report transition count, reversal count, sign-change timing, and
  gripper delta statistics.

### 5. Action-Space Validity Must Be Postprocessor-Aware

Coherence on normalized actions can look legal while postprocessed actions
violate the actual 7D interface or Base-relative caps.

Required condition:

- DCCG must audit both normalized and postprocessed action validity.
- Stage 0 must report translation, rotation, and gripper deltas from Base.
- Any nonfinite action, shape mismatch, or invalid postprocessed action is an
  implementation failure, not a scientific result.

### 6. No-Headroom Is A Valid Early Stop

If Base and ACG already produce coherent validation chunks, DCCG has no local
room to improve. The method may not rescue itself by inventing new jitter,
changing tasks, or moving thresholds.

Required condition:

- Stage 0B must compare Base, ACG, DCCG, no-demo-calibration, and smoothing on
  the same validation rows.
- If ACG or smoothing dominates DCCG on the frozen proxy and action validity,
  stop as `DESIGN_FAILURE`.
- If Base and ACG leave no meaningful coherence or task proxy headroom, stop
  as `NO_HEADROOM`.

### 7. ACG Proxy Must Be Transparent

Official ACG may not directly run under the local SmolVLA/LIBERO stack. A
proxy is acceptable only if it faithfully implements the published perturbation
guidance mechanism and is labeled as a transparent local proxy.

Required condition:

- The protocol must first attempt to identify official ACG assets and code
  compatibility.
- If exact official reproduction is unavailable, the proxy must document every
  mismatch from the paper.
- DCCG may not compare against a weak smoothing-only stand-in for ACG.

### 8. Do Not Reopen Closed Methods

DCCG must not rescue MHS, NICE, S2C, HEST, LCG, AFID, BRID, or any previous
closed method by changing their labels, thresholds, tasks, or interpretation.

Required condition:

- DCCG may cite those failures only as motivation and overlap constraints.
- It may not reuse a closed method's failed labels as a tuned rescue signal.

## Required Rebuttal Commitments

Researcher A must explicitly accept:

1. ACG as policy 2 and action smoothing as policy 5.
2. No demonstration time index or privileged phase at inference.
3. A differentiable or subgradient-safe coherence guidance audit.
4. Gripper-event preservation as a hard gate.
5. Normalized and postprocessed action validity checks.
6. No-headroom, ACG dominance, smoothing dominance, or ablation dominance as
   valid stops.
7. Transparent official/proxy ACG provenance.
8. No deterministic-action KL.
9. No confirmatory-test tuning.
10. No rescue of MHS, NICE, or other closed methods.

## Conditional Pass

Reviewer B conditionally passes DCCG to Researcher A rebuttal. The rebuttal
must accept every condition above. If accepted, the next stage is a
mathematical mechanism audit. No implementation, validation search, rollout,
or confirmatory access is allowed before the rebuttal and audit are complete.
