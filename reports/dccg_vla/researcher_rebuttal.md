# DCCG-VLA Researcher A Rebuttal

Date: 2026-07-16 KST

Decision: `DCCG_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Method: `DCCG-VLA`, Demonstration-Calibrated Coherence Guidance for SmolVLA.

Proposal: `reports/dccg_vla/researcher_proposal.md`

Reviewer attack: `reports/dccg_vla/reviewer_attack.md`

Proposal SHA-256:
`AE5DBB13F0B4C19E3DD8BD054433DCFBCC301F4C4293D7B98883D76CA4A1390E`

## Response Summary

Researcher A accepts every Reviewer B condition.

DCCG will proceed only as a narrow ACG extension: a frozen-SmolVLA,
identity-preserving, demonstration-calibrated action-coherence guidance
mechanism for flow-based action chunks. LoRA remains implementation
infrastructure only. DCCG will not claim generic smoothing, temporal
ensembling, chunk consistency, or ordinary low-rank adaptation as the
scientific mechanism.

No DCCG implementation, validation search, rollout, simulator evaluation, or
confirmatory-test tuning has happened before this rebuttal.

## Accepted Closest Prior And Simple Killer

Researcher A accepts that ACG remains policy 2 and action smoothing remains
policy 5 in the first serious comparison:

1. `smolvla_base`
2. `acg_official_proxy`
3. `dccg_full`
4. `dccg_no_demo_calibration_ablation`
5. `action_smoothing_simple_killer`

DCCG must report a mechanism metric showing that its demonstration-calibrated
coherence energy changes chunks differently from both ACG and smoothing. If
ACG or the smoothing baseline explains the gain, the DCCG paper claim is
killed under the frozen protocol.

## Accepted Legal Inference Inputs

Researcher A accepts that deployment bin selection may use only legal
nonprivileged inputs:

- current generated action features;
- ordinary instruction or task-family information;
- queue or chunk index available to the deployed policy;
- nonprivileged action history.

Demonstration time index may stratify training diagnostics only. It is not an
inference input. The mathematical audit must freeze the exact deployment bin
function and must prove that no reset identity, future observation, future
expert action, reward, success bit, simulator state, or confirmatory outcome is
used.

## Accepted Differentiability Requirement

Researcher A accepts that robust statistics such as median, IQR, p95, and
transition counts are not automatically valid flow-guidance gradients.

The mathematical audit must specify differentiable or subgradient-safe
implementations for every guided feature. Stage 0A must show finite nonzero
gradients on real action chunks. Any nondifferentiable feature may gate, score,
or audit the method only if it is not used as an unverified gradient source.

No deterministic-action KL is allowed. Deterministic 7D actions and SmolVLA
flow vectors are not probability distributions.

## Accepted Gripper And Action-Validity Gates

Researcher A accepts gripper-event preservation as a hard pre-rollout gate.

DCCG must report:

- transition count;
- reversal count;
- sign-change timing;
- gripper delta statistics;
- translation, rotation, and gripper deltas from Base.

The action-smoothing simple killer may use a gripper-event-preserving variant
if that is the strongest reviewer-killer.

DCCG must audit both normalized actions and postprocessed actions. Any nonfinite
action, shape mismatch, invalid postprocessed action, or violation of the
frozen Base-relative caps is an `IMPLEMENTATION_FAILURE`, not a scientific
result.

## Accepted Stop Conditions

Researcher A accepts the following as valid stops before rollout or paper
claiming:

- `NO_HEADROOM` if Base and ACG leave no meaningful coherence or task-proxy
  room;
- `DESIGN_FAILURE` if ACG, smoothing, or the no-demo-calibration ablation
  dominates DCCG under matched validation rows;
- `DATA_FAILURE` if coherence features, bins, gripper events, or legal
  contrasts collapse;
- `IMPLEMENTATION_FAILURE` if source loading, gradient, checkpoint,
  postprocessing, action validity, or JSON persistence fails.

These failures may not be rescued by changing tasks, thresholds, identities,
feature definitions, policy order, or decision rules.

## Accepted ACG Provenance Rule

Researcher A accepts that the protocol must first attempt to identify official
ACG assets and code compatibility.

If official ACG cannot run exactly in the local SmolVLA/LIBERO stack, policy 2
must be labeled as a transparent local proxy. The proxy must document every
material mismatch from the published ACG mechanism and may not degrade into a
weak smoothing-only stand-in.

## Accepted Closed-Method Boundary

DCCG may cite MHS, NICE, S2C, HEST, LCG, AFID, BRID, and other closed methods
only as motivation or overlap constraints.

DCCG may not reopen, rescue, relabel, retune, reinterpret, or reuse any closed
method's failed labels, thresholds, tasks, identities, or decision rules as a
tuned DCCG signal.

## Accepted No Confirmatory-Test Tuning

Researcher A accepts that no confirmatory task, reset identity, outcome,
threshold, metric, or partial result may alter:

- DCCG features;
- bin definitions;
- guidance scale;
- gates;
- clipping caps;
- policy order;
- ablation definition;
- smoothing baseline;
- ACG proxy status;
- task list;
- decision rule.

A major redesign after confirmatory access is a new method cycle.

## Immediate Next Stage

Proceed to the DCCG mathematical mechanism audit before preregistration,
prototype protocol, implementation, validation search, rollout, or
confirmatory-test access.
