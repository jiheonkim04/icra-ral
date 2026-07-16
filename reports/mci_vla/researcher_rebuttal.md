# MCI-VLA Researcher A Rebuttal

Date: 2026-07-16 KST

Decision: `MCI_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Method: `MCI-VLA`, Multi-Consistency Invariance for Base-preserving SmolVLA.

Proposal: `reports/mci_vla/researcher_proposal.md`

Reviewer attack: `reports/mci_vla/reviewer_attack.md`

Proposal SHA-256:
`88CB11CC6236D19BA05602217C65C1819A68BEA53B041E17BA12796403BA0B9A`

## Response Summary

Researcher A accepts all ten Reviewer B conditions without modification.

MCI will proceed only as a narrow RoVLA-anchored extension: Base-preserving
multi-consistency invariance for frozen SmolVLA action chunks. LoRA or another
small adapter may be used only as implementation infrastructure. It is not the
scientific mechanism.

No MCI implementation, training, validation search, rollout, simulator
evaluation, or confirmatory-test access has happened before this rebuttal.

## Accepted Closest Prior And Policy Order

RoVLA remains policy 2 in the first serious comparison. The first serious
comparison remains exactly:

1. `smolvla_base`
2. `rovla_multiconsistency_proxy`
3. `mci_full`
4. `mci_no_consistency_code_ablation`
5. `augmentation_only_lora_killer`

The RoVLA policy must enter before any broad internal-control suite. If RoVLA
or the augmentation-only killer explains the gain, MCI is not a paper
candidate.

## Accepted Novelty Boundary

MCI claims only:

`Base-preserving multi-consistency invariance for frozen SmolVLA action chunks`.

It will not claim generic VLA robustness, generic data augmentation, generic
LoRA adaptation, full RoVLA reproduction, object binding, short-horizon
memory, critical-step residual repair, or chunk-boundary smoothing.

## Accepted RoVLA Proxy Rule

Researcher A accepts that policy 2 must be either an official RoVLA-compatible
run or a transparent local proxy. If official RoVLA cannot be run faithfully in
the local SmolVLA/LIBERO budget, the proxy must be named
`rovla_multiconsistency_proxy` and must preserve the essential RoVLA mechanism:

- instruction consistency;
- observation/proprioception consistency;
- action-evolution consistency.

Every material mismatch from the official RoVLA mechanism must be listed. A
weak generic augmentation baseline may not be renamed as RoVLA.

## Accepted Transformation Freeze

All instruction paraphrases, image/proprioception perturbations, and
action-evolution perturbations must be deterministic or fully logged,
task-preserving, and generated only from discovery/validation identities.

No unlogged LLM paraphrase generation, prompt iteration, or transform
selection may occur after confirmatory results. Any transformation that changes
task or action semantics invalidates the row and stops the audit as
`DATA_OR_SUPERVISION_FAILURE` or `DESIGN_FAILURE`.

## Accepted Data And Label Health Gates

Before training or rollout, Stage 0 must report by transformation family:

- pair count;
- positive and negative contrast count;
- task and demo coverage;
- representation variance;
- mask or gate activation rate;
- duplicate key count;
- train/validation/confirmatory overlap count;
- all-zero/all-one target checks.

Collapsed transforms, collapsed labels, or identical full/ablation targets are
data failures, not closed-loop scientific evidence.

## Accepted Deployment Observability Rule

The consistency signal must be predictable from legal deployment inputs only:
current RGB or cached legal visual features, proprioception, task/language
string, and the frozen Base chunk.

MCI may not use object pose, simulator state, reward, success, done, reset
identity, future observation, future expert action, or confirmatory outcome.
Trivial baselines must include task identity, demo/frame phase as an audit-only
baseline, action-magnitude statistics, and augmentation-family identity.

If those trivial baselines explain the signal, MCI is not ready for rollout.

## Accepted Simple Killer

`augmentation_only_lora_killer` remains policy 5. It must use the same
development data, same legal augmentations, same low-compute scaffold where
technically valid, and matched training budget, but without the
multi-consistency code mechanism.

If policy 5 explains the gain, MCI is not a paper candidate. It may not be
replaced with an easier baseline.

## Accepted Identity-Preservation Gates

Before rollout, MCI must show:

- exact Base passthrough at initialization;
- unchanged frozen Base weights;
- checkpoint persistence and disk reload;
- bounded translation, rotation, and gripper deltas;
- postprocessed action validity;
- gripper behavior sanity;
- clean validation retention.

A module that changes nearly every action is an `IMPLEMENTATION_FAILURE` or
`DESIGN_FAILURE`, not evidence against the closed-loop claim.

## Accepted Objective And Gradient Audit

The mathematical audit must define variables, tensor shapes, formulas, units,
scales, gradient paths, simpler alternatives, and required ablations for:

- `L_code`;
- `L_act`;
- `L_fit`;
- `L_keep`;
- `L_var`;
- `L_bound`.

It must estimate term magnitudes and gradient norm ratios on a small batch,
check representation noncollapse, and document any gradient conflict relevant
to the adapter.

No deterministic 7D-action KL is allowed. SmolVLA flow vectors and
deterministic actions are not probability distributions.

## Accepted Contamination Boundary

Confirmatory identities, rewards, success flags, done flags, simulator states,
object poses, future observations, and confirmatory outcomes may not tune MCI.

The Windows gaming / Efficiency Mode interval remains a resource-contention
interval. Latency, throughput, wall-clock efficiency, and resource utilization
from overlapping measurements are not final paper evidence.

## Immediate Next Stage

Proceed to the MCI mathematical mechanism audit before preregistration,
prototype protocol, implementation, validation search, rollout, or
confirmatory-test access.
