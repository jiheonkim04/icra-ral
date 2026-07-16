# MCI-VLA Reviewer B Attack

Date: 2026-07-16 KST

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Method: `MCI-VLA`

Reviewed proposal: `reports/mci_vla/researcher_proposal.md`

Proposal SHA-256:
`88CB11CC6236D19BA05602217C65C1819A68BEA53B041E17BA12796403BA0B9A`

## Summary

MCI is not rejected before development. The proposal has a plausible new
mechanism for this campaign: RoVLA-anchored multi-consistency invariance in a
zero-gated Base-preserving SmolVLA adapter.

The main risks are not enough to pre-kill the method, but they are serious:

- MCI could collapse into ordinary augmentation plus LoRA;
- a local RoVLA proxy could be too weak or too different from the prior;
- paraphrase, observation, proprioception, and action-evolution transforms
  could change task semantics or leak validation choices;
- the consistency code could collapse to a constant representation;
- action invariance could preserve wrong actions instead of improving policy
  behavior;
- clean Base behavior could be disrupted by a small adapter that acts
  everywhere.

Researcher A must rebut these risks before mathematical audit,
preregistration, implementation, validation search, rollout, or confirmatory
access.

## Closest-Prior Assessment

RoVLA remains the closest positive prior. It explicitly introduces
Instructional, Evolutionary, and Observational Consistency to improve VLA
robustness under task-preserving language, action-generation, visual, and
proprioceptive perturbations. That mechanism is closer to MCI than IntentVLA's
short-horizon intent memory or OA-WAM's object-addressable world-action model.

MCI can be novel only as a constrained transfer of RoVLA's multi-consistency
mechanism into a frozen-SmolVLA, Base-preserving adapter. It cannot claim to be
an official RoVLA reproduction unless the official code, backbone assumptions,
data semantics, and training conditions are faithfully matched.

## Required Conditions

1. Keep RoVLA as policy 2 in the first serious comparison.

The first serious comparison must remain exactly:

1. `smolvla_base`
2. `rovla_multiconsistency_proxy`
3. `mci_full`
4. `mci_no_consistency_code_ablation`
5. `augmentation_only_lora_killer`

Do not defer RoVLA until after internal ablations. Do not add a sixth
standard-LoRA policy before this comparison unless the proposal changes into a
PEFT scientific claim, which is currently forbidden.

2. Narrow the novelty claim.

MCI may claim only:

`Base-preserving multi-consistency invariance for frozen SmolVLA action chunks`.

It may not claim generic VLA robustness, generic data augmentation, generic
LoRA adaptation, full RoVLA reproduction, object binding, short-horizon memory,
critical-step residual repair, or chunk-boundary smoothing.

3. Make the RoVLA proxy transparent and mechanism-faithful.

If official RoVLA cannot be run on the local SmolVLA backbone and budget, the
proxy must be explicitly labeled a transparent local proxy. It must preserve
the essential mechanism:

- instruction consistency;
- observation/proprioception consistency;
- action-evolution consistency.

It may omit incompatible full-backbone details only if the mismatch is listed.
It may not be a weak generic augmentation baseline renamed as RoVLA.

4. Freeze legal transformation generators before training.

Instruction paraphrases, image/proprio perturbations, and action-evolution
perturbations must be deterministic or fully logged, task-preserving, and
generated from discovery/validation identities only. No unlogged LLM
paraphrase generation, prompt iteration, or transform selection after seeing
confirmatory results is allowed.

A transform that changes task semantics or action semantics invalidates the
row and must stop the audit as `DATA_OR_SUPERVISION_FAILURE` or
`DESIGN_FAILURE`, not become closed-loop evidence.

5. Prove transformation and label health.

Stage 0 must report, by transformation family:

- pair count;
- positive and negative contrast count;
- task and demo coverage;
- representation variance;
- mask or gate activation rate;
- duplicate key count;
- train/validation/confirmatory overlap count;
- all-zero/all-one target checks.

Collapsed transforms, collapsed labels, or identical full/ablation targets are
data failures.

6. Prove deployment-time observability.

The consistency signal must be predictable from legal inference inputs:
current RGB or cached legal visual features, proprioception, task/language
string, and the frozen Base chunk. It may not depend on object pose, simulator
state, reward, success, done, reset identity, future observation, future expert
actions, or confirmatory outcomes.

Trivial baselines must include at least task identity, demo/frame phase used
only as an audit baseline, action-magnitude statistics, and augmentation
family identity. If those explain the signal, the method is not ready for
rollout.

7. Keep the augmentation-only LoRA killer live.

Policy 5, `augmentation_only_lora_killer`, must use the same development data,
same legal augmentations, same low-compute scaffold where technically valid,
and matched training budget without the multi-consistency code mechanism.

If policy 5 explains the gain, MCI is not a paper candidate. Do not replace it
with an easier baseline.

8. Preserve Base identity before any rollout.

MCI must show exact Base passthrough at initialization, unchanged frozen Base
weights, checkpoint persistence and reload, bounded per-group action deltas,
postprocessed action validity, gripper behavior sanity, and clean validation
retention. A module that changes nearly every action is an implementation or
design failure.

9. Audit objective scale, gradients, and representation collapse.

The mathematical audit must define all variables, tensor shapes, formulas,
units, gradient paths, objective scales, and gradient norm ratios for
`L_code`, `L_act`, `L_fit`, `L_keep`, `L_var`, and `L_bound`.

No deterministic 7D-action KL is allowed. The audit must include a simpler
alternative, a required ablation, and a representation noncollapse check.

10. No hidden confirmatory-test access or resource-metric contamination.

Confirmatory identities, rewards, success flags, done flags, simulator states,
object poses, future observations, and confirmatory outcomes may not tune MCI.
The Windows gaming / Efficiency Mode interval remains a resource-contention
interval: latency, throughput, wall-clock efficiency, and resource utilization
from overlapping measurements are not final paper evidence.

## Reviewer Decision

Conditional pass to Researcher A rebuttal. Researcher A must explicitly accept
all ten conditions or narrow the proposal before mathematical audit,
preregistration, implementation, validation search, rollout, or confirmatory
test access.
