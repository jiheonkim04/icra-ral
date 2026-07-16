# MCI-VLA Researcher A Proposal

Date: 2026-07-16 KST

Decision: `MCI_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

Method: `MCI-VLA`

Full name: Multi-Consistency Invariance for Base-preserving SmolVLA

Contribution type: `PRIOR_EXTENSION`

Closest positive prior: RoVLA

Primary sources:

- https://arxiv.org/abs/2605.19678
- https://arxiv.org/html/2605.19678v1
- https://github.com/HCPLab-SYSU/RoVLA

## Fixed Claim

MCI-VLA tests whether RoVLA-style multi-consistency invariance can improve
frozen SmolVLA manipulation robustness by making a small identity-preserving
adapter stable under task-preserving language, observation/proprioception, and
action-evolution perturbations.

The scientific mechanism is multi-consistency invariance. LoRA or another
small adapter may only parameterize the low-compute implementation. Removing
LoRA from the description must not remove the novelty.

The narrow paper claim, if supported, is:

> A Base-preserving SmolVLA adapter trained with multi-consistency invariance
> improves matched LIBERO robustness over a RoVLA-style consistency proxy,
> a no-consistency-code ablation, and an augmentation-only LoRA alternative.

MCI does not claim a new VLA backbone, a generic LoRA method, a residual
critical-step rescue, a chunk-scheduling method, a latent drift monitor, or a
generic history memory.

## External Prior Anchor

RoVLA is the closest positive prior. It targets VLA brittleness under visual
observation changes, paraphrased language instructions, and compounded
perturbations. Its mechanism introduces multi-consistency constraints:

- Instructional Consistency for equivalent instruction rewrites;
- Evolutionary Consistency for action-intent stability across flow-matching
  stages;
- Observational Consistency for visual and proprioceptive perturbations.

The positive result reported by RoVLA is improved robustness on LIBERO-Plus,
RoboTwin 2.0, and real-world manipulation tasks. The public repository contains
training/evaluation code and consistency-learning components.

MCI extends the same claim axis but changes the technical setting:

- RoVLA: large GR00T/InternVL-style VLA trained with multi-consistency
  constraints.
- MCI-VLA: frozen SmolVLA plus a zero-gated, Base-preserving adapter trained
  to learn a compact consistency code from existing LIBERO demonstrations.

This is a prior extension, not a claim that local MCI is an official RoVLA
reproduction. If official RoVLA code cannot be run on the local SmolVLA
backbone, policy 2 must be labeled a transparent RoVLA-style proxy.

## Closed-Method Boundary

CSPR is closed as `CSPR_STAGE_0_IMPLEMENTATION_FAILURE`. MCI does not repair
CSPR's gradient-scale failure, reuse its criticality labels, change its
thresholds, relaunch its worker, or reinterpret its result.

MCI also avoids already-exercised routes:

- VLA-Corrector / NICE latent drift monitoring;
- AAC / EAC adaptive chunk scheduling;
- SEAM / ChunkFlow / S2C boundary smoothing;
- MHS / RAR generic history residual memory;
- DCCG demonstration-calibrated action coherence;
- standard LoRA as a scientific mechanism.

## Evidence Partitions

Discovery and validation may use only existing LIBERO demonstrations and
cached/legal SmolVLA Base outputs. The initial local development partition is
restricted to the verified cache-covered identities unless a later
preregistered source audit proves new coverage:

- `libero_10/task_5`
- `libero_goal/task_5`
- `libero_object/task_3`
- `libero_spatial/task_3`

with demo ids `0..9`.

Default development split for the first audit:

- discovery demos: `0..7`;
- validation demos: `8..9`;
- confirmatory rollout identities: sealed until after proposal, review,
  rebuttal, mathematical audit, preregistration, prototype protocol,
  implementation validation, and final configuration freeze.

No confirmatory outcome may change transformations, labels, coefficients,
architecture, policy list, thresholds, tasks, reset identities, metrics, or
decision rules.

## Legal Inputs

Allowed at training and validation:

- current RGB observations or cached legal visual features;
- proprioception;
- task/language strings;
- frozen SmolVLA Base action chunk `B`;
- demonstration action chunk `Y` from development demonstrations;
- task-preserving perturbation metadata generated from discovery/validation
  identities only.

Allowed at inference:

- the ordinary SmolVLA deployment inputs;
- the frozen Base action chunk currently produced by SmolVLA;
- the learned MCI consistency code, adapter output, and gate.

Prohibited at inference:

- simulator object state or privileged `states` arrays;
- reward, success, done, timeout, or reset identity;
- future observations or future expert actions;
- human correction labels;
- confirmatory-test outcomes or hidden confirmatory identities.

## Task-Preserving Transformations

MCI uses paired same-task transformations as supervision. The exact generators
must be frozen in preregistration before any confirmatory access.

Instructional transformations:

- deterministic task-preserving paraphrase templates over the known task
  string;
- official equivalent language strings if locally available;
- no unlogged LLM paraphrase generation after validation.

Observational transformations:

- bounded RGB brightness, contrast, crop/resize, and low-amplitude noise;
- optional PGD-style image perturbation only if the implementation is
  transparent and uses development data only;
- bounded proprioceptive jitter within deployment-valid ranges.

Action-evolution transformations:

- flow-time or action-noise perturbations of the current Base chunk feature
  path;
- small valid action perturbations used only to enforce consistency of intent,
  not to create a new action target from confirmatory outcomes.

Every transformation must preserve task identity and action semantics. If a
transformation changes the task, collapses labels, or cannot be inferred from
deployment inputs, Stage 0 must stop before rollout.

## Mechanism

Let:

- `N`: batch size;
- `H = 50`: SmolVLA action horizon;
- `D = 7`: normalized action dimension;
- `o`: current RGB observation or legal cached visual feature;
- `p in R^[N, 8]`: proprioception;
- `l`: task/language embedding or deterministic task string feature;
- `B in R^[N, H, D]`: frozen SmolVLA Base action chunk;
- `Y in R^[N, H, D]`: demonstration action chunk for development supervision;
- `T_k`: a task-preserving transformation from family `k`.

MCI learns:

- consistency encoder `z_phi(o, p, l, B) -> R^[N, d_z]`;
- adapter residual proposal `r_theta(o, p, l, B, z) -> R^[N, H, D]`;
- gate `g_eta(o, p, l, B, z) -> [0, 1]^[N, H, D]`.

The emitted chunk is:

`A = postprocess(B + g_eta * cap_group(tanh(r_theta), delta_max))`.

At initialization:

- `r_theta = 0`;
- gate bias makes `g_eta = 0`;
- therefore `A = B` exactly.

The consistency code should remain stable across legal transformations while
still separating different tasks, phases, and action regimes enough to avoid
representation collapse.

## Objective

The proposal fixes the objective family; exact coefficients are selected only
by bounded validation search after mathematical audit and preregistration.

For a paired sample `x = (o, p, l, B, Y)` and transformed sample `T_k(x)`,
define:

- `z = z_phi(x)`;
- `z_k = z_phi(T_k(x))`;
- `A = A_theta(x)`;
- `A_k = A_theta(T_k(x))`.

Required terms:

- consistency-code loss `L_code = mean_k ||z - stopgrad(z_k)||_2^2` with a
  symmetric variant if gradient balance passes the small-batch audit;
- action-consistency loss `L_act = mean_k Huber(A - A_k)`;
- demonstration fit loss `L_fit = Huber(A - Y)` on development demonstrations;
- Base-retention loss `L_keep = Huber(A - B)` with higher weight on clean or
  low-confidence states;
- representation noncollapse or variance floor `L_var` to prevent all samples
  sharing one code;
- action-bound penalty `L_bound` before postprocessing.

No KL divergence is used between deterministic 7D actions. SmolVLA flow
vectors are not treated as normalized probability distributions.

Before any real training, MCI must estimate term magnitudes and gradient norms
on a small batch, identify whether one term dominates by scale, and freeze a
validation-only coefficient envelope.

## Identity-Preserving Integration

MCI is Base-preserving by construction:

- frozen SmolVLA backbone;
- zero-initialized residual branch;
- gate initialized to exact Base passthrough;
- exact Base candidate retained for diagnostics;
- groupwise caps for translation, rotation, and gripper deltas;
- clean-retention objective;
- disk reload and Base-hash checks before rollout.

A configuration that changes nearly all actions, violates action bounds,
changes gripper behavior unintentionally, or collapses clean validation is an
implementation or design failure, not a closed-loop scientific result.

## Bounded Validation Search

Maximum search budget: `6` configurations.

Allowed factors:

- consistency loss coefficient: at most `3` values;
- adapter capacity or latent dimension: at most `2` choices.

No broad combinatorial sweep is allowed. If a LoRA parameterization is used,
rank and target modules must be fixed unless a single capacity insufficiency
check proves the low-compute scaffold cannot express the unchanged scientific
method. Confirmatory-test identities remain untouched.

Validation score:

`S_val = 0.30 * success_or_best_legal_proxy
       + 0.20 * clean_retention
       + 0.20 * consistency_activation
       + 0.15 * action_validity
       + 0.10 * prior_and_ablation_margin
       + 0.05 * compute_overhead`.

All terms are scaled to `[0, 1]`. Ties break by clean retention, then lower
action delta, then lower adapter parameter count.

All tried configurations and negative results must be saved. Offline action L2
alone cannot select the final configuration.

## First Serious Comparison

Exactly five policies:

1. `smolvla_base`
2. `rovla_multiconsistency_proxy`
3. `mci_full`
4. `mci_no_consistency_code_ablation`
5. `augmentation_only_lora_killer`

Policy 2 must preserve the closest prior mechanism: multi-consistency
constraints over task-preserving instruction, observation/proprioception, and
action-evolution perturbations. If official RoVLA cannot run on SmolVLA under
local constraints, it must be called a transparent proxy, not an official
reproduction.

Policy 4 removes the learned consistency code while keeping the same adapter
surface, data, and training budget.

Policy 5 tests the strongest simple alternative explanation: generic
adaptation to the same augmented development data without the multi-consistency
code mechanism.

## Stage 0 Development Audit

Before expensive training or rollout, MCI must run a bounded development-only
audit. It may use discovery and validation identities only.

Required checks:

- source and split integrity;
- zero train/validation/confirmatory overlap;
- noncollapsed transformation pairs by family;
- representation variance and positive/negative contrast health;
- consistency-code predictability above trivial task/demo baselines;
- Base and RoVLA proxy leave meaningful residual headroom on the claim axis;
- initial exact Base passthrough after disk reload;
- finite nonzero gradients for expected adapter parameters;
- action delta from Base by translation, rotation, and gripper groups;
- action-bound validity;
- clean validation retention;
- full differs from Base, RoVLA proxy, ablation, and augmentation-only LoRA;
- no privileged inference input and no confirmatory identity read.

Stop before rollout for:

- collapsed transformations or labels: `DATA_OR_SUPERVISION_FAILURE`;
- no Base/prior/oracle headroom: `CONDITION_TOO_SEVERE_OR_NO_HEADROOM`;
- nonacting gradients, reload mismatch, invalid action values, or wrong
  checkpoint: `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- globally destructive action changes or trivial equivalence:
  `DESIGN_FAILURE` or `SIMPLE_BASELINE_EXPLAINS_METHOD`.

These are development classifications, not closed-loop scientific kills.

## Mechanism Smoke Before Rollout

For every trainable or inference-time variant, report:

- Base action chunk and MCI action chunk;
- residual norm and gate value;
- dimensions changed and per-group delta p95;
- consistency-code distance for original versus transformed pairs;
- clean versus shifted behavior;
- action validity and gripper-transition effects;
- checkpoint path and reload result;
- exact policy identity and no accidental checkpoint reuse.

Do not launch hundreds of rollouts merely because unit tests pass.

## Confirmatory Stages

After proposal, review, rebuttal, mathematical audit, preregistration,
prototype protocol, implementation validation, Stage 0 audit, and bounded
validation selection, freeze one configuration and one paired manifest.

Stage A:

- approximately 10 paired episodes per policy;
- detects catastrophic harm, no headroom, mechanism invalidity, obvious prior
  dominance, ablation dominance, or exact trivial equivalence;
- small differences advance to Stage B.

Stage B:

- at least 40 paired episodes per key policy;
- paired wins/losses/ties;
- bootstrap confidence interval;
- effect size and failure-rate reduction;
- per-task breakdown;
- mechanism activation and clean retention;
- latency and VRAM outside resource-contention intervals.

One expansion to 80 paired episodes per key policy is allowed only when the
frozen uncertainty rule declares Stage B unresolved.

## Paper-Candidate Gate

MCI becomes a serious paper candidate only if:

- MCI beats Base;
- MCI beats the RoVLA proxy on the matched claim axis;
- MCI beats the no-consistency-code ablation;
- augmentation-only LoRA does not explain the gain;
- clean behavior and action validity are retained;
- mechanism evidence supports the intended multi-consistency explanation;
- novelty remains defensible after final literature refresh.

After prototype GO, immediately verify Quantized OpenVLA-OFT INT4 plus MCI,
add one claim-specific second condition or benchmark, add recent direct
baselines when feasible, measure compute and latency, and prepare
figure/table-ready evidence.

## Resource Policy

The Windows gaming and Efficiency Mode interval remains recorded as a
resource-contention interval. Latency, throughput, wall-clock efficiency, CUDA
utilization, and resource utilization measured during or overlapping that
interval are not final paper evidence. Synchronous closed-loop success rows may
remain valid only after timeout, exception, action-semantics, identity,
duplicate-key, and manifest checks.

## Current Status

No MCI implementation, training, validation search, rollout, simulator access,
or confirmatory-test access has happened. The next step is independent
Reviewer B attack on this frozen proposal.
