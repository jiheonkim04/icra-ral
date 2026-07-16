# Epoch 4 Cycle 38 Candidate Generation

Date: 2026-07-16 KST

Decision: `MCI_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING`

Candidate count: exactly `3`

Previous method: `CSPR-VLA`

Previous decision: `CSPR_STAGE_0_IMPLEMENTATION_FAILURE`

Governance: current performance-oriented governance with one genuinely new
scientific mechanism, LoRA only as low-compute implementation infrastructure,
and the closest external prior entering the first serious comparison. CSPR is
closed without rescue and is not repaired, retuned, relaunched, or
reinterpreted.

Design constraint: the selected method must be reproducible from existing
LIBERO demonstrations without privileged inference inputs. Development may use
discovery and validation identities only; confirmatory-test identities remain
sealed.

## Candidate 1: MCI-VLA

Full name: Multi-Consistency Invariance for Base-preserving SmolVLA

Closest prior: RoVLA

Primary sources:

- https://arxiv.org/abs/2605.19678
- https://arxiv.org/html/2605.19678v1
- https://github.com/HCPLab-SYSU/RoVLA

Positive prior: RoVLA reports that multi-consistency constraints improve VLA
robustness under paraphrased instructions, visual/proprioceptive perturbations,
and action-evolution shifts on LIBERO-Plus, RoboTwin 2.0, and real-world
manipulation tasks. Its public repository contains GR00T-based training and
evaluation code, LIBERO/LIBERO-Plus examples, PGD adversarial training code,
and consistency-learning modules.

Contribution type: `PRIOR_EXTENSION`

Scientific method: train a compact consistency-code adapter around frozen
SmolVLA features and action chunks. The adapter is forced to preserve task-
relevant action intent under three legal same-task transformations:
instruction paraphrases, bounded image/proprio perturbations, and action-
evolution or flow-time perturbations. It is initialized as exact Base
passthrough through a zero gate and may be implemented with LoRA or another
small adapter only as low-compute infrastructure. The scientific mechanism is
multi-consistency invariance, not LoRA, residual critical-step repair,
chunk-boundary smoothing, latent drift monitoring, or generic action history.

Minimal difference from prior: RoVLA applies multi-consistency learning inside
a large GR00T/InternVL-style VLA. MCI keeps the same claim axis and prior
mechanism but tests whether a smaller frozen-SmolVLA identity-preserving
adapter can reproduce the useful invariance mechanism under matched LIBERO
demonstration-derived transformations.

Mechanism chain:

- problem condition: SmolVLA can be brittle when equivalent language, small
  observation/proprio changes, or action-generation perturbations change the
  local action intent;
- intermediate failure mechanism: shallow correlations make internal features
  and action chunks unstable under task-preserving transformations;
- policy representation/action behavior: MCI learns a consistency code whose
  representation and bounded action delta remain stable across those
  transformations, while the Base action remains the default;
- expected closed-loop improvement: fewer failures caused by paraphrase,
  visual/proprio noise, and action-generation instability, with clean Base
  behavior retained when no inconsistency is detected.

Data and supervision viability: existing LIBERO demonstrations provide RGB
streams, proprioception, language/task strings, Base action chunks, and 7D
demonstration chunks. Legal transformation pairs can be generated from
discovery and validation identities only. No reward, success flag, done flag,
object pose, simulator state, future observation, or confirmatory-test identity
is used at inference. Stage 0 must verify noncollapsed transformation pairs,
representation predictability above trivial baselines, split separation, and
action-validity retention before any training expansion or rollout.

Identity-preserving integration: the consistency adapter starts at exact Base
passthrough, the gate is initialized to zero, action deltas are capped by
translation, rotation, and gripper groups, and clean-retention loss is required.
The module may not globally replace strong pretrained actions.

First serious comparison:

1. `smolvla_base`
2. `rovla_multiconsistency_proxy`
3. `mci_full`
4. `mci_no_consistency_code_ablation`
5. `augmentation_only_lora_killer`

Scores:

- provisional novelty: `22 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `10 / 10`
- total: `92 / 100`

Rationale: MCI is selected because it is strongly prior-anchored, technically
distinct from the closed CSPR/NICE/EAC/S2C/MHS/DCCG routes, uses only
deployment-observable inputs, preserves Base identity by default, and admits a
bounded local audit before expensive training or rollout. Unknown empirical
performance is not a rejection reason.

## Candidate 2: ICR-VLA

Full name: Intent-Commitment Routing for Base-preserving SmolVLA

Closest prior: IntentVLA

Primary sources:

- https://arxiv.org/abs/2605.14712
- https://github.com/ZGC-EmbodyAI/IntentVLA

Positive prior: IntentVLA reports that short-horizon intent representations
improve rollout stability under aliased observations, with strong reported
AliasBench performance and standard-benchmark results including LIBERO-Long.
The repository currently releases AliasBench code and states that full model
training/evaluation code is coming soon.

Contribution type: `PRIOR_EXTENSION`

Scientific method: learn an intent-commitment code from recent legal RGB,
proprioception, language, and Base action context, then use a zero-initialized
route gate to maintain local action intent only when a current observation is
ambiguous under development-only aliasing diagnostics.

Minimal difference from prior: IntentVLA conditions a VLA on compact recent
visual intent. ICR would use the same aliasing claim axis but convert it into a
Base-preserving selective route gate for frozen SmolVLA rather than full
history-conditioned chunk generation.

Mechanism chain:

- problem condition: visually similar current states can require different
  action continuations because of recent phase or intent;
- intermediate failure mechanism: frame-conditioned chunks resample local
  intent across replanning boundaries;
- policy representation/action behavior: ICR predicts an intent-commitment
  code and only activates a bounded route when aliasing is detected;
- expected closed-loop improvement: fewer inconsistent chunk transitions in
  aliased states while exact Base behavior is retained elsewhere.

Data and supervision viability: LIBERO demonstrations provide legal histories,
task strings, proprioception, images, and action chunks. The risk is that
local cache-covered LIBERO tasks may not contain noncollapsed aliasing
contrast; if the aliasing labels collapse, Stage 0 must stop as
`DATA_OR_SUPERVISION_FAILURE`. This route is also close to previous generic
history-residual failures, so the aliasing contrast must be the mechanism, not
ordinary history memory.

Identity-preserving integration: zero route at initialization, Base
passthrough outside detected aliasing, bounded action deltas, and clean-
retention audit before rollout.

First serious comparison:

1. `smolvla_base`
2. `intentvla_aliasing_proxy`
3. `icr_full`
4. `icr_no_intent_commitment_ablation`
5. `history_only_lora_killer`

Scores:

- provisional novelty: `21 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `17 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `7 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `85 / 100`

Rationale: ICR is plausible and prior-anchored, but code availability and
local aliasing-label health are weaker than MCI. It is not selected.

## Candidate 3: OAB-VLA

Full name: Object-Addressed Binding for Base-preserving SmolVLA

Closest prior: OA-WAM

Primary sources:

- https://arxiv.org/abs/2605.06481
- https://arxiv.org/html/2605.06481v1

Positive prior: OA-WAM reports that object-addressable slot states and
address-only slot routing improve robustness on LIBERO, SimplerEnv, and
LIBERO-Plus geometric shifts, with causal slot-intervention evidence for
object binding.

Contribution type: `CROSS_DOMAIN_MECHANISM_TRANSFER`

Scientific method: build a deployment-observable object-address proxy from
image-derived features and language nouns, then gate Base action chunks by a
stable target-binding code. The method would test whether object addressability
can be transferred from a world-action model into a frozen SmolVLA action
adapter without simulator object state at inference.

Minimal difference from prior: OA-WAM uses explicit object slots and
address-only attention inside a large world-action model. OAB would use a
smaller nonprivileged image/language object-binding proxy and a Base-preserving
action gate around SmolVLA.

Mechanism chain:

- problem condition: scene or layout shifts can make the policy bind the
  instruction to the wrong object or wrong spatial relation;
- intermediate failure mechanism: holistic features entangle target identity
  with background and neighboring context;
- policy representation/action behavior: OAB learns a stable target-binding
  code and permits bounded action edits only when the binding is confident and
  action-relevant;
- expected closed-loop improvement: fewer wrong-object or wrong-relation
  actions under object/layout perturbations.

Data and supervision viability: the main weakness is legal object-slot
construction. Simulator poses would be privileged at inference and are not
allowed. A local route would need image-derived slots or a transparent noun/
feature proxy with noncollapsed coverage. If the slot proxy is unavailable or
not predictive from deployment inputs, Stage 0 must stop before rollout.

Identity-preserving integration: exact Base candidate always remains, gate
starts at passthrough, and action edits are bounded by groupwise caps.

First serious comparison:

1. `smolvla_base`
2. `oa_wam_addressable_proxy`
3. `oab_full`
4. `oab_no_address_binding_ablation`
5. `noun_only_binding_killer`

Scores:

- provisional novelty: `23 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `4 / 10`
- decisive experiment feasibility: `7 / 10`
- total: `84 / 100`

Rationale: OAB is novel and strongly anchored, but the legal slot-extraction
dependency is too risky for the immediate next cycle. It is not selected.

## Selection

Selected method: `MCI-VLA`

Selected score: `92 / 100`

Selection decision: `MCI_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING`

MCI is selected because it provides the strongest honest next experiment under
the sharpened Cycle 38 constraint: one genuinely new mechanism, RoVLA as a
positive external prior in the first serious comparison, no privileged
inference input, LoRA only as low-compute infrastructure, identity-preserving
integration, and a bounded development-only audit before training or rollout.
No MCI proposal, implementation, training, validation search, rollout, or
confirmatory-test access has happened.
