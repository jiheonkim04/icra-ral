# Epoch 4 Cycle 16 Candidate Generation

Date: 2026-07-15 KST

Decision: `SELECT_IARC_VLA`

Exactly three candidates were generated and scored under the active
performance-oriented and post-COVI LoRA-role governance. LIFT remains closed as
`LIFT_COMPUTE_INFEASIBLE`; no candidate changes LIFT or rescues a prior local
method.

## Candidate 1: IARC-VLA

Name: `IARC-VLA`, Interference-Aware Robustness Consolidation for VLA policies.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Closest external prior: STRONG-VLA,
https://arxiv.org/abs/2604.10055.

Positive external result: STRONG-VLA reports seen/unseen perturbation gains up
to `12.60 / 7.77` points on OpenVLA, `14.48 / 13.81` on OpenVLA-OFT, and
`16.49 / 5.58` on pi0. It uses a perturbation curriculum followed by clean task
refinement across all three backbones.

Secondary mechanism prior: Gradient Episodic Memory,
https://arxiv.org/abs/1706.08840, with official code at
https://github.com/facebookresearch/GradientEpisodicMemory.

Official artifact status: no STRONG-VLA author code or checkpoint was verified.
The closest-prior arm must be a transparent proxy of the published Stage I
curriculum and Stage II clean refinement, never an official reproduction.

### Scientific Method

Stage I follows the locally frozen STRONG proxy and obtains a robustness-adapted
checkpoint. During Stage II, each clean action batch is paired with a replay
batch carrying the same task semantics under a sampled development perturbation.
Let

- `g_c = grad_theta L_action(B_clean; theta)`;
- `g_r = grad_theta L_action(B_robust_replay; theta)`;
- `d = <g_c, g_r>`.

IARC applies

`g_iarc = g_c - min(0,d) * g_r / (||g_r||_2^2 + epsilon)`.

Thus an agreeing clean update is unchanged, while a conflicting update is
projected to avoid a first-order increase in replay robust loss. This directly
operationalizes the gradient conflict STRONG-VLA motivates but does not
constrain during clean refinement.

The key ablation uses the same paired batches and compute but applies the
unprojected joint gradient `(g_c + g_r) / 2`. It asks whether conflict-aware
projection, rather than replay data or extra gradient computation, is
necessary.

This method does not add an inference-time component. Removing the words LoRA
and QLoRA leaves the scientific mechanism unchanged.

### Low-Compute Parameterization

- frozen SmolVLA base with one default, locally validated LoRA rank and target
  set;
- mixed precision, batch size `1`, and accumulation as needed;
- gradient projection only over trainable adapter parameters;
- disk-persistent Stage I and selected Stage II checkpoints;
- no rank sweep and at most one capacity adjustment after a demonstrated
  subset-fit failure;
- the exact same adapter scaffold for Prior, Ours, ablation, and standard LoRA.

Standard LoRA is required as the conditional fifth policy because Ours updates
weights, uses LoRA infrastructure, and receives perturbation-augmented training;
ordinary matched adaptation is a plausible alternative explanation.

### Quality Screen

Provisional novelty:

- IARC is not a claim to invent gradient projection or two-stage robustness
  training.
- Its narrow method novelty is a VLA-specific Stage II constrained update that
  protects perturbation robustness while restoring clean task fidelity.
- It is materially different from the unselected Cycle 12 DCR sketch: DCR
  proposed a generic identity-preserving adapter and clean retention, whereas
  IARC defines a falsifiable per-update conflict condition and action.
- Novelty fails if the projection almost never activates, is equivalent to
  unprojected replay, or recent VLA work already applies the same rule on the
  same robustness/clean axis.

Prior-anchor strength:

- STRONG-VLA supplies a strong positive same-domain result across OpenVLA,
  OpenVLA-OFT, and pi0.
- GEM supplies a positive, reproducible constrained-gradient mechanism.
- The local STRONG arm is transparent rather than official, but Prior and Ours
  can share backbone, data, perturbations, adapter capacity, tasks, and budget.

Mechanism plausibility:

- Perturbed curriculum -> robust invariances enter adapter parameters.
- Clean-only Stage II -> some clean gradients oppose perturbed action gradients.
- Opposing updates -> robustness can be forgotten while clean fidelity returns.
- Closed loop -> perturbed observations again produce brittle action chunks.
- IARC -> project only conflicting clean updates against a current robust replay
  gradient.
- Internal effect -> negative cosine events become nonnegative update alignment
  without suppressing already agreeing clean updates.
- Action effect -> clean action error falls while perturbed action behavior is
  retained.
- Expected closed-loop effect -> better perturbed success than STRONG, joint
  replay, and standard LoRA without a material clean-success loss.

Data and supervision viability:

- Demonstration actions already provide both clean and perturbed action targets;
  no new labels, oracle states, future frames, or privileged inference inputs
  are needed.
- Text perturbations must preserve task semantics; semantic drift and changed
  object/goal instructions are evaluation-only, not training pairs.
- Visual perturbations are generated from discovery/training records only.
- Stage 0 must audit balance by modality, family, severity, task, and phase;
  duplicates and all split overlap must be zero.

Identity-preserving integration:

- Every trainable policy starts from the same zero-effect LoRA initialization,
  reproducing Base before training.
- IARC changes only adapter optimization and has no inference-time intervention.
- Clean retention, action bounds, translation/rotation/gripper deltas, and
  checkpoint reload are mandatory gates.

Decisive experiment feasibility:

- A small-batch Stage 0 can measure gradient cosine, conflict rate, projected
  first-order constraint residual, finite gradients, subset fit, and action
  validity before rollout.
- A bounded validation-only search needs at most three values of one projection
  strength coefficient and at most two seeds, capped at six configurations.
- The five policies have distinct questions and can use one paired manifest.

Score:

- provisional novelty: `22 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `19 / 20`
- data/supervision feasibility: `10 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `95 / 100`

## Candidate 2: SAAF-VLA

Name: `SAAF-VLA`, Sparse 2D Affordance-Aligned Flow for low-compute VLA
adaptation.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: AffordanceVLA,
https://arxiv.org/abs/2606.06155.

Positive external result: AffordanceVLA reports `95.8%` average LIBERO success,
`4.33` average CALVIN ABCD chain length, and large positive real-world gaps over
pi0. Official code is available at
https://github.com/Skywalker-yqz/AffordanceVLA/.

### Scientific Method

Predict one instruction-conditioned 2D interaction heatmap from a SmolVLA
visual feature and align the action-flow representation to the heatmap-pooled
feature. Unlike full AffordanceVLA, the candidate omits Which2Act and How2Act and
uses only a compact Where2Act bottleneck. The key ablation keeps the same LoRA
and training data but removes the heatmap objective.

### Low-Compute Parameterization

- cached development-only 2D masks or points;
- one small heatmap head and a fixed SmolVLA LoRA scaffold;
- no affordance head or privileged label at inference;
- standard LoRA required because ordinary adaptation with the same data and
  compute is a plausible explanation.

### Quality Screen

Provisional novelty:

- A sparse 2D-only flow bottleneck is simpler than AffordanceVLA's three-part
  MoT architecture.
- The route is close to Cycle 11 SAR and G3P's spatial-supervision family, so
  novelty and non-cosmetic distinction are weaker than IARC.

Prior-anchor strength:

- The prior has strong positive simulation and real-world results plus official
  code.
- A 2D-only SmolVLA proxy omits most of the published mechanism and training
  curriculum, reducing fidelity.

Mechanism plausibility:

- Direct visual-action mapping can attend to distractors.
- A task-conditioned interaction map can localize the relevant region and
  improve precise action generation.

Data and supervision viability:

- Approximate labels can be generated, but local records lack official
  AffordanceVLA annotations.
- Cycle 11 and G3P require a stringent noncollapse and source-legality audit.

Identity-preserving integration:

- Zero-effect LoRA and a zero-influence head preserve Base initially.
- The heatmap is training-only and clean action retention is mandatory.

Decisive experiment feasibility:

- Label and subset-fit audits are bounded.
- Building a fair prior proxy and enough healthy task/phase labels is materially
  harder than IARC's action-supervision-only design.

Score:

- provisional novelty: `18 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `6 / 10`
- decisive experiment feasibility: `6 / 10`
- total: `80 / 100`

## Candidate 3: SWIR-VLA

Name: `SWIR-VLA`, Sparse World-Imagination Regularization for low-compute VLA
adaptation.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: Self-Correcting VLA,
https://arxiv.org/abs/2602.21633.

Positive external result: SC-VLA reports `9%` higher success than its strongest
compared baseline, `16%` fewer steps, and a `14%` real-world gain. Official code
is available at https://github.com/Kisaragi0/SC-VLA.

### Scientific Method

Predict a compact progress and future-trajectory-direction target from a
SmolVLA hidden state during imitation training, but omit SC-VLA's online SAC
residual policy. The auxiliary heads are training-only. The key ablation uses
the same LoRA scaffold without sparse imagination targets.

### Low-Compute Parameterization

- cached simulator-derived progress and direction labels;
- two small training-only heads and fixed SmolVLA LoRA;
- standard LoRA required because generic adaptation and extra supervision are
  plausible explanations.

### Quality Screen

Provisional novelty:

- Removing online residual RL makes the method locally cheaper but also removes
  the prior's central self-correction path.
- The design overlaps EvoState, CALA, and RAR predictive targets and risks being
  a cosmetic reentry.

Prior-anchor strength:

- SC-VLA has positive simulation/real results and official code.
- Its tasks, backbone, online RL, and reward mechanism are not a matched local
  SmolVLA/LIBERO comparison.

Mechanism plausibility:

- Sparse physical predictions could improve temporal representations.
- Without online refinement, the causal route from prediction to closed-loop
  action improvement is weaker than the published prior.

Data and supervision viability:

- Simulator state can generate labels, but prior local progress and residual
  predictability margins were negative.
- A fresh source and predictability audit is mandatory before training.

Identity-preserving integration:

- Heads are training-only and LoRA starts at Base behavior.
- Clean retention and action bounds remain required.

Decisive experiment feasibility:

- Offline predictability can be tested cheaply.
- A fair SC-VLA proxy would require online RL and exceeds the minimum-sufficient
  local experiment; omitting it weakens prior fidelity.

Score:

- provisional novelty: `15 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `15 / 20`
- data/supervision feasibility: `5 / 10`
- decisive experiment feasibility: `5 / 10`
- total: `72 / 100`

## Selection

`IARC-VLA` is selected with `95 / 100`.

It has the strongest same-claim positive anchor, requires no new labels, adds no
inference component, separates the scientific method from LoRA infrastructure,
and turns a published optimization diagnosis into one falsifiable constrained
update. It also supports a fair matched comparison on SmolVLA now and on
Quantized OpenVLA-OFT INT4 after prototype GO. Unknown empirical performance is
not a rejection reason.

## Baseline Rationale

| Comparison | Scientific question |
| --- | --- |
| Base vs Ours | Does IARC improve the same SmolVLA backbone under perturbation while retaining clean behavior? |
| Prior vs Ours | Does explicit Stage II conflict protection improve over transparent STRONG-style distribution decoupling? |
| Ablation vs Ours | Is conflict-aware projection necessary beyond matched clean/perturbed replay and gradient compute? |
| Standard LoRA vs Ours | Can ordinary adaptation with matched checkpoint, demonstrations, steps, optimizer, rank, and target modules explain the gain? |

The first serious comparison therefore contains exactly five policies:

1. `smolvla_base`
2. `strong_vla_transparent_proxy`
3. `iarc_vla_full`
4. `iarc_unprojected_joint_replay_ablation`
5. `standard_lora_clean_only`

No additional internal control is authorized at selection time.

## Frozen Next Gate

Researcher A must write one bounded proposal. Before expensive training or
rollout, discovery/validation-only evidence must establish:

- meaningful Base and transparent-STRONG residual failure under selected
  semantics-preserving perturbations;
- noncollapsed perturbation families, severities, tasks, and phases;
- zero discovery/validation/confirmatory identity overlap;
- exact perturbation semantics and no target-changing train transform;
- finite nonzero clean and robust gradients on the intended adapter parameters;
- a nontrivial gradient-conflict rate with confidence or an unresolved-safe
  continuation rule;
- projection constraint residual within numerical tolerance;
- small-subset fit, disk reload, Base identity at initialization, bounded action
  delta, action validity, and clean retention;
- no privileged inference input and no confirmatory-test tuning;
- a predeclared search cap of at most six configurations and one selected frozen
  validation score before confirmatory evaluation.

