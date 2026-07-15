# Epoch 4 Cycle 12 Candidate Generation

Date: 2026-07-15 KST

Decision: `SELECT_CALA_VLA`

Governance applied: current performance-oriented and honest-positive-result
governance. Exactly three candidates were generated and scored. G3P-VLA remains
stopped as `DATA_OR_SUPERVISION_FAILURE`; it must not be rescued by changing
point labels, thresholds, source gates, validation criteria, or interpretation.

## Candidate 1: CALA-VLA

Name: `CALA-VLA`, Context-Gated Action-Latent Adapter for frozen SmolVLA.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: CAC-VLA, https://arxiv.org/abs/2607.04816.

Secondary priors: VLS, https://arxiv.org/abs/2602.03973; World Pilot,
https://arxiv.org/abs/2606.12403.

Positive prior result: CAC-VLA reports that predicting latent actions from
visual-language context and injecting them through a context gate achieves
`98.3%` average success on LIBERO and `89.5%` on LIBERO-Plus. The paper's
ablation framing supports both latent-action horizon and context-gated
conditioning as meaningful components.

Official code/checkpoint/reproducible mechanism: no official CAC-VLA code or
checkpoint was located during the Cycle 12 scan. The mechanism is reproducible
from the primary paper: encode future action segments into latent actions,
predict those latents from current observation and language, project them into
the action expert, and inject them through context-gated residual conditioning.
Local comparison must call the closest-prior policy a faithful transparent
proxy unless exact official equivalence is later established.

Assumption or limitation extended: CAC-VLA assumes a training/inference path
where latent-action conditioning is integrated into the VLA action expert. The
local limitation is frozen SmolVLA safety: arbitrary hidden-state conditioning
can damage a strong pretrained flow policy. CALA extends the prior with an
identity-preserving gate, bounded action-delta audit, and Stage 0 latent-label
health gate before any expensive training or rollout.

Minimal technical difference proposed by Ours:

- build a deterministic OAT-lite action-latent encoder from future 7D action
  segments on discovery/validation identities only;
- train a small predictor from deployment-observable RGB/proprioception,
  language, and Base features to the latent action;
- inject the predicted latent through a zero-initialized context-gated
  hidden-state adapter, not as direct executable actions;
- default to exact Base behavior when the gate is closed, confidence is low, or
  source/label audits fail;
- compare against Base, a CAC-style latent-action proxy, CALA full,
  no-context-gate ablation, and a task-mean latent-action baseline.

Why it could improve the same claim axis: CAC-VLA's positive result suggests
that action-structured latent guidance can bridge visual-language understanding
and continuous motor control. Local SmolVLA has repeatedly resisted output
correction, queue scheduling, memory, route, and point-label methods. A
Base-preserving latent-action interface targets an earlier internal
conditioning bottleneck while keeping the final pretrained action expert as the
default behavior.

### Quality Screen

Provisional novelty:

- Distinct from the closest prior because the local contribution is a frozen
  SmolVLA-compatible identity-preserving action-latent adapter with explicit
  source, label, and disruption gates.
- Distinct from G3P because it uses future action segment structure, not point
  labels or spatial target inference.
- Distinct from MARC, DAGR, MTF, PESA, EAC, RAC, FANG, CAVM, and RCV because
  it does not correct final 7D actions, route arm/gripper components, select
  frames, query priors, schedule chunks, use consequence histories, use failure
  memories, or replay nearest actions.
- Novelty risk remains: if a task-mean latent prior or no-gate ablation
  explains the gain, the method must be killed.

Prior-anchor strength:

- Very strong same-benchmark positive prior on LIBERO and LIBERO-Plus.
- Secondary priors support action-interface steering rather than generic visual
  preprocessing.
- Official code/checkpoint is not verified, so transparent proxy status is
  mandatory.

Mechanism plausibility:

- Problem condition -> SmolVLA fails when visual-language context must become
  precise multi-step motor structure.
- Intermediate failure mechanism -> action expert receives generic
  visual-language features but not an explicit summary of imminent action
  structure.
- Policy behavior -> Base emits plausible chunks that may be phase- or
  contact-misaligned.
- Closed-loop failure -> early chunk errors compound into missed grasp,
  misplaced object, or failed multi-step completion.
- Proposed method -> predict a latent future-action summary from legal current
  inputs and inject it through a bounded context gate.
- Intended internal change -> hidden action states receive action-structured
  conditioning only when the gate opens.
- Intended action behavior -> chunks become better phase-aligned while clean
  states remain Base-like.
- Expected closed-loop improvement -> higher task-balanced success with clean
  retention and bounded latency overhead.

Data and supervision viability:

- Future 7D action segments exist in local LeRobot LIBERO demonstrations.
- RGB, proprioception, language, Base actions, train/validation/test manifests,
  and official rollout infrastructure exist locally.
- Stage 0 must prove latent variance, task/phase coverage, predictability above
  trivial baselines, zero split overlap, and no future action use at inference.
- If latent targets are collapsed or not predictable from deployment inputs,
  the correct classification is `DATA_OR_SUPERVISION_FAILURE` or
  `DESIGN_FAILURE`, not a closed-loop scientific result.

Identity-preserving integration:

- Context gate is initialized to exact Base passthrough.
- The adapter is zero-initialized and bounded by residual scale.
- Translation, rotation, and gripper deltas are audited separately before
  rollout.
- Clean validation behavior is a hard gate.

Decisive experiment feasibility:

- Stage 0 can be run without rollout or heavy training: latent label health,
  predictability, source legality, and initial action-delta checks.
- Bounded validation search can test at most six configurations over latent
  horizon, gate scale, and one architecture choice.
- First serious comparison uses exactly five policies: Base, CAC-style proxy,
  CALA full, no-context-gate ablation, and task-mean latent-action baseline.
- Second-backbone path: if SmolVLA reaches GO, port the same action-latent gate
  to Quantized OpenVLA-OFT INT4.
- Second condition: LIBERO-Plus or a frozen perturbation/long-horizon slice
  after the SmolVLA prototype GO.

Score:

- provisional novelty: `23 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `19 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `94 / 100`

## Candidate 2: DCR-VLA

Name: `DCR-VLA`, Decoupled Clean-Robustness Adaptation for frozen SmolVLA.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: STRONG-VLA, https://arxiv.org/abs/2604.10055.

Positive prior result: STRONG-VLA reports that decoupled robustness acquisition
followed by clean task-aligned refinement improves VLA success under multimodal
perturbations, including gains up to `12.60%` on OpenVLA, `14.48%` on
OpenVLA-OFT, and `16.49%` on `pi0` for seen perturbations, plus gains under
unseen perturbations.

Official code/checkpoint/reproducible mechanism: no official code/checkpoint
was verified during this Cycle 12 scan. The paper provides a reproducible
mechanism: generate text and vision perturbation curricula, train on increasing
perturbation difficulty, then clean-refine to restore fidelity.

Assumption or limitation extended: STRONG-VLA studies general VLA robustness.
The local extension would be an identity-preserving SmolVLA adapter with
strict clean-retention and perturbed-condition headroom gates.

Minimal technical difference proposed by Ours:

- create discovery/validation-only visual and language perturbation partitions;
- train a small adapter in two stages: perturbation curriculum, then clean
  refinement;
- initialize adapter near Base and preserve clean behavior with retention;
- compare against Base, a STRONG-style proxy, no-clean-refinement ablation, and
  one simple perturbation-normalization baseline.

Why it could improve the same claim axis: the prior demonstrates that training
schedule matters for robustness/clean tradeoffs. A local version could improve
shifted LIBERO behavior without relying on point labels or output correction.

### Quality Screen

Provisional novelty:

- Meaningful as a frozen-SmolVLA, clean-retention, local perturbation-governed
  extension of STRONG-VLA.
- Risk: it could collapse into a PSE/GCAP-style visual perturbation rescue if
  multimodal robustness, clean refinement, and matched STRONG proxy are not
  preserved.

Prior-anchor strength:

- Strong external result across multiple VLA backbones and perturbation types.
- Official source fidelity is weaker than CAC-VLA because no local code path
  was verified.

Mechanism plausibility:

- Problem condition -> visual/language perturbations shift the input
  distribution.
- Intermediate failure mechanism -> joint robustness training can damage clean
  sensitivity or fail unseen perturbations.
- Proposed method -> decouple robustness acquisition from clean refinement.
- Expected action behavior -> perturbed observations produce less brittle
  chunks while clean behavior remains close to Base.

Data and supervision viability:

- RGB, language, and action labels exist; perturbations are locally generable.
- Need Stage 0 to prove Base actually fails under the selected perturbations
  and that clean retention does not collapse.
- Prior local visual-robustness failures lower the feasibility score.

Identity-preserving integration:

- Near-zero adapter or gated adapter can preserve Base initially.
- Clean refinement and retention are mandatory.

Decisive experiment feasibility:

- Stage 0 perturbation headroom and clean-retention audit is feasible.
- Closed-loop perturbation rollout is more expensive because perturbation
  operators and seen/unseen splits must be frozen before test.

Score:

- provisional novelty: `21 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `88 / 100`

## Candidate 3: RGS-VLA

Name: `RGS-VLA`, Reward-Guided Sampling Steering for frozen SmolVLA flow
actions.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: VLS, https://arxiv.org/abs/2602.03973.

Positive prior result: VLS reports training-free inference-time steering of
frozen diffusion or flow-matching policies with VLM-synthesized
trajectory-differentiable rewards, achieving a `31%` improvement on CALVIN and
a `13%` gain on LIBERO-PRO, plus real-world Franka results.

Official code/checkpoint/reproducible mechanism: no local official code or
checkpoint was verified during this Cycle 12 scan. The primary paper gives the
mechanism: synthesize a differentiable trajectory reward from the
vision-language goal and steer the policy sampling process without modifying
weights.

Assumption or limitation extended: VLS assumes a reliable VLM reward source and
differentiable access to policy sampling. Local SmolVLA flow inference may not
expose a faithful steering hook, and the campaign cannot use simulator object
state or confirmatory rewards as hidden inference input.

Minimal technical difference proposed by Ours:

- source-gate a deployment-observable reward proxy from image/language/Base
  features;
- use a bounded steering budget over SmolVLA flow samples only if the sampling
  hook is faithful;
- default to Base when reward confidence or compute budget fails;
- compare against Base, VLS proxy, RGS full, no-reward-gradient ablation, and
  one simple deterministic sample-rerank baseline.

Why it could improve the same claim axis: if SmolVLA already contains useful
motor skills but sampling is misaligned under shifted conditions, a bounded
reward-guided sampler could select better chunks without retraining.

### Quality Screen

Provisional novelty:

- Distinct from prior local methods because it preserves weights and steers the
  generative sampling process rather than training adapters or correcting
  output actions.
- Risk: a non-differentiable reranker or heuristic reward would no longer be a
  faithful VLS extension.

Prior-anchor strength:

- Strong positive prior on flow/diffusion policies and LIBERO-PRO.
- Official local reproducibility is currently unverified.

Mechanism plausibility:

- Problem condition -> pretrained flow policy samples plausible but
  task-misaligned chunks under spatial or semantic shift.
- Intermediate failure mechanism -> imitation prior lacks test-time
  requirement steering.
- Proposed method -> steer sampling with a deployment-observable reward.
- Expected action behavior -> choose chunks with better spatial/task
  compliance while preserving learned motion priors.

Data and supervision viability:

- Local data can support diagnostics, but faithful VLM reward synthesis and
  differentiable sampling access are uncertain.
- Privileged reward leakage risk is high.

Identity-preserving integration:

- Policy weights remain frozen and Base is default when steering is inactive.
- Sampling-time perturbations must be bounded and audited for action validity.

Decisive experiment feasibility:

- A source-fidelity and sampling-hook Stage 0 can reject before rollout.
- Full validation is harder than CALA because steering may require heavy
  inference and reward synthesis.

Score:

- provisional novelty: `22 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `5 / 10`
- decisive experiment feasibility: `6 / 10`
- total: `84 / 100`

## Selection

Selected method: `CALA-VLA`.

Selection reason:

- It has the strongest same-benchmark positive prior among locally feasible
  options.
- It avoids the G3P data failure because its primary labels are future 7D
  action segments, which are present in the local demonstrations.
- It changes more than two core dimensions relative to prior killed methods:
  representation, supervision, policy conditioning, and claim axis all change.
- It preserves Base by construction through a zero-initialized context gate and
  bounded hidden-state residual.
- It has a decisive pre-rollout Stage 0: latent-label health, predictability,
  source legality, initial action deltas, and simple task-mean baseline checks.
- Unknown empirical performance is not a rejection reason; the Stage 0 audit can
  stop as `DATA_OR_SUPERVISION_FAILURE`, `DESIGN_FAILURE`, or
  `NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE` before rollout if the mechanism
  is not viable.

Immediate next steps:

1. Freeze a `CALA-VLA` Researcher A proposal and hash it.
2. Reviewer B attacks novelty and source fidelity against CAC-VLA, action-level
   reasoning, latent-action pretraining, VLS/World Pilot, task-mean action
   prototypes, and prior local kills.
3. Researcher A provides one rebuttal if the method remains nontrivial and
   locally feasible.
4. Write `reports/cala_vla/mathematical_mechanism_audit.md`,
   preregistration, and prototype protocol before any expensive training or
   rollout.
5. Implement only Stage 0 first: latent-label health, source legality,
   discovery/validation/test split proof, latent predictability above trivial
   baselines, Base passthrough, gradient-path smoke, action-delta bounds, and
   no confirmatory-test identity use.
