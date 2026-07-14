# Epoch 4 Cycle 7 Candidate Generation

Date: 2026-07-14 KST

Decision: `SELECT_DAGR_VLA`

Governance applied: post-RAC honest positive-result governance. Exactly three candidates were generated and scored. MTF-VLA remains archived as `MTF_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD` and must not be rescued.

## Candidate 1: DAGR-VLA

Name: `DAGR-VLA`, Dynamic Arm-Gripper Routing for frozen SmolVLA adaptation.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: DAM-VLA, https://arxiv.org/abs/2603.00926.

Positive prior result: DAM-VLA reports that dynamic action routing, specialized arm/gripper action models, and dual-scale action weighting improve complex manipulation success in simulation and real-world settings.

Official code/checkpoint/reproducible mechanism: no compatible local official DAM-VLA checkpoint is available. A faithful transparent local proxy is feasible: split SmolVLA's 7D action into translation, rotation, and gripper groups; train a static component-weighted adapter; and compare it against a dynamic route-gated full method under matched data and inference budget.

Assumption or limitation extended: DAM-VLA changes the action decoder architecture. DAGR-VLA tests the minimal frozen-policy version of the same claim axis: whether group-specific action routing can improve a strong pretrained SmolVLA without replacing the backbone.

Minimal technical difference proposed by Ours:

- infer deployment-observable route logits for translation, rotation, and gripper groups;
- train group-specific residual adapters on expert-minus-base residual targets;
- initialize residuals and route gates to base passthrough;
- clip group residuals by preregistered translation, rotation, and gripper limits;
- compare against a static DAM-style component-weighted proxy, a shared-residual ablation, and one gripper-transition heuristic simple killer.

Why it could improve the same claim axis: DAM-VLA's positive result suggests that manipulation performance improves when action components are modeled differently. Local SmolVLA failures often happen at grasp/release or approach/rotation transitions; DAGR can alter only the relevant action group while preserving base behavior elsewhere.

### Quality Screen

Provisional novelty:

- Distinct from DAM-VLA because it is a frozen-policy, route-gated residual adapter rather than a full dynamic action decoder.
- Distinct from MTF because it changes action generation and supervision, not frame selection or retained-frame sampling.
- Distinct from generic residual correction because route labels, group-specific losses, and component-clipped residuals are first-class and ablated.
- Distinct from RAC because it does not infer deployment calibration from action consequences.

Prior-anchor strength:

- Strong positive prior from DAM-VLA on dynamic action routing and specialized arm/gripper models.
- A faithful local proxy can preserve the core action-factorization mechanism even without official code.
- The comparison can be matched under the same backbone, data, tasks, and inference budget.

Mechanism plausibility:

- Problem condition -> a single shared action adapter treats translation, rotation, and gripper dimensions as if they share the same timing and supervision difficulty.
- Intermediate failure mechanism -> gripper timing and wrist/arm approach errors are diluted by easier action dimensions or corrected globally when only one group needs a change.
- Policy behavior -> ordinary adapters make small global action changes that do not fix contact, grasp, or release timing.
- Closed-loop failure -> missed grasps, premature closure/opening, or approach misalignment.
- Proposed method -> predict which action group needs intervention and apply bounded group-specific residuals.
- Intended internal change -> route head activates selectively and residual heads specialize by action group.
- Intended action behavior -> base-like action outside routed groups, stronger but bounded corrections for active arm/gripper groups.
- Expected closed-loop improvement -> better manipulation transition success with clean retention.

Data and supervision viability:

- Expert 7D actions, base action predictions, robot state, task keys, and chunk phases exist in local SmolVLA artifacts.
- Arm, rotation, and gripper residual targets can be generated from expert-minus-base action differences on discovery/validation data.
- Route labels can be derived from group-wise residual magnitude, gripper sign changes, and group-specific action variance.
- Privileged simulator success is not required at inference.
- Stage 0 must verify noncollapsed route labels, group coverage, task coverage, zero split overlap, and above-trivial route predictability from deployment inputs.

Identity-preserving integration:

- Residual heads are initialized to zero.
- Route gates default to base passthrough.
- Residuals are clipped separately for translation, rotation, and gripper groups.
- Clean validation action delta, action validity, and always-on activation are hard gates before rollout.

Decisive experiment feasibility:

- Stage 0 audit: route-label health, residual target scale, group coverage, route predictability, initial identity, and action-delta bounds.
- Bounded validation search: at most six configs over route threshold and residual alpha, selected by clean retention, route activation, full-versus-ablation separation, action validity, and a validation closed-loop proxy when feasible.
- First serious comparison uses exactly five policies: Base, DAM-style static component proxy, DAGR full, no-dynamic-route shared residual ablation, and gripper-transition heuristic simple killer.
- Second backbone path: if SmolVLA reaches GO, port the small route/residual module to Quantized OpenVLA-OFT INT4 with the same action-group semantics.
- Second condition: transition-heavy grasp/release slice or controlled gripper-timing condition, frozen before use.

Score:

- provisional novelty: `22 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `89 / 100`

## Candidate 2: CAFP-VLA

Name: `CAFP-VLA`, Contact-Affordance Field Proxy for frozen VLA action calibration.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Closest external prior: AffordanceVLA, https://arxiv.org/abs/2606.06155.

Secondary positive prior: GEAR-VLA, https://arxiv.org/abs/2606.08530.

Positive prior result: AffordanceVLA reports strong simulated and real-world results from Which2Act, Where2Act, and How2Act affordance forecasting. GEAR-VLA reports strong LIBERO, LIBERO-Plus, RoboTwin, and real-robot performance from geometry-aware action representations and embodiment canonicalization.

Official code/checkpoint/reproducible mechanism: no compatible local affordance or 3D geometry teacher is available. A local proxy would use weak contact/affordance labels from gripper state, end-effector motion, and object-interaction timing in existing traces.

Assumption or limitation extended: the positive priors assume rich affordance or geometry supervision. CAFP-VLA would test whether a lightweight weak affordance proxy can guide small action calibration without dense teachers.

Minimal technical difference proposed by Ours:

- derive weak contact/approach labels from gripper closure, low end-effector speed near action changes, and task phase;
- train a small contact-affordance predictor from deployment-observable inputs;
- use the predictor as a bounded action calibration gate;
- compare against state/action phase baselines and a no-affordance ablation.

Why it could improve the same claim axis: affordance and geometry priors suggest that manipulation succeeds when actions are tied to functional contact regions and phases. A local proxy may recover enough of this signal for simple LIBERO tasks.

### Quality Screen

Provisional novelty:

- Meaningful if the contact-affordance proxy predicts a manipulation-relevant latent rather than generic phase.
- Risk of collapsing into a hand-engineered state/phase adapter.

Prior-anchor strength:

- External priors are strong, but local reproduction is weak because true affordance/3D labels are missing.
- The local proxy must be labeled as a weak proxy, not an official AffordanceVLA or GEAR-VLA reproduction.

Mechanism plausibility:

- Problem condition -> policy lacks explicit contact or affordance state.
- Intermediate failure mechanism -> action corrections are not tied to functional interaction moments.
- Proposed method -> infer contact-affordance phase and gate bounded action calibration.
- Expected action behavior -> better contact/approach precision.

Data and supervision viability:

- Weak labels can be generated, but label correctness is uncertain.
- Stage 0 may reveal collapsed or phase-only targets.
- No privileged inference inputs are required if the predictor uses only RGB, state, and instruction.

Identity-preserving integration:

- Zero-initialized residual and base-passthrough gate are feasible.
- Clean retention and activation sparsity must be hard gates.

Decisive experiment feasibility:

- Feasible if Stage 0 label health passes.
- Less decisive than DAGR because reviewers may reject the weak affordance proxy as too far from the closest priors.

Score:

- provisional novelty: `21 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `17 / 20`
- technical mechanism quality: `16 / 20`
- data/supervision feasibility: `5 / 10`
- decisive experiment feasibility: `6 / 10`
- total: `79 / 100`

## Candidate 3: CPDF-VLA

Name: `CPDF-VLA`, Continuous delayed-feedback perturbation for SmolVLA flow actions.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: PDF, Test-Time Perturbation Learning with Delayed Feedback for Vision-Language-Action Models, https://arxiv.org/abs/2604.18107.

Positive prior result: PDF reports LIBERO success gains from uncertainty-based augmentation, action voting, adaptive augmentation scheduling, and delayed-feedback perturbation learning while keeping the base VLA frozen.

Official code/checkpoint/reproducible mechanism: PDF reports code availability, but the local SmolVLA action interface is continuous 7D flow actions rather than discrete action logits. A local proxy must use robust continuous-action consensus and cannot use invalid KL or entropy over deterministic 7D vectors.

Assumption or limitation extended: PDF assumes action logits and voting. CPDF-VLA would adapt the delayed-feedback idea to continuous action chunks through robust action statistics and a frozen perturbation scheduler.

Minimal technical difference proposed by Ours:

- sample a small set of observation perturbations;
- compute continuous-action consensus and uncertainty from base action chunks;
- train a delayed-feedback perturbation scheduler from development rollouts or diagnostics;
- gate to base passthrough when consensus is unstable;
- compare against the strongest single perturbation and no-feedback perturbation ensemble.

Why it could improve the same claim axis: PDF's positive result suggests that perturbation plus delayed feedback can reduce overconfident trajectory errors. A mathematically valid continuous-action version could make the idea compatible with SmolVLA.

### Quality Screen

Provisional novelty:

- Real technical adaptation because the action space changes from logits to continuous chunks.
- Local novelty risk remains high because PSE already tested a nearby photometric perturbation route and lost to a simple bright-single baseline.

Prior-anchor strength:

- Strong positive prior and potential code availability.
- Direct reproduction is not compatible with local continuous actions without substantial adaptation.

Mechanism plausibility:

- Problem condition -> visual or trajectory uncertainty causes overconfident actions.
- Proposed method -> perturb observations, estimate continuous consensus, and use delayed feedback to schedule perturbation.
- Expected action behavior -> more robust actions under uncertainty while preserving clean base.

Data and supervision viability:

- Perturbations are easy to generate.
- Delayed-feedback labels are not currently verified and may require additional development rollouts.
- Must avoid privileged success signals at inference.

Identity-preserving integration:

- Base passthrough is feasible when uncertainty is low.
- Multiple base policy calls increase latency and must be reported.

Decisive experiment feasibility:

- Feasible, but less attractive because it is close to PSE and requires extra delayed-feedback infrastructure before a fair Stage A.

Score:

- provisional novelty: `18 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `16 / 20`
- data/supervision feasibility: `6 / 10`
- decisive experiment feasibility: `5 / 10`
- total: `77 / 100`

## Selection

Selected method: `DAGR-VLA`.

Selection reason:

- It has the best combination of strong positive prior anchor, local label availability, identity-preserving integration, and decisive first comparison.
- It changes the mechanism away from MTF: action-component routing and group-specific residual generation replace frame selection and retained-frame sampling.
- It directly tests a paper-relevant claim axis from DAM-VLA while remaining locally bounded and fair to the closest prior via a transparent static component proxy.
- It can be killed before rollout if route labels collapse, route prediction is trivial, residuals are globally destructive, or the full method does not differ from the shared-residual ablation.

Immediate next steps:

1. Freeze a DAGR-VLA Researcher A proposal and hash it.
2. Reviewer B attacks novelty against DAM-VLA, generic residual adapters, arm/gripper loss weighting, gripper-threshold heuristics, MTF, and RAC.
3. Researcher A provides one rebuttal if the method remains nontrivial and locally feasible.
4. Write `reports/dagr_vla/mathematical_mechanism_audit.md`, preregistration, and prototype protocol.
5. Implement only a Stage 0 development audit first: split proof, route-label health, residual scale, route predictability, identity initialization, and action-delta smoke before any expensive training or rollout.
