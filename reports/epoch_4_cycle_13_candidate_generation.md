# Epoch 4 Cycle 13 Candidate Generation

Date: 2026-07-15 KST

Decision: `SELECT_RAR_VLA`

Governance applied: current performance-oriented and honest-positive-result
governance. Exactly three candidates were generated and scored. CALA-VLA
remains stopped as `DESIGN_FAILURE`; it must not be rescued by changing latent
labels, prediction features, thresholds, source gates, validation configs, or
baseline interpretation.

## Candidate 1: RAR-VLA

Name: `RAR-VLA`, Re-Anchored Autoregressive Residuals for frozen SmolVLA.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: AR-VLA, https://arxiv.org/abs/2603.10126.

Secondary priors: ReactVLA, https://arxiv.org/abs/2606.14255; DSWAM,
https://arxiv.org/abs/2607.04927.

Positive prior result: AR-VLA reports that a standalone autoregressive Action
Expert with long-lived memory and re-anchoring can replace reactive chunk-based
action heads, produce smoother trajectories, and maintain or exceed
state-of-the-art reactive VLA task success on simulated and real manipulation
tasks.

Official code/checkpoint/reproducible mechanism: AR-VLA lists code and videos
through its project website, but no official local code/checkpoint equivalence
has been established in this repository. The local comparison must call the
closest-prior policy `ar_vla_reanchored_expert_proxy` until exact equivalence is
verified. The reproducible mechanism is causal continuous action memory plus
re-anchoring to refreshed vision-language context.

Assumption or limitation extended: AR-VLA replaces the action expert. RAR-VLA
extends the same claim axis to a frozen SmolVLA setting where replacing the
action head is disruptive and expensive. The local method wraps the frozen
action chunk with a bounded causal residual memory initialized to exact Base
passthrough.

Minimal technical difference proposed by Ours:

- keep frozen SmolVLA as the default action generator;
- maintain a causal memory state from previous emitted Base/Ours actions and
  current proprioception;
- re-anchor that memory whenever a new Base action chunk arrives;
- predict a bounded residual or hidden adapter update from current Base chunk,
  causal memory, proprioception, and task identity;
- initialize the gate and residual to zero so initial behavior equals Base;
- compare against Base, AR-style proxy, RAR full, no-reanchor-memory ablation,
  and `ema_action_history_baseline`.

Why it could improve the same claim axis: CALA Stage 0 showed that action
history is the strongest trivial signal for local future-action structure. AR-VLA
provides a positive external prior that causal action memory and re-anchoring
can improve action generation. RAR tests whether a Base-preserving version can
convert that signal into closed-loop success while surviving the
action-history-only killer baseline.

### Quality Screen

Provisional novelty:

- Distinct from AR-VLA because it is a frozen-backbone residual memory adapter,
  not a full replacement action expert.
- Distinct from EAC/RCV/MTF because it does not schedule chunk length, replan
  frequency, or retained frames; it changes the causal action-generation state.
- Distinct from CALA because it uses causal past actions and current Base
  re-anchoring, not future-action latent labels or latent prediction.
- Novelty risk remains: if EMA/history-only explains the gain, the method must
  be killed.

Prior-anchor strength:

- Strong positive action-generation prior from AR-VLA, accepted at RSS 2026.
- ReactVLA and DSWAM support the broader importance of real-time action
  generation and asynchronous/re-anchored control.
- Official local equivalence is unverified, so transparent proxy status is
  mandatory.

Mechanism plausibility:

- Problem condition -> frozen SmolVLA emits reactive chunks that can be
  temporally inconsistent across observation refreshes or flow samples.
- Intermediate failure mechanism -> no persistent causal action state maintains
  kinematic intent between slow perception updates and fast action execution.
- Policy behavior -> chunk boundary discontinuities or stale action syntax can
  cause missed grasps, late gripper changes, or overshoot.
- Closed-loop failure -> small boundary errors compound into failed
  manipulation.
- Proposed method -> causal residual memory re-anchored to each new Base chunk.
- Intended internal change -> memory predicts only bounded corrections and
  starts as exact passthrough.
- Intended action behavior -> smoother, more temporally coherent actions while
  preserving Base when the learned gate is closed.
- Expected closed-loop improvement -> higher task-balanced success than Base,
  AR proxy, ablation, and simple EMA baseline.

Data and supervision viability:

- Development records contain ordered frames, Base actions, target actions,
  task indices, and splits.
- Causal history features can be constructed without future action inference.
- Stage 0 must prove residual predictability and action-discontinuity headroom
  above EMA/linear history baselines before validation search.

Identity-preserving integration:

- Residual branch and gate are zero-initialized.
- Base passthrough is exact at initialization.
- Translation, rotation, and gripper deltas are bounded separately.
- Clean validation behavior is a hard gate.

Decisive experiment feasibility:

- Stage 0 can audit discontinuity headroom, causal feature legality, residual
  predictability, and zero-delta identity without rollout.
- Bounded validation search can test at most six configs over memory horizon,
  residual scale, and linear versus small-MLP adapter.
- First serious comparison uses exactly five policies: Base, AR proxy, RAR
  full, no-reanchor-memory ablation, and EMA action-history baseline.

Score:

- provisional novelty: `22 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `91 / 100`

## Candidate 2: AMF-VLA

Name: `AMF-VLA`, Action-Manifold Filter for frozen SmolVLA chunks.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: ABot-M0 Action Manifold Learning,
https://arxiv.org/abs/2602.11236.

Positive prior result: ABot-M0 reports that robot actions lie on a
low-dimensional smooth manifold and that Action Manifold Learning predicts clean
continuous action sequences, improving decoding speed and policy stability.

Official code/checkpoint/reproducible mechanism: ABot-M0 lists a project
website and code release intent. No official local code/checkpoint equivalence
has been established. The reproducible mechanism is action-manifold projection
or direct clean-sequence prediction.

Assumption or limitation extended: ABot-M0 learns an action generator around a
large curated multi-embodiment dataset. AMF-VLA would add only a small
Base-preserving action-manifold filter around frozen SmolVLA to test whether
local failures are due to off-manifold action chunks.

Minimal technical difference proposed by Ours:

- learn a task-conditioned low-rank manifold from development action chunks;
- audit whether failed or high-error Base chunks are measurably off-manifold;
- apply a bounded gated projection only when the chunk is out-of-manifold;
- compare against Base, an AML-style proxy, AMF full, no-manifold ablation, and
  PCA/EMA projection baseline.

Why it could improve the same claim axis: if SmolVLA emits physically plausible
but locally off-manifold chunks under hard conditions, a bounded manifold filter
could improve stability without replacing the policy.

### Quality Screen

Provisional novelty:

- Meaningful only if it demonstrates a closed-loop useful manifold mechanism,
  not generic smoothing or PCA.
- Risk is high because prior local ActionMap-style approximation and simple
  baselines have already killed related output-space routes.

Prior-anchor strength:

- Strong positive prior in ABot-M0, but not a matched frozen-SmolVLA adapter.

Mechanism plausibility:

- Problem condition -> Base action chunks occasionally leave the demonstrated
  smooth task manifold.
- Proposed method -> project only the off-manifold component with a bounded
  identity-preserving gate.
- Expected behavior -> fewer discontinuities and invalid motions while clean
  Base actions pass through.

Data and supervision viability:

- Local ordered action chunks exist.
- Stage 0 must prove off-manifold headroom and that the projection is not
  explained by PCA/EMA.

Identity-preserving integration:

- Projection gate initialized closed.
- Projection magnitude bounded per action component.

Decisive experiment feasibility:

- Stage 0 is cheap, but closed-loop risk is high because output-space filters
  have repeatedly failed locally.

Score:

- provisional novelty: `20 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `16 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `84 / 100`

## Candidate 3: RFCD-VLA

Name: `RFCD-VLA`, Reactive Flow-Consistency Distillation for frozen SmolVLA.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: ReactVLA, https://arxiv.org/abs/2606.14255.

Positive prior result: ReactVLA reports improved Mean Flow action generation and
Attention Residuals, outperforming similarly sized VLA baselines including
SmolVLA and increasing inference speed on simulation and real-world tasks.

Official code/checkpoint/reproducible mechanism: no local official code or
checkpoint equivalence has been established. The reproducible mechanism is a
mean-flow action generator with residual feature routing.

Assumption or limitation extended: ReactVLA changes the VLA architecture.
RFCD-VLA would test a smaller frozen-SmOLVLA-compatible distillation: learn a
bounded correction that makes one/few-step action predictions consistent with a
development-time multi-sample or multi-step flow proxy.

Minimal technical difference proposed by Ours:

- probe SmolVLA flow/action-sample dispersion on development records;
- train a small residual consistency head only if there is headroom beyond a
  deterministic fixed-noise and smoothing baseline;
- initialize to Base passthrough and cap residuals;
- compare against Base, ReactVLA-style proxy, RFCD full, no-flow-consistency
  ablation, and deterministic fixed-noise/simple smoothing baseline.

Why it could improve the same claim axis: if frozen SmolVLA loses success
because reactive flow samples are noisy or slow to settle, a local
flow-consistency residual could preserve the learned action prior while
improving reactivity.

### Quality Screen

Provisional novelty:

- Distinct from ECHO candidate ranking because it does not merely choose among
  existing candidates; it distills a flow-consistency update.
- Risk remains because prior ECHO candidate-oracle headroom was zero for
  selecting better stochastic actions.

Prior-anchor strength:

- Strong ReactVLA prior, including SmolVLA comparison.
- Local source fidelity is uncertain because flow internals may not expose the
  required hooks cleanly.

Mechanism plausibility:

- Problem condition -> reactive flow action generation is noisy or too slow for
  fine control.
- Proposed method -> a lightweight consistency head aligns one/few-step outputs
  with a better multi-step proxy under bounded residuals.
- Expected behavior -> lower latency/dispersion and improved task success.

Data and supervision viability:

- Base predictions exist, but faithful flow-step/multi-sample supervision may
  require additional inference and careful caching.
- Stage 0 must stop if the sampling hook is unavailable or candidate headroom
  is still absent.

Identity-preserving integration:

- Residual initialized to zero and disabled when confidence is low.

Decisive experiment feasibility:

- Stage 0 can test hook availability and dispersion headroom.
- Validation/training is more fragile than RAR because it depends on SmolVLA
  internals and inference budget.

Score:

- provisional novelty: `21 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `6 / 10`
- decisive experiment feasibility: `7 / 10`
- total: `83 / 100`

## Selection

Selected method: `RAR-VLA`.

Selection reason:

- It has the strongest locally feasible positive external prior after CALA's
  design failure.
- It uses the strongest signal revealed by CALA's audit, causal action history,
  while requiring a direct simple history baseline to prevent self-deception.
- It changes more than two core dimensions relative to killed methods:
  representation, causal memory state, objective, and action-generation
  mechanism change.
- It is identity-preserving by construction and can be stopped before rollout
  if the causal residual is not predictable beyond EMA/linear history.
- Unknown empirical performance is not a rejection reason; Stage 0 can classify
  failure as `DATA_OR_SUPERVISION_FAILURE`, `NO_USABLE_HEADROOM_OR_CONDITION_TOO_SEVERE`,
  `DESIGN_FAILURE`, or `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` before rollout.

Immediate next steps:

1. Freeze a `RAR-VLA` Researcher A proposal and hash it.
2. Reviewer B attacks novelty and source fidelity against AR-VLA, ReactVLA,
   ABot-M0/AML, DSWAM, EMA/action-history smoothing, and prior local kills.
3. Researcher A provides one rebuttal if the method remains nontrivial and
   locally feasible.
4. Write `reports/rar_vla/mathematical_mechanism_audit.md`, preregistration,
   and prototype protocol before any expensive training or rollout.
5. Implement only Stage 0 first: causal source legality, action-discontinuity
   headroom, residual predictability above EMA/linear history baselines, Base
   passthrough, gradient-path smoke, action-delta bounds, clean retention, and
   no confirmatory-test identity use.
