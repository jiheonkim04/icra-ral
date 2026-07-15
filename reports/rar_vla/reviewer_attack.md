# RAR-VLA Reviewer B Attack

Date: 2026-07-15 KST

Method under review: `RAR-VLA`, Re-Anchored Autoregressive Residuals for frozen
SmolVLA.

Proposal: `reports/rar_vla/researcher_proposal.md`

Proposal hash: `723C16C3885A974E2CA12D90BC36267FA6E86827AC9D2A1E0E0E475E16FB0E56`

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

## Sources Checked

Primary closest sources:

- AR-VLA, https://arxiv.org/abs/2603.10126
- REMAC / Real-Time Robot Execution with Masked Action Chunking,
  https://arxiv.org/abs/2601.20130
- Temporal Action Selection for Action Chunking,
  https://arxiv.org/abs/2511.04421
- ReactVLA, https://arxiv.org/abs/2606.14255
- DSWAM, https://arxiv.org/abs/2607.04927
- ABot-M0 Action Manifold Learning, https://arxiv.org/abs/2602.11236

## Attack 1: Closest Prior May Be AR-VLA Itself

AR-VLA already claims the central idea: a continuous autoregressive action
expert with long-lived memory, refreshable vision-language prefixes, and
re-anchoring for perception staleness. If RAR claims broad novelty for
autoregressive action memory, context-aware action generation, or re-anchoring,
the claim is not defensible.

Conditional ruling:

- RAR may not claim novelty for autoregressive action experts or re-anchoring.
- RAR's only defensible novelty is a frozen-SmolVLA, identity-preserving,
  development-gated residual memory adapter that tests whether AR-style action
  memory can be made locally useful without replacing the action head.
- The closest-prior comparison must remain `ar_vla_reanchored_expert_proxy`
  unless exact official AR-VLA equivalence is established.

## Attack 2: REMAC May Be Closer Than The Proposal Admits

REMAC directly studies real-time action chunk execution. It argues that
inter-chunk smoothing is insufficient and identifies intra-chunk inconsistency
between intended action chunks and current perception. It learns corrective
adjustments on a pretrained policy through masked action chunking and uses
prefix-preserved sampling to reinforce continuity.

This is dangerous for RAR:

- RAR's "re-anchoring to new Base chunks" could be only prefix preservation;
- RAR's residual could be only masked chunk correction;
- RAR's claimed failure mode could collapse into the same asynchronous
  chunk-mismatch problem.

Conditional ruling:

- RAR must explicitly distinguish its causal residual memory from REMAC-style
  masked chunk training and prefix preservation.
- If implementation becomes prefix-preserved smoothing or masked chunk repair,
  it should be reclassified as a REMAC proxy or killed for insufficient
  novelty.
- Stage 0 must report inter-chunk and intra-chunk diagnostics separately.

## Attack 3: TAS May Explain The Mechanism More Simply

Temporal Action Selection caches predicted action chunks from multiple
timesteps and uses a lightweight selector to trade off reactivity, consistency,
and motion coherence. If RAR is merely choosing or blending among recent chunks,
TAS is the simpler equivalent method.

Conditional ruling:

- RAR must not be implemented as a chunk selector, cached-chunk router, or
  weighted blend among recent chunks.
- The `ema_action_history_baseline` is necessary but may not be sufficient; the
  AR proxy or ablation must expose whether re-anchored memory adds value beyond
  cached-action selection.
- If RAR's learned component never uses the re-anchored memory state beyond
  selecting a recent action, it is a trivial equivalence and should stop as
  `DESIGN_FAILURE`.

## Attack 4: The Strongest Simple Killer Is Action-History Smoothing

CALA Stage 0 already showed that `action_history_only` beat a more elaborate
deployment-observable predictor for future latent action structure. RAR is
therefore at high risk of being a mathematical dressing on a simple
action-history baseline.

Conditional ruling:

- `ema_action_history_baseline` must remain in the first five-policy comparison.
- Stage 0 residual predictability must beat EMA and linear-history baselines by
  a preregistered margin before validation search.
- If EMA/history is best, the correct outcome is `DESIGN_FAILURE`, not a call
  for more features or tuned thresholds.

## Attack 5: Prior Local Kills Are Highly Relevant

Prior local methods killed related ideas:

- EAC and RCV explored action/chunk scheduling and replan-style interventions;
- MTF and DAGR explored frame retention and component-level routing;
- MARC explored action mixture/anchor behavior;
- CALA found that a richer latent predictor lost to action-history-only.

Conditional ruling:

- RAR must not alter chunk length, queue policy, retained-frame ratio, or
  component routing.
- RAR's implementation must expose the residual memory state, gate value,
  action deltas, and activation context.
- If the method only changes final 7D actions globally, it should stop as
  `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.

## Required Rebuttal Commitments

Researcher A must accept:

1. narrowed novelty: frozen-SmolVLA identity-preserving AR-style residual memory
   adapter only;
2. no broad novelty for autoregressive action memory, re-anchoring, action
   smoothing, or action chunking;
3. explicit REMAC/TAS distinction;
4. `ema_action_history_baseline` as the simple killer;
5. `ar_vla_reanchored_expert_proxy` as closest-prior proxy unless official
   equivalence is established;
6. Stage 0 inter-chunk and intra-chunk diagnostics;
7. no future action, success, reset, object-pose, or confirmatory outcome use at
   inference;
8. no validation search, training, manifest freeze, or rollout before Stage 0
   passes.

## Final Reviewer Ruling

Do not kill RAR before implementation. The external prior is strong, CALA's
audit provides a concrete reason to examine causal action history directly, and
a development-only Stage 0 can cheaply reject trivial equivalence.

However, RAR is conditionally alive only under the narrowed claim. If
Researcher A refuses the REMAC/TAS distinctions or the EMA/action-history killer
baseline, RAR should be killed before implementation.
