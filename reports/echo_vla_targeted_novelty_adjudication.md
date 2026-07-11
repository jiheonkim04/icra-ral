# ECHO-VLA Targeted Novelty Adjudication

Date: 2026-07-11 KST

Branch: `codex/implement-echo-vla-first-prototype`

Current main input commit: `5fcc87b93b627dbf09eb69676801e4412909bda4`

## Decision

`ECHO_NOVELTY_SURVIVES_TARGETED_GATE`

ECHO survives only under the narrowed claim below:

> ECHO-VLA learns a phase-conditioned, explicit physical-effect mediator from same-state action interventions and uses that mediator for pre-execution candidate credit/ranking.

The claim is not action-effect history, future-frame prediction, candidate validity prediction, action advantage prediction, phase classification, counterfactual data augmentation, or generic world-model planning.

## Primary Sources Checked

| Work | Date | Source | Core mechanism |
| --- | --- | --- | --- |
| Reflective VLA | 2026-06-23 | https://arxiv.org/abs/2606.25215 | observation-action-consequence triplets as in-context history for deployment generalization |
| Action-Effect Memory | 2026-06-10 | https://arxiv.org/abs/2606.12499 | compact history pretraining from interleaved vision-action sequences via masked modeling |
| Causal World Modeling / LingBot-VA | 2026-01-29, rev. 2026-03-22 | https://arxiv.org/abs/2601.21998 | autoregressive video-action world model with shared latent space and action execution |
| Pre-VLA | 2026-05-21 | https://arxiv.org/abs/2605.22446 | preemptive verifier predicting safety confidence and critic advantage for candidate chunks |
| CoVer / CoVer-VLA | 2026-02-12, rev. 2026-02-18 | https://arxiv.org/abs/2602.12281 | contrastive verifier and test-time prompt/action candidate selection |
| Move-Then-Operate | 2026-04-26 | https://arxiv.org/abs/2604.23620 | move/operate phase router and dual-expert VLA policy |
| Dream2Fix | 2026-03-13 | https://arxiv.org/abs/2603.13528 | generative world-model counterfactual failure synthesis for recovery data |
| VLA-Corrector | 2026-07-02 | https://arxiv.org/abs/2607.01804 | latent visual dynamics monitor, chunk truncation, and corrective replanning |

## Comparison Matrix

| Work | Models action consequences? | Predicts physical effects? | Actual interventions? | Ranks candidate chunks? | Models execution phase? | Direct success/value? | Intervention timing | Requires world model? | Interpretable effect representation? | Effect as explicit causal mediator? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reflective VLA | Yes, as past observation-action-consequence context | Observed consequences are stored; no explicit effect vector | At deployment, realized past interactions; training uses trajectory triplets | No | No explicit phase model | No | Before next action, via context | No | No, consequence is observation frame/context | No |
| Action-Effect Memory | Yes, via vision-action history | Implicit state evolution in a compact memory | No same-state candidate interventions | No | No | No | Policy conditioning | No | No, latent memory | No |
| LingBot-VA | Yes, through video-action dynamics | Predicts future visual dynamics and actions | No same-state candidate intervention protocol in ECHO sense | Not primary | No explicit task phase | Not direct candidate value | During rollout/prediction | Yes | Future frames/latents, not explicit predicate effects | No |
| Pre-VLA | Indirectly, through safety and advantage labels | No explicit physical-effect vector | Candidate execution labels/critic signals, not same-state physical mediator | Yes | No explicit phase-required effect | Yes, advantage-like | Before execution | Optional WM rollout support | No | No |
| CoVer | No physical consequences; instruction-action alignment | No | No | Yes | No | No, verifier alignment/task progress | Before execution | No | No | No |
| Move-Then-Operate | No action-consequence model | No | No | No | Yes, move vs operate | No | During generation via phase router | No | Phase labels only | No |
| Dream2Fix | Yes, synthetic failure rollouts | Predicts failure/recovery trajectories, not effect vector | Counterfactual in a generative world model, not exact same simulator state interventions | No pre-execution candidate credit | Failure/recovery structure, not phase-required effect | Recovery target, not candidate value baseline | After failure / recovery | Yes | Failure type and recovery trajectory, not predicate-effect vector | No |
| VLA-Corrector | Yes, predicted vs actual latent visual evolution | Latent visual deviation, not explicit effect vector | Real execution feedback, but no same-state action comparison | No pre-execution ranking | Adaptive horizon, not task phase | No direct value | During/after chunk execution | Uses latent dynamics monitor | No | No |
| ECHO-VLA | Yes | Yes, explicit effect vector | Yes: restore identical state and execute K candidates | Yes | Yes: phase-required effect target | No direct success as primary mediator | Before executing selected chunk | No world model required | Yes | Yes |

## Exact Difference From Closest Recent Papers

### Reflective VLA

Reflective VLA is the closest action-consequence work. It stores previous observation-action-consequence triplets so the model can infer deployment-specific sensing/control factors. ECHO differs because its training unit is not historical context but a same-state intervention set: multiple candidate chunks are executed from exactly the same simulator state and labeled by realized physical effects. ECHO then learns an explicit effect mediator and ranks future candidates before execution.

### Action-Effect Memory

AEM pretrains a compact temporal memory from interleaved vision-action histories using masked reconstruction. ECHO does not pretrain a history representation; it learns an interpretable predicate-effect vector for each candidate chunk and optimizes same-state pairwise ranking.

### Pre-VLA

Pre-VLA is the closest candidate-ranking baseline. It predicts safety confidence and critic-derived advantage for action chunks and resamples low-quality actions. ECHO must not be framed as generic validity or advantage. Its difference is that ranking is mediated by phase-required physical effect components, not by direct action quality/value.

### Move-Then-Operate

Move-Then-Operate validates that phase structure can matter. ECHO uses richer phases, but phase alone is not its novelty. ECHO's novel element is phase-conditioned effect credit from same-state interventions.

### Dream2Fix and LingBot-VA

Both use world-model style action consequences. ECHO does not synthesize future RGB or train recovery trajectories from generated failures. It observes actual realized effects from restored simulator states and deploys a lightweight effect predictor without querying a world model online.

### VLA-Corrector

VLA-Corrector detects execution drift after a chunk starts. ECHO ranks candidate chunks before execution based on predicted phase-compatible physical effects.

## Gate Result

The novelty gate passes only with the following implementation constraints:

- every training pair called counterfactual/interventional must come from identical restored state;
- ordinary demonstration transitions may provide priors or warm-start labels, but not causal claims;
- direct success/value prediction must be a baseline and cannot be the ECHO mediator;
- phase labels are useful only as conditioning, not as the contribution;
- local Pre-VLA-style validity/advantage is a proxy baseline unless official Pre-VLA is faithfully reproduced.

Proceed to counterfactual protocol and candidate-headroom gate.
