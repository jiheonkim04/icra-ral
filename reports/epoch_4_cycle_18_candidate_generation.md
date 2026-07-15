# Epoch 4 Cycle 18 Candidate Generation

Date: 2026-07-15 KST

Decision: `SELECT_PCAV_VLA`

Exactly three candidates were generated and scored under the active
performance-oriented, false-negative, and post-COVI method-design governance.
FAMR-VLA remains closed as an endpoint implementation/action-validity failure.
No FAMR coefficient, action threshold, task, or checkpoint is reused as a
development variant here.

## Prior And Failure Synthesis

The recent campaign provides four constraints rather than a negative result
about all VLA adaptation:

1. EAC-VLA showed that an inference scheduler can act safely, but its Stage B
   gain was explained by a simpler fixed policy.
2. G3P-VLA lacked viable non-privileged grounding supervision; CALA-VLA and
   RAR-VLA were nonacting design failures.
3. COVI-VLA, IARC-VLA, and FAMR-VLA reached implementation or optimization
   failures before a fair closed-loop scientific test. Their results do not
   establish that adaptation is ineffective.
4. IARC and FAMR both exposed the risk of globally changing strong Base
   actions before proving that the new action remains on a valid successful
   support.

The strongest current positive priors instead generate several action
candidates and verify them without replacing the Base policy globally:

- TACO: https://arxiv.org/abs/2512.02834, official code at
  https://github.com/breez3young/TACO;
- VLA-ATTC: https://arxiv.org/abs/2605.01194;
- ProgressVLA: https://arxiv.org/abs/2603.27670;
- RoboMonkey: https://arxiv.org/abs/2506.17811, official code at
  https://github.com/robomonkey-vla/RoboMonkey.

TACO reports gains of `1.8` points with pi0.5 and `6.0` points with OpenVLA on
LIBERO-Long using a lightweight pseudo-count verifier. VLA-ATTC reports more
than `50%` failure-rate reduction on LIBERO-Long using adaptive candidate
generation and a relative action critic. ProgressVLA reports gains on LIBERO,
CALVIN, and real hardware by predicting task progress and guiding action
generation through a latent world model. RoboMonkey reports `9` points on
in-distribution simulation and `25` points on real OOD tasks using action
sampling and verification.

## Candidate 1: PCAV-VLA

Name: `PCAV-VLA`, Progress-Conditioned Anti-exploration Verification for VLA
policies.

Contribution type: `CROSS_PAPER_SYNTHESIS` plus `PRIOR_EXTENSION`.

Closest external prior: TACO, with ProgressVLA as the single mechanism source
for the extension.

Positive external result: TACO demonstrates that selecting the highest-support
candidate with a coupled pseudo-count head improves flow-based and
autoregressive VLA success. ProgressVLA demonstrates that an explicit progress
estimator and latent action-consequence model can improve manipulation success.

Official code or reproducible mechanism: TACO provides official code. The
ProgressVLA paper specifies a progress estimator, latent world model, and
progress-guided action objective. PCAV does not claim official ProgressVLA
equivalence; its matched closest-prior arm is a transparent local TACO proxy.

### Scientific Method

PCAV keeps the frozen Base SmolVLA and samples a small fixed candidate set that
always includes the deterministic Base chunk. A TACO-style support head first
forms the eligible set. Within that set, a progress-consequence head predicts
which candidate advances the task most. Ours replaces Base only when the best
eligible candidate beats Base by a validation-frozen progress margin;
otherwise it returns Base exactly.

The decision is lexicographic, not an uncalibrated weighted sum:

1. reject candidates below the frozen support threshold or outside the
   absolute/Base-relative action-validity envelope;
2. among eligible candidates, maximize predicted progress advantage;
3. replace Base only when that advantage exceeds the frozen abstention margin.

Primary scientific difference from TACO: support density answers whether an
action resembles successful demonstrations, while progress consequence asks
whether an in-support action advances the current task state. TACO's maximum
density assumption can favor frequent early-phase or low-motion actions. PCAV
tests whether progress resolves that ambiguity without permitting
out-of-support action changes.

Key ablation: `pcav_support_only`, the same candidates, support head, thresholds,
and Base inclusion, but selection is by maximum support with no progress head.
This is also the faithful transparent TACO proxy, so the closest prior enters
the first serious comparison directly rather than as a late control.

Strongest simple reviewer-killer: `standard_lora_new_task`, because ordinary
matched target-task adaptation could explain any gain from using target-task
demonstrations.

### Mechanism Chain

- stochastic flow sampling after task adaptation -> multiple locally plausible
  action chunks for one observation;
- highest demonstration density alone -> a frequent but phase-inappropriate
  chunk can be preferred;
- phase-inappropriate selection -> hesitation, insufficient progress, or
  repeated approach motion causes closed-loop failure;
- TACO-style support filter -> remove candidate actions inconsistent with
  successful demonstration support;
- action-conditioned latent consequence model -> predict the near-future task
  representation for each remaining chunk;
- temporal progress model -> order those consequences by expected task
  advancement;
- Base-relative abstention -> intervene only when predicted improvement is
  sufficiently clear;
- expected action effect -> bounded, sparse substitutions drawn from the Base
  candidate distribution;
- expected closed-loop effect -> fewer stalled or phase-inappropriate choices
  than Base or support-only TACO while retaining clean Base behavior.

### Data And Supervision Viability

- local target demonstrations contain two RGB streams, 8D proprioception, 7D
  actions, frame index, task identity, and ordered episode trajectories;
- support supervision uses successful discovery demonstrations and TACO-style
  random coin targets on frozen internal candidate features;
- consequence supervision uses `(observation_t, action_chunk_t,
  observation_t+delta)` triples from the same discovery episode;
- progress supervision uses within-episode temporal ordering and local
  continuity, never task success, future frames, object poses, or reset identity
  at inference;
- symmetric pair construction prevents positional label leakage;
- validation episodes choose thresholds and at most one critical coefficient;
- reserved confirmatory episodes and reset identities remain sealed;
- candidate generation, support, consequence, and progress must all be
  noncollapsed before rollout.

### Identity-Preserving Integration

- the deterministic Base action is always candidate index zero;
- initial and invalid heads force exact Base passthrough;
- candidates outside the frozen action envelope are ineligible;
- a validation-frozen margin controls intervention rather than post-test
  threshold changes;
- no Base parameter is updated;
- no privileged simulator signal is used at inference.

### Bounded Search

At most six validation-only configurations:

- support percentile in `{50, 70}` crossed with progress margin in
  `{0.00, 0.05}`: four configurations;
- candidate count in `{4, 8}` at the selected support/margin pair: two
  configurations;
- one fixed head architecture and one training seed during selection;
- one second seed only for the selected lightweight configuration;
- no combinatorial architecture search and no confirmatory identity use.

The validation score combines target validation success or the closest
available paired proxy, clean retention, mechanism activation, action validity,
and inference overhead. Offline action L2 alone cannot select a configuration.

### Score

- provisional novelty: `23 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `19 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `95 / 100`

## Candidate 2: CAVR-VLA

Name: `CAVR-VLA`, Constrained Abstaining Verification and Reranking for VLA
policies.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: VLA-ATTC,
https://arxiv.org/abs/2605.01194.

Positive external result: VLA-ATTC reports more than `50%` failure-rate
reduction for pi0.5 on LIBERO-Long and `17.3` points on its real-robot tasks
using a lightweight relative critic and adaptive deliberation.

Scientific method: train a small pairwise action critic on frozen SmolVLA
context and symmetric action pairs, include Base in every tournament, reject
invalid candidates, and abstain unless the tournament winner beats Base by a
validation-frozen margin.

Key ablation: the same relative critic and candidates without abstention.

Main strength: direct positive LIBERO prior, simple labels from solver-step or
expert-distance ordering, and exact Base fallback.

Main limitation: VLA-ATTC already owns uncertainty-triggered relative action
comparison. The support envelope and abstention improve safety and local
feasibility but provide less scientific novelty than PCAV's explicit separation
of support from progress.

Score:

- provisional novelty: `17 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `10 / 10`
- decisive experiment feasibility: `10 / 10`
- total: `90 / 100`

## Candidate 3: SPCN-VLA

Name: `SPCN-VLA`, Stage-Partitioned Coupled Pseudo-Counts for VLA policies.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: TACO,
https://arxiv.org/abs/2512.02834, with official code at
https://github.com/breez3young/TACO.

Positive external result: TACO's coupled pseudo-count verifier improves
multiple VLA families and benchmarks, including LIBERO.

Scientific method: infer a coarse task phase from frozen visual-language and
proprioceptive context, train one coupled pseudo-count head per phase, and
select the highest-count candidate under the inferred phase rather than under
one global demonstration density.

Key ablation: one global pseudo-count head with identical candidates and
training rows.

Main strength: all labels can be generated from ordered demonstrations and the
method remains gradient-free with respect to Base at inference.

Main limitation: phase bins are a weaker proxy for advancement than a learned
action-conditioned consequence. Temporal bins can correlate with episode
length or collection style, and a misclassified phase can reject the correct
mode. This yields a less decisive mechanism than PCAV.

Score:

- provisional novelty: `19 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `86 / 100`

## Selection

`PCAV-VLA` is selected with `95 / 100`.

It has the strongest mechanism for addressing the observed campaign failure
pattern without rescuing a stopped method: keep candidates on successful
demonstration support, distinguish support from task advancement, include Base
in every decision, and abstain when the progress evidence is weak. The method
is verification and consequence modeling; no LoRA parameterization defines the
scientific contribution.

Unknown empirical performance is not a rejection reason. Selection authorizes
only a frozen Researcher A proposal, independent Reviewer B attack, rebuttal,
mathematical audit, preregistration, and bounded Stage 0 implementation. It
does not authorize confirmatory decoding or rollout.

## First Serious Comparison

| Policy | Scientific question |
| --- | --- |
| `smolvla_base` | Does candidate verification improve over the untouched policy? |
| `taco_support_proxy` | Does the closest positive support-only prior improve under the same candidates and backbone? |
| `pcav_full` | Does progress-conditioned in-support selection improve over Base and TACO? |
| `pcav_progress_only` | Is the support constraint necessary, or does progress scoring alone explain the result? |
| `standard_lora_new_task` | Can ordinary matched target-task adaptation explain any gain? |

Exactly five policies are justified. No sixth internal control enters the first
serious comparison without a concrete, decision-relevant alternative
explanation.
