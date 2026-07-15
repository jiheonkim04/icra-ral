# RAP-VLA Reviewer B Attack

Date: 2026-07-16 KST

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

RAP is not rejected before rebuttal because it is anchored to a strong current
positive prior with official code/assets, and because residualizing around
retrieved legal action anchors is not an exact duplicate of OptimusVLA's GPM +
LCM formulation. However, the paper risk is high unless Researcher A accepts
the following constraints.

## 1. OptimusVLA Proximity Is The Central Novelty Risk

RAP's closest prior already retrieves action priors and improves VLA action
generation. A residualized local adapter is not automatically novel. RAP must
prove that the claimed mechanism is:

- not merely Global Prior Memory with a different name;
- not Local Consistency Memory with a smaller action-history encoder;
- not direct nearest-demonstration replay;
- not ordinary LoRA fine-tuning over the same demonstrations.

Reviewer requirement:

- the first serious comparison must keep the transparent OptimusVLA memory
  prior proxy as policy 2;
- policy 4 must be anchor-only/no-residual to isolate retrieval copying;
- policy 5 must be matched standard LoRA to isolate ordinary adaptation;
- RAP must report residual magnitude, gate value, retrieval confidence, top-k
  diversity, and anchor-vs-residual contribution for scored states.

## 2. Memory Construction Can Leak If Not Frozen Early

Retrieval memory is powerful enough to become hidden test-set tuning. RAP must
freeze memory partitions before any validation or confirmatory outcome is
decoded.

Reviewer requirement:

- discovery/training demos only may enter the memory index used for training;
- validation demos may be queried only for development scoring, never inserted
  into the memory used by the candidate;
- confirmatory task/reset identities may not be embedded, indexed, inspected,
  or used for FAISS tuning before final freeze;
- memory feature normalization, top-k, distance metric, task filters, and phase
  features must be fixed before validation search;
- duplicate-key, source-overlap, and frame-overlap audits must be persisted.

## 3. Retrieval Headroom Must Beat Trivial Priors

If retrieved anchors do not beat a task/phase mean action chunk, then RAP has no
local reason to exist. If anchor-only matches RAP, then residual learning is
unnecessary.

Reviewer requirement:

- Stage 0 must show nearest/retrieved anchors beat task/phase mean chunks by
  the preregistered margin;
- residual targets must have nonzero variance after subtracting anchors;
- a deployment-input residual probe must beat zero-residual prediction;
- anchor-only and RAP must be empirically distinct before rollout;
- direct action L2 alone cannot select the final configuration.

## 4. Action Validity Must Use A Single Predeclared Unit System

Recent methods repeatedly failed action-validity gates. RAP must not silently
move between raw normalized chunks and environment postprocessed actions.

Reviewer requirement:

- define whether action validity is raw normalized action validity,
  postprocessed 7D LIBERO action validity, or both;
- justify the chosen unit system before Stage 0 execution;
- report Base action, anchor action, RAP action, residual norm, gate, changed
  dimensions, and validity context for representative states;
- no clipping rescue, threshold widening, or post-hoc validity reinterpretation
  after Stage 0 begins.

## 5. OptimusVLA Proxy Fidelity Must Be Honest

If official OptimusVLA assets are unavailable locally, the comparison cannot be
called an official reproduction. If they are available, RAP must not quietly use
a weaker proxy to make Ours easier to beat.

Reviewer requirement:

- first check whether the official released memory assets/checkpoints can be
  installed within local budget;
- if official assets are not used, label policy 2 as a transparent proxy and
  list every deviation from official OptimusVLA;
- match memory sources, inference budget, action postprocessing, task/reset
  manifest, and retrieval features as far as locally possible;
- do not select a weaker prior proxy after seeing validation performance.

## 6. Standard LoRA Is Mandatory Here

RAP updates weights through an adapter. Generic fine-tuning is therefore a
plausible explanation for any gain.

Reviewer requirement:

- matched standard LoRA must use the same demos, optimizer steps, rank, target
  modules, clean-retention coefficient where applicable, and checkpoint
  selection budget;
- if standard LoRA matches or beats RAP under the same manifest, RAP does not
  become a paper candidate;
- RAP may still be a useful engineering variant, but not the claimed method.

## 7. Memory Latency And Storage Are Part Of The Claim

OptimusVLA reports speed improvements. RAP cannot ignore memory overhead if it
uses retrieval at every action step.

Reviewer requirement:

- report index size, memory-action size, feature dimensionality, retrieval time,
  total policy latency, and peak memory;
- Stage A may kill for unacceptable overhead if RAP is slower than Base without
  a commensurate success signal;
- timing or resource metrics overlapping resource-contention intervals remain
  ineligible for final paper evidence.

## 8. Conditional Pass

RAP may proceed to Researcher A rebuttal only if Researcher A accepts:

1. OptimusVLA proxy as the first serious prior comparison;
2. anchor-only/no-residual ablation;
3. matched standard LoRA as the single simple reviewer-killer baseline;
4. frozen discovery/validation/test memory separation;
5. predeclared action-validity unit system;
6. retrieval headroom and residual predictability Stage 0 gates;
7. honest official-vs-transparent OptimusVLA status;
8. memory latency/storage reporting;
9. no VDR, KITE, RAR, LIFT, or EAC rescue through RAP.

If any of these are rejected, RAP should stop before implementation as
`RAP_REVIEWER_REJECTED_OR_NEEDS_REDESIGN`.
