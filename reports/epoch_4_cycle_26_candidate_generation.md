# Epoch 4 Cycle 26 Candidate Generation

Date: 2026-07-16 KST

Status: development-only candidate selection. No training, validation search,
rollout, confirmatory testing, RAP repair, or RAP rescue happened here.

Cycle 26 design constraint: exactly one genuinely new mechanism; LoRA may be
used only as implementation infrastructure; the closest external prior must
enter the first serious comparison.

## Candidate 1: AMP-VLA

Full name: Action-Manifold Projection for VLA action-flow adaptation.

Closest external prior: ABot-M0 / ABot-M0.5 Action Manifold Learning.

Positive prior: ABot-M0 reports action manifold learning for stable continuous
robot action prediction, with official repository assets for weights, inference,
training code, and data. ABot-M0.5 further reports strong LIBERO-family results.

Proposed mechanism: learn a discovery-only low-dimensional action manifold over
LIBERO action chunks and constrain a SmolVLA adapter through an identity-preserving
projection or gated residual. The method changes action support, not merely the
optimizer or LoRA rank. LoRA is allowed only to parameterize the bounded residual
or gate.

Falsifiable chain:

RAP revealed that action validity can dominate even when a prior anchor has
headroom. If SmolVLA adaptation produces off-support postprocessed chunks, closed
loop behavior can fail before task semantics matter. A demonstrated action
manifold should bound residual updates to legal local action support, preserve
base behavior when uncertain, reduce invalid deltas, and improve task success
only when the manifold is predictive.

Data and supervision viability: existing LIBERO demonstrations provide action
chunks, proprioception, images, language, task identities, and reset identities.
No privileged inference input is required; the manifold target is generated from
training demonstrations only.

Identity-preserving integration: default gate initializes to base-policy
passthrough; residual starts at zero; projection magnitude is bounded; clean
retention is an explicit validation term.

First serious comparison:

1. `smolvla_base`
2. `abot_m0_action_manifold_proxy`
3. `amp_full`
4. `amp_no_manifold_projection`
5. `standard_lora`

Score:

- provisional novelty: 24 / 25
- importance of problem: 15 / 15
- strength of positive prior anchor: 20 / 20
- technical mechanism quality: 19 / 20
- data/supervision feasibility: 9 / 10
- decisive experiment feasibility: 8 / 10

Total: 95 / 100.

## Candidate 2: PEQ-VLA

Full name: Prior-Expert Query adaptation for VLA policies.

Closest external prior: PriorVLA.

Positive prior: PriorVLA reports a frozen Prior Expert plus Adaptation Expert
with expert queries, improving LIBERO, RoboTwin 2.0, and real-world settings
while updating fewer parameters than full fine-tuning.

Proposed mechanism: freeze SmolVLA as a prior expert and expose base flow/action
summaries as expert queries into a small adaptation expert. The adaptation expert
is identity-preserving and can default to the frozen base action.

Falsifiable chain:

Fine-tuning can erase useful pretrained motor priors. A query interface to a
frozen prior should preserve base action geometry while allowing adaptation only
where deployment-observable evidence supports it. If the prior query is useful,
Ours should beat both a transparent PriorVLA-style proxy and a no-query ablation.

Data and supervision viability: local demonstrations can supervise adaptation
without privileged inference inputs, but the official PriorVLA repository did
not provide a directly reusable implementation at selection time, weakening the
matched prior comparison.

Identity-preserving integration: prior expert frozen; adaptation path initialized
near zero; mixture defaults to base behavior.

First serious comparison:

1. `smolvla_base`
2. `priorvla_expert_query_proxy`
3. `peq_full`
4. `peq_no_expert_query`
5. `standard_lora`

Score:

- provisional novelty: 23 / 25
- importance of problem: 14 / 15
- strength of positive prior anchor: 17 / 20
- technical mechanism quality: 18 / 20
- data/supervision feasibility: 9 / 10
- decisive experiment feasibility: 9 / 10

Total: 90 / 100.

## Candidate 3: SGW-VLA

Full name: Spatial-Grounded Waypoint adaptation for VLA action chunks.

Closest external prior: InternVLA-M1.

Positive prior: InternVLA-M1 reports spatially guided VLA training, code,
checkpoints, and LIBERO / robot-control gains from spatial grounding and spatial
prompting.

Proposed mechanism: generate development-only spatial waypoint or contact-proxy
supervision from LIBERO demonstrations, then condition a bounded SmolVLA adapter
on predicted waypoints. LoRA is only the adapter infrastructure.

Falsifiable chain:

Instruction-conditioned visual policies can fail because action updates are not
tethered to spatially meaningful intermediate targets. A deployment-observable
waypoint representation should improve local action direction and phase behavior.
If the waypoint signal is real, Ours should beat a spatial prior proxy and a
no-waypoint ablation while retaining clean behavior.

Data and supervision viability: LIBERO observations and proprioception are
available, but true object or pixel grounding labels are not guaranteed in the
current HDF5 demonstrations, so supervision may be weaker or synthetic.

Identity-preserving integration: bounded waypoint-conditioned residual, zero
initial residual, and base passthrough when waypoint confidence is low.

First serious comparison:

1. `smolvla_base`
2. `internvla_spatial_guidance_proxy`
3. `sgw_full`
4. `sgw_no_waypoint`
5. `standard_lora`

Score:

- provisional novelty: 22 / 25
- importance of problem: 14 / 15
- strength of positive prior anchor: 19 / 20
- technical mechanism quality: 17 / 20
- data/supervision feasibility: 7 / 10
- decisive experiment feasibility: 9 / 10

Total: 88 / 100.

## Selection

Selected candidate: `AMP-VLA`.

Rationale: AMP-VLA is the most direct response to the newest fixed-protocol
evidence. RAP had retrieval-anchor headroom but failed the hard action-validity
gate; therefore Cycle 26 should make action support the mechanism itself rather
than treating validity as a downstream diagnostic. AMP-VLA also has the strongest
official prior anchor, the cleanest local supervision from existing
demonstrations, and the most decisive bounded Stage 0 audit.

The selected method remains a proposal-stage result only. It must still pass a
Researcher A proposal, Reviewer B attack, rebuttal, mathematical objective audit,
preregistration, and bounded development-only audits before any confirmatory
test.
