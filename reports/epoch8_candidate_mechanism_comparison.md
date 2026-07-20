# Epoch 8 Candidate Mechanism Comparison

Decision time: 2026-07-20T20:15:00+09:00
Evidence state: split manifests frozen; validation language manually audited; confirmation text and outcomes sealed; no Ours model outcome observed.

## Decision

Candidate 2, **Paired Counterfactual Action Transport (PCAT)**, is selected for the cheapest decisive Stage 0. Candidate 1 is rejected before empirical execution as a paper contribution because its claimed mechanism is an incremental combination of already occupied object-grounding components.

This decision does not name PCAT as a successful method. It authorizes one frozen, bounded causal mechanism test. The primary endpoint remains official closed-loop success under held-out paraphrases; the offline Stage 0 can only authorize that experiment, not replace it.

## Candidate 1: two-sided binding-posterior mediation

### Causal hypothesis

Paraphrases fail because the instruction-conditioned posterior over scene targets moves to the wrong visual entity. A target posterior that is invariant to audited equivalent wording, selective under genuine target swaps, and injected into the action decoder should stabilize downstream control.

For visual tokens `v_k` and instruction `l`, the candidate predicts `p_phi(k | o,l)`. Training-only simulator segmentation supplies a legal target mask. Equivalent instructions minimize Jensen-Shannon divergence between posteriors. Target-swapped instructions receive distinct target-mask supervision. The posterior-weighted target vector `b=sum_k p_phi(k|o,l)v_k` enters the X-VLA action transformer through a zero-initialized residual gate. Deployment receives only RGB, language, proprioception, and the ordinary action history.

### Decisive falsifier

On at least five tasks and two manipulated-object families, require mask/target accuracy and posterior equivalence to improve over Base probes and a capacity-matched head, target swaps to move probability to the correct new entity, decoder interventions to measurably change legal actions, and canonical clean-action loss to remain within five percent of Base. Failure of posterior improvement, causal action effect, or retention closes this exact construction.

### Why it is not selected

- GuidedVLA already trains mask-supervised object attention and fuses the specialized path through a zero-initialized action residual.
- ProGAL-VLA already applies entity-level grounding with contrastive supervision before goal-conditioned control.
- Direct grounded-point action-head work already injects an explicit grounded target into a diffusion action head.
- Adding positive paraphrase invariance and negative target swaps gives a useful ablation, but the current audit does not support a sufficiently distinct RA-L mechanism claim.
- The retained X-VLA path also lacks a validated Florence-token-to-simulator-mask alignment, making this route slower and less decisive locally.

Status: `REJECTED_BEFORE_OUTCOMES_OCCUPIED_AND_INCREMENTAL`. This is a scientific novelty rejection, not an empirical falsification.

## Candidate 2: Paired Counterfactual Action Transport (PCAT)

### Causal hypothesis

The missing constraint is not another target classifier. The action field should be locally invariant along a meaning-preserving language intervention and should move by the *real expert action displacement* induced by a genuine target change. Training this signed response directly should improve instruction-to-action selectivity while retaining the Base policy.

Let the frozen X-VLA clean-action predictor be `f_0(o,l,x_t,t)`. A zero-initialized residual adapter produces

`f_phi = f_0 + R_phi(h_0(o,l), x_t, p, t, f_0)`.

`h_0` is a pooled frozen X-VLA multimodal encoder feature. The adapter is a per-action-step MLP with a 1024-to-128 context projection, two width-256 hidden layers, a fixed 16-dimensional sinusoidal time feature, and a 10-dimensional left-arm output; the right-arm residual is identically zero. It has fewer than 0.3 million trainable parameters. Base weights remain frozen.

For an audited equivalent instruction `l_tilde`, the same real action chunk supervises `f_phi(o,l)` and `f_phi(o,l_tilde)`, with an explicit vector-consistency term.

For paired real demonstrations `(o_i,l_i,a_i)` and `(o_j,l_j,a_j)` from the same released Goal world and closely matched initial end-effector pose, define the standardized real transport vector

`Delta_ij = S(a_j[1:30,0:9] - a_i[1:30,0:9])`.

Using shared diffusion time and noise, define the signed instruction response

`delta_phi = 0.5 * S((f_phi(o_i,l_j)-f_phi(o_i,l_i)) + (f_phi(o_j,l_j)-f_phi(o_j,l_i)))`.

PCAT minimizes a vector Smooth-L1 loss between `delta_phi` and `Delta_ij`. It does not score, rank, or margin-separate a factual action under negative text. The swapped instruction receives no fabricated action target. Factual canonical and paraphrase branches use only their real demonstration action. At deployment there is one ordinary instruction-conditioned branch and no counterfactual query, retrieval model, visual-only subtraction, target detector, or privileged input.

### Frozen supervision

`reports/epoch8_action_response_supervision.json` freezes:

- 48 one-to-one training action pairs: 12 for each of three bowl-destination pairs and 12 for the wine-destination pair;
- four untouched validation action pairs covering five tasks and two manipulated-object families;
- 15 training and 15 manually audited validation equivalence pairs, one per task and paraphrase family;
- frame 0 and the following 30-action real chunk, before interaction;
- fixed initial-pose matching scales and data-derived action-coordinate scales;
- immutable demo paths, file hashes, frame indices, action hashes, and pair IDs.

The preflight passes. Every training pair group has median normalized real action displacement at least 0.25; every held-out pair exceeds 0.25. Initial-pose matches pass the frozen physical tolerance. The released X-VLA-format files do not contain full simulator state, so this remains a matched-demonstration causal proxy and cannot replace paired closed-loop resets.

### Exact distinction from closest work

- **Not RobustVLA/RoVLA:** they cover semantic-preserving consistency or paraphrase sampling; PCAT adds a signed response to true intent changes.
- **Not CAG/RSS:** they subtract or steer conditional and unconditioned predictions at inference; PCAT trains one deployment branch to reproduce a real differential response.
- **Not CAST:** CAST constructs synthetic counterfactual action labels. PCAT never fabricates a target action; it transports the vector difference between two real expert chunks.
- **Not Anchor-Align:** Anchor-Align predicts absolute motion-direction language and anchors representations. PCAT supervises the differential of the numeric action field under an instruction intervention.
- **Not GPLA or Candidate A:** there is no scalar language-action score, energy, margin, or ranking objective.
- **Not ProGAL/GuidedVLA/grounded-point injection:** PCAT has no entity posterior, mask head, symbolic plan, detector, or target-point input.

The novelty claim is deliberately narrow: real-demonstration, two-sided semantic intervention supervision of the *signed action-field response*. Whether that difference is useful is now an empirical question.

## Stage 0 comparison roles

All trainable roles use the identical zero-initialized adapter, optimizer budget, cached Base features, frozen time/noise bank, and training pairs.

1. Base: unmodified released X-VLA.
2. Capacity control: canonical factual clean-action fitting only.
3. Paraphrase-augmentation control: canonical and paraphrase factual fitting, no response constraint.
4. Equivalence-only ablation: augmentation plus vector equivalence, no counterfactual transport.
5. PCAT: factual fitting, equivalence, signed real action transport, and a small Base anchor.

This isolates adapter capacity, ordinary augmentation, positive consistency, and the claimed two-sided mechanism.

## Stage 0 authorization boundary

The frozen protocol is `reports/epoch8_pcat_stage0_protocol.json`. A positive offline gate authorizes only the already frozen serial validation rollouts at reset indices 3 and 4. Confirmation language and outcomes remain sealed. A negative result closes only the exact PCAT adapter, matching rule, loss, and budget after the defect-repair policy is applied; it does not close language grounding generally.
