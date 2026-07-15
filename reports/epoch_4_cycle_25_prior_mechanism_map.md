# Epoch 4 Cycle 25 Prior And Mechanism Map

Date: 2026-07-16 KST

## Preserved Boundary

VDR-VLA is closed unchanged as
`VDR_STAGE_0A_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`. Its Stage 0A result is a
development-only implementation/optimization failure, not a scientific kill.
Stage 0B, rerun, threshold change, clipping, action-validity reinterpretation,
and VDR rescue are forbidden.

Closed or high-risk neighborhoods remain live boundaries:

- LIFT/CAG language-guidance and counterfactual flow steering;
- EAC/AAC entropy-only adaptive chunk execution;
- RAR action-history residual memory;
- HEST/HASTE phase, keyframe, and event-timing supervision;
- KITE action-to-future-end-effector realization;
- VDR/FutureVLA dynamic visual-feature residual alignment;
- COVI/LIBERO-Occ complementary-view or occlusion reconstruction;
- generic LoRA/QLoRA adaptation as the contribution.

Cycle 25 keeps the sharpened design constraint: one genuinely new mechanism,
LoRA only as implementation infrastructure, and the closest positive prior in
the first serious comparison.

## Positive Primary Priors

### OptimusVLA

Primary source: https://arxiv.org/abs/2602.20200

Official repository: https://github.com/iLearn-Lab/CVPR26-OptimusVLA

Official project/checkpoint release: the repository states that inference code
and LIBERO assets/checkpoints were released in May 2026, with GPM memory,
LCM checkpoint, FAISS index, memory actions, and LIBERO evaluation scripts.

Positive prior result: OptimusVLA reports that Global Prior Memory replaces
Gaussian action-generation priors with task-level retrieved trajectory priors,
and Local Consistency Memory models executed action history to enforce temporal
coherence. The paper reports `98.6%` average LIBERO success, `13.5%` CALVIN
improvement over pi0, RoboTwin 2.0 Hard gains, real-world Generalization and
Long-horizon gains, and `2.9x` inference speedup.

Mechanism map:

| Axis | OptimusVLA |
| --- | --- |
| observation/input | current observation, task/language, historical executed actions |
| representation | multimodal query for memory retrieval and short action-history encoding |
| supervision/assets | memory index, task head, memory actions, LCM checkpoint |
| objective | retrieval-guided flow prior plus local consistency correction |
| policy component changed | flow sampler prior and optional LCM correction |
| action-generation mechanism | denoise from a retrieved action distribution rather than isotropic noise |
| inference-time intervention | yes, but only legal current observation/language/history and stored training memory |
| demonstrated causal link | reported LIBERO and real-world gains plus reduced NFE |
| local extension opening | residualize the policy around retrieved anchors instead of copying or smoothing memory actions |

### Past-Token Prediction For Long-Context Diffusion Policies

Primary source: https://arxiv.org/abs/2505.09561

Official project: https://long-context-dp.github.io/

Official repository: https://github.com/long-context-dp/ldp

Positive prior result: Past-Token Prediction (PTP) trains policies to predict
past action tokens alongside future ones, improving temporal modeling in
diffusion policies. The project reports that PTP improves long-context policy
performance and adds test-time self-verification by selecting candidate
sequences that reconstruct already executed actions.

Mechanism map:

| Axis | PTP |
| --- | --- |
| observation/input | long history of observations and actions |
| representation | cached visual embeddings and action-token history |
| supervision | past and future action tokens from demonstrations |
| objective | auxiliary past-token prediction plus future action prediction |
| policy component changed | policy head and candidate selector |
| action-generation mechanism | retain temporal dependencies between past executed actions and future chunk |
| inference-time intervention | self-verification against already executed action history |
| demonstrated causal link | improved long-context diffusion policy performance |
| local risk | close to RAR/action-history residual memory and may collapse on short LIBERO histories |

### AutoHorizon

Primary source: https://arxiv.org/abs/2602.21445

Official repository: https://github.com/hatchetProject/AutoHorizon

Positive prior result: AutoHorizon uses action self-attention weights in
flow-based VLAs to estimate chunk-specific execution horizons. It reports
that execution horizon strongly affects VLA success, and that attention-guided
dynamic horizon selection improves simulated and real robot manipulation with
low overhead.

Mechanism map:

| Axis | AutoHorizon |
| --- | --- |
| observation/input | current VLA attention maps during action chunk generation |
| representation | bidirectional soft pointers over action self-attention |
| supervision | no new labels required |
| objective | test-time horizon estimation |
| policy component changed | execution scheduler only |
| action-generation mechanism | execute only the reliable prefix of each generated chunk |
| inference-time intervention | yes, dynamic replan horizon |
| demonstrated causal link | reported success improvements over fixed horizon baselines |
| local risk | close to EAC/AAC adaptive chunking, and SmolVLA attention hooks may not expose the same signal |

## Local Failure Synthesis

VDR showed that full future-feature residualization is locally brittle: the
static predictor and FutureVLA proxy did not leave the required positive
development headroom, while action validity failed a fixed gate before any
training or rollout. Future methods should avoid heavy future-feature targets
and should audit deployment-observable action-generation signals before any
optimizer step.

KITE showed that local demonstrations contain meaningful action-to-consequence
structure, but directly reconstructing future state from generated chunks was
too fragile under the frozen action-validity gate. The useful lesson is to keep
physical/action distribution structure close to legal expert actions rather
than inventing unconstrained reconstructed actions.

LIFT and EAC warn that inference-only steering or horizon control can be killed
by action validity or simple baselines. A Cycle 25 candidate should therefore
make the strongest simple explanation explicit early: standard LoRA if weights
are updated, or a direct retrieval/action-replay baseline if memory explains
the result.

## Design Implication

Cycle 25 should prioritize an OptimusVLA-anchored retrieved-action-prior method
because it has a strong, current, primary-source anchor with official code and
LIBERO assets. The local method must not merely attach a nearest-demonstration
replay module to SmolVLA. A defensible extension is to use retrieved expert
chunks as an action-distribution anchor and train only a bounded residual
around that anchor, preserving Base by zero-initialized gating and comparing
early against an OptimusVLA proxy, an anchor-only ablation, and standard LoRA.
