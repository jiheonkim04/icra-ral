# Epoch 4 Cycle 32 Prior Mechanism Map

Date: 2026-07-16 KST

Previous method: `S2C-VLA`

Previous result: `S2C_STAGE_0_DATA_OR_SUPERVISION_FAILURE`

S2C is closed. No S2C repair, rescue, threshold change, task change, proxy
change, or reinterpretation is allowed.

## Primary-Source Anchors

### Counterfactual Action Guidance

Primary source: `https://arxiv.org/abs/2602.17659`

Mechanism: Counterfactual Action Guidance contrasts a standard VLA policy with
a language-unconditioned vision-action branch to reduce vision-shortcut and
counterfactual language-following failures.

Positive result: the paper reports improved LIBERO-CF language-following
accuracy and task success, plus real-world reductions in counterfactual
failures.

Local reproduction path: use SmolVLA instruction-conditioned Base actions and a
legal language-null or counterfactual-language branch as a transparent proxy.
No reward, success, done, object pose, or confirmatory identity is needed.

### TAG

Primary source: `https://arxiv.org/html/2603.24584v1`

Mechanism: Target-Agnostic Guidance contrasts policy predictions under the
original observation and an object-erased observation, using the difference as
an inference-time residual steering signal for clutter robustness.

Positive result: the paper reports robustness improvements on LIBERO,
LIBERO-Plus, and VLABench under clutter, near-miss, and wrong-object failures.

Local reproduction path: use transparent observation-ablation proxies derived
from existing image inputs. Risk: reliable object-erasure may require masks or
segmentation not present in local demonstrations.

### ProgressVLA

Primary source: `https://arxiv.org/abs/2603.27670`

Mechanism: ProgressVLA estimates task progress and uses differentiable progress
guidance through an inverse dynamics world model to refine action tokens.

Positive result: the paper reports progress-estimator residual `0.07` on a
`[0,1]` scale and success-rate gains on CALVIN, LIBERO, and real-world tasks.

Local reproduction path: train an observation-feature progress probe from
demonstration frame position on discovery/validation only. Risk: using true
phase at inference is privileged, so deployment must infer progress from
available observations.

### CAC-VLA

Primary source: `https://arxiv.org/html/2607.04816v1`

Mechanism: CAC-VLA uses a latent-action interface and context-gated action
conditioning to connect VLM representations to continuous action generation.

Positive result: the paper reports LIBERO and LIBERO-Plus validation with
ablations showing latent-action horizon and context-gated conditioning matter.

Local use: not selected for this cycle because previous CALA-style latent-action
adaptation was already closed. CAC remains background context, not a rescue
route.

### GEAR-VLA

Primary source: `https://arxiv.org/html/2606.08530v1`

Mechanism: geometry-aware action representations combine coarse-to-fine action
learning, semantic-aligned 3D integration, and embodiment canonicalization.

Positive result: the paper reports strong LIBERO, LIBERO-Plus, cross-embodiment,
and grasping success.

Local use: less suitable for immediate implementation because the existing
LIBERO demonstrations do not provide reliable deployment-time 3D geometry or
embodiment-canonical supervision without adding a new perception stack.
