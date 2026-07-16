# Epoch 4 Cycle 31 Prior Mechanism Map

Date: 2026-07-16 KST

Current frozen predecessor: `URF-VLA` stopped as
`URF_STAGE_0_NO_USABLE_HEADROOM`. The result is preserved unchanged in
`reports/urf_vla/stage_0_result.json`; no URF repair, rerun, rescue, threshold
change, proxy change, or reinterpretation is allowed.

Design constraint for Cycle 31:

- one genuinely new mechanism;
- LoRA only as implementation infrastructure;
- closest prior enters the first serious comparison;
- use existing LIBERO demonstrations only for development;
- no privileged inference inputs;
- no confirmatory-test tuning.

## Primary-Source Anchors

### ChunkFlow

Primary source: `https://arxiv.org/html/2607.12992v1`

Project page: `https://cytoderm-ai.github.io/chunkflow`

Positive prior result: ChunkFlow reports a seam-aware training-and-execution
framework for chunked policies, with deterministic overlap blending, boundary
consistency losses, history corruption, and continuity regularization. On
LIBERO, it reports `93.4%` long-horizon success, low boundary jump, low
high-frequency energy, and `4.43 ms` amortized reasoning latency.

Mechanism signal relevant to this repository: action-chunk boundary structure
can matter even when one-step Base residual headroom is small. The prior
directly targets cross-chunk disagreement, smoothness, and execution-indexed
semantics.

Local reproducibility path: a faithful transparent proxy can be implemented
from SmolVLA decoded chunks and existing LIBERO demonstrations by measuring
overlap disagreement, deterministic overlap blending, first/second-order action
smoothness, and boundary-jump metrics. No reward, success, done, future
observation, or object-pose input is required.

### SEAM

Primary source: `https://arxiv.org/abs/2607.04609`

Positive prior result: SEAM reports a training-free inference-time method for
flow-matching VLAs that uses the previous chunk tail as an analytic consistency
reference. On LIBERO-10 with pi_0.5, it reports `28%` lower boundary jerk,
`27%` lower transition discontinuity, baseline-level task success, and near
baseline denoising-loop cost.

Mechanism signal relevant to this repository: the previous unexecuted tail is
a deployment-available reference. This is useful for SmolVLA if adjacent decoded
chunks disagree around replanning boundaries.

Local reproducibility path: use previous-tail / current-head overlap metrics
and a transparent SEAM-style overlap correction proxy. The official method is
flow-step-specific; the local proxy must be labeled as a proxy unless official
SEAM assets are installed and verified.

### IntentVLA

Primary source: `https://arxiv.org/abs/2605.14712`

Repository: `https://github.com/ZGC-EmbodyAI/IntentVLA`

Positive prior result: IntentVLA identifies observation aliasing in chunked
imitation and introduces compact short-horizon history-conditioned intent
representations. Its repository reports AliasBench code, IntentVLA results of
`45.8` average on AliasBench versus lower history-frame controls, and
`97.4` LIBERO-Long Avg@500 success.

Mechanism signal relevant to this repository: if current-frame SmolVLA chunks
resample inconsistent local continuations, a compact recent-history commitment
state can reduce conflict without requiring future observations.

Local reproducibility path: use only past observations/proprio/actions from
LIBERO demonstrations and SmolVLA Base chunks. The full model training code is
not yet released, so the local comparison must use a transparent proxy unless
official model code/checkpoints become available.

### RynnVLA-002

Primary source: `https://arxiv.org/html/2511.17502v3`

Repository: `https://github.com/alibaba-damo-academy/RynnVLA-002`

Positive prior result: RynnVLA-002 reports a unified VLA/world model that
achieves `97.4%` LIBERO simulation success and shows controlled gains from a
world-modeling objective, including continuous-action success improvement from
`91.6%` to `94.6%` under matched backbone/data/training recipe.

Mechanism signal relevant to this repository: action-aware prediction of future
state or visual change can provide a useful action plausibility signal without
reading rollout success labels.

Local reproducibility path: train a small development-only proprioceptive
consequence model from current proprioception and candidate action chunks, then
compare Base and candidate chunks by predicted next-proprio consistency. At
inference this uses only current proprioception and candidate actions.

### ABot-M0 / Action Manifold Learning

Primary source: `https://arxiv.org/abs/2602.11236`

Repository: `https://github.com/amap-cvlab/ABot-Manipulation`

Positive prior result: ABot-M0 introduces Action Manifold Learning, directly
predicting clean continuous action sequences rather than denoising noise, with
a stated aim of improving efficiency and policy stability.

Mechanism signal relevant to this repository: demonstration action chunks may
define a low-dimensional feasible-action manifold. However, URF found small
Base-to-expert residual headroom on the current development tasks, so a
manifold-only correction risks becoming a smoothness-only result unless it is
paired with an execution-structure diagnostic.

## Exclusions For Cycle 31

- Do not revive URF, CCIF, TSC, CFR, AMP, RAP, VDR, KITE, HASTE, HEST, NICE,
  SPARC, PCAV, FAMR, IARC, LIFT, COVI, RAR, CALA, G3P, EAC, PESA, MARC, DAGR,
  MTF, RAC, EvoState, FANG, CAVM, or RCV by renaming a component.
- Do not select a method whose only scientific content is LoRA.
- Do not select a method that requires object poses, future observations,
  rollout success labels, reward/done flags, or confirmatory identities at
  inference.
- Do not select a method whose first serious comparison omits the closest
  external prior.
