# Epoch 4 Cycle 33 Prior Mechanism Map

Date: 2026-07-16 KST

Previous method: `LCG-VLA`

Previous decision: `LCG_STAGE_0_DESIGN_FAILURE`

Previous result: `reports/lcg_vla/stage_0_result.json`

LCG is closed without rescue. The frozen Stage 0 result completed `5120 / 5120`
rows with zero exceptions and exact manifest/partial key equality, but failed
design gates because the language gate activated nearly everywhere
(`0.99978125`) and `standard_lora_proxy` explained or exceeded LCG.

Cycle 33 must select a genuinely new mechanism. LoRA may be used only as
implementation infrastructure. The closest positive prior must enter the first
serious comparison.

## Primary-Source Anchors

### ACoT-VLA

Source: https://arxiv.org/html/2601.11404v2

Positive prior: ACoT-VLA formulates reasoning directly in action space through
explicit coarse reference trajectories and implicit action priors. The paper
reports LIBERO evaluation under the official protocol and an average of `98.5`
in Table 1 for the full ACoT configuration, with code linked at
https://github.com/AgibotTech/ACoT-VLA.

Local relevance: existing LIBERO demonstrations include action chunks from
which coarse action rationales can be derived. Risk: prior mechanism is close
to earlier coarse-intent attempts unless the new method changes the
supervision target and integration point.

### FineVLA

Source: https://arxiv.org/html/2605.27284v1

Positive prior: FineVLA shows that fine-grained instruction supervision can
improve steerable VLA control without sacrificing goal success. The paper
reports that fine-grained-only improves over raw-only by `+1.4` to `+8.1`
success-rate points, and that mixed fine-grained/raw supervision peaks around
`1:2` to `1:1` ratios.

Local relevance: LIBERO has only goal-level language, but the demonstrations
contain action/proprio traces from which bounded action-factor labels can be
derived on discovery/validation partitions. At inference, any factor must be
predicted from deployment-observable RGB/proprio/language/Base chunks, not read
from future actions.

### GEAR-VLA

Source: https://arxiv.org/html/2606.08530v2

Positive prior: GEAR-VLA learns geometry-aware action representations through
coarse-to-fine action learning, semantic-aligned 3D integration, and embodiment
canonicalization. It reports `98.7%` LIBERO average success and `88.7%`
zero-shot LIBERO-Plus average, outperforming listed baselines.

Local relevance: LIBERO demonstrations expose proprioceptive end-effector
state and RGB, but not reliable object-level geometry or depth in the current
cached Stage 0 path. A local proxy must avoid privileged object poses and must
not invent 3D state unavailable at inference.

## Selection Implications

The best Cycle 33 candidate should attack the LCG failure mode by making the
intervention sparse, semantically specific, and label-auditable before any
rollout. FineVLA provides the strongest local mechanism fit because action
factor labels can be derived from existing demonstrations and used to supervise
a sparse predictor, while the first comparison can include a faithful
FineVLA-style action-factor instruction proxy.
