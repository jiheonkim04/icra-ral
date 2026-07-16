# Epoch 4 Cycle 30 Prior Mechanism Map

Date: 2026-07-16 KST

Previous method: `CCIF-VLA`

Previous decision: `CCIF_STAGE_0_DESIGN_FAILURE`

CCIF remains closed under its fixed Stage 0 result. No rescue, threshold
change, intent reconstruction change, task/reset change, proxy change, or
retroactive reinterpretation is allowed.

Cycle 30 keeps the active design constraint: exactly one genuinely new
mechanism; LoRA or another lightweight adapter may be used only as
implementation infrastructure; and the closest external prior or a faithful
transparent proxy must enter the first serious comparison.

## Primary-Source Anchors Checked

### SUREFlow

Primary paper: `https://arxiv.org/abs/2607.10504`

Official repository: `https://github.com/tanvirnwu/SUREFlow`

Positive prior: SUREFlow is accepted to IROS 2026 and reports a 179M
state-space VLA with uncertainty-aware residual flow matching. The official
repository reports LIBERO success rates of `94.8 / 91.0 / 93.8 / 90.2` on
Spatial/Object/Goal/Long, `92.5%` average, and a LIBERO-PRO average success
rate around `0.49` with 179.1M parameters.

Mechanism map:

| Axis | SUREFlow |
| --- | --- |
| observation/input | RGB observations, proprioception, and task-language embeddings |
| representation | shared latent state-space backbone with uncertainty-aware residual flow |
| supervision | LIBERO demonstrations and continuous action targets |
| objective | joint action-velocity prediction and input-dependent residual uncertainty |
| policy component changed | action generator and residual flow head |
| action-generation mechanism | selectively refine unreliable action dimensions without environment feedback |
| inference-time intervention | no privileged simulator state; uses deployment observations |
| demonstrated causal link | reported LIBERO and LIBERO-PRO success gains with small parameter count |
| local extension opening | preserve SmolVLA identity and apply bounded residual transport only where predicted residual uncertainty supports intervention |

Local implication: SUREFlow is the strongest current anchor for a
demonstration-only SmolVLA overlay because existing LIBERO demonstrations
directly supply current inputs, Base chunks, and expert residual targets. A
transparent proxy can preserve the essential heteroscedastic residual-flow
mechanism even if the full Mamba backbone is not transplanted.

### VLA-IAP

Primary paper: `https://arxiv.org/abs/2603.22991`

Project page: `https://chengjt1999.github.io/VLA-IAP.github.io/`

Positive prior: VLA-IAP reports training-free interaction-aligned visual token
pruning, `97.8%` success with `1.25x` speedup on LIBERO, and up to `1.54x`
speedup while maintaining comparable performance. The project page reports
LIBERO, CALVIN, VLABench, and real-world rollouts.

Mechanism map:

| Axis | VLA-IAP |
| --- | --- |
| observation/input | current image tokens, semantic/motion alignment, structural anchors |
| representation | interaction-aligned union mask over visual tokens |
| supervision | training-free geometric and semantic-motion priors |
| objective | none; inference-time token selection |
| policy component changed | visual token set passed to the VLA policy |
| action-generation mechanism | preserve manipulation-critical tokens while pruning redundant tokens |
| inference-time intervention | yes, visual-token pruning without privileged environment state |
| demonstrated causal link | reported success retention plus speedup across several environments |
| local extension opening | derive interaction saliency from Base action motion and SmolVLA/OpenVLA visual features |

Local risk: SmolVLA token or feature-map hooks may not expose a comparable
token interface. This is promising for a later efficiency axis, especially on
Quantized OpenVLA-OFT INT4, but weaker as the immediate primary SmolVLA method.

### ReactVLA / Low-Step Mean Transport

Primary paper: `https://arxiv.org/abs/2606.14255`

Project page: `https://game-loader.github.io/ReactVLA/`

Related primary paper: `https://arxiv.org/abs/2606.05737`

Positive prior: ReactVLA reports improved Mean Flow action generation plus
Attention Residual routing, one-to-few-step generation, performance gains over
similarly sized VLA baselines including SmolVLA, more than `4x` inference-speed
increase, and real-world policy latency below `38.6 ms`. Let It Be Simple
reports that high-noise-biased one-step VLA action generation can match or
exceed multi-step decoding across LIBERO-family evaluations, including
`95.6%` on LIBERO-Long for its 1.4B setting.

Mechanism map:

| Axis | ReactVLA / one-step VLA |
| --- | --- |
| observation/input | rich multimodal observations, language, proprioception/state |
| representation | improved Mean Flow finite-interval transport and depth-wise attention residuals |
| supervision | demonstrations and action chunks |
| objective | mean-transport or high-noise-biased flow training |
| policy component changed | action generator and/or depth-wise feature routing |
| action-generation mechanism | fewer transport evaluations from noise to action |
| inference-time intervention | low-step action decoding, no simulator state |
| demonstrated causal link | reported success/latency tradeoff gains on LIBERO and real robots |
| local extension opening | train a small Base-preserving mean-transport residual head for SmolVLA chunks |

Local risk: a low-step sampler is mainly an efficiency mechanism unless the
claim-specific condition includes delay or reactive control. It also risks
being too close to ordinary flow-training schedule changes unless the
finite-interval transport mechanism is isolated.

### ACoT-VLA, ZR-0, and RynnVLA-002

ACoT-VLA (`https://arxiv.org/abs/2601.11404`,
`https://github.com/AgibotTech/ACoT-VLA`) reports action chain-of-thought
reasoning with explicit action trajectories and latent action priors, reaching
`98.5%` LIBERO average. ZR-0 (`https://arxiv.org/abs/2606.30552`,
`https://github.com/RUCKBReasoning/ZR-0`) reports dense embodied
chain-of-thought supervision and `97.8%` LIBERO. RynnVLA-002
(`https://arxiv.org/abs/2511.17502`,
`https://github.com/alibaba-damo-academy/RynnVLA-002`) reports a unified
VLA/world model with `97.4%` LIBERO and released LIBERO training/evaluation
code.

These are strong positive priors, but the immediate local extensions are
lower priority because:

- ACoT-style coarse action-intent supervision is too close to the just-closed
  CCIF coarse-intent route unless a materially different mechanism is proven.
- Dense ECoT requires reasoning annotations or teacher generation beyond the
  existing LIBERO demonstrations.
- RynnVLA-style future-image/world modeling overlaps VDR/FutureVLA-like
  future-feature routes that already proved locally brittle.

## Cycle 30 Design Bias

The newest fixed result says coarse intent was not the right local signal: it
was predictable poorly from deployment inputs and endpoint-only diagnostics
explained much of the observed structure. The next method should therefore not
add another intent, waypoint, or future-feature target.

The strongest fresh axis is SUREFlow-style uncertainty-aware residual flow, but
adapted conservatively: keep SmolVLA as the default action source, learn a
heteroscedastic residual model from existing LIBERO demonstrations, and route
bounded residual transport only where the predicted residual is both useful and
trustworthy. This changes action generation, not merely confidence reporting.

Proceed to exactly three Cycle 30 candidates:

1. `URF-VLA`: Uncertainty-Routed Residual Flow anchored to SUREFlow.
2. `IAF-VLA`: Interaction-Aligned Feature retention anchored to VLA-IAP.
3. `HNT-VLA`: High-Noise Mean-Transport residual action generation anchored to
   ReactVLA / one-step VLA.

Prefer `URF-VLA` if scoring confirms the strongest prior anchor, one
non-LoRA mechanism, local supervision from existing LIBERO demonstrations,
identity-preserving integration, and a decisive bounded Stage 0 audit.
