# Epoch 4 Cycle 34 Prior Mechanism Map

Date: 2026-07-16 KST

Previous method: `AFID-VLA`

Previous decision: `AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE`

Previous result: `reports/afid_vla/stage_0_result.json`

AFID is closed without rescue. The frozen Stage 0 result completed
`5120 / 5120` rows with zero exceptions and exact manifest/partial key
equality, but failed the implementation/objective-scale gate because
`action_deltas_bounded = false`. It also failed the required factor-prediction
baselines. This was development-only and not a closed-loop scientific result.

Cycle 34 must select a genuinely new mechanism. LoRA may be used only as
implementation infrastructure. The closest positive prior must enter the first
serious comparison.

## Primary-Source Anchors

### Diffusion Policy

Sources:

- https://diffusion-policy.cs.columbia.edu/
- https://github.com/real-stanford/diffusion_policy

Positive prior: Diffusion Policy represents visuomotor policy learning as
conditional denoising over action sequences. The project reports consistent
outperformance across 12 tasks from four manipulation benchmarks, with an
average success-rate improvement of `46.9%`, and provides official code,
experiment configs, logs, and checkpoints.

Local relevance: existing LIBERO demonstrations provide action chunks, and
the repository already has verified cached SmolVLA Base chunks. A local proxy
can train a diffusion or score-matching model over residual action chunks
without reward, success, done, object poses, future observations, or
confirmatory identities.

### FAST / FAST+

Source: https://arxiv.org/abs/2501.09747

Positive prior: FAST introduces frequency-space action sequence tokenization
with a DCT-based compression scheme and FAST+, a universal action tokenizer
trained on large robot action trajectories. The paper reports that FAST-based
VLA training can match diffusion VLA performance while reducing training time.

Local relevance: LIBERO action chunks can be transformed into frequency
coefficients from existing demonstrations. A local proxy can compare raw
action residual modeling against frequency-space residual token modeling.
Risk: tokenizer benefits may require official FAST+ assets or larger-scale
autoregressive training than the local Stage 0 budget permits.

### ACT

Source: https://github.com/tonyzhaozh/act

Positive prior: ACT, Action Chunking with Transformers, provides official code
for training and evaluating chunked imitation policies in simulated and real
ALOHA-style manipulation settings. It is a strong prior for chunk-level
temporal modeling and temporal ensembling.

Local relevance: existing LIBERO demonstrations and SmolVLA action chunks can
support a lightweight ACT-style chunk correction proxy. Risk: ACT is not a VLA
prior and a local residual transformer can collapse into a generic sequence
model unless the new mechanism changes the representation or integration point.

### OpenVLA-OFT

Sources:

- https://arxiv.org/abs/2502.19645
- https://github.com/moojink/openvla-oft

Positive prior: OpenVLA-OFT reports optimized fine-tuning that improves
OpenVLA average LIBERO success from `76.5%` to `97.1%` while increasing action
generation throughput by `26x`, with public code and LIBERO checkpoints.

Local relevance: this remains the mandatory later backbone check for serious
paper candidates. For Cycle 34 method selection, however, OpenVLA-OFT is a
less suitable closest prior because a SmolVLA-local candidate must contribute
a new mechanism rather than an optimized fine-tuning recipe.

## Selection Implications

AFID failed partly because a sparse classifier/gate was too brittle and because
bounded action-scale enforcement failed. The next candidate should avoid
hard class gates as the core mechanism, preserve Base identity by construction,
and let the closest prior enter early. Diffusion Policy is the best local
anchor because it directly models action sequence distributions, handles
multimodality, has official code/checkpoints/logs, and can be proxied from
existing LIBERO demonstration chunks without privileged inference inputs.
