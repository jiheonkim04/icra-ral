# Epoch 4 Cycle 20 Prior And Mechanism Map

Date: 2026-07-15 KST

Decision scope: discovery and method design only. No confirmatory identity was
read, decoded, trained on, or used for selection.

## Campaign Boundary

Cycle 19 SPARC-VLA is closed unchanged as
`SPARC_STAGE_0A_IMPLEMENTATION_OR_PROTOTYPE_ACTION_VALIDITY_FAILURE_NO_SCIENTIFIC_KILL`.
Its single implementation repair was consumed. Cycle 20 may not change the
SPARC operator, ridge, aperture, beta, hook site, smoke construction, or action
thresholds and call the result a SPARC continuation.

The new cycle must also respect these local lessons:

1. RCV-VLA and EAC-VLA showed that an adaptive execution horizon can be
   matched or explained by a simpler fixed replanning policy when its trigger
   does not measure deployment drift directly.
2. DAGR, MARC, IARC, FAMR, and SPARC showed that a mechanism can act while
   failing action safety. Exact Base passthrough and post-intervention safety
   must be separate gates.
3. G3P, CALA, and RAR showed that plausible supervision is not sufficient when
   the target cannot be inferred from deployment-observable inputs.
4. CAVM and FANG close success/failure action-memory and learned 7D failure
   field routes. Cycle 20 may not rename either route.
5. LoRA is an implementation mechanism or control, not novelty by itself.

## Closest Positive Prior: VLA-Corrector

Primary paper:
https://arxiv.org/abs/2607.01804

Official project and code:

- https://zju-omniai.github.io/vla-corrector/
- https://github.com/ZJU-OmniAI/vla-corrector
- inspected official commit:
  `9d23a0ba6fad562d3ed1a68fc52c8a12459abb41`
- license: Apache-2.0

The official repository is a LeRobot-based implementation with latent
extraction, corrector training, modified VLA evaluation, SmolVLA integration,
and Online Gradient Guidance. The shallow source audit contained `600` files
and about `28.2 MB`; datasets and trained checkpoints are not bundled.

VLA-Corrector freezes the VLA and trains an external latent dynamics model to
predict a short-horizon visual residual from current visual tokens and the
executed action. A Latent-space Vision Monitor compares predicted and observed
residuals. A detected mismatch truncates the stale queue and applies OGG to the
next recovery replan.

Positive result already demonstrated:

- MetaWorld PI0.5: `48.70 -> 64.35`, `+15.65` points;
- MetaWorld SmolVLA: `61.90 -> 66.65`, `+4.75` points;
- MetaWorld X-VLA: `55.55 -> 59.60`, `+4.05` points;
- LIBERO PI0.5 few-shot: `94.00 -> 97.80`, `+3.80` points;
- real AgileX PiPER: `55.6 -> 73.3`, `+17.7` points.

The paper also reports that truncation alone improved the main MetaWorld
average from `48.70` to `60.35`, truncation plus OGG reached `64.35`, and
`83.7%` of truncations occurred in manually labeled critical phases.

### Exact Prior Limitation

The inspected official `CircuitBreaker` computes one scalar:

`E_t = 1 - cosine(delta_z_pred, delta_z_real)`.

It appends that score to an episode-local rolling window and triggers above:

`median(E) + k_MAD * MAD(E)`.

The trigger does not model predictive variance. It does not normalize latent
innovation by expected uncertainty conditioned on action magnitude, visual
state, instruction, or chunk phase. During the first ten observations its
threshold is infinite. The source contains MSE/cosine mean-dynamics objectives
but no heteroscedastic covariance head or conformal innovation calibration.

This creates a falsifiable extension opportunity: normal latent dynamics can
have different residual scales in free-space motion, contact, grasp closure,
and release. A global cosine-MAD detector can trigger on expected high-variance
motion or miss a smaller but unlikely error in a low-variance critical state.

## Adjacent Prior: WIZARD

Paper and project:

- https://arxiv.org/abs/2606.07217
- https://fascetta.github.io/WIZARD/

WIZARD maps an instruction and short demonstration video to task-specific LoRA
weights in one forward pass. Its meta-dataset pairs task evidence with expert
LoRA updates. The reported design preserves multimodal parameter structure,
predicts layer scale, and trains with MSE, scale, and cosine alignment.

Positive result already demonstrated:

- up to about `2x` on unseen LIBERO dataset collections;
- up to about `14x` on unseen tasks;
- real-robot average `0.33` versus `0.17` for the matched baseline.

The local repository does not have the required task-expert meta-dataset. The
checkpoint inventory contains several method checkpoints but only one full
FAMR target endpoint, not a broad set of task-specific expert LoRA updates.
Creating dozens of new task experts would dominate the current cycle budget.
No official WIZARD code repository was found on the paper or project page.

## Adjacent Prior: Harness VLA

Paper:
https://arxiv.org/abs/2607.08448

Harness VLA exposes a frozen VLA as a retryable contact-rich primitive and
uses a memory-guided agent with a fixed analytic primitive library for
grounding, staging, transport, navigation, and release. It learns primitive
operating ranges from execution traces, global success rules, and failure
models.

Positive result already demonstrated:

- `+38.6` points over the strongest relevant LIBERO-Pro baseline;
- `+25.4` points on RoboCasa365;
- `58.4%` on RoboTwin C2R.

The local LIBERO runner has no non-privileged analytic grounding, staging, or
transport primitive library. Simulator object state could implement such
primitives, but using it at inference would violate the campaign input policy.
No official Harness VLA code repository was found in the primary-source audit.

## Local Data And Reproduction Audit

The official local LIBERO root is `C:/assets/data/libero` and contains:

- `130` task HDF5 files across LIBERO-10, LIBERO-90, Goal, Object, and Spatial;
- `50` demonstrations per audited task file;
- hundreds of frames per audited demonstration;
- `128 x 128` agent-view and eye-in-hand RGB streams;
- 7D actions, proprioception, reward/done records, and episode boundaries.

This is sufficient to form within-episode `k`-step latent pairs without
privileged inference state. Raw simulator `states` exist but are prohibited as
method inputs and may be used only for an explicitly labeled diagnostic oracle
if preregistered.

The local checkpoint inventory has:

- a working frozen SmolVLA checkpoint;
- multiple historical adapter/control checkpoints;
- one full FAMR target LoRA endpoint;
- no broad task-expert LoRA meta-dataset;
- no trained VLA-Corrector checkpoint.

Therefore a VLA-Corrector extension has viable raw supervision and official
code, while a WIZARD extension does not yet have viable local weight targets
and a Harness extension does not yet have legal analytic primitive inputs.

## Provisional Novelty Search

Searches covered VLA latent dynamics monitoring, heteroscedastic prediction,
normalized innovation, covariance calibration, conformal anomaly detection,
adaptive horizon, weight-space adaptation, and memory-guided agents. The
closest VLA method remained VLA-Corrector. Adjacent robotics runtime monitoring
uses stochastic dynamics or failure classifiers, but the audit found no prior
that combines the VLA-Corrector intervention with action-conditioned
heteroscedastic latent innovation and a frozen split-conformal trigger.

This is a provisional, not final, novelty conclusion. Reviewer B must attack
the selected mechanism against VLA-Corrector, EAC, RCV, generic anomaly
detection, heteroscedastic regression, conformal calibration, and a fixed
short-horizon baseline before implementation.

## Resource Evidence

The two user-reported Windows gaming and Efficiency Mode intervals remain in
`reports/resource_contention_intervals.json`. No timing, throughput,
wall-clock efficiency, resource utilization, or latency from unknown or
overlapping intervals may support selection or a paper claim.
