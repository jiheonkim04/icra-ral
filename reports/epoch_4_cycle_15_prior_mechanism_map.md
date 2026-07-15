# Epoch 4 Cycle 15 Prior Mechanism Map

Date: 2026-07-15 KST

Purpose: select the first method governed by the post-COVI LoRA-role and
minimum-sufficient-design correction. COVI is closed under its fixed Stage 0
protocol as
`COVI_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE_NO_SCIENTIFIC_KILL`.
Nothing in this cycle rescues COVI, changes its losses, or reuses its sealed
one-check or confirmatory records.

## Local Constraints From Prior Results

The next method must change the scientific axis. It must not be:

- another action-history, residual, EMA, chunk-boundary, or output-action
  repair;
- another future-action latent, waypoint, material-point, or context-gated
  label rescue;
- another scheduler, replanning, retained-frame, or action-queue variant;
- another static action mixture, uncertainty gate, or generic confidence head;
- another occlusion-view adapter, random-erasing defense, or COVI objective
  repair;
- a LoRA or QLoRA contribution in disguise.

For every candidate, the scientific method is separated from its low-compute
parameterization. Standard LoRA is included only when ordinary adaptation is a
plausible alternative explanation.

## Close Sources

### Counterfactual Action Guidance and LIBERO-CF

Full title: When Vision Overrides Language: Evaluating and Mitigating
Counterfactual Failures in VLAs.

Paper: https://arxiv.org/abs/2602.17659

Project link reported by the paper: https://vla-va.github.io/

AUTHOR_STATED:

- LIBERO-CF assigns alternative feasible instructions in visually plausible
  LIBERO layouts and measures both target grounding and task success.
- Counterfactual Action Guidance (CAG) combines a language-conditioned VLA
  action with a vision-only action. Its training-free branch drops language
  from the same policy; its stronger branch uses a separately trained
  vision-action model.
- The paper reports average gains of `9.7` percentage points in grounding and
  `3.6` points in success for training-free CAG, and `15.5` and `8.5` points
  respectively with the trained vision-action branch. It reports a `17.2`
  point average real-world success gain.
- CAG is deliberately architecture-agnostic and applies its guidance rule to
  policy actions after the two policies generate them.

LOCAL VERIFICATION:

- The installed SmolVLA implementation exposes the complete flow path in
  `SmolVLAModel.sample_actions`: language and image prefixes are cached, then a
  conditional vector field is integrated for `10` steps from shared noise.
- Empty-task tokenization is supported by the installed LeRobot processor, so
  conditioned and language-dropped branches can be evaluated without changing
  policy weights.
- A two-branch field-level comparison is implementable with the same backbone,
  observation, noise, number of denoising steps, and branch count as the local
  final-action CAG proxy.
- The paper's project URL did not expose downloadable code or LIBERO-CF assets
  during this audit, and no official GitHub repository was found. The local
  baseline must therefore be labeled a transparent training-free CAG
  implementation, not an official reproduction. Published results are context,
  not a direct numerical baseline.

CROSS_DOMAIN_SYNTHESIS:

- CAG applies language guidance after action generation. SmolVLA exposes a
  stronger mechanistic intervention point: guide the continuous action vector
  field at every integration step while both branches share the same latent
  state.
- Field-level guidance is classifier-free guidance transferred to a VLA action
  flow, not a claim to invent classifier-free guidance. The only defensible
  novelty is whether pathwise language transport improves VLA counterfactual
  control over final-action CAG under matched inference budget.

### Joint Learning With Motion Image Diffusion

Full title: Robotic VLA Benefits from Joint Learning with Motion Image
Diffusion.

Paper: https://arxiv.org/abs/2512.18007

Project: https://vla-motion.github.io/

AUTHOR_STATED:

- The method jointly trains a normal action-flow head and a DiT motion head that
  predicts optical-flow motion images from a shared VLM representation.
- The motion head is absent from the normal action inference path.
- It reports `97.5%` average LIBERO success for its stronger backbone, `58.0%`
  on RoboTwin, and a `23%` real-world improvement.
- The reported implementation uses a roughly `400M`-parameter motion head,
  DROID warm-up, full joint optimization, and eight H200 GPUs.

LOCAL IMPLICATION:

- Cached optical flow from local future frames is a viable training-only label,
  but a small SmolVLA proxy cannot be represented as an official reproduction
  of the large dual-head method.
- Ordinary data-matched LoRA is a mandatory optional-control choice here because
  the method updates policy weights and introduces extra supervision.
- This route is scientifically plausible but less decisive locally: a compact
  motion-token objective may simply act as generic auxiliary regularization.

### GeoPredict

Full title: GeoPredict: Leveraging Predictive Kinematics and 3D Gaussian
Geometry for Precise VLA Manipulation.

Paper: https://arxiv.org/abs/2512.16811

Project: https://jingjingqian75.github.io/GeoPredict-Page/

AUTHOR_STATED:

- GeoPredict uses history-conditioned 3D keypoint trajectory queries and a
  predictive 3D Gaussian workspace representation supervised by future depth.
- Expensive 3D decoding and rendering are training-only; lightweight query
  tokens remain at inference.
- It reports `96.5%` average LIBERO success versus `93.9%` for its reproduced
  base and positive results on RoboCasa Human-50 and real-world tasks.

LOCAL IMPLICATION:

- The positive prior strongly supports predictive geometry, but the local
  official LIBERO records do not directly provide the complete calibrated depth
  and 3D keypoint supervision used by GeoPredict.
- Simulator replay could generate these labels on discovery and validation
  identities, but prior G3P point-label failure and the multi-module supervision
  burden make this the least minimum-sufficient Cycle 15 route.
- Standard LoRA and a data-matched ordinary objective would be required because
  policy weights, additional labels, and training compute all change.

## Cycle 15 Opportunity

The best bounded opportunity is `LIFT-VLA`, Language-Induced Flow Transport for
SmolVLA.

Its scientific method is one inference-time mechanism. At every SmolVLA flow
step, evaluate the conditional and language-dropped vector fields from the same
latent state and shared observation, then integrate

`v_lift(x_t,t) = v_u(x_t,t) + omega * (v_c(x_t,t) - v_u(x_t,t))`.

The closest prior is training-free CAG:

`a_cag = a_u + omega * (a_c - a_u)`.

The two are not generally equivalent because SmolVLA's vector field is nonlinear
in the evolving latent action. LIFT changes the entire transport path; CAG mixes
two completed paths. This is a falsifiable mechanism difference under the same
two-branch, ten-step inference budget.

LIFT is inference-only. The SmolVLA backbone remains frozen, no LoRA or QLoRA is
used, no extra training data is introduced, and `omega = 1` recovers Base up to
floating-point tolerance. Standard LoRA is omitted because generic adaptation
does not test whether pathwise flow guidance is better than final-action CAG.

The first serious comparison is exactly four policies:

1. `frozen_smolvla`
2. `training_free_cag_proxy`
3. `lift_full_pathwise_guidance`
4. `lift_last_step_only_ablation`

No fifth policy is justified: final-action CAG is already the strongest simple
inference-time alternative and the last-step ablation tests whether transport
through the full path is necessary.

