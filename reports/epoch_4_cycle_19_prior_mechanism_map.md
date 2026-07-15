# Epoch 4 Cycle 19 Prior And Mechanism Map

Date: 2026-07-15 KST

Decision: `CYCLE_19_PRIOR_MAP_COMPLETE`

This map was completed before selecting or implementing a Cycle 19 method.
PCAV-VLA remains closed as `PCAV_STAGE_0A_NO_USABLE_HEADROOM`; no candidate
below changes its noise, generator, support threshold, progress model, tasks,
or Stage 0A evidence.

## Campaign Constraints

The next method must satisfy five empirical constraints from the campaign:

1. It may not depend on frozen SmolVLA multi-noise candidates being useful.
   PCAV found only `7 / 96` rows with a material candidate improvement.
2. It must avoid globally replacing Base actions before proving action
   validity. IARC and FAMR did not reach a fair scientific test after their
   learned action mechanisms failed implementation or validity gates.
3. It must use deployment-observable signals. G3P, CALA, and RAR showed that a
   plausible label is not useful when it cannot be inferred from legal inputs.
4. It must not revive CAVM's action memory or FANG's learned 7D action field.
5. It must enter the closest positive external prior in the first serious
   comparison and preserve discovery, validation, and confirmatory identities.

The previously highlighted `libero_10/task_4` and `libero_spatial/task_4`
failures are not used as the new problem claim. Quantized OpenVLA-OFT INT4 had
already solved those bounded identities, so centering another method on them
would overfit a SmolVLA-specific weakness.

## Prior 1: COAST

Primary source: https://arxiv.org/abs/2605.17144

Full paper: https://arxiv.org/html/2605.17144

Title: Contrastive Conceptor Activation Steering: Unlocking Vision-Language-
Action Models through Hidden States.

Positive result:

- more than `20` absolute points mean simulation improvement and more than
  `40` points mean real-robot improvement across flow-matching VLA,
  autoregressive VLA, and Diffusion Policy families;
- strict fitting/evaluation separation with `15` fitting rollouts and `30`
  held-out evaluation rollouts per task;
- gains over Contrastive Activation Addition, sparse-autoencoder steering,
  and same-trajectory LoRA SFT;
- cross-task transfer when source and target failure subspaces overlap.

Reproducible mechanism:

1. capture one action-expert residual-stream activation vector per denoising
   step after token mean pooling;
2. mean-center success and failure activation matrices separately;
3. construct the closed-form conceptor
   `C = R (R + alpha^-2 I)^-1`;
4. compose `C_success AND NOT C_failure` with Moore-Penrose inverses;
5. apply `M = (1 - beta) I + beta C` as a multiplicative residual-stream
   gate at every action-generation pass.

Important demonstrated geometry:

- outcome-relevant computation is low-rank but not rank-one;
- success geometry is substantially task-specific;
- failure geometry can be shared across tasks;
- failure-subspace containment correlates with transfer gain;
- a source task's full contrastive conceptor can transfer to a target task,
  but the source success subspace is still carried into that gate.

Open limitation used by Cycle 19:

COAST self-fitting requires target successes and target failures. Its transfer
experiment reuses a complete source contrastive conceptor even though the
paper's own geometry says success is task-specific while failure is the more
transferable part. It does not test a task-balanced multi-source failure
conceptor composed with a target-specific success conceptor.

Local compatibility:

- frozen SmolVLA has a 16-layer action expert with width `720`;
- `policy.model.vlm_with_expert.lm_expert.layers[layer].mlp` is invoked at
  every denoising step and supports capture/replacement through a forward hook;
- the intervention is closed-form and changes no Base weight;
- exact Base behavior is available at `beta = 0`.

## Prior 2: Latent Policy Steering

Primary source: https://arxiv.org/abs/2603.05296

Project: https://jellyho.github.io/LPS/

Official code: https://github.com/jellyho/LPS

Positive result:

LPS reports `56.2%` real-robot success on its DROID setting, compared with
`31.2%` for Flow-BC, `28.7%` for MF-BC, and `35.0%` for DSRL. It converts the
base policy to one-step MeanFlow, learns an original-action-space critic, and
backpropagates the critic gradient into a spherical latent actor.

Open limitation used by Cycle 19:

LPS does not make Base identity the default policy. A locally attractive
extension would use a zero-initialized tangent residual around the Base latent
and a validation-frozen trust radius. The local campaign, however, does not
currently contain the rewarded transition buffer and validated one-step
MeanFlow conversion needed for a faithful decisive test.

## Prior 3: World Pilot

Primary source: https://arxiv.org/abs/2606.12403

Project: https://world-pilot.github.io/

Official code: https://github.com/ZefuLin/WorldPilot

Positive result:

World Pilot reports `84.7%` on LIBERO-Plus and strong real-robot performance.
Its world-action model supplies a scene-evolution latent and anticipated
trajectory token to the VLA rather than merely fitting an output action
residual.

Open limitation used by Cycle 19:

The published mechanism is not explicitly Base-preserving when its world
prior is uncertain. A plausible extension is a zero-initialized confidence
gate around world-action token injection. Locally, this requires a compatible
world-action checkpoint or a substantial new pretraining/distillation stage,
so the first decisive experiment is less bounded than conceptor steering.

## Additional Priors Screened

- ZPRL, https://arxiv.org/abs/2605.19919, official code at
  https://github.com/manutdmoon/ZPRL: positive online-RL residual control in a
  bottleneck latent, but it requires a new online-RL phase and overlaps the
  latent-residual axis of the LPS candidate.
- Self-Correcting VLA, https://arxiv.org/abs/2602.21633: progress and future
  heads plus residual RL improve task success, but the campaign has already
  tested several progress/recovery routes and lacks a stronger local
  differentiator than the COAST gap.
- Action Draft and Verify, https://arxiv.org/abs/2603.18091: strong positive
  sampling-and-reranking evidence, but PCAV's frozen candidate audit directly
  found insufficient local candidate headroom. Reusing an adapted generator
  would be a different method and is not selected here.
- Mostly Harmless VLA, https://arxiv.org/abs/2606.12299: positive language-
  feedback steering with conformal acceptance, but it needs iterative prompt
  search and is closer to already-exhausted language/task steering axes.

## Mechanism Gap Selected For Candidate Generation

The highest-value local gap is not generic activation steering. It is the
specific mismatch exposed by COAST itself:

`task-specific success geometry + cross-task failure geometry`

yet the prior's cross-task intervention transfers:

`source success geometry + source failure geometry`.

Cycle 19 therefore evaluates whether a target-success conceptor should be
combined with a task-balanced multi-source failure conceptor. This remains a
provisional novelty claim until independent Reviewer B analysis.
