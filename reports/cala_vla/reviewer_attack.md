# CALA-VLA Reviewer B Attack

Date: 2026-07-15 KST

Reviewed frozen proposal: `reports/cala_vla/researcher_proposal.md`

Proposal hash: `5B3933C9C0FD5AE5F07FDB0CEC447B48040238FB6D872D97E545E3D93E257E76`

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Primary sources reviewed:

- CAC-VLA, https://arxiv.org/abs/2607.04816
- RotVLA, https://arxiv.org/abs/2605.13403
- LARA, https://arxiv.org/abs/2606.07100
- VLS, https://arxiv.org/abs/2602.03973
- World Pilot, https://arxiv.org/abs/2606.12403
- STRONG-VLA, https://arxiv.org/abs/2604.10055
- VLA Grounder, https://arxiv.org/abs/2607.04517

## Summary Ruling

Do not kill CALA-VLA before implementation. The closest prior is a strong
positive anchor: CAC-VLA directly supports latent-action conditioning through a
context gate as a useful interface between visual-language context and
continuous action experts on LIBERO and LIBERO-Plus.

However, CALA's novelty is narrow. CAC-VLA already contains the central
latent-action prediction plus context-gated action conditioning idea. RotVLA,
LARA, and villa-style latent-action work further reduce any broad claim that
latent actions are new for VLA control. CALA is viable only as a local
frozen-SmolVLA, identity-preserving, source-gated adaptation of CAC-style
conditioning, and only if Stage 0 proves that latent-action labels are legal,
noncollapsed, predictable from deployment inputs, and not explained by
task-mean or action-history shortcuts.

## Attack 1: Closest-Prior Duplication Risk

CAC-VLA already claims the main mechanism:

- encode future action segments into coarse-to-fine latent actions;
- predict those latent actions from visual-language context;
- condition the action expert through a context gate;
- report strong LIBERO and LIBERO-Plus success.

CALA cannot claim broad novelty for latent actions, action-tokenization,
coarse-to-fine action guidance, or context-gated action conditioning. The only
defensible novelty is local:

- frozen SmolVLA integration;
- exact or near-exact Base passthrough initialization;
- bounded hidden-state residual rather than unconditional final-action
  replacement;
- strict source gating that forbids future-action inference leakage;
- a fair local comparison against a CAC-style proxy, no-context-gate ablation,
  and task-mean latent-action baseline.

Required rebuttal:

- Explicitly disclaim broad latent-action and context-gating novelty.
- State that `cac_vla_latent_action_proxy` dominance kills or archives the
  local contribution.
- State that official CAC-VLA reproduction is not claimed unless official
  code/checkpoints and protocol equivalence are later verified.

## Attack 2: Latent-Action Prior Art Narrows The Claim Further

RotVLA reports continuous rotational latent actions that guide flow-matching
action generation and strong LIBERO/RoboTwin results. LARA reports alignment
between latent action models and VLAs, with positive improvements across
simulation and real-world manipulation. These priors make "latent action helps
VLA" a crowded claim.

CALA may proceed only if its contribution is framed as:

- a CAC-anchored local SmolVLA adaptation;
- a source-gated training-label/inference separation mechanism;
- a Base-preserving context gate on the local flow action interface;
- an honest matched experiment showing whether this adaptation beats its
  closest proxy, key ablation, and task-mean baseline.

Any paper claim must avoid implying that CALA invented latent actions, latent
planners, action-tokenization, or context-gated action conditioning.

## Attack 3: Future-Action Leakage Is The Central Failure Mode

CALA trains from future 7D action segments. In LIBERO demonstrations, it is
easy to accidentally leak deployment-invalid information through sample index,
episode progress, reset identity, task name, success label, hidden HDF5 future
actions, or cached labels.

Hard requirements:

- Stage 0 must list every source used to build `A_{t:t+H-1}`, `z_t`,
  `zhat_t`, Base features, action previews, and gates.
- Confirmatory inference may use only current deployment RGB, proprioception,
  language, and Base features available through the official runner.
- Future HDF5 actions, latent labels, future observations, simulator state,
  reward, success flags, reset identity, and manifest metadata are forbidden at
  inference.
- The final runner must contain a source gate proving no future-action or
  privileged fields are accessed.

Reject before rollout as `DATA_OR_SUPERVISION_FAILURE` or
`DESIGN_FAILURE` if a deployment-observable latent predictor cannot be built.

## Attack 4: Task Mean Or Action History May Explain The Method

The proposal's simple reviewer-killer is correct but must be kept alive. A
large fraction of LIBERO behavior can be task- or phase-regular. If a task-mean
latent prototype or action-history-only predictor explains the same behavior,
the context-gated latent-action claim is not supported.

Required constraints:

- Keep `task_mean_latent_action_baseline` as the one mandatory simple policy
  killer in the first five-policy comparison.
- Stage 0 latent predictability must compare against task-mean, phase-only,
  action-only/action-history, and majority/trivial predictors as diagnostics.
- Do not add a second mandatory simple policy baseline before the first
  comparison unless a concrete Reviewer B objection cannot be resolved by
  Stage 0 diagnostics.
- If task-mean latent actions match CALA in validation/rollout, CALA must be
  killed or archived as explained by a simple task prior.

## Attack 5: Local CAC Proxy Could Be Unfair

No official CAC-VLA code or checkpoint is verified locally. A weak proxy could
inflate CALA, and an overpowered proxy could hide a useful local safety
extension.

Hard requirements:

- Label `cac_vla_latent_action_proxy` as a faithful transparent local proxy,
  not an official reproduction.
- Use the same data partitions, latent encoder, inference input restrictions,
  and comparable inference budget for proxy and full method.
- Document the exact technical difference between proxy and CALA full.
- If official code/checkpoints become available later, official-equivalence
  claims require a separate source/protocol equivalence audit before
  confirmatory testing.

## Attack 6: SmolVLA Integration May Not Expose A Compatible Hook

CAC-VLA conditions a VLA action expert in its own architecture. Local SmolVLA
may not expose the same hidden states, action-expert interface, or gradient
path. A final-action residual would collapse CALA into a MARC-like correction
method, and a global hidden update could damage Base.

Hard requirements:

- Stage 0 or the mathematical audit must identify the exact SmolVLA hook:
  feature tensor, hidden state, action preview, or adapter input.
- CALA full must not be implemented as an unconditional final 7D action
  residual.
- Initial CALA behavior must be exact or near-exact Base passthrough.
- Translation, rotation, and gripper deltas must be bounded and reported
  separately.
- Intended parameters must receive finite nonzero gradients while unintended
  frozen parameters remain unchanged.
- Disk reload must preserve the same policy identity and action-delta
  statistics before rollout.

Reject before rollout as `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` if the only
available implementation is a destructive final-action wrapper or a globally
active hidden-state perturbation.

## Attack 7: Latent Encoder Must Not Be Decorative

An OAT-lite latent encoder can become a decorative DCT/PCA transform with no
mechanistic value. If latent targets collapse to task identity, timestep, or
mean action, CALA has no independent mechanism.

Stage 0 must report:

- latent dimensionality and horizon;
- per-dimension variance and explained-variance concentration;
- high/low contrast counts or noncollapsed cluster occupancy;
- task and phase coverage;
- duplicate sample/frame counts;
- train/validation/test overlap;
- predictability from deployment inputs above trivial baselines;
- whether latent labels differ materially from task-mean or action-history
  prototypes.

Reject before rollout if the latent encoder is collapsed, not predictable, or
only encodes a trivial task/phase index.

## Attack 8: Key Ablation Must Be Matched

The no-context-gate ablation must not be a weak strawman. It should keep the
same latent labels, training records, Base policy, data partitions, and
comparable parameter budget while removing or disabling the context-dependent
gate.

Required reporting:

- full-versus-ablation latent predictions;
- gate values by task/phase;
- full-versus-ablation action L2 by translation, rotation, and gripper;
- contexts where full activates and ablation does not;
- whether full and ablation differ on validation before rollout.

If full and ablation are action-equivalent on validation, stop as exact trivial
equivalence before rollout.

## Attack 9: Mathematical Objective Risks

The mathematical audit must define actual tensors and gradients. Do not add
ornamental KL, mutual information, or entropy terms. Do not compute KL directly
between deterministic 7D actions.

Required definitions:

- image tensor, proprioception tensor, language embedding, Base feature tensor,
  future action segment tensor, latent action tensor, predicted latent tensor,
  gate tensor, Base action chunk, and adapted action chunk shapes;
- deterministic latent encoder formula and whether it is PCA, DCT, k-means,
  vector quantization, or a learned encoder;
- latent prediction loss, gate formula, hidden residual formula, action
  imitation/retention loss, and bounded action-delta penalty;
- objective magnitudes and gradient norms on a small batch;
- simpler alternative objective and required ablation;
- whether the latent encoder, predictor, and adapter are trained jointly,
  frozen, or generated offline.

## Required First Comparison

The first serious comparison must remain exactly:

1. `frozen_smolvla`
2. `cac_vla_latent_action_proxy`
3. `cala_full`
4. `cala_no_context_gate_ablation`
5. `task_mean_latent_action_baseline`

Additional policy baselines may not precede this comparison unless Stage 0
exposes a concrete implementation ambiguity that would otherwise invalidate the
five-policy test.

## Required Stage 0 Stop Rules

Stage 0 must stop before training search or rollout for any of:

- latent labels are unavailable, collapsed, duplicated, or split-leaking;
- latent prediction is not observable above task-mean, phase-only,
  action-history, or majority/trivial baselines;
- diagnostic headroom shows no usable latent-action benefit;
- train/validation/test or reset identity separation fails;
- source gate detects future-action or privileged inference access;
- SmolVLA integration hook cannot preserve Base behavior;
- intended parameters receive no finite nonzero gradients;
- action deltas are global or unbounded;
- full, proxy, ablation, or simple baseline are action-equivalent in a way that
  invalidates the frozen comparison.

## Reviewer B Decision

`REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

CALA may proceed to Researcher A rebuttal because it has a strong positive
external prior, changes the mechanism axis relative to G3P, and has locally
available action-segment supervision. It cannot proceed to mathematical audit,
implementation, Stage 0, validation search, training, or rollout until
Researcher A accepts the narrowed novelty, source-fidelity requirements,
future-action leakage gate, simple-baseline constraints, and
identity-preserving integration requirements above.
