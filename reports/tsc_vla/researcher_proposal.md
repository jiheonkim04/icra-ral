# TSC-VLA Researcher A Proposal

Date: 2026-07-16 KST

Method: `TSC-VLA`, Temporal-Spatial masked action completion for continuous VLA
chunks.

Decision: `TSC_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

This proposal begins after CFR-VLA is closed as
`CFR_STAGE_0_NO_USABLE_HEADROOM`. It does not repair, rescue, retune, or
reinterpret CFR.

## Research Claim

TSC-VLA tests whether a pretrained continuous VLA's decoded action chunk can be
improved by changing only a sparse, deployment-observable subset of
time-dimension action cells while clamping all other cells exactly to the Base
policy.

The scientific mechanism is continuous temporal-spatial masked action
completion. LoRA, QLoRA, or a lightweight adapter may only be used as
identity-preserving implementation infrastructure.

## Closest Prior

Closest external prior: TS-Mask VLA, `https://arxiv.org/abs/2607.09818`.

Positive prior result: TS-Mask VLA reports a Discrete Diffusion Action Expert,
Bridge Attention conditioning, and a 2D temporal-spatial masking strategy over
action tokens. Its arXiv abstract reports `95.7%` average LIBERO success with a
`0.5B` parameter model and CALVIN average sequence length `4.19`.

TSC-VLA extends the same temporal-spatial action-structure principle to an
existing continuous flow-matching SmolVLA policy. Instead of training a native
discrete diffusion action expert, TSC starts from the Base continuous decoded
chunk and performs Base-clamped masked completion in the `[time, action_dim]`
grid.

## Mechanism

Let:

- `H = 50`: SmolVLA action chunk horizon.
- `D = 7`: LIBERO action dimension.
- `x_t`: deployment observation tuple of RGB streams, proprioception, and
  language instruction available to SmolVLA at time `t`.
- `A_B(x_t) in R^[H,D]`: frozen Base SmolVLA decoded action chunk after the
  official postprocessor.
- `A_E in R^[H,D]`: demonstration expert action chunk from the same training or
  validation frame.
- `M_theta(x_t, A_B) in [0,1]^[H,D]`: predicted sparse action-cell error mask.
- `Delta_phi(x_t, A_B, M) in R^[H,D]`: masked completion field.
- `alpha in [0,1]`: bounded correction scale selected on validation only.

The action chunk before execution is:

`A_TSC = (1 - M) * A_B + M * ProjectOfficial(A_B + alpha * Delta_phi)`.

`ProjectOfficial` means the same official action-postprocessing semantics used
by SmolVLA. TSC must not introduce ad hoc numeric action bounds that are not
exposed by the official policy stack.

The initial policy is exactly Base:

- mask logits initialized below the activation threshold;
- completion head initialized to zero effect;
- `alpha = 0` until validation selects a nonzero configuration;
- unselected cells copied from Base without arithmetic modification.

## Supervision

Development supervision uses only existing official LIBERO demonstrations and
frozen Base SmolVLA predictions:

1. Run frozen Base on discovery and validation demonstration frames to cache
   `A_B`.
2. Align each cached chunk with the corresponding expert chunk `A_E`.
3. Construct development-only sparse error labels from Base-vs-expert
   per-cell residuals using thresholds frozen before confirmatory rollout.
4. Train the mask predictor to identify likely incorrect time-dimension cells.
5. Train the completion field only on masked cells, with clean-retention loss
   on unmasked cells.

No reward, success label, simulator object pose, reset identity, future
observation, privileged state, or confirmatory-test outcome may enter training
or inference.

## Falsifiable Mechanism Chain

Problem chain:

Base condition -> a sparse subset of chunk cells is wrong while most cells are
already useful -> full-chunk residual/refinement either overchanges clean cells
or lacks headroom -> gripper/rotation/late-translation errors persist ->
closed-loop task failure.

Method chain:

TSC predicts where the Base chunk is likely wrong -> completion updates only
those cells while clamping the rest -> useful sparse corrections appear without
global policy disruption -> action validity and clean behavior are retained ->
closed-loop success can improve.

## First Serious Comparison

The first serious comparison must use exactly these policy families unless a
Reviewer B objection requires a cheaper additional control:

1. `smolvla_base`
2. `ts_mask_continuous_proxy` or official `ts_mask_vla` if installed
3. `tsc_full`
4. `tsc_no_targeted_mask_ablation`
5. `standard_lora`

`ts_mask_continuous_proxy` is a transparent local proxy for temporal-spatial
masked action modeling without TSC's Base-error targeted mask. It must not be
labeled as official TS-Mask VLA unless official code/checkpoints are locally
integrated and verified.

`standard_lora` is required because TSC uses trainable lightweight policy
infrastructure; ordinary adaptation remains a relevant simple reviewer-killer.

## Required Ablations And Baselines

Key ablation: `tsc_no_targeted_mask_ablation`, which performs continuous
completion with a non-targeted or uniform mask schedule while preserving the
same compute and parameterization where possible.

Closest-prior proxy: `ts_mask_continuous_proxy`, which tests whether ordinary
temporal-spatial masked modeling explains the gain without the
Base-error-targeted mechanism.

Simple reviewer-killer baseline: matched `standard_lora`, plus a
non-trainable smoothing or per-dimension residual diagnostic during Stage 0
when it is cheaper than rollout and can expose trivial equivalence.

## Stage 0 Development Audit

Before bounded validation search or rollout, Stage 0 must verify:

- discovery/validation split separation and no confirmatory identities;
- Base-vs-expert residual alignment and finite `[50,7]` chunks;
- per-cell error labels are noncollapsed;
- positive/negative mask counts across tasks and phases;
- mask predictor beats trivial-majority and magnitude-only baselines;
- completion improves masked-cell validation loss over prior proxy and
  no-targeted-mask ablation;
- unmasked cells remain exactly or nearly Base;
- action deltas are bounded and sparse rather than globally destructive;
- official action validity is preserved;
- checkpoint save/reload works;
- expected parameters receive finite nonzero gradients;
- no privileged inference inputs are used.

Stage 0 stop classes:

- `TSC_STAGE_0_DATA_OR_SUPERVISION_FAILURE`
- `TSC_STAGE_0_NO_USABLE_HEADROOM`
- `TSC_STAGE_0_DESIGN_FAILURE`
- `TSC_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`
- `TSC_STAGE_0_PASS_TO_BOUNDED_VALIDATION`

If labels collapse, completion does not beat simple baselines, the mask is
nonacting, or the method changes nearly every cell destructively, stop before
rollout and classify the failure correctly.

## Bounded Validation Search

If Stage 0 passes, validation search is limited to at most six configurations.
Possible factors:

- mask positive-rate target or threshold;
- `alpha` correction scale;
- completion latent dimension;
- clean-retention coefficient;
- one lightweight architecture choice for the mask/completion module.

Selection score must combine validation closed-loop success if already
authorized, otherwise the closest feasible proxy, clean retention, mechanism
activation, action validity, and compute overhead. Offline action L2 alone is
not a valid selection score.

## Confirmatory Discipline

After selecting one configuration:

- freeze method, configuration, policies, ablation, task/reset identities,
  metrics, and thresholds;
- save all tried configurations and negative results;
- do not tune on confirmatory outcomes;
- treat any major redesign after confirmatory test as a new method cycle.

## Researcher A Position

TSC-VLA is worth advancing to Reviewer B because it is a fresh mechanism family
relative to CFR. CFR tried continuous full-chunk iterative refinement and found
no usable headroom. TSC instead assumes errors are sparse over the temporal
action-dimension grid and tests whether targeted masked completion can improve
without disturbing the rest of the pretrained policy.

The method is risky but falsifiable: if error masks are collapsed, completion
does not beat the TS-Mask proxy and no-targeted-mask ablation, or clean
behavior is not retained, the method should stop before rollout.
