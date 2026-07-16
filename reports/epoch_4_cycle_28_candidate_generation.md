# Epoch 4 Cycle 28 Candidate Generation

Date: 2026-07-16 KST

Previous method: `CFR-VLA`

Previous decision: `CFR_STAGE_0_NO_USABLE_HEADROOM`

Decision: `TSC_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING`

Cycle 28 generates exactly three candidates under the post-COVI
minimum-sufficient and post-RAC positive-prior governance. None repairs CFR.
Unknown empirical performance is not a rejection reason.

## Candidate 1: TSC-VLA

Full name: Temporal-Spatial masked action completion for continuous VLA chunks

Contribution type: `PRIOR_EXTENSION`

Closest prior: TS-Mask VLA, `https://arxiv.org/abs/2607.09818`.

Prior positive result: TS-Mask VLA reports a Discrete Diffusion Action Expert,
Bridge Attention conditioning, and 2D temporal-spatial masking over action
tokens, with `95.7%` average LIBERO success and CALVIN average sequence length
`4.19`.

Actual mechanism: learn a deployment-input action-cell error mask over the
Base decoded chunk `A_base in R^[50,7]`, then run a continuous masked action
completion field that changes only the selected time-dimension cells while
clamping the rest exactly to Base. The mechanism is targeted temporal-spatial
action completion, not LoRA, not full-chunk residual refinement, not adaptive
chunk length, and not confidence-only detection.

Minimal technical difference from prior: TS-Mask models discrete action-token
dependencies with temporal-spatial masking inside a native discrete diffusion
action expert. TSC-VLA adapts the same structural principle to an existing
continuous flow-matching VLA by using Base-clamped masked completion in the
continuous `[time, action-dimension]` grid. It learns from existing LIBERO
demonstrations without privileged inference inputs.

Falsifiable chain:

Base condition -> only a sparse subset of time-dimension action cells is
wrong, but full residual/refinement either overchanges the chunk or fails to
identify where correction is useful -> small structured errors in gripper,
rotation, or late translation cells persist -> closed-loop task failure.

TSC method -> a learned deployment-observable error mask selects likely
wrong cells and a masked completion field inpaints only those cells while
clamping all others to Base -> targeted chunk repair preserves clean behavior
while correcting sparse structured errors -> expected closed-loop improvement.

Data/supervision viability: existing LIBERO HDF5 demonstrations provide images,
language, proprioception, and expert action chunks. Existing SmolVLA produces
Base chunks. Discovery labels are Base-vs-demo per-cell error masks and masked
completion targets. No reward, success, simulator object pose, future
observation, or confirmatory reset identity is required at inference.

Identity-preserving integration: default mask is all-zero and default
completion delta is zero, so initial inference is exactly Base. Any LoRA or
adapter parameters are only implementation infrastructure for the mask and
completion field.

Decisive local experiment: Stage 0 checks label health for per-cell error
masks, noncollapsed positive/negative cells, prediction above trivial majority
and magnitude baselines, completion headroom over a transparent TS-Mask
continuous proxy, distinction from a no-targeted-mask ablation, bounded action
deltas, action validity, checkpoint reload, finite gradients, and clean
retention before any rollout.

Scores:

- provisional novelty: `23 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `91 / 100`

Reviewer-risk notes: the official TS-Mask code is not available from the arXiv
record during this pass, so policy 2 must be transparently labeled
`ts_mask_continuous_proxy` until an official implementation is locally
integrated. TSC must beat that proxy, the no-targeted-mask ablation, and one
simple baseline before paper viability.

## Candidate 2: SFR-VLA

Full name: Spectral Frequency Residual flow for VLA action chunks

Contribution type: `PRIOR_EXTENSION`

Closest prior: Frequency-Aware Flow Matching, `https://arxiv.org/abs/2606.20135`.

Prior positive result: FAFM reports DCT-domain flow matching, cosine-basis
reconstruction, and a Sobolev-type first-derivative constraint, improving
LIBERO success, smoothness, convergence, mixed-frequency robustness,
mechanical-bias robustness, and real Franka deployment.

Actual mechanism: learn a bounded frequency-domain residual over Base action
chunks. Instead of changing all timesteps directly, SFR decomposes Base-vs-demo
residuals into DCT bands, allows only selected low/mid-frequency coefficients
to move, and penalizes high-frequency derivative energy before reconstructing a
bounded action chunk.

Minimal technical difference from prior: FAFM trains flow matching natively in
frequency space. SFR-VLA is a lightweight identity-preserving residual wrapper
around a pretrained continuous VLA, using frequency-selective corrections
rather than a new full action generator.

Falsifiable chain:

Base condition -> action chunk errors have structured temporal frequency
content and direct time-domain residuals either overfit or introduce abrupt
changes -> unstable or biased execution -> task failure.

SFR method -> frequency-selective residuals suppress harmful high-frequency
changes while correcting useful low/mid-frequency drift -> smoother, valid
chunks with less clean disruption -> improved closed-loop behavior.

Data/supervision viability: DCT coefficients can be constructed directly from
existing LIBERO action chunks and Base predictions. No privileged inference
input is required. Risk: CFR Stage 0 already found poor direct residual
headroom, so SFR must show that frequency structure creates headroom rather
than merely reparameterizing a dead residual.

Identity-preserving integration: zero residual coefficients and a zero gate
produce exact Base behavior.

Decisive local experiment: Stage 0 checks spectral residual variance, whether
frequency-band probes beat time-domain residual and smoothing baselines,
whether derivative energy decreases without target loss collapse, and whether
actions remain valid.

Scores:

- provisional novelty: `20 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `86 / 100`

Reviewer-risk notes: if SFR only smooths Base actions, a moving-average or
Savitzky-Golay baseline will explain it. If it is only another residual after
CFR's no-headroom stop, it should be killed at Stage 0 rather than rescued.

## Candidate 3: LWM-VLA

Full name: Latent Woven Memory action conditioning for VLA chunks

Contribution type: `PRIOR_EXTENSION`

Closest prior: LaMem-VLA, `https://arxiv.org/abs/2607.07608`.

Prior positive result: LaMem-VLA reports short-term and long-term memory vaults,
retrieval, condensation into latent memory tokens, and injection of those
tokens into the VLA context, with reported superiority on SimplerEnv and
LIBERO.

Actual mechanism: build compact short-term and long-term latent action-memory
tokens from LIBERO demonstration histories and current deployment-observable
features, then inject them through a bounded identity-preserving adapter before
action chunk decoding.

Minimal technical difference from prior: LaMem-VLA is a native latent-memory
architecture. LWM-VLA would be a low-compute, frozen-backbone approximation
that distills demo history into small memory tokens without retraining the full
VLA.

Falsifiable chain:

Base condition -> current observation is locally ambiguous or short-horizon
biased -> policy lacks useful task-history context -> repeated or premature
actions cause failure.

LWM method -> relevant demonstration memory tokens provide phase and action
history evidence inside the policy context -> chunk generation better respects
long-horizon structure -> fewer temporal repetition and premature-transition
failures.

Data/supervision viability: existing LIBERO trajectories can supply memory
items. Risk is substantial: prior local MTF/RAP-style routes already tested
informative-frame and retrieval-adaptation ideas, and SmolVLA latent-context
injection may be harder than TSC or SFR.

Identity-preserving integration: zero memory gate and no retrieved token effect
produce exact Base behavior.

Decisive local experiment: Stage 0 checks whether memory retrieval labels are
noncollapsed, whether latent tokens predict held-out action/phase targets above
current-frame and nearest-neighbor baselines, whether injected memory changes
actions only in relevant states, and whether clean behavior is retained.

Scores:

- provisional novelty: `21 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `17 / 20`
- technical mechanism quality: `15 / 20`
- data/supervision feasibility: `7 / 10`
- decisive experiment feasibility: `7 / 10`
- total: `80 / 100`

Reviewer-risk notes: LWM is not selected because its local version risks
collapsing into another retrieval or memory-token adapter, and prior cycles
already exposed reviewer pressure around memory, retrieval, and milestone
selection.

## Selection

Selected candidate: `TSC-VLA`

Rationale: TSC-VLA has the best combination of recent positive primary-source
anchor, a single clear action-generation mechanism, local supervision from
existing LIBERO demonstrations, identity-preserving integration, and a bounded
Stage 0 audit. It also best satisfies the current design constraint: the
scientific mechanism is continuous temporal-spatial masked action completion,
LoRA is only infrastructure, and the closest prior enters the first serious
comparison.

Frozen first serious comparison order after Stage 0 and bounded validation:

1. `smolvla_base`
2. `ts_mask_continuous_proxy` or official `ts_mask_vla` if installed
3. `tsc_full`
4. `tsc_no_targeted_mask_ablation`
5. `standard_lora`

Next action: freeze TSC-VLA Researcher A proposal before Reviewer B attack,
mathematical audit, preregistration, or implementation.
