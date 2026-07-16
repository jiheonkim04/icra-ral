# Epoch 4 Cycle 28 Prior Mechanism Map

Date: 2026-07-16 KST

Previous method: `CFR-VLA`

Previous decision: `CFR_STAGE_0_NO_USABLE_HEADROOM`

CFR is not repaired or rescued. Cycle 28 starts from fresh primary-source
anchors whose positive claims are outside CFR's failed continuous full-chunk
refinement headroom path. The design constraint remains sharp: one genuinely
new mechanism, LoRA only as implementation infrastructure, and the closest
prior or a faithful transparent proxy must enter the first serious comparison.

## Primary-Source Anchors Checked

### TS-Mask VLA

Primary source: `https://arxiv.org/abs/2607.09818`

Positive result: TS-Mask VLA reports a Discrete Diffusion Action Expert with
Bridge Attention and a 2D temporal-spatial masking strategy over action tokens.
The arXiv abstract reports `95.7%` average LIBERO success with a `0.5B`
parameter model and CALVIN average sequence length `4.19`.

Relevant mechanism: mask-and-complete action generation over the joint
time-by-action-token structure, rather than independent next-token prediction
or post-hoc smoothing.

Local extension opportunity: transfer the 2D temporal-spatial masking principle
from discrete action tokens to continuous SmolVLA `[50,7]` chunks by clamping
most Base action cells and completing only a bounded predicted error mask. The
local method can train from existing LIBERO demonstrations and deployment
inputs without simulator state, reward, object pose, or confirmatory resets.

### Frequency-Aware Flow Matching

Primary source: `https://arxiv.org/abs/2606.20135`

Positive result: FAFM transforms discrete action sequences into DCT frequency
coefficients, performs flow matching in that frequency domain, reconstructs
continuous actions through cosine basis expansion, and adds a first-derivative
Sobolev-style regularizer. The abstract reports improvements across LIBERO,
mixed-frequency input, robustness to mechanical bias, motion smoothness, and a
real Franka robot deployment.

Relevant mechanism: frequency-domain action representation and temporal
smoothness regularization for flow-matching policies.

Local extension opportunity: learn a bounded frequency-domain residual around
Base chunks rather than another direct time-domain residual. This is distinct
from CFR because it does not iteratively refine the full chunk in action space;
it constrains which temporal frequencies can be altered.

### Guided Action Flow

Primary source: `https://arxiv.org/abs/2607.02092`

Positive result: Guided Action Flow keeps a pretrained SmolVLA policy frozen
and uses a learned action-chunk critic to guide the reverse-time flow sampler.
The abstract reports single-task gains from `68.0%` to `82.0%` and from
`82.0%` to `86.0%`, a validation gain from `46.0%` to `56.0%`, and a locked
held-out test gain from `65.0%` to `67.5%`.

Relevant mechanism: inference-time action-gradient guidance through a frozen
flow-matching VLA sampler.

Local extension opportunity: potentially combine a critic with deployment-time
uncertainty gating, but this is not selected for Cycle 28 because the closest
prior depends on success/failure rollout labels. The user's current design
constraint favors mechanisms reproducible from existing LIBERO demonstrations
before additional closed-loop data collection.

### Perturbation-Based Uncertainty For VLA Failure Detection

Primary source: `https://arxiv.org/abs/2606.20754`

Positive result: this paper injects Gaussian perturbations into transformer
hidden activations and estimates epistemic uncertainty from disagreement across
perturbed action predictions. The abstract reports stronger failure detection
on LIBERO and LIBERO-PRO than sampling-based uncertainty under distribution
shift.

Relevant mechanism: label-free hidden-activation perturbation disagreement as
an uncertainty signal.

Local extension opportunity: useful as a diagnostic or future gate, but not
sufficient as the core Cycle 28 method because failure detection alone does not
change action generation and prior campaign governance rejects confidence-only
methods.

### LaMem-VLA

Primary source: `https://arxiv.org/abs/2607.07608`

Positive result: LaMem-VLA reconstructs historical experience into short-term
and long-term latent memory tokens and interweaves them with VLA reasoning. The
abstract reports superiority on SimplerEnv and LIBERO.

Relevant mechanism: memory-native latent tokens inside the VLA context rather
than external retrieval attached after multimodal reasoning.

Local extension opportunity: construct compact action-history/demo memory
tokens from existing LIBERO trajectories. This is lower priority because prior
local cycles already tested milestone/retention and retrieval-style action
adaptation routes; a new memory method would need very crisp novelty and
nontrivial integration access to SmolVLA latent context.

## Cycle 28 Design Constraint

The next method must use one genuinely new action-generation mechanism. LoRA
may only be identity-preserving implementation infrastructure. The closest
external prior or a faithful transparent proxy must appear in the first serious
comparison.

## Map Decision

Proceed to exactly three Cycle 28 candidates:

1. `TSC-VLA`: Temporal-Spatial masked action completion anchored to TS-Mask VLA.
2. `SFR-VLA`: Spectral frequency residual flow anchored to FAFM.
3. `LWM-VLA`: Latent woven memory action conditioning anchored to LaMem-VLA.

Prefer `TSC-VLA` if candidate scoring confirms that it has the strongest
combination of current positive prior, one non-LoRA mechanism, local
supervision from existing LIBERO demonstrations, identity-preserving
integration, and a decisive Stage 0 audit.
