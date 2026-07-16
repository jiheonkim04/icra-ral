# Epoch 4 Cycle 27 Candidate Generation

Date: 2026-07-16 KST

Previous method: `AMP-VLA`

Previous decision: `AMP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`

Decision: `CFR_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING`

Cycle 27 generates exactly three candidates under the post-RAC/post-COVI
performance-oriented governance. None repairs RAP or AMP. Unknown empirical
performance is not a rejection reason.

## Candidate 1: CFR-VLA

Full name: Continuous Full-Chunk Refinement for VLA action-flow decoding

Contribution type: `PRIOR_EXTENSION`

Closest prior: DFM-VLA, `https://arxiv.org/html/2603.26320v1`, project page
`https://chris1220313648.github.io/DFM-VLA/`.

Prior positive result: DFM-VLA reports iterative full-sequence action-token
refinement via discrete flow matching, `95.7%` LIBERO average, `4.44` CALVIN
average success length, and `70.8%` real-world average on its project page.

Actual mechanism: learn a continuous residual velocity/refinement field over a
full `[50,7]` SmolVLA action chunk. Starting from a Base decoded chunk, CFR
performs a fixed small number of refinement steps that may update every action
dimension at every future timestep, then applies a deterministic validation
projection/consistency check. The mechanism is iterative full-chunk correction,
not LoRA, not chunk-size selection, and not action-manifold projection.

Minimal technical difference from prior: DFM-VLA refines discrete action tokens
using token probability velocity. CFR-VLA refines continuous action chunks
using a bounded vector field around an existing flow-matching VLA policy. It is
trained and audited from existing LIBERO demonstrations without privileged
inference inputs.

Falsifiable chain:

Base condition -> early chunk errors or inconsistent later chunk elements
cannot be corrected after the chunk is decoded -> action sequence contains
persistent translation/rotation/gripper drift -> closed-loop task failure.

CFR method -> continuous refinement field observes deployment inputs plus Base
chunk and predicts bounded iterative corrections -> later chunk elements can be
changed before execution -> smoother and more target-consistent closed-loop
behavior.

Data/supervision viability: existing LIBERO HDF5 demonstrations provide current
images, proprioception, language, and action chunks. Training targets are
Base-chunk-to-demo residual/refinement directions generated on discovery data;
validation uses held-out demos only. No reward, success, done, object pose, or
confirmatory reset identity is required.

Identity-preserving integration: initialize the residual field/gate to zero so
initial behavior equals Base. LoRA may parameterize the adapter, but the method
claim is the iterative continuous refinement objective and decoding procedure.

Decisive local experiment: Stage 0 checks whether Base chunks leave residual
headroom, whether a deployment-input refinement probe predicts demo residuals
better than a DFM-style transparent proxy and no-refinement ablation, whether
corrections are bounded/action-valid, and whether identity/reload/gradients are
healthy before any rollout.

Scores:

- provisional novelty: `23 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `92 / 100`

Reviewer-risk notes: official DFM-VLA code is not yet available, so policy 2
must be transparently labeled `dfm_vla_continuous_refinement_proxy` until an
official release is locally installed. CFR must beat the proxy, the
no-refinement ablation, and one simple baseline before paper viability.

## Candidate 2: EOC-VLA

Full name: Entropy-gated Overlap Composition for VLA action execution

Contribution type: `PRIOR_EXTENSION`

Closest prior: Adaptive Action Chunking, `https://arxiv.org/html/2604.04161v2`,
project/code page `https://lance-lot.github.io/adaptive-chunking.github.io/`.

Prior positive result: AAC reports entropy-based inference-time chunk-size
selection that improves simulated and real-world manipulation by balancing
reactivity and temporal consistency.

Actual mechanism: instead of selecting only how many actions to execute, EOC
would compose overlapping predicted chunks using entropy-conditioned temporal
weights, continuity penalties, and a bounded change detector. It changes the
executed action sequence by fusing multiple overlapping chunk predictions; it
is not merely adaptive chunk length.

Minimal technical difference from prior: AAC gates execution horizon; EOC gates
continuous overlap composition and continuity-preserving action blending.

Falsifiable chain:

Base condition -> adjacent chunk predictions disagree near boundaries -> fixed
execution creates mode jumps or delayed corrections -> task failure.

EOC method -> entropy and overlap disagreement select smooth per-timestep
mixture weights -> executed actions preserve low-entropy consistency while
reacting to high-entropy states -> fewer boundary failures.

Data/supervision viability: existing SmolVLA can generate overlapping chunks on
LIBERO frames. Demonstrations provide action targets. However, entropy for
continuous SmolVLA chunks is a proxy from stochastic flow samples rather than a
native calibrated probability, increasing measurement risk.

Identity-preserving integration: default weights reproduce current SmolVLA
execution; learned composer initialized to Base passthrough.

Decisive local experiment: Stage 0 checks overlap disagreement headroom,
entropy reliability above trivial baselines, boundary-delta reduction, action
validity, and clean retention.

Scores:

- provisional novelty: `18 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `17 / 20`
- technical mechanism quality: `14 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `78 / 100`

Reviewer-risk notes: governance already warns against a contribution that is
only adaptive chunk size, so EOC is acceptable only if the overlap-composition
mechanism, not horizon selection, carries the result.

## Candidate 3: LGR-VLA

Full name: Lie-Group Residual action representation for VLA chunk adaptation

Contribution type: `CROSS_DOMAIN_MECHANISM_TRANSFER`

Closest prior: RotVLA, `https://arxiv.org/abs/2605.13403`.

Prior positive result: RotVLA reports continuous rotational latent actions on
`SO(n)`, triplet-frame learning, a latent planner coupled to a flow-matching
action head, `98.2%` LIBERO, and strong RoboTwin2.0 clean/randomized results.

Actual mechanism: represent local orientation/translation residuals through a
Lie-algebra action chart with composition consistency, then train an
identity-gated residual adapter to predict bounded corrections in the structured
chart before mapping back to 7D LIBERO actions.

Minimal technical difference from prior: RotVLA uses large-scale continuous
rotational latent action pretraining. LGR-VLA would be a lightweight local
structured residual chart for an existing SmolVLA policy, learned from LIBERO
demonstrations.

Falsifiable chain:

Base condition -> Euclidean residuals mix rotation, translation, and gripper
changes in a poorly structured space -> small orientation errors compound ->
closed-loop manipulation fails.

LGR method -> residuals are learned in a composition-aware chart -> corrections
respect local action geometry -> orientation/translation consistency improves.

Data/supervision viability: LIBERO demonstrations provide 7D action chunks and
proprioception. The method can construct local charts without privileged
inference. However, RotVLA's strongest evidence depends on large-scale latent
pretraining and human videos, which are not locally reproduced.

Identity-preserving integration: zero chart residual and gate produce exact
Base behavior at initialization.

Decisive local experiment: Stage 0 checks whether chart residuals are
predictable above Euclidean residual baselines, whether composition consistency
improves validation action targets, whether action validity is retained, and
whether the method beats a simple SE(3)/Euclidean residual baseline.

Scores:

- provisional novelty: `21 / 25`
- importance of problem: `12 / 15`
- strength of positive prior anchor: `15 / 20`
- technical mechanism quality: `16 / 20`
- data/supervision feasibility: `7 / 10`
- decisive experiment feasibility: `7 / 10`
- total: `78 / 100`

Reviewer-risk notes: without official RotVLA code/assets, this is a mechanism
transfer rather than a faithful reproduction; the prior comparison would need a
transparent local proxy.

## Selection

Selected candidate: `CFR-VLA`

Rationale: CFR has the strongest current positive prior, a new mechanism that
is not AMP/RAP/chunk-size-only, direct local supervision from existing LIBERO
demonstrations, identity-preserving integration, and a bounded Stage 0 audit.
It also best satisfies the user's sharpened constraint: one genuinely new
mechanism, LoRA only as infrastructure, and the closest prior entering the first
serious comparison.

Frozen first serious comparison order after Stage 0 and bounded validation:

1. `smolvla_base`
2. `dfm_vla_continuous_refinement_proxy` or official `dfm_vla` if installed
3. `cfr_full`
4. `cfr_no_iterative_refinement`
5. `standard_lora`

Next action: freeze CFR-VLA Researcher A proposal before Reviewer B attack,
mathematical audit, preregistration, or implementation.
