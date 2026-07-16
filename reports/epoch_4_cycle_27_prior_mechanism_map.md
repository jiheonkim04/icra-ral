# Epoch 4 Cycle 27 Prior Mechanism Map

Date: 2026-07-16 KST

Previous method: `AMP-VLA`

Previous decision: `AMP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`

AMP is not repaired or rescued. Cycle 27 starts from fresh primary-source
anchors whose positive claims are outside AMP's failed action-manifold
projection path.

## Primary-Source Anchors Checked

### DFM-VLA

Primary source: `https://arxiv.org/html/2603.26320v1`

Project page: `https://chris1220313648.github.io/DFM-VLA/`

Positive result: DFM-VLA reports discrete flow matching for iterative action
refinement, allowing full action-token sequences to be revised across
refinement steps rather than committed after one prediction. The paper reports
`95.7%` LIBERO average success, `4.44` CALVIN average success length, and a
project-page real-world average of `70.8%`. The project page states code is
coming soon, so a local first comparison would require a faithful transparent
proxy until official code is available.

Relevant mechanism: reversible/iterative full-sequence action refinement via a
learned velocity field and two-stage refinement-plus-validation decoding.

Local extension opportunity: transfer the refinement principle from discrete
action-token probability velocities to continuous SmolVLA action chunks by
learning a bounded residual vector field around Base chunk predictions using
only LIBERO demonstrations and deployment inputs.

### Adaptive Action Chunking

Primary source: `https://arxiv.org/html/2604.04161v2`

Project/code entry: `https://lance-lot.github.io/adaptive-chunking.github.io/`

Positive result: AAC reports an inference-time strategy that uses action
entropy to select chunk size dynamically, using smaller chunks under high
uncertainty and longer chunks when entropy is low. The project page links code
and benchmark repositories.

Relevant mechanism: uncertainty-conditioned execution horizon selection for
reactivity versus temporal consistency.

Local extension opportunity: not selected as a standalone method because prior
campaign governance rejects a contribution that is only adaptive chunk size.
A defensible extension would need a new action-composition mechanism, not just
horizon selection.

### RotVLA

Primary source: `https://arxiv.org/abs/2605.13403`

Positive result: RotVLA proposes continuous rotational latent actions modeled
on `SO(n)` with triplet-frame learning, and reports `98.2%` LIBERO plus strong
RoboTwin2.0 clean/randomized results. No official code was found on the arXiv
record during this pass.

Relevant mechanism: continuous structured latent action representation used as
a high-level planner conditioning a flow-matching action head.

Local extension opportunity: use a Lie-algebra residual representation for
orientation/translation action chunks from LIBERO demonstrations, with an
identity-preserving adapter around SmolVLA. Feasibility is lower than DFM-style
continuous refinement because RotVLA's strongest result depends on large-scale
pretraining and latent-action supervision not locally available.

### GEAR-VLA

Primary source: `https://arxiv.org/abs/2606.08530`

Repository: `https://github.com/babynabeauty/GEAR-VLA`

Positive result: GEAR-VLA reports geometry-aware action representations,
coarse-to-fine action learning, semantic-aligned 3D integration, embodiment
canonicalization, and strong LIBERO/RoboTwin/generalization results. The GitHub
repository currently says `coming soon`.

Relevant mechanism: geometry-aware action representation and embodiment
canonicalization.

Local extension opportunity: weaker for this repository because reliable 3D
spatial backbone assets/depth/embodiment-transfer supervision are not already
available for the current SmolVLA-LIBERO setup.

### Multi-view-VLA / AML

Primary source: `https://arxiv.org/abs/2605.11832`

Repository: `https://github.com/junjxiao/Multi-view-VLA`

Positive result: Multi-view-VLA reports Geometry-Guided Gated Transformer and
Action Manifold Learning, with released code/weights/data and `98.6` LIBERO,
`85.7` LIBERO-PLUS, and `86.1` RoboTwin2.0 results on its README. It is not
selected as a near-term Cycle 27 anchor because AMP already tested a local
action-manifold projection route and failed its frozen Stage 0 gates; a new
candidate must not be a cosmetic AMP rescue.

## Cycle 27 Design Constraint

The next method must use one genuinely new mechanism. LoRA may only be
identity-preserving implementation infrastructure. The closest prior or a
faithful transparent proxy must enter the first serious comparison.

## Map Decision

Proceed to exactly three Cycle 27 candidates:

1. `CFR-VLA`: Continuous Full-Chunk Refinement anchored to DFM-VLA.
2. `EOC-VLA`: Entropy-gated Overlap Composition anchored to AAC but extending
   beyond chunk-size selection.
3. `LGR-VLA`: Lie-Group Residual action representation anchored to RotVLA.

Prefer `CFR-VLA` if candidate scoring confirms that it has the strongest
positive prior, one clear new mechanism, local supervision from existing
LIBERO demonstrations, and a decisive Stage 0 audit.
