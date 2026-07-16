# Epoch 4 Cycle 38 Prior Mechanism Map

Date: 2026-07-16 KST

Previous method: `CSPR-VLA`

Previous decision: `CSPR_STAGE_0_IMPLEMENTATION_FAILURE`

Previous result: `reports/cspr_vla/stage_0_result.json`

CSPR is closed without rescue. Its frozen Stage 0 completed `5760 / 5760`
rows with exit code `0`, zero exceptions, exact manifest/partial key equality,
and no duplicate, missing, extra, or split-overlap keys. The raw runner wrote
`CSPR_STAGE_0_DESIGN_FAILURE`, but adjudication corrected the final class to
`CSPR_STAGE_0_IMPLEMENTATION_FAILURE` because
`weighted_gradient_norm_ratio_max = 129.38210738906673` exceeded the frozen
`100.0` objective-scale limit. This is not a closed-loop scientific kill.
CSPR may not be repaired, retuned, relaunched, or reinterpreted.

## Boundary From Prior Cycles

Cycle 38 must not rename these closed or already-exercised routes:

- VLA-Corrector / NICE-style latent drift monitors;
- AAC / EAC adaptive horizon selection;
- SEAM / ChunkFlow / S2C boundary smoothing and overlap editing;
- MHS / RAR generic history-state residual memory;
- CSPR critical-step residual refinement;
- DCCG demo-calibration guidance;
- standard LoRA as the scientific mechanism.

The next method must use one genuinely new scientific mechanism. LoRA may be
used only as low-compute implementation infrastructure. The closest external
prior or a faithful transparent proxy must enter the first serious comparison.

## Current Primary-Source Anchors

### RoVLA

Primary sources:

- https://arxiv.org/abs/2605.19678
- https://arxiv.org/html/2605.19678v1
- https://github.com/HCPLab-SYSU/RoVLA

AUTHOR_STATED: RoVLA targets VLA brittleness under visual observation changes,
paraphrased language instructions, and compounded perturbations. It introduces
multi-consistency constraints: Instructional Consistency for equivalent
instruction rewrites, Evolutionary Consistency for action-intent stability
across flow-matching stages, and Observational Consistency for visual and
proprioceptive perturbations. The paper reports superior robustness on
LIBERO-Plus, RoboTwin 2.0, and real-world tasks. The official repository is
public, contains GR00T-based training/evaluation code, dataset conversion
scripts, and command-line flags for PGD and consistency learning.

INDEPENDENTLY_INFERRED: The positive prior is not LoRA and not a generic
augmentation recipe. Its mechanism is invariant policy learning: action
generation is forced to remain stable under task-preserving transformations.
This is meaningfully different from CSPR's gradient-scaled residual, S2C's
chunk-boundary bridge, and NICE's latent drift trigger.

CROSS_PAPER_SYNTHESIZED: Local LIBERO demonstrations already contain the
inputs needed for a bounded development audit: RGB images, proprioception,
language/task strings, and 7D action chunks. Synthetic paraphrases,
small image/proprio perturbations, and flow-time/action-noise perturbations can
be generated from development identities only. No success, reward, done,
object pose, future observation, simulator state, or confirmatory identity is
needed at inference.

Mechanism map:

- observation/input: current RGB streams, proprioception, language/task
  instruction, and frozen SmolVLA Base chunk;
- learned representation: a compact consistency code that must remain stable
  across task-preserving language, observation, and action-evolution
  perturbations;
- supervision: paired same-task transformations generated from discovery and
  validation demonstrations only;
- objective: preserve Base behavior while making adapter features and action
  deltas invariant under legal task-preserving transformations;
- policy component changed: identity-preserving adapter/gate around frozen
  SmolVLA features or action interface, not the base VLA identity;
- inference intervention: exact Base passthrough when the gate is zero or the
  consistency confidence is low;
- primary metric: consistency-code separation, action invariance under legal
  perturbations, clean retention, action validity, and paired success if Stage
  0 passes;
- demonstrated causal link in prior: explicit multi-consistency constraints
  improve robustness under language, visual, and trajectory shifts;
- untested local causal link: a smaller SmolVLA adapter can reproduce the
  useful consistency mechanism without full GR00T-scale fine-tuning.

### IntentVLA

Primary sources:

- https://arxiv.org/abs/2605.14712
- https://github.com/ZGC-EmbodyAI/IntentVLA

AUTHOR_STATED: IntentVLA identifies short-horizon observation aliasing in robot
imitation, where similar current observations can correspond to different
action chunks because of recent intent, phase, or context. It encodes recent
visual observations into a compact short-horizon intent representation and
reports improved rollout stability across AliasBench, SimplerEnv, LIBERO, and
RoboCasa. Its repository currently releases AliasBench code and states that
full model training/evaluation code is coming soon.

INDEPENDENTLY_INFERRED: IntentVLA is a strong positive prior for compact
history-conditioned intent, but the local campaign has already tested generic
history residuals in MHS and RAR. A new candidate must therefore use aliasing
contrast and intent commitment directly, not residual action memory.

Local relevance: existing LIBERO demonstrations provide legal histories,
images, proprioception, actions, and task strings. The main feasibility risk is
whether local tasks contain noncollapsed aliasing contrast; if not, Stage 0
must stop as a data/supervision failure.

### OA-WAM

Primary sources:

- https://arxiv.org/abs/2605.06481
- https://arxiv.org/html/2605.06481v1

AUTHOR_STATED: OA-WAM decomposes each frame into robot and object slot states
with persistent address vectors and time-varying content vectors. Addressable
attention routes cross-slot interaction through address-only keys and resets
the address slice at every layer. The paper reports `97.8%` LIBERO success,
`79.3%` SimplerEnv success, strong LIBERO-Plus geometric-axis robustness, and
a causal slot-intervention swap-binding cosine of `0.87` versus at most `0.09`
for holistic baselines.

INDEPENDENTLY_INFERRED: OA-WAM is a strong object-binding prior, but the local
repository does not currently have verified legal object-slot extraction for
LIBERO images. Simulator object poses would make the route easy but would be
privileged at inference. A local method must either use deployment-observable
image-derived slots or fail before rollout.

Local relevance: object-addressed consistency could target spatial/object
confusions, but supervision viability is weaker than RoVLA unless a legal slot
extractor is available or a transparent noun/feature proxy proves predictive.

### GEAR-VLA

Primary sources:

- https://arxiv.org/abs/2606.08530
- https://arxiv.org/html/2606.08530v1
- https://github.com/babynabeauty/GEAR-VLA

AUTHOR_STATED: GEAR-VLA introduces geometry-aware action representations,
coarse-to-fine action learning, semantic-aligned 3D integration, and embodiment
canonicalization. It reports state-of-the-art LIBERO and zero-shot LIBERO-Plus
performance, plus real-world success on AgileX and LDT-01.

INDEPENDENTLY_INFERRED: The positive result is compelling, but the local
SmolVLA/LIBERO path lacks a verified 3D spatial backbone, depth/multiview
alignment, and embodiment-transfer supervision. This is a useful reviewer
baseline and future expansion route, not the best immediate Cycle 38 candidate.

## Selection Implications

The strongest Cycle 38 direction is RoVLA-anchored multi-consistency learning
implemented as a frozen-SmolVLA identity-preserving adapter. It changes the
scientific mechanism away from CSPR residual gradient scaling and away from
closed chunking/monitor routes. It uses only deployment-observable inputs and
existing LIBERO demonstrations, has official code, and admits a decisive
development audit before training or rollout.
