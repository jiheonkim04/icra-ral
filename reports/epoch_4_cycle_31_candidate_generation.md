# Epoch 4 Cycle 31 Candidate Generation

Date: 2026-07-16 KST

Previous method: `URF-VLA`

Previous fixed result: `URF_STAGE_0_NO_USABLE_HEADROOM`

Previous result artifact: `reports/urf_vla/stage_0_result.json`

URF is closed. Its no-headroom stop is not a closed-loop scientific kill, but
bounded validation and rescue are forbidden for this formulation.

## Candidate Count

Exactly three candidates were generated.

## Candidate 1: S2C-VLA

Full name: Seam-Supervised Chunk Consistency for Base-preserving SmolVLA
execution.

Contribution type: `PRIOR_EXTENSION`

Closest prior: ChunkFlow

Closest prior source: `https://arxiv.org/html/2607.12992v1`

Closest prior positive result: ChunkFlow reports `93.4%` LIBERO long-horizon
success with lower boundary jump, high-frequency energy, and smoothness
metrics than chunked VLA baselines, while preserving low-latency inference.

Secondary prior: SEAM, `https://arxiv.org/abs/2607.04609`, which reports lower
boundary jerk and transition discontinuity on LIBERO-10 using the previous
chunk tail as an analytic reference.

Mechanism: S2C learns a development-only overlap edit mask and tail-anchored
bridge for SmolVLA chunks. At inference, the previous executed/unexecuted tail
and the current Base chunk are both deployment-available. S2C freezes the
already-committed prefix, edits only the overlap cells selected by the learned
mask, and leaves the future zone Base-preserving unless the boundary-consistency
gate activates. The initialized gate is exact Base passthrough.

Minimal technical difference from ChunkFlow: ChunkFlow trains a chunked policy
with seam losses, history corruption, deterministic blending, and optional RL
fine-tuning. S2C does not replace SmolVLA or require RL. It adds a
Base-preserving overlap edit layer that can be fitted from existing LIBERO
demonstrations and compared against a transparent ChunkFlow/SEAM proxy under
the same SmolVLA Base chunks.

Falsifiable mechanism chain:

problem condition -> adjacent SmolVLA chunks disagree in their overlap and
produce boundary jumps or high-frequency action artifacts;

intermediate failure mechanism -> the executed tail from the previous chunk and
the current chunk head encode incompatible local continuations;

policy behavior -> raw replanning switches modes or injects discontinuity;

closed-loop failure -> boundary artifacts degrade contact-rich or long-horizon
execution.

proposed method -> overlap edit mask and tail-anchored bridge reduce
pre-execution boundary disagreement while preserving unselected Base cells;

intended action behavior -> lower boundary jump, lower high-frequency energy,
bounded action delta, no global smoothing;

expected closed-loop improvement -> smoother replanning without suppressing
task-relevant motion.

Data/supervision viability: existing LIBERO demonstrations provide action
sequences for expert overlap consistency and smoothness targets. SmolVLA Base
chunks can be decoded for paired neighboring windows. No future observation,
object pose, reward, success, done flag, or confirmatory identity is required.

Identity-preserving integration: zero edit gate equals Base. Deterministic
ChunkFlow/SEAM proxy remains policy 2; S2C full is policy 3; key ablation is
S2C with no learned overlap mask; standard LoRA is retained because S2C trains
a small adapter/head on demonstrations.

Decisive experiment feasibility: Stage 0 can measure boundary jump, first- and
second-order discontinuity, high-frequency ratio, action validity, clean
retention, gate locality, and paired proxy-vs-ours-vs-ablation metrics before
any rollout.

Scores:

- provisional novelty: `23 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `10 / 10`

Total: `95 / 100`

## Candidate 2: HIC-VLA

Full name: History-Indexed Commitment for aliased SmolVLA chunks.

Contribution type: `PRIOR_EXTENSION`

Closest prior: IntentVLA

Closest prior source: `https://arxiv.org/abs/2605.14712`

Closest prior repository: `https://github.com/ZGC-EmbodyAI/IntentVLA`

Closest prior positive result: IntentVLA reports improved rollout stability and
strong benchmark performance, including `97.4` LIBERO-Long Avg@500 success and
AliasBench gains over history-frame baselines.

Mechanism: HIC learns a compact recent-history commitment vector from past
proprioception, executed actions, and optionally cached visual features. The
commitment vector gates a small Base-preserving chunk selector so that adjacent
replanning steps maintain a consistent short-horizon continuation under
observation aliasing.

Minimal technical difference from IntentVLA: IntentVLA conditions a full VLA on
recent visual history. HIC does not feed long image histories through SmolVLA.
It learns a small deployment-observable commitment state and uses it only to
gate a bounded chunk-level consistency adapter.

Falsifiable mechanism chain: local observations are aliased; frame-conditioned
SmolVLA chunks switch continuations; a recent-history commitment state predicts
which continuation should remain active; closed-loop stability improves when
mode switching drops.

Data/supervision viability: LIBERO demonstrations provide recent action and
proprio histories. Visual history features would require extra feature
extraction, but a proprio/action-only proxy is available locally. The official
IntentVLA model implementation is not fully released, so policy 2 would be a
transparent proxy unless assets become available.

Identity-preserving integration: zero commitment gate equals Base.

Decisive experiment feasibility: Stage 0 can measure short-horizon continuation
classification, chunk-to-chunk consistency, clean retention, and ablation
equivalence.

Scores:

- provisional novelty: `22 / 25`
- importance of problem: `15 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `9 / 10`

Total: `90 / 100`

## Candidate 3: PCM-VLA

Full name: Proprioceptive Consequence Model for SmolVLA action plausibility.

Contribution type: `CROSS_PAPER_SYNTHESIS`

Closest prior: RynnVLA-002

Closest prior source: `https://arxiv.org/html/2511.17502v3`

Closest prior repository: `https://github.com/alibaba-damo-academy/RynnVLA-002`

Closest prior positive result: RynnVLA-002 reports `97.4%` LIBERO simulation
success and controlled gains from world-model data, including continuous-action
success improving from `91.6%` to `94.6%` under a matched setting.

Mechanism: PCM trains a compact forward consequence model from current
proprioception and candidate action chunks to predicted next-proprio summaries.
It then gates Base-preserving action candidates by whether their predicted
consequence remains consistent with demonstration dynamics and task phase.

Minimal technical difference from RynnVLA-002: RynnVLA unifies world modeling
and action generation in a large model. PCM uses only a small proprioceptive
consequence checker attached to SmolVLA, with no image future prediction and no
privileged inference input.

Falsifiable mechanism chain: Base chunks may be action-valid but imply
implausible proprioceptive progress; a learned consequence model detects this
from current proprioception and candidate actions; rejecting or softly gating
implausible candidates improves closed-loop behavior.

Data/supervision viability: existing LIBERO demonstrations contain current
proprioception, action chunks, and future proprioception within the same
demonstration. Future proprioception is used only for training labels, not
inference.

Identity-preserving integration: calibrated gate initializes to Base
passthrough.

Decisive experiment feasibility: Stage 0 can test forward-prediction accuracy,
Base-vs-demo consequence discrepancy, action validity, no privileged inference,
and no-consequence ablation.

Scores:

- provisional novelty: `21 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `8 / 10`
- decisive experiment feasibility: `8 / 10`

Total: `87 / 100`

## Selection

Selected method: `S2C-VLA`

Selection decision: `S2C_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING`

Reason: S2C has the strongest current primary-source anchor, the clearest local
headroom after URF's residual no-headroom stop, and the most decisive
development-only audit. It targets a different failure axis from URF:
cross-chunk boundary consistency rather than expert residual magnitude. The
closest prior, ChunkFlow, must enter the first serious comparison.

Frozen first serious comparison for the design:

1. `smolvla_base`
2. `chunkflow_overlap_proxy` or official ChunkFlow if locally installed and
   verified
3. `s2c_full`
4. `s2c_no_learned_overlap_mask_ablation`
5. `standard_lora`

Immediate next stage: freeze the S2C-VLA Researcher A proposal before Reviewer
B attack, mathematical audit, preregistration, prototype protocol,
implementation, training, validation search, or rollout.
