# Epoch 6 Current Literature and Novelty Map

Audit date: 2026-07-19 KST
Source policy: primary papers, proceedings, project pages, and official
repositories. This is a preselection audit; all candidate-specific equations,
code paths, and licenses must be rechecked before a protocol is frozen.

## Mandatory action-map and candidate-selection audit

| Prior | Verified mechanism / artifact state | Epoch 6 disposition |
|---|---|---|
| [ActionMap](https://arxiv.org/abs/2606.06904), [official code](https://github.com/showlab/ActionMap) | Voxelized translation/rotation heatmaps with Gaussian targets and top-k soft-argmax. The repository exposes the head but, at audit time, not a complete checkpointed training/evaluation stack. | Direct collision with TCA/action-heatmap work; the historical axis is closed. |
| [Action Map Policy](https://arxiv.org/abs/2607.10706) | Multi-view projected 3-D gripper-keypoint heatmaps, triangulation, soft labels, and equivariant image/label augmentation. No official public implementation was found. | Direct collision with projected-keypoint/action-map decoders. |
| [BridgeVLA](https://arxiv.org/abs/2506.07961), [official code](https://github.com/BridgeVLA/BridgeVLA) | Text-conditioned heatmaps over RGB-D orthographic views with coarse-to-fine inference and discretized rotation/gripper outputs. The checkpoint bundle is about 50.5 GB and PaliGemma access is gated. | Both collision and budget/artifact mismatch; not a feasible Epoch 6 Base/Prior. |
| [Selected Diffusion Noise](https://arxiv.org/abs/2606.14084) | Best-of-N diffusion sampling scored with object-mask counterfactuals, kNN density ratios, and jerk. Requires repeated policy calls plus GroundingDINO/SAM2; no official implementation was found. | Generic action-candidate selection is crowded and costly. |
| [CALAMARI](https://proceedings.mlr.press/v229/wi23a.html), [official code](https://github.com/MMintLab/calamari) | Language-conditioned spatial contact/action-affordance maps on a custom RLBench-era stack. | Occupies language-conditioned contact maps; not a practical local Prior. |
| [VoxPoser](https://voxposer.github.io/), [official code](https://github.com/huangwl18/VoxPoser) | LLM/VLM-generated 3-D value maps and planning; released RLBench demo uses privileged masks and an API key. | Not budget/fidelity compatible and not a novelty opening. |
| [InSpire](https://arxiv.org/abs/2505.13888), [official code](https://github.com/Koorye/Inspire) | Spatial-QA prompting plus answer/action alignment; official scripts/checkpoints include a 1B miniVLA path. | Plausible smaller Base after resource/provenance validation, but coarse object-relative spatial reasoning is already its contribution. |
| [OpenVLA-OFT](https://arxiv.org/abs/2502.19645), [official code](https://github.com/moojink/openvla-oft) | Parallel continuous-action decoding, chunking, L1 objective, and optional FiLM. | Strong frozen Base. Official training memory exceeds this GPU; locally selected use is inference unless a separately verified low-memory path exists. |

Conclusion: do not reopen target maps, action heatmaps, projected keypoint maps,
object-mask candidate ranking, or map-conditioned decoding. A new paper cannot
be obtained by renaming this representation family.

## Additional collision map

- Recovery and failure monitoring are crowded by
  [SAFE](https://arxiv.org/abs/2506.09937),
  [VLA-Corrector](https://arxiv.org/abs/2607.01804), FLARE, FAR, and B2FF.
  SAFE has current detector code; this refresh changes artifact availability but
  does not turn the historical blocked recovery path into a scientific failure.
- Execution-horizon, chunk-consistency, and replanning routes are occupied by
  AAC, AutoHorizon, Mixture of Horizons, SEAM, SDN, VLA-ATTC, and ReconVLA.
  The asynchronous-delay axis is independently closed locally.
- Generic splines are no longer a safe novelty opening: B-spline Policy
  ([paper](https://arxiv.org/abs/2607.09648),
  [code](https://github.com/BarisYazici/b-spline-policy)), Spline Policy, and
  ABPolicy now occupy learned spline action generation. A controller-validity
  study would require a separate equation/code-level collision audit.
- Generic efficiency/pruning is crowded by Drop-Then-Recovery, MoLe-VLA,
  SemanticVLA, AC2-VLA, AVA-VLA, and EfficientVLA. Generic language robustness
  is also crowded by LIBERO-Para, LIBERO-CF, ST4VLA, STRONG-VLA, RoVLA, and
  RobustVLA, while the local canonicalization route is exhausted.
- Camera-centric action frames collide with MCF-Proto, CamVLA, GEAR-VLA, and
  current action-space design studies. Geometry/depth/future-state auxiliaries
  and generic progress/phase/gripper-transition supervision are likewise
  crowded by GAM, ELAN4D, EvoScene-VLA, ThinkingVLA, ProgVLA, ProgressVLA,
  Green-VLA, StaKe, FrameSkip, and LA4VLA.

## Residual opportunities after hard filters

### 1. Outcome-separated VLA evaluation

- Problem: checkpoint/configuration selection and final reporting can reuse the
  same outcomes, overstating performance and even changing rankings.
- Runnable basis: [vla-eval](https://arxiv.org/abs/2603.13966) with frozen
  OpenVLA-OFT. Action Map Policy's described best-of-20 evaluation is a current
  motivation, not evidence of misconduct.
- Distinct residual: nested outcome-disjoint selection, paired reset manifests,
  and a selection-adjusted episode-level lower confidence bound. vla-eval
  standardizes execution semantics but does not correct winner's-curse
  checkpoint reporting. RouterVLA uses outcome-disjoint cross-fitting for
  routing, not checkpoint-reporting inference.
- Gate: demonstrate preregistered ranking reversals or material selection
  optimism repeatedly on outcome-disjoint folds. Otherwise this is only an
  audit/tooling note.
- Independent hard-filter result: `TOO_OVERLAPPING_OR_TRIVIAL`. RouterVLA's
  outcome-disjoint cross-fitting is a near-exact protocol collision, including
  explicit controls for leakage through profiles, normalization, filtering,
  hyperparameters, and tie-breaking. RSS 2026 N-SCORE already occupies safe,
  anytime-valid, sample-efficient robot-policy comparison. vla-eval already
  provides the execution ledger and provenance layer. Ordinary independent
  holdout is statistically sufficient for inference on one frozen selected
  policy; a paired manifest or generic lower bound alone is not a new method.
  No experiment was authorized or run.

### 2. Counterfactual preservation of task-causal low-motion actions — rejected after official code audit

- Problem: [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) treats complete
  no-op filtering as important to avoid freezing, whereas
  [VLA-Arena](https://arxiv.org/abs/2512.22539) reports that deleting all no-ops
  harms replay and preserves fixed neighborhoods around transitions.
- Intended residual: use exact-init simulator replay to label near-zero segments
  by their downstream physical efficacy when removed, rather than by magnitude,
  gripper proximity, phase, or duration.
- Strong controls: retain all, remove all, and VLA-Arena's fixed-N
  gripper-neighborhood preservation. FrameSkip is a heuristic frame-importance
  Prior; SIEVE selects whole demonstrations rather than within-trajectory
  actions.
- Official-code finding: pinned VLA-Arena revision
  `babe582ebffc82b979b77964a7e56417d02f63a4` already filters all no-ops,
  progressively restores 4/8/12/16 post-gripper actions, replays each variant,
  and retains the first successful trajectory. This is outcome-based causal
  preprocessing, not merely a fixed-N heuristic as the preliminary abstract-
  level audit suggested.
- Decision: segment-level or non-gripper refinement is incremental to the same
  central supervision, intervention, replay adjudicator, and causal claim.
  Classification: `TOO_OVERLAPPING_OR_TRIVIAL`. No Epoch 6 experiment was run.

### 3. Controller-dynamics contract adaptation — rejected after collision audit

- Problem: an unchanged policy and action convention may degrade when realized
  controller dynamics change despite identical action metadata.
- Intended residual: infer a low-dimensional dynamics signature from commanded
  versus realized proprioceptive transitions and condition a small residual
  adapter on that signature.
- Strong controls: fixed global action scaling and controller randomization
  without conditioning.
- Closest work: [Tune to Learn](https://openreview.net/forum?id=K7rg8zA5nJ)
  studies collection-time controller gains across BC/RL/sim-to-real;
  RobustVLA studies input/output perturbations; Same Weights, Different Robot
  studies normalization metadata and controller-facing conventions.
- Fatal collision: [APEX](https://arxiv.org/abs/2606.16504) already inserts a
  plug-and-play layer between black-box learned policies and controllers,
  reconstructs dynamically feasible references, and adapts online from
  low-level state feedback. It reports tracking and task-success gains across
  four policy classes. The proposed signature-conditioned residual adapter is
  therefore near-exact in problem, inputs, intervention location, and causal
  claim. Classification: `TOO_OVERLAPPING_OR_TRIVIAL`. Do not put this route in
  the active portfolio.

### 4. Training-only non-gripper contact-mode distillation

- Problem: missed or premature environmental contact may be underrepresented by
  ordinary action loss even when simulator contact labels are available.
- Distinct residual: a training-only nonspatial, non-gripper contact-mode head
  distilled into vision/proprioception features and removed at inference.
- Strong controls: unstructured binary contact and a StaKe-style stage head.
- Closest work: CALAMARI uses spatial contact maps;
  [TacCoRL](https://arxiv.org/abs/2606.11743) uses tactile input and simulation
  RL; StaKe predicts manipulation stages and next gripper events.
- Gate: non-gripper contact must explain residual failures beyond gripper state
  and be predictable without tactile input.
- Independent hard-filter result: `BACKUP_GATE_ONLY`, not active-thesis GO.
  [FD-VLA](https://arxiv.org/abs/2602.02142) predicts force tokens from
  vision/state and deploys without a force sensor, while
  [HapticVLA](https://arxiv.org/abs/2603.15257) distills a tactile teacher into
  a conventional VLA and deploys without tactile input. CALAMARI, TacCoRL,
  [StaKe](https://arxiv.org/abs/2606.26801), and
  [GAP](https://arxiv.org/abs/2602.12032) further crowd contact, stage, and
  phase supervision.
- A defensible residual, if a frozen pre-Ours gate passes, is limited to
  training-only *non-gripper object-environment contact-transition topology*
  extracted from MuJoCo geom pairs. It must exclude force magnitude, tactile
  features, robot/gripper contacts, spatial maps, and inference-time contact
  tokens, and must beat binary-contact, stage/gripper, phase, shuffled-label,
  action-only, and gripper-history explanations. This is an unverified backup,
  not a selected thesis.

## Ranking implication

The initial method portfolio is exhausted. VLA-Arena removes the low-motion
route, APEX removes controller-dynamics adaptation, RouterVLA plus N-SCORE
remove generic outcome-separated selection/reporting, and FD-VLA/HapticVLA
remove generic sensor-free contact distillation. The narrow non-gripper
contact-topology residual remains only a gated backup.

The subsequent benchmark/systems refresh selected schedule-invariant
stochastic VLA evaluation for problem verification only. Its current residual
is narrower than a generic reproducibility claim: test whether request order
remaps process-global stochastic action samples strongly enough to change
closed-loop robot-policy outcomes or comparisons. The pinned evaluation harness
and X-VLA source establish the mechanism's plausibility; JAX-style counter-based
randomness and batch-invariant serving are non-robotics precedents, so keyed
randomness alone is not a contribution. The frozen outcome-suppressed gate must
pass before any closed-loop problem outcome is accessed, and both problem gates
must pass before any Ours design or paper authorization.
