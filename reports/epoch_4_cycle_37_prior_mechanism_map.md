# Epoch 4 Cycle 37 Prior Mechanism Map

Date: 2026-07-16 KST

Previous method: `DCCG-VLA`

Previous decision: `DCCG_STAGE_0_DATA_FAILURE`

Previous result: `reports/dccg_vla/stage_0_result.json`

DCCG is closed without rescue. Its fixed Stage 0 result wrote `0 / 0`
completed/planned rows with exit code `0`, zero exceptions, exact
manifest/partial key equality, and zero duplicate, missing, extra, or
split-overlap keys. The stop was a development-only cache coverage failure:
the frozen DCCG task/demo identities did not match the available cached
SmolVLA Base chunks. This is not a closed-loop scientific kill, and DCCG may
not be rescued by changing identities, thresholds, cache source, or protocol.

Cycle 37 must generate exactly three candidates. The selected method must use
one genuinely new scientific mechanism. LoRA may be used only as
implementation infrastructure. The closest positive external prior must enter
the first serious comparison. The local data path is constrained by available
cached SmolVLA Base rows: `640` rows across `libero_10/task_5`,
`libero_goal/task_5`, `libero_object/task_3`, and `libero_spatial/task_3`,
with demo ids `0..9`.

## Primary-Source Anchors

### DySL-VLA

Sources:

- https://arxiv.org/abs/2602.22896
- https://github.com/PKU-SEC-Lab/DYSL_VLA

AUTHOR_STATED: DySL-VLA observes that actions within a task have different
importance: critical steps need high precision, while less important steps can
tolerate more variance. It dynamically skips VLA layers based on action
importance, keeps informative layers, selectively skips incremental layers,
uses prior-post skipping guidance, and trains with skip-aware two-stage
knowledge distillation. The paper reports `2.1%` success-length improvement on
CALVIN, `85.7x` fewer trainable parameters, and `3.75x` speedup at
iso-accuracy. Official code is available.

INDEPENDENTLY_INFERRED: The positive prior is not LoRA and not generic
efficiency tuning. The useful mechanism is action-importance-conditioned
allocation of model capacity. A local extension can transfer that idea from
conditional compute to conditional action refinement: keep Base exactly on
noncritical steps, and spend bounded correction capacity only on predicted
critical action cells.

CROSS_PAPER_SYNTHESIZED: DCCG failed because its frozen identities had no
matching cached Base chunks. DySL-VLA suggests a mechanism that can be audited
on the available cache because action importance can be derived from cached
Base action chunks, demonstration action deltas, gripper events, and
deployment-observable visual/proprio features.

Mechanism map:

- observation/input: ordinary SmolVLA observation, language, proprioception,
  cached Base action chunk, and optional cached visual/proprio features;
- learned representation: a critical-step score over `[50, 7]` action cells
  or timestep groups;
- supervision: demonstration action deltas, Base-vs-demo error, gripper-event
  boundaries, and smoothness/curvature diagnostics on discovery and
  validation identities only;
- objective: learn a deployment-observable criticality predictor and a
  zero-initialized bounded residual that acts only on high-criticality cells;
- policy component changed: action-interface residual/gate, not the base VLA
  identity and not a global replacement action head;
- inference intervention: exact Base passthrough when criticality is low or
  residual scale is zero; bounded residual only when critical-step evidence is
  above a validation-frozen threshold;
- primary metric: clean retention, critical-step activation localization,
  action validity, separation from DySL proxy and simple criticality
  heuristics, then paired closed-loop success;
- demonstrated causal link in prior: action importance can guide selective
  capacity allocation while preserving accuracy and improving efficiency;
- untested local causal link: action importance can guide selective action
  refinement while preserving clean Base behavior.

Local relevance: the available CCIF cached rows include `base_chunk_cache_path`,
`feature_cache_path`, task identity, demo id, frame index, 7D Base chunks, and
960D visual features for the four cache-covered tasks. No success, reward,
done, object pose, simulator state, or confirmatory identity is required at
inference.

### ProgressVLA

Sources:

- https://arxiv.org/abs/2603.27670
- https://arxiv.org/html/2603.27670v1

AUTHOR_STATED: ProgressVLA argues that most VLA models lack task-progress
awareness. It combines robust progress estimation with differentiable progress
guidance: an inverse dynamics world model maps predicted action tokens to
future latent visual states, a progress estimator scores those latents, and a
maximal-progress regularizer guides action tokens. The paper reports
substantial gains on CALVIN, LIBERO, and real-world deployment.

INDEPENDENTLY_INFERRED: The mechanism is progress-conditioned action guidance,
not generic imitation or LoRA. Local progress labels can be derived from
demonstration frame index, but inference must use a predictor from current
deployment-observable inputs, not the demonstration time index.

CROSS_PAPER_SYNTHESIZED: Cycle 18 PCAV already used TACO plus a
ProgressVLA-motivated progress-consequence extension and stopped for no usable
headroom. A new ProgressVLA-anchored candidate must therefore avoid reopening
PCAV or reusing its candidate-oracle framing.

Local relevance: frame-index progress labels are locally viable on the cache
covered demos. Novelty risk is high because progress-aware mechanisms have
already been exercised in this campaign.

### ForesightFlow

Sources:

- https://arxiv.org/abs/2606.04968
- https://arxiv.org/html/2606.04968v1

AUTHOR_STATED: ForesightFlow augments flow-matching VLA action chunks with
learned success-potential trajectories. The same flow proposes and scores
candidate actions for best-of-`K` inference without an external critic. It
uses decoupled advantage-weighted flow matching, a one-step boundary
estimator, and reports improved simulation and real-world success plus `38%`
training-compute reduction.

INDEPENDENTLY_INFERRED: The positive prior is self-scoring action generation
from mixed-quality experience. The local data issue is severe: the existing
cached LIBERO demonstrations and cached Base chunks do not provide
noncollapsed success/failure/advantage labels across the cache-covered
identities.

CROSS_PAPER_SYNTHESIZED: A local candidate can treat ForesightFlow as a prior
for action-potential ranking, but the current cache is likely insufficient for
a faithful success-potential reproduction without adding new rollouts or
privileged labels.

Local relevance: strong mechanism, weak immediate supervision viability.

### STRONG-VLA

Sources:

- https://arxiv.org/abs/2604.10055
- https://arxiv.org/html/2604.10055v1

AUTHOR_STATED: STRONG-VLA separates robustness acquisition from clean
task-aligned refinement. Stage I progressively exposes the model to
multimodal perturbations; Stage II re-aligns to clean data. It reports LIBERO
success gains across OpenVLA, OpenVLA-OFT, and pi0, plus real-world AIRBOT
validation.

INDEPENDENTLY_INFERRED: The prior strongly supports perturbation robustness
and clean retention, but this campaign already used STRONG-VLA as the closest
prior for IARC and used several adjacent robustness/occlusion/language
contrast mechanisms.

Local relevance: useful reviewer pressure and possible baseline source, but a
new Cycle 37 selection should not be a renamed robustness LoRA.

## Selection Implications

The strongest Cycle 37 direction is DySL-anchored critical-step selective
policy refinement. It uses a positive official-code prior, directly respects
the current "one new mechanism, LoRA only infrastructure" constraint, fits the
available cached Base rows, preserves identity by default, and avoids
rescuing DCCG, PCAV, IARC, or other closed methods. ProgressVLA and
ForesightFlow remain valuable anchors, but ProgressVLA has higher historical
overlap and ForesightFlow has weaker local supervision viability.
