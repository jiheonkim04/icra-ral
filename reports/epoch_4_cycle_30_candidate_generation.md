# Epoch 4 Cycle 30 Candidate Generation

Date: 2026-07-16 KST

Previous method: `CCIF-VLA`

Previous decision: `CCIF_STAGE_0_DESIGN_FAILURE`

Decision: `URF_CANDIDATE_SELECTED_RESEARCHER_PROPOSAL_PENDING`

Cycle 30 generates exactly three candidates under the active prior-first,
performance-oriented, minimum-sufficient governance. None repairs CCIF, TSC,
CFR, AMP, RAP, or VDR. Unknown empirical performance is not a rejection reason.

## Candidate 1: URF-VLA

Full name: Uncertainty-Routed Residual Flow for Base-preserving SmolVLA chunks

Contribution type: `PRIOR_EXTENSION`

Closest external prior: SUREFlow, `https://arxiv.org/abs/2607.10504`,
official code `https://github.com/tanvirnwu/SUREFlow`.

Prior positive result: SUREFlow reports uncertainty-aware residual flow
matching in a 179M state-space VLA, `92.5%` average LIBERO success, and
LIBERO-PRO robustness results using the same broad observation/action family.

Actual mechanism: learn a heteroscedastic residual-flow overlay around a
decoded SmolVLA Base action chunk `A_base in R^[50,7]`. From legal deployment
inputs plus `A_base`, URF predicts a residual mean and residual uncertainty per
time-action cell. At inference it applies a bounded residual transport only
where the predicted residual is nontrivial and the predicted residual
uncertainty says the correction is trustworthy; otherwise the cell remains
exactly or nearly Base.

The method is not a confidence head. The uncertainty field changes which
action cells are allowed to move and how large the bounded residual transport
can be.

Minimal technical difference from prior: SUREFlow trains a native
state-space/Mamba VLA whose action generator jointly predicts velocities and
uncertainty. URF-VLA keeps the pretrained SmolVLA policy as the default action
source and uses uncertainty to route a conservative residual transport around
Base chunks. LoRA or a small adapter is only implementation infrastructure for
the residual and uncertainty heads.

Falsifiable chain:

Base condition -> some Base chunk cells contain useful residual headroom while
others are already strong or dangerous to alter.

Intermediate failure -> ordinary residual adaptation either changes too many
cells or applies residuals where the residual model is uncertain.

Policy behavior -> global residual changes disrupt clean behavior or gripper
timing, while no residual leaves correctable errors untouched.

Proposed method -> heteroscedastic residual flow estimates where residual
transport is useful and reliable.

Intended action behavior -> bounded cell-selective corrections with Base
passthrough in uncertain or already-good cells.

Expected closed-loop improvement -> better task success than Base, a SUREFlow
transparent proxy, a no-uncertainty residual ablation, and ordinary matched
LoRA while retaining clean behavior.

Data and supervision viability:

- existing LIBERO demonstrations provide images, proprioception, language/task
  identity, and expert action chunks;
- Base chunks can be generated from the frozen SmolVLA checkpoint;
- residual targets `A_expert - A_base` and heteroscedastic residual losses can
  be constructed from discovery/validation demonstrations only;
- no reward, success, done flag, object pose, future observation, or
  confirmatory reset identity is needed at inference.

Identity-preserving integration:

- residual branch initialized to zero;
- uncertainty gate initialized to Base passthrough;
- residual magnitude bounded per translation/rotation/gripper group;
- clean-retention objective required if training proceeds;
- disk reload must reproduce Base within the frozen tolerance before training.

First serious comparison after Stage 0 and bounded validation:

1. `smolvla_base`
2. `sureflow_uncertainty_residual_proxy` or official `sureflow` if installed
3. `urf_full`
4. `urf_no_uncertainty_route_ablation`
5. `standard_lora`

Standard LoRA is required because URF trains an adapter/head on the same
demonstrations and ordinary adaptation is a plausible alternative explanation.

Scores:

- provisional novelty: `22 / 25`
- importance of problem: `14 / 15`
- strength of positive prior anchor: `20 / 20`
- technical mechanism quality: `18 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `92 / 100`

## Candidate 2: IAF-VLA

Full name: Interaction-Aligned Feature retention for VLA policies

Contribution type: `PRIOR_EXTENSION`

Closest external prior: VLA-IAP, `https://arxiv.org/abs/2603.22991`, project
page `https://chengjt1999.github.io/VLA-IAP.github.io/`.

Prior positive result: VLA-IAP reports interaction-aligned visual token pruning
with `97.8%` LIBERO success and `1.25x` speedup, with up to `1.54x` speedup
while maintaining comparable performance.

Actual mechanism: derive an interaction saliency map from legal deployment
inputs and Base-predicted end-effector motion, then preserve only visual
features or tokens aligned with the predicted physical interaction region
while pruning or downweighting visually redundant background. The method is
inference-time feature routing, not LoRA and not task-language routing.

Minimal technical difference from prior: VLA-IAP is a training-free token
pruning method for tokenized VLA backbones. IAF would adapt the interaction
alignment principle to the current SmolVLA/OpenVLA feature interfaces and test
whether physical-interaction feature retention improves the success/latency
tradeoff.

Data and supervision viability: no training labels are required for the
inference-only version, and existing demonstrations can provide development
diagnostics for whether interaction saliency overlaps future motion/contact
proxies. However, current SmolVLA feature hooks may not expose a comparable
visual-token selection path.

Identity-preserving integration: default mask keeps all features; pruning is
disabled unless diagnostics show bounded action deltas and clean retention.

First serious comparison:

1. `smolvla_base`
2. `vla_iap_interaction_pruning_proxy`
3. `iaf_full`
4. `iaf_no_interaction_alignment_ablation`
5. `fixed_retention_pruning_baseline`

Standard LoRA is omitted because this is a frozen inference-time feature
routing method; ordinary LoRA does not test the pruning/interaction claim.

Scores:

- provisional novelty: `20 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `16 / 20`
- data/supervision feasibility: `7 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `82 / 100`

## Candidate 3: HNT-VLA

Full name: High-Noise Mean-Transport residual generation for SmolVLA chunks

Contribution type: `CROSS_PAPER_SYNTHESIS`

Closest external prior: ReactVLA, `https://arxiv.org/abs/2606.14255`, with
supporting one-step VLA evidence from `https://arxiv.org/abs/2606.05737`.

Prior positive result: ReactVLA reports one-to-few-step improved Mean Flow
action generation, performance gains over similarly sized VLA baselines
including SmolVLA, more than `4x` inference speedup, and real-world policy
latency below `38.6 ms`. Let It Be Simple reports that high-noise-biased
one-step VLA training can match or exceed multi-step decoding across
LIBERO-family experiments.

Actual mechanism: train a small Base-preserving finite-interval mean-transport
head that maps high-noise action samples directly toward expert chunks while
regularizing toward the Base decoded chunk. The claim axis is reactive
low-step action generation under a fixed closed-loop budget or controlled
delay, not generic LoRA.

Minimal technical difference from prior: ReactVLA replaces the native action
generator and transformer routing. HNT would preserve SmolVLA as the default
policy and add only a bounded low-step residual transport path around Base.

Data and supervision viability: existing LIBERO demonstrations provide action
chunks and legal inputs, but the primary gain may be latency/reactivity rather
than clean success in a synchronous simulator. That makes the decisive first
experiment less direct than URF.

Identity-preserving integration: zero residual head and default Base
passthrough; low-step head is enabled only after validation proves bounded
action deltas and clean retention.

First serious comparison:

1. `smolvla_base`
2. `reactvla_mean_transport_proxy`
3. `hnt_full`
4. `hnt_no_high_noise_transport_ablation`
5. `standard_lora`

Scores:

- provisional novelty: `18 / 25`
- importance of problem: `12 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `16 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `7 / 10`
- total: `80 / 100`

## Selection

Selected candidate: `URF-VLA`

Selection score: `92 / 100`

Rationale: URF has the strongest current positive primary-source anchor with
official code, a single mechanism that changes action generation rather than
only reporting confidence, direct supervision from existing LIBERO
demonstrations, identity-preserving integration, and a bounded Stage 0 audit.
It also avoids immediately reusing CCIF coarse intent, TSC masked completion,
CFR full-chunk refinement, AMP action-manifold projection, RAP retrieval
anchors, or VDR future-feature residuals.

Frozen first serious comparison order after Stage 0 and bounded validation:

1. `smolvla_base`
2. `sureflow_uncertainty_residual_proxy` or official `sureflow` if installed
3. `urf_full`
4. `urf_no_uncertainty_route_ablation`
5. `standard_lora`

Next action: freeze the URF-VLA Researcher A proposal before Reviewer B attack,
mathematical audit, preregistration, or implementation.
