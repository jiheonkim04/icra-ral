# IARC-VLA Reviewer B Attack

Date: 2026-07-15 KST

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Frozen proposal hash:
`A1B0CF8BCBCF6A88F27B31EF5E38BAF408A3E62BB34206A1AC9F051EA6B57408`.

## Sources Reviewed

- STRONG-VLA, https://arxiv.org/abs/2604.10055
- Gradient Episodic Memory, https://arxiv.org/abs/1706.08840
- A-GEM, https://arxiv.org/abs/1812.00420
- PCGrad, https://arxiv.org/abs/2001.06782
- Conflict-aware Gradient Agreement Augmentation,
  https://arxiv.org/abs/2308.01194
- local DCR candidate record,
  `reports/epoch_4_cycle_12_candidate_generation.md`
- local COVI, G3P, CALA, RAR, PSE, and LIFT outcomes.

## Strongest Fair Reading

The strongest fair interpretation is not that IARC invents gradient surgery.
It is a narrow VLA robustness synthesis:

1. STRONG-VLA provides a positive same-domain result and identifies
   clean/perturbed optimization conflict.
2. GEM/A-GEM/PCGrad provide established projection mechanisms.
3. IARC applies one asymmetric constraint specifically during the clean
   realignment stage so that robustness replay is protected only when clean
   and robust action objectives conflict.

This is a plausible empirical contribution if it beats a transparent STRONG
proxy, unprojected joint replay, and matched standard LoRA on the same backbone
and perturbation condition. It is not yet implementation-ready.

## Essential Blocking Findings

### 1. The AdamW Guarantee Is Invalid As Written

The proposal projects the raw Euclidean gradient and then states that the
robust loss is first-order non-increasing. AdamW does not apply the raw gradient
as the parameter step. Momentum, second-moment preconditioning, and decoupled
weight decay can rotate or add to the actual update. Therefore
`<g_IARC,g_r> >= 0` does not imply that the realized AdamW parameter delta is
non-increasing for `L_r`.

The `epsilon` denominator also leaves a small negative dot product on a
conflict row. The proposal's tolerance makes this numerically testable but does
not restore the exact mathematical claim.

Required resolution: choose one minimum-sufficient path before the mathematical
audit:

- use explicit Stage II SGD with no momentum and no weight decay for Prior,
  Ours, ablation, and the matched Stage II portion of standard LoRA; or
- project the complete proposed optimizer step, including momentum,
  preconditioning, and weight decay, and prove/test the realized constraint.

Reviewer B recommends the first path. It is simpler, faithful to the claimed
geometry, and avoids a new optimizer-state module. The exact projection must
divide by `||g_r||^2` only when that norm exceeds a frozen floor; a below-floor
robust gradient is a nonacting-mechanism or capacity diagnostic, not an
epsilon-regularized proof.

This issue is essential and blocks implementation.

### 2. SmolVLA Flow Stochasticity Is An Uncontrolled Confound

Clean and perturbed gradient pairs must use:

- the exact same demonstration row and action chunk;
- the exact same flow noise tensor;
- the exact same flow time tensor;
- the exact same preprocessing except for the declared perturbation;
- the same mixed-precision and loss-reduction path.

Otherwise negative cosine may be caused by different stochastic flow-matching
draws rather than input perturbation. The proposal freezes action-target hashes
but does not yet freeze shared flow noise/time. The mathematical audit and
runner must do so.

This issue is essential and blocks mechanism interpretation.

### 3. Mixed Precision And Missing Gradients Need An Exact Vector Contract

The proposal does not define:

- whether gradients are unscaled before dot products;
- how `None` gradients are represented;
- parameter ordering and shape identity across `g_c` and `g_r`;
- accumulation boundaries;
- clipping order;
- what happens when `||g_r||` is below the floor;
- whether projection is performed once per physical batch or once per
  accumulated optimizer step.

Required resolution: accumulate clean and robust gradients separately in
`float32` over the complete logical batch, unscale before vector operations,
represent missing gradients as exact zeros in a frozen parameter order, project
once per optimizer step, forbid gradient clipping before the mechanism audit,
and report finite/nonzero tensors and norm contributions by resolved target
module.

This issue is essential and blocks implementation.

### 4. Conflict Is Not Yet Evidence Of Useful Headroom

A negative gradient dot product proves only local interference. It does not
show that:

- Base has meaningful closed-loop perturbation failure;
- Stage I improves that condition;
- clean Stage II actually forgets it;
- preventing the local gradient increase improves action behavior.

Offline action-flow loss is useful but cannot be the only headroom evidence for
a closed-loop claim. Before full five-policy training, freeze a development-only
paired Base headroom screen of approximately `10` clean and `10` perturbed
episodes on fresh validation reset identities. The simulator must be
synchronous, task/reset pairs identical, and perturbations fixed before seeing
outcomes. This screen may kill only for decisive no-headroom or invalid
perturbations under the active false-negative safeguard.

The micro Stage I checkpoint is an implementation diagnostic, not a faithful
STRONG baseline and may not be used to claim prior dominance or permanently
kill the scientific method.

This issue is essential before expensive training, but does not block writing
the mathematical audit.

### 5. Perturbation Semantics And Relevance Are Under-Specified

`instruction_repetition` and the fixed context wrapper are likely
semantics-preserving, but they may be too weak to create headroom. Image
translation and Gaussian noise may create a measurable shift, but the exact
padding, normalization domain, camera streams, and severity units are not yet
defined.

Required resolution:

- perturb raw RGB before the official image processor;
- perturb both policy camera streams with the same family/severity but
  independently seeded pixel noise where appropriate;
- define pixel range, noise standard deviation, translation pixels, padding,
  and all text strings exactly;
- hash clean and perturbed processor inputs;
- preserve task text and action targets under an explicit allowlist;
- freeze one family-balanced discovery/validation assignment;
- reject any transform that changes goal semantics or collapses to an identical
  processor input.

No family or severity may be selected because of confirmatory outcomes. If the
frozen four-family set has no development headroom, classify `NO_HEADROOM` or
`UNDERPOWERED_OR_UNRESOLVED`; do not substitute a stronger perturbation after
seeing test results.

### 6. Prior Fidelity And Baseline Compute Must Be Transparent

The local Prior must reproduce the published mechanism that is locally
specified: hierarchical perturbation curriculum followed by clean refinement
with the same action objective. It cannot be called official STRONG-VLA.

For fair comparison:

- Prior, Ours, and ablation start Stage II from byte-identical Stage I adapter
  weights;
- all use the same Stage II clean rows and ordering;
- Ours and ablation use the same robust replay rows and ordering;
- Prior may compute and discard `g_r` for diagnostic compute matching, but that
  fact and overhead must be reported;
- ablation uses the simple average only after clean/robust gradient scales are
  audited; otherwise its result could be a scale artifact;
- standard LoRA uses the same checkpoint, demonstrations, total optimizer
  steps, adapter rank, target modules, and selection rule;
- the joint replay ablation, not standard LoRA, is the direct control for extra
  perturbed data.

No additional optimizer or gradient-surgery comparator is essential at the
prototype gate. PCGrad, A-GEM, and CG2A constrain novelty; they do not each
justify another policy.

### 7. The Novelty Claim Must Be Narrowed Again

GEM/A-GEM already protect prior tasks with a reference gradient. PCGrad already
projects conflicting task gradients. CG2A already addresses augmentation
gradient conflict in visual reinforcement learning. STRONG-VLA already frames
VLA clean/perturbed conflict.

IARC may claim only:

- a cross-paper synthesis specialized to VLA robustness consolidation;
- an asymmetric clean-refinement constraint against a perturbation-replay
  action objective;
- matched closed-loop evidence across VLA backbones if results support it.

It may not claim a new generic optimizer, gradient surgery, continual-learning
principle, or augmentation-conflict discovery. A latest-literature recheck is
required before paper packaging.

This narrower claim remains provisionally defensible.

### 8. The Offline And Rollout Partitions Must Be Separate And Explicit

The offline `train/val/test` split is episode-disjoint, but rollout identities
are a second partition. Before any development rollout, freeze:

- development headroom task/reset identities;
- Stage A task/reset identities;
- Stage B task/reset identities;
- any allowed expansion identities.

They must have zero reset-identity overlap and no outcomes may select tasks.
The `1200` offline test rows remain sealed until method, checkpoint, policies,
metrics, thresholds, and rollout manifest are frozen.

### 9. Resource-Contention Evidence Must Stay Quarantined

The audit in `reports/resource_contention_intervals.json` found no active worker
and accepted the already completed EAC success rows without rerun. The exact
start of the Windows Efficiency Mode interval is unknown. Any IARC or inherited
latency, throughput, wall-clock, VRAM-utilization, or resource-efficiency metric
whose overlap cannot be excluded is not final paper evidence. This does not
invalidate synchronous, exception-free task-success rows that pass duplicate
and manifest checks.

## Required Mechanism Evidence

The mathematical audit and Stage 0 must add:

- exact actual-update derivation under the selected Stage II optimizer;
- shared clean/perturbed noise and time tensors;
- exact parameter-vector order and mixed-precision contract;
- analytic and finite-difference tests for agreeing, conflicting, orthogonal,
  zero-reference, and nonfinite gradients;
- before/after robust loss for a tiny realized clean, joint, and IARC step on a
  held-out development batch;
- conflict rate and cosine interval by family/task/phase;
- projected update norm ratio and module contributions;
- clean and perturbed action deltas;
- disk reload, Base identity, action validity, and clean retention;
- validation-only closed-loop Base headroom before full training.

## False-Negative Calibration

The proposed `4 / 40` conflict threshold is a mechanism-activation screen, not a
proof of paper viability. A micro Stage I checkpoint can underrepresent the
conflict that appears after adequate robustness training. Therefore:

- `4+` healthy conflicts may pass the mechanism screen;
- `1-3` conflicts receive the one fixed check exactly as proposed;
- zero conflicts after both checks may stop only the current low-compute
  implementation when Stage I subset fit or robustness acquisition is weak;
- a permanent `DESIGN_FAILURE_NONACTING_MECHANISM` requires healthy Stage I
  acquisition, adequate independent records, and an interval excluding the
  frozen useful conflict rate;
- micro-optimization weakness is
  `LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT` or
  `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`, not a scientific kill.

Risk of false positive: high if conflict is measured with unmatched flow noise,
if perturbations change semantics, or if offline loss shifts do not produce
closed-loop failure.

Risk of false negative: moderate to high if the `20`-step micro Stage I has not
acquired meaningful robustness or if the four perturbations are too weak.

## Evidence Priority

Essential paper/prototype evidence:

- actual-update mathematical validity;
- same-noise/time gradient audit;
- development closed-loop headroom;
- Base, Prior, Ours, ablation, and standard LoRA matched comparison;
- clean retention, action validity, mechanism activation, and split integrity.

Useful diagnostics:

- per-module conflict contributions;
- finite-difference robust-loss change;
- Stage I versus Stage II conflict trajectories;
- family-level offline action-loss breakdown.

Optional supplementary evidence:

- additional severities after the primary protocol closes;
- optimizer sensitivity after the paper claim is established;
- extra perturbation families on a new method cycle.

Irrelevant at the prototype gate:

- a LoRA rank sweep;
- KL between deterministic `7D` actions;
- several gradient-surgery baselines answering the same objection;
- a new auxiliary head, gate, memory, or divergence;
- Quantized OpenVLA-OFT INT4 before SmolVLA prototype GO.

## Reviewer Decision

`REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`.

IARC is not killed before implementation. Researcher A must accept the narrow
cross-paper-synthesis claim, repair the actual-update mathematics, freeze
same-noise/time pairing, define exact perturbations, distinguish micro Stage I
from a real STRONG proxy, add a bounded validation-only closed-loop headroom
screen, preserve the five-policy baseline logic, and maintain the resource
quarantine. Only then may the method proceed to a mathematical audit.

