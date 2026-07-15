# IARC-VLA Researcher A Proposal

Date: 2026-07-15 KST

Decision: `IARC_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

## Method Identity

Name: `IARC-VLA`, Interference-Aware Robustness Consolidation for VLA
policies.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Primary claim axis: closed-loop task robustness under semantics-preserving
visual and textual perturbations, with clean task fidelity retained.

Closest external prior: STRONG-VLA,
https://arxiv.org/abs/2604.10055.

Secondary mechanism prior: Gradient Episodic Memory,
https://arxiv.org/abs/1706.08840.

The STRONG-VLA arm is a transparent local proxy, not an official reproduction.
No official STRONG-VLA code or checkpoint was verified. Published numbers are
context and are not a direct local baseline.

## Research Question

STRONG-VLA attributes the clean/robust tradeoff to conflicting optimization and
separates robustness acquisition from clean task refinement. Its Stage II,
however, applies ordinary clean-data gradients and does not constrain whether a
clean update erases robustness acquired in Stage I.

Can clean Stage II refinement be made first-order non-increasing for a matched
perturbation-replay action loss, only when the two gradients conflict, and does
that improve perturbed closed-loop success over STRONG-style refinement without
materially harming clean success?

## Scientific Method

### Stage I: Transparent STRONG Proxy

Starting from the same Base policy, optimize the ordinary SmolVLA action-flow
loss under a frozen curriculum of semantics-preserving visual and textual
perturbations. Perturbation probability and severity increase monotonically
over the predeclared Stage I phases.

No new labels, auxiliary heads, or privileged state are introduced. Clean and
perturbed examples share the same demonstration action target because every
training perturbation is required to preserve task semantics.

### Stage II: Interference-Aware Consolidation

For trainable parameters `theta`, pair a clean batch `B_c` with a perturbed
replay batch `B_r` from the same task distribution. Define

`L_c(theta) = L_action(B_c; theta)`

`L_r(theta) = L_action(B_r; theta)`

`g_c = grad_theta L_c(theta)`

`g_r = grad_theta L_r(theta)`

`d = <g_c, g_r>`.

The IARC update gradient is

`g_IARC = g_c - min(0,d) * g_r / (||g_r||_2^2 + epsilon)`.

`epsilon` is a fixed numerical stabilizer, not a searched coefficient. When
`d >= 0`, IARC is exactly ordinary clean refinement. When `d < 0`, the clean
gradient is projected against the robust replay gradient. With negligible
`epsilon`, `<g_IARC,g_r> = 0`; the descent update is first-order
non-increasing for `L_r`.

Only trainable adapter gradients enter the dot products. The full precision
accumulator used for dot products and norms must be `float32` even when model
forward/backward uses mixed precision.

### Narrow Novelty

IARC does not claim to invent gradient projection, constrained continual
learning, perturbation curricula, or LoRA. The narrow claim is the explicit
VLA Stage II consolidation rule on the clean-versus-perturbed action objective,
tested against STRONG-style clean refinement and matched unprojected replay.

This is materially different from Cycle 12's unselected DCR sketch. DCR was an
identity-preserving two-stage adapter proposal; IARC defines a per-update
conflict measurement, an exact constrained update, and an observable mechanism
activation event.

## Low-Compute Parameterization

The scientific method is realized locally through:

- official SmolVLA checkpoint at `C:\assets\checkpoints\smolvla_libero`;
- frozen Base weights;
- the official LeRobot `wrap_with_peft` LoRA path;
- fixed LoRA rank `4`;
- the default SmolVLA PEFT target modules returned by that official path;
- LoRA alpha and target selection frozen by the official wrapper;
- mixed precision;
- physical batch size `1` and gradient accumulation where needed;
- AdamW;
- disk-persistent, reloadable adapter checkpoints;
- no full-model fine-tuning and no rank sweep.

The Stage 0 runner must record the exact resolved target-module names and
trainable parameter count. If the official wrapper targets modules that cannot
express the measured clean and robust gradients, one rank or target-capacity
adjustment is allowed only after classification as
`LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT`. The scientific update rule may not
change.

Removing the words LoRA and QLoRA does not remove IARC's scientific method.

## Data Partitions

Authoritative source:
`reports/official_smolvla_split_manifest.json` and the already downloaded
official `lerobot/libero` data at
`C:\assets\datasets\lerobot_libero`.

Frozen episode-disjoint source counts:

- training/discovery source: `1200` rows;
- validation source: `400` rows;
- confirmatory test source: `1200` rows.

Stage 0 may decode only:

- `80` training-source rows: two deterministic mid-phase rows per task across
  all `40` tasks;
- `40` validation-source rows: one deterministic mid-phase row per task.

Within the `80` training rows:

- `40` rows, one per task, form the micro Stage I fit set;
- the other `40` rows form the independent gradient-conflict audit set.

The `40` validation rows are used only for action-loss headroom, clean
retention, perturbation health, and action diagnostics. They may not select or
change Stage 0 thresholds.

All `1200` test rows remain sealed during Stage 0, training, and validation
search. Their observations, actions, policy outputs, and perturbations may not
be decoded or computed before the final configuration is frozen.

The runner must prove zero sample, frame, episode, task-reset identity, and
generated-perturbation seed overlap across the three evidence partitions.

## Frozen Development Perturbations

Stage 0 and later training use exactly four semantics-preserving families:

1. `gaussian_sensor_noise`: additive zero-mean RGB noise with clipping;
2. `image_translation`: bounded image shift with edge padding, never changing
   the task instruction or simulator state;
3. `instruction_repetition`: repeat the exact operative instruction once;
4. `context_wrapper`: prepend the fixed non-imperative text
   `Context note: the workspace contains several objects. Task:` to the exact
   original instruction.

Each family has three predeclared severities. Numeric parameters are frozen in
the mathematical audit before implementation. No semantic drift, changed
object, changed spatial relation, adversarial instruction override, or
goal-changing transform is allowed for training.

Family assignment is deterministic, task-balanced, and seed-derived from the
proposal hash, partition, task, and row identity. Stage 0 must report:

- count by family, severity, task, and phase;
- pixel delta or token delta distribution;
- duplicate image and tokenized-instruction hashes;
- unchanged action-target hash for every clean/perturbed pair;
- nonzero perturbation variance;
- no all-clean or all-identical family;
- no target-changing pair.

Evaluation-only perturbations may later include held-out families or severities,
but they must be frozen before confirmatory rollout and never enter training or
configuration selection.

## Stage 0 Development Audit

Stage 0 is a bounded implementation and mechanism audit, not confirmatory
testing and not a paper result.

### Fixed Micro Stage I

Use the `40` micro-fit rows for exactly `20` optimizer steps at learning rate
`1e-4`, seed `1601`, rank `4`, batch size `1`, and the frozen Stage I curriculum.
This micro fit is not a validation configuration and may not select a
hyperparameter.

Required checks:

- before-training adapter identity versus Base;
- finite action loss;
- finite nonzero gradients on expected LoRA parameters;
- loss decrease on the fixed micro subset;
- no Base-weight updates;
- peak CUDA allocation below `15.5 GiB`;
- saved adapter checkpoint, SHA256 manifest, disk reload, and output equality.

### Gradient Conflict Audit

At the reloaded micro Stage I checkpoint, evaluate one clean and one assigned
perturbed action-loss gradient for each of the `40` independent audit rows.

For each pair report:

- clean and robust losses;
- `||g_c||_2`, `||g_r||_2`, dot product, cosine, and conflict indicator;
- projection coefficient;
- `||g_IARC||_2`;
- pre- and post-projection `<g,g_r>`;
- finite/nonzero gradient tensor counts;
- parameter-group contributions;
- family, severity, task, phase, and sample identity.

The exact projection implementation must also pass deterministic tensor unit
cases for agreeing, conflicting, orthogonal, zero-robust, and nonfinite inputs.

Mechanism activation passes when:

- at least `4 / 40` audit pairs have negative cosine below `-0.01`;
- at least two perturbation families contain a conflict event;
- all projected conflict rows satisfy
  `<g_IARC,g_r> >= -1e-6 * max(1, ||g_IARC||_2 ||g_r||_2)`;
- IARC differs from both ordinary clean and unprojected joint gradients on every
  conflict row;
- no projection is applied on agreeing rows.

If `1-3` conflicts are observed or only one family activates, classify
`UNDERPOWERED_OR_UNRESOLVED` and permit exactly one fixed check on the remaining
`40` deterministic training-source rows after `20` additional Stage I micro
steps. No threshold, perturbation, learning-rate, seed, or adapter change is
allowed. Zero conflicts across both checks, with healthy nonzero gradients and
perturbations, is `DESIGN_FAILURE_NONACTING_MECHANISM`; collapsed perturbations
or gradients are `DATA_OR_SUPERVISION_FAILURE` or
`LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT`, not a scientific kill.

### Headroom And Policy Disruption

On the `40` validation rows, report:

- Base clean and perturbed action-flow loss;
- micro-STRONG clean and perturbed action-flow loss;
- diagnostic oracle loss obtained by using the clean input for the same target;
- clean-to-perturbed action delta;
- Base-to-adapter translation, rotation, and gripper deltas;
- finite and in-range action fraction;
- clean validation retention;
- inference latency and memory as diagnostics only.

The oracle is diagnostic and is never an inference policy. Stage 0 may not stop
for `NO_HEADROOM` unless the Base and micro-STRONG perturbation-loss increase is
nonpositive with an episode/task bootstrap upper interval below the frozen
practical threshold. A wide or mixed interval is unresolved, not a kill.

Any timing or resource metric overlapping the interval recorded in
`reports/resource_contention_intervals.json`, or whose overlap cannot be
excluded, is quarantined from final paper evidence. Task and action diagnostics
remain usable if they pass their own validity checks.

## Bounded Validation-Only Search

Validation search is forbidden until Stage 0 passes.

Maximum search budget: `6` total training trials.

- Stage II learning rates: `{5e-5, 1e-4, 2e-4}`;
- seeds: `{1601, 1602}`;
- fixed Cartesian total: `3 x 2 = 6` trials;
- no architecture, rank, target-module, perturbation-family, stage-length,
  optimizer, or threshold sweep.

Each trial uses the same Stage I checkpoint construction, training rows, and
Stage II paired data. Save all six trial outcomes and negative results.

The preregistered validation score is

`0.40 * robust_loss_improvement`

`+ 0.25 * clean_retention`

`+ 0.15 * constraint_satisfaction`

`+ 0.10 * action_validity`

`+ 0.10 * mechanism_activation`

with every component normalized by frozen discovery-only scales documented in
the mathematical audit. Closed-loop validation may replace the loss proxy only
if a validation-task rollout partition is frozen before any trial. Offline
action L2 alone may not select the configuration.

Select exactly one learning rate by mean score across the two seeds. Freeze the
selected rule and one designated seed before confirmatory evaluation. Do not
select by the better seed alone.

## First Serious Comparison

The first paper-oriented matched comparison contains exactly five policies:

1. `smolvla_base`
2. `strong_vla_transparent_proxy`
3. `iarc_vla_full`
4. `iarc_unprojected_joint_replay_ablation`
5. `standard_lora_clean_only`

Policy definitions:

- Base: unchanged official SmolVLA.
- Prior: frozen Stage I curriculum followed by ordinary clean-only Stage II.
  It may compute `g_r` for diagnostic compute matching but must discard it from
  the update.
- Ours: the same Stage I checkpoint and the IARC Stage II projected clean
  update.
- Ablation: the same paired Stage II batches and gradient evaluations, updated
  with `(g_c + g_r) / 2` without projection.
- Standard LoRA: ordinary clean-only LoRA adaptation from Base for the same
  total optimizer-step budget, checkpoint, demonstrations, optimizer, rank,
  target modules, batch/accumulation, and selection rule.

The joint-replay ablation controls perturbation data and gradient compute. The
standard LoRA control is required because generic policy adaptation remains a
plausible explanation.

| Comparison | Scientific question |
| --- | --- |
| Base vs Ours | Does IARC improve the same backbone under perturbation while retaining clean behavior? |
| Prior vs Ours | Does explicit conflict protection improve over STRONG-style stage separation? |
| Ablation vs Ours | Is projection needed beyond paired robust replay and extra gradient compute? |
| Standard LoRA vs Ours | Can ordinary matched adaptation explain the gain? |

No sixth policy is authorized before Stage A.

## Training Budget After Stage 0

The exact Stage I and Stage II step counts are frozen in the mathematical audit
after a source-throughput preflight and before validation search. The budget
must fit the one-RTX-5080 campaign limit and remain identical across Prior,
Ours, and ablation. Standard LoRA receives the same total optimizer steps.

No policy may use confirmatory tasks, reset identities, outcomes, observations,
or actions for training or selection.

## Stage A And Stage B

After one configuration is frozen:

- Stage A: approximately `10` paired episodes per policy on a frozen manifest;
- Stage B: at least `40` paired episodes per key policy if Stage A is
  noncatastrophic;
- one expansion to `80` only when Stage B is genuinely unresolved under the
  active governance.

Report clean and perturbed success, paired wins/losses/ties, bootstrap interval,
effect size, failure-rate reduction, per-task results, mechanism conflict and
projection rates, action validity, clean retention, latency, VRAM, parameters,
and training cost. Resource metrics from the recorded Windows Efficiency Mode
interval are excluded from final evidence.

Stage A may permanently kill only under the active catastrophic criteria. Small
differences advance. Confirmatory outcomes may not retune this method.

## Paper-Candidate Gate

IARC becomes a serious paper candidate only if:

- SmolVLA plus IARC beats SmolVLA;
- IARC beats the transparent STRONG proxy on the matched perturbation claim;
- IARC beats unprojected joint replay;
- standard LoRA does not explain the gain;
- clean behavior is retained;
- projection activates in relevant states and satisfies the constraint;
- novelty remains defensible against current literature.

After SmolVLA prototype GO, immediately verify Quantized OpenVLA-OFT INT4 versus
Quantized OpenVLA-OFT INT4 plus IARC, add one claim-specific second condition or
benchmark, include directly relevant recent baselines where feasible, and
prepare statistical and efficiency evidence.

## Stop And Classification Rules

Use the active governance labels, including:

- `DATA_OR_SUPERVISION_FAILURE` for collapsed or target-changing perturbations;
- `NO_HEADROOM` only after the frozen decisive headroom rule;
- `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` for invalid gradients, projection,
  checkpoint, or integration;
- `LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT` when rank-4 PEFT cannot express
  the unchanged method;
- `DESIGN_FAILURE_NONACTING_MECHANISM` only after the fixed one-check rule;
- `SIMPLE_BASELINE_EXPLAINS_METHOD` when standard LoRA or unprojected replay
  accounts for the gain;
- `PROTOTYPE_GO` only after the full evidence gate.

Do not rescue a valid kill by changing perturbations, labels, thresholds,
partitions, policy list, stage lengths, or test identities. A major redesign
after confirmatory evaluation begins a new method cycle.

## Next Action

Reviewer B must now attack novelty, STRONG proxy fidelity, GEM/gradient-surgery
proximity, perturbation semantics, conflict observability, the projection
derivation, LoRA capacity, baseline fairness, source partitions, false-negative
risk, and compute feasibility before any implementation.

