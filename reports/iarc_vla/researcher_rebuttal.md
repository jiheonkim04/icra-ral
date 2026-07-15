# IARC-VLA Researcher A Rebuttal

Date: 2026-07-15 KST

Decision: `IARC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Frozen proposal hash:
`A1B0CF8BCBCF6A88F27B31EF5E38BAF408A3E62BB34206A1AC9F051EA6B57408`.

Researcher A accepts every essential Reviewer B condition. This rebuttal
narrows and repairs the frozen proposal before mathematical preregistration. It
does not add another module, loss, gate, memory, policy, perturbation family, or
hyperparameter sweep.

## 1. Actual-Update Mathematics

Accepted. Raw Euclidean gradient projection followed by AdamW does not justify
the proposal's realized-step claim.

The minimum-sufficient resolution is frozen:

- Stage I optimizer: AdamW, as ordinary robustness acquisition;
- Stage II optimizer for Prior, Ours, and ablation: explicit SGD;
- Stage II SGD momentum: `0`;
- Stage II weight decay: `0`;
- no gradient clipping;
- one projected logical-batch gradient per optimizer step.

Standard LoRA uses the same optimizer schedule: clean-only AdamW during its
Stage-I-equivalent step budget and clean-only zero-momentum, zero-decay SGD
during its Stage-II-equivalent budget.

For `r = ||g_r||_2^2`, the corrected IARC rule is:

- if `r < 1e-12`, do not claim a valid reference direction and classify the
  row as `robust_gradient_below_floor`;
- if `r >= 1e-12` and `d = <g_c,g_r> >= 0`, use `g_IARC = g_c`;
- if `r >= 1e-12` and `d < 0`, use
  `g_IARC = g_c - (d / r) * g_r`.

There is no epsilon in the projection denominator. With the explicit SGD
parameter delta `Delta theta = -eta * g_IARC`, a conflict row satisfies

`Delta L_r approximately g_r dot Delta theta = -eta * <g_r,g_IARC> = 0`

up to float32 numerical tolerance. The mathematical audit will distinguish
this first-order local constraint from a guarantee on finite-step closed-loop
performance.

Prior and ablation use the same Stage II SGD optimizer. The ablation applies
`(g_c + g_r) / 2` after the objective-scale audit. This isolates projection,
not optimizer choice.

## 2. Shared Flow Stochasticity

Accepted. Every clean/perturbed gradient pair uses:

- the exact same dataset row and demonstration action chunk;
- one shared native flow-noise tensor of shape `[1, 50, 32]`;
- one shared flow-time tensor of shape `[1]`;
- the same noise/time dtype and device;
- the same loss reduction and autocast scope;
- identical proprioception and all nonperturbed inputs;
- only the allowlisted image or text transform changed.

The noise/time seed is a deterministic hash of proposal hash, partition,
sample identity, logical optimizer step, and accumulation index. Clean and
perturbed calls receive cloned tensors with identical hashes. Different pairs
receive different deterministic seeds.

Any clean/perturbed noise, time, action-target, or nonallowlisted-input hash
mismatch is `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE` and stops the audit.

## 3. Gradient Vector Contract

Accepted. The mathematical audit and implementation freeze:

- the lexicographically sorted names and shapes of all trainable LoRA
  parameters;
- exact zero tensors for `None` gradients;
- separate clean and robust accumulation buffers in float32;
- no GradScaler dependence in the vector operation; if scaling is active,
  gradients are unscaled before accumulation and projection;
- projection after the complete logical batch is accumulated;
- no clipping before or after projection;
- finite checks before dot products and before the SGD step;
- resolved-module norm and dot-product contributions;
- explicit zero-reference handling at squared norm `< 1e-12`.

The optimizer receives the reconstructed projected gradients in the original
parameter shapes and order. Unit tests must round-trip flatten/unflatten without
error.

## 4. Development Closed-Loop Headroom

Accepted. Offline action-flow loss cannot be the only headroom evidence.

After the pure implementation and gradient smoke passes, but before full
five-policy training, run a frozen Base-only development headroom screen:

- suites: `libero_spatial` and `libero_goal`;
- task IDs per suite: `[0, 2, 4, 6, 8]`;
- reset identity: `20261601` for every selected task;
- pair count: `10`;
- conditions: one clean and one assigned perturbed episode per pair;
- planned episode count: `20`;
- simulator: one synchronous environment with `use_async_envs=False`;
- same initial state, task, reset identity, action semantics, and horizon within
  each pair;
- policy: frozen Base only;
- no policy selection or configuration tuning from this screen.

Perturbation family is assigned by sorted pair index modulo four. Severity is
the fixed middle severity. Visual perturbations are deterministically generated
per frame; text perturbations remain fixed for the episode.

The screen demonstrates headroom when perturbed success is below clean success
by at least `0.10`, or when at least two clean successes become perturbed
failures. With only `10` pairs, smaller or mixed differences are
`UNDERPOWERED_OR_UNRESOLVED`, not a permanent kill. Zero clean successes is
`NO_SCOREABLE_HEADROOM`; zero perturbation effect on processor inputs is
`DATA_OR_SUPERVISION_FAILURE`.

The micro Stage I checkpoint remains an implementation diagnostic. It is not
the closest-prior baseline and cannot establish prior dominance or a scientific
kill. After full Stage I training, validation must show that the transparent
STRONG proxy improves at least one frozen perturbation metric while leaving
residual failure; otherwise classify prior-acquisition failure or no residual
headroom before confirmatory rollout.

## 5. Exact Perturbation Contract

Accepted. Images are transformed in raw float RGB `[0,1]` before the official
image processor. Both policy camera streams receive the same family and
severity. The exact frozen severities are:

### Gaussian Sensor Noise

- standard deviations: `[0.02, 0.05, 0.10]` in raw `[0,1]` units;
- independent zero-mean Gaussian draws per camera stream;
- deterministic seed per pair, camera, and frame;
- clip to `[0,1]` after addition.

### Image Translation

- absolute shifts: `[4, 8, 16]` pixels;
- direction selected deterministically from up/down/left/right;
- same direction and magnitude for both camera streams in a pair;
- edge-replication padding;
- no resize, crop, or simulator-state change.

### Instruction Repetition

- append the exact original instruction `[1, 2, 3]` additional times;
- separate copies with ` ; `;
- no new verb, object, relation, or negation.

### Context Wrapper

- prepend the exact string
  `Context note: the workspace contains several objects. Task:`;
- prefix repetitions: `[1, 2, 3]` separated by one space;
- append the original instruction exactly once after the prefixes.

Family assignment is task-balanced: sorted task index modulo four, giving ten
offline audit rows per family across `40` tasks. Severity cycles within each
family by sorted within-family index modulo three. The headroom rollout uses the
same assignment rule over its ten sorted pairs and middle severity only.

The runner hashes raw and processed images, raw and tokenized text, flow
noise/time, proprioception, and action targets. Any target change, semantic
allowlist violation, all-identical processed family, duplicate generated pair,
or partition-seed overlap stops as a data/implementation failure.

No family or severity is replaced after outcomes. Evaluation-only unseen
families, if later added, are frozen before confirmatory evaluation and do not
enter training or validation selection.

## 6. Prior And Baseline Fidelity

Accepted.

- `strong_vla_transparent_proxy` is never called official.
- Prior, Ours, and ablation load byte-identical Stage I adapter weights.
- All three share Stage II clean rows, ordering, SGD optimizer, step count, and
  learning rate.
- Ours and ablation share robust replay rows, ordering, flow noise/time, and
  gradient evaluations.
- Prior computes and discards the matched robust gradient only for diagnostic
  compute accounting; its parameter update remains clean-only.
- The ablation's clean and robust loss scales are audited before training. The
  frozen arithmetic mean is retained only if their discovery median gradient
  norm ratio lies in `[0.25, 4.0]`; outside that interval is an objective-scale
  implementation failure requiring normalization in the mathematical audit
  before any trial, not post-result tuning.
- Standard LoRA matches checkpoint, demonstrations, total steps, optimizer
  schedule, rank, target modules, batch/accumulation, and selection rule.

The five policies remain exactly those in the proposal. No PCGrad, A-GEM, CG2A,
rank, or optimizer variant is added at the prototype gate.

## 7. Narrow Novelty

Accepted. The provisional novelty statement is now:

`IARC is a cross-paper synthesis that applies an asymmetric, actual-step
constraint during VLA clean refinement against a paired perturbation-replay
action objective, and tests whether this improves the robustness/clean tradeoff
over STRONG-style stage separation under matched closed-loop evaluation.`

IARC does not claim a new generic optimizer, gradient surgery method,
continual-learning principle, perturbation curriculum, or augmentation-conflict
discovery. A primary-source novelty recheck is mandatory before paper packaging.

## 8. Evidence Partitions

Accepted. Offline partitions remain `1200 / 400 / 1200` train/validation/test
rows with test sealed.

Rollout identities are separate:

- development headroom: reset `20261601`, frozen ten task pairs above;
- Stage A reset identities: reserved `20261611` and `20261612`;
- Stage B reset identities: reserved `20261613` and `20261614`;
- one optional expansion identities: reserved `20261615` and `20261616`.

The exact Stage A/B task allocation is frozen only after one validation
configuration is selected, but tasks may not be selected by observed outcomes.
All reset sets are disjoint. Development headroom outcomes may not tune the four
perturbation families or severities.

## 9. Resource-Contention Quarantine

Accepted. `reports/resource_contention_intervals.json` is authoritative. Any
latency, throughput, wall-clock, VRAM utilization, or efficiency metric whose
overlap with the start-unknown Windows Efficiency Mode interval cannot be
excluded is removed from final paper evidence. It may remain labeled as a local
diagnostic. Synchronous, exception-free, manifest-matched task-success rows
remain eligible after duplicate audit.

Every future detached run must record PID, child PID, heartbeat, status,
partial result, final result, logs, exit code, and exact missing-key resume rule.
No completed row may be repeated after interruption.

## 10. False-Negative Safeguard

Accepted. The `4 / 40` threshold is only an activation pass. A low-conflict
micro checkpoint cannot permanently kill IARC unless Stage I acquisition,
perturbation health, adapter capacity, record independence, and the frozen
confidence rule all pass.

The one fixed check remains exactly one. No method, perturbation, threshold,
seed, learning-rate, rank, or target-module change is permitted during it.

Classification priority:

- collapsed transform or target mismatch -> `DATA_OR_SUPERVISION_FAILURE`;
- zero/invalid adapter gradient or failed subset fit ->
  `LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT` or
  `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`;
- weak micro Stage I acquisition -> implementation/optimization failure;
- healthy acquisition but no conflict after adequate independent evidence ->
  current-formulation nonacting design failure;
- closed-loop Base perturbation tie with a narrow interval excluding useful
  harm -> `NO_HEADROOM`;
- wide or mixed interval -> `UNDERPOWERED_OR_UNRESOLVED`.

## Rebuttal Decision

`IARC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`.

The central method is now one exact Stage II projected SGD update. LoRA remains
low-compute infrastructure. All Reviewer B essential conditions are accepted.
Proceed only to the mathematical mechanism audit; no training, validation
search, headroom rollout, or confirmatory evaluation is yet authorized.

