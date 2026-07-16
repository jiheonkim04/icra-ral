# LCG-VLA Mathematical Mechanism Audit

Date: 2026-07-16 KST

Decision: `LCG_MATHEMATICAL_AUDIT_PREREGISTERED`

Proposal: `reports/lcg_vla/researcher_proposal.md`

Proposal SHA-256:
`F0D980AA0760F143D781C723DB632BC324C1E18F390D9C33C5DA94F3A897D11E`

Reviewer attack: `reports/lcg_vla/reviewer_attack.md`

Researcher rebuttal: `reports/lcg_vla/researcher_rebuttal.md`

This audit freezes LCG-VLA's mathematical mechanism before preregistration,
prototype protocol, implementation, training, validation search, rollout,
simulator access, or confirmatory-test access.

## Scope

LCG's allowed novelty is only:

`A frozen-SmolVLA, Base-preserving, identity-initialized action-cell gate that
learns when deployment-observable original-versus-null language contrast
permits bounded residual edits, with exact Base passthrough when the contrast
is absent or unreliable.`

LCG is not official CAG, not ordinary LoRA, not counterfactual relabeling, not
a new VLA backbone, and not a rescue or reinterpretation of S2C or any previous
closed method.

## Constants

- `H = 50`: SmolVLA action chunk horizon.
- `D = 7`: official LIBERO action dimension.
- `B`: batch size.
- `l_null = ""`: the empty instruction string, passed through the same local
  SmolVLA prompt/tokenizer wrapper as ordinary task text.
- `eps = 1e-6`: numerical floor.
- `d_trans = {0,1,2}`: translation coordinates.
- `d_rot = {3,4,5}`: rotation coordinates.
- `d_grip = {6}`: gripper coordinate.

Default residual/action caps in normalized action units:

- `rho_trans = 0.02`;
- `rho_rot = 0.05`;
- `rho_grip = 0.25`.

Default language-mask threshold:

- `tau_lang = 0.25` after discovery-only contrast normalization.

## Variables And Shapes

For each legal development row:

| Symbol | Shape | Source | Gradient path | Meaning |
| --- | --- | --- | --- | --- |
| `o_t` | tuple | legal current observation | frozen Base only unless adapter hooks are declared | current visual input |
| `q_t` | vector | legal current proprioception/state exposed to SmolVLA | frozen Base only unless adapter hooks are declared | current robot state |
| `l_t` | string | legal task instruction | tokenizer only | original instruction |
| `B_t` | `[B,H,D]` | frozen SmolVLA with `l_t` | stopgrad | original-instruction Base chunk |
| `N_t` | `[B,H,D]` | frozen SmolVLA with `l_null` | stopgrad | null-instruction Base chunk |
| `E_t` | `[B,H,D]` | demonstration action chunk | target only | expert action chunk |
| `V_t` | `[B,H,1]` | action-valid row mask | no gradient | valid horizon cells |
| `U_t` | `[B,H,D]` | `B_t - N_t` | stopgrad | raw language-contrast action difference |
| `s_lang_d` | `[D]` | discovery-only `p95(abs(U_t))` | no gradient | contrast normalization scale |
| `C_t` | `[B,H,D]` | `abs(U_t) / s_lang_d` | stopgrad | normalized language contrast |
| `M_lang` | `[B,H,D]` | `1[C_t >= tau_lang]` | no gradient | language-sensitive cell mask |
| `R_t` | `[B,H,D]` | `E_t - B_t` | target only | Base-to-demonstration residual |
| `x_t` | implementation-defined | legal deployment inputs, `B_t`, `N_t`, `C_t` | trainable only through declared LCG adapter/head | gate/residual features |
| `Delta_theta` | `[B,H,D]` | LCG residual head | trainable | proposed bounded residual edit |
| `G_theta` | `[B,H,D]` | LCG gate head | trainable | action-cell intervention gate |
| `A_LCG` | `[B,H,D]` | edited chunk | trainable through LCG only | LCG output chunk |

No future observation, object pose, reward, success flag, done flag, expert
future action at inference, confirmatory identity, or confirmatory outcome may
enter any variable above.

## Contrast Scale And Mask

Compute discovery-only contrast scales:

`s_lang_d = clamp(p95_discovery(abs(U_t[:,:,d])), 1e-4, 10.0)`.

Validation and later partitions use frozen discovery scales.

The normalized contrast is:

`C_t[:,:,d] = abs(U_t[:,:,d]) / s_lang_d[d]`.

The language mask is:

`M_lang = 1[C_t >= tau_lang]`.

Stage 0 must report `M_lang` positive fractions by task, phase, timestep, and
action group. If the global positive fraction is below `0.05` or above `0.95`,
or if any development task has all-zero or all-one mask, the method stops
before validation search as `LCG_STAGE_0_DATA_OR_SUPERVISION_FAILURE`.

## Identity-Preserving Gate

The residual head is bounded by action group:

`Delta_theta = clip_group(raw_Delta_theta, rho_trans, rho_rot, rho_grip)`.

The gate is:

`G_theta = eta * sigmoid(Z_theta(x_t))`,

where:

- `Z_theta(x_t)` has shape `[B,H,D]`;
- `eta in [0,1]` is a scalar or per-group multiplier initialized exactly to
  `0`;
- any parameterization of `eta` must reproduce exact Base within `1e-6` before
  training and after disk reload.

The output chunk is:

`A_LCG = B_t + M_lang * G_theta * Delta_theta`.

All multiplication is elementwise. Gradients flow only into `theta` and `eta`.
No gradient flows into frozen SmolVLA Base, `B_t`, `N_t`, `E_t`, `U_t`,
`C_t`, `M_lang`, or discovery scales.

If `M_lang = 0` or `eta = 0`, LCG must output exact Base.

## CAG Proxy

The closest-prior proxy is:

`A_CAG(beta) = B_t + beta * clip_group(U_t, rho_trans, rho_rot, rho_grip)`.

Allowed validation-only coefficients:

`beta in {0.25, 0.5, 1.0}`.

Stage 0 may report all three fixed coefficients but may not select or retune
using confirmatory data. If official CAG assets are installed and verified
before confirmatory testing, official CAG must replace or supplement this proxy
unless a protocol-incompatibility reason is recorded before seeing
confirmatory outcomes.

## Objective Terms

All objectives use discovery/training rows only before validation selection.
All reductions are masked by `V_t` and are coordinate means unless stated.

### 1. Language-Masked Residual Huber

Variables: `A_LCG`, `E_t`, `M_lang`, each `[B,H,D]`.

Formula:

`L_res = mean V_t * M_lang * Huber_delta((A_LCG - E_t) / sigma_d)`.

Default scales:

- `sigma_trans = 0.02`;
- `sigma_rot = 0.05`;
- `sigma_grip = 1.0`;
- `delta = 1.0`.

Units: dimensionless normalized action error.

Gradient path: `A_LCG -> G_theta, Delta_theta -> theta`.

Intended effect: learn bounded residual edits only in language-sensitive cells.

Simpler alternative: matched standard LoRA on demonstrations.

Required ablation: `standard_lora`.

### 2. Clean Retention

Variables: `A_LCG`, `B_t`, `M_lang`.

Formula:

`L_clean = mean V_t * (1 - M_lang) * Huber_delta((A_LCG - B_t) / sigma_d)`.

Scale and units: same as `L_res`.

Gradient path: `A_LCG -> G_theta, Delta_theta -> theta`.

Intended effect: preserve Base where language contrast is absent.

Simpler alternative: exact Base passthrough.

Required diagnostic: inactive-gate exact-Base report.

### 3. Gate Sparsity

Variable: `G_theta`, shape `[B,H,D]`.

Formula:

`L_gate = mean V_t * G_theta`.

Scale: dimensionless.

Gradient path: `G_theta -> theta`.

Intended effect: prevent global action edits.

Simpler alternative: fixed CAG coefficient.

Required ablation: `counterfactual_action_guidance_proxy`.

### 4. Action Validity Penalty

Let `post(A)` be the official SmolVLA/LIBERO postprocessor. Let
`invalid(A)` be zero for finite in-bound actions and positive for NaN, inf, or
postprocessor-bound violations.

Formula:

`L_valid = mean invalid(post(A_LCG))`.

Scale: normalized action-bound violation.

Gradient path: implementation-dependent only through `A_LCG`; if the
postprocessor is nondifferentiable, this term is reported as a smoke metric and
not used for gradient updates.

Required diagnostic: action-validity report by action group.

## Total Objective

Default Stage 0 small-fit objective:

`L_total = L_res + 1.0 L_clean + 0.01 L_gate + 1.0 L_valid`.

Before training, Stage 0 must estimate term magnitudes and gradient norms on a
small development batch. If any weighted gradient norm is more than `20x` the
median weighted term norm, the run must stop as
`LCG_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE` unless a bounded
validation-only search is preregistered.

No KL divergence is used. Deterministic 7D actions and SmolVLA flow vectors are
not probability distributions.

## Contrast-Residual Headroom Diagnostics

Stage 0 must test whether language contrast is useful rather than merely
nonzero.

Required diagnostics:

- Spearman correlation between per-cell `C_t` and `abs(R_t / sigma_d)`;
- validation Huber of a contrast-conditioned residual probe versus a task/phase
  residual baseline;
- best fixed `A_CAG(beta)` Huber versus Base;
- oracle masked residual Huber using `M_lang` and clipped `R_t`.

Pass to bounded validation is disallowed if:

- contrast-residual Spearman is below `0.05`;
- contrast-conditioned residual probe fails to beat task/phase residual
  baseline by at least `1%`;
- best CAG proxy leaves no measurable residual headroom for the masked oracle;
- null-branch actions are invalid or collapsed.

The oracle is diagnostic only and is not an inference method.

## Stage 0 Pass And Stop Gates

Stage 0 may pass to bounded validation only if all are true:

- proposal hash matches
  `F0D980AA0760F143D781C723DB632BC324C1E18F390D9C33C5DA94F3A897D11E`;
- no reward, success, done, simulator result, object pose, future observation,
  or confirmatory identity is read;
- `B_t`, `N_t`, `U_t`, `C_t`, `M_lang`, and `E_t` parse with expected shapes;
- null-branch actions are finite and postprocessor-valid;
- Base/null contrast is noncollapsed;
- residual labels are noncollapsed;
- contrast predicts residual headroom above the task/phase baseline;
- CAG proxy leaves residual headroom;
- initialized and disk-reloaded LCG equals Base within `1e-6`;
- expected LCG parameters receive finite nonzero gradients;
- frozen SmolVLA Base parameters receive no gradients;
- LCG after a small fit differs from Base, CAG proxy, no-language-contrast
  ablation, and standard LoRA;
- action deltas respect group caps;
- clean retention passes on inactive-mask rows.

Stop classes:

- `LCG_STAGE_0_DATA_OR_SUPERVISION_FAILURE`: collapsed contrast, collapsed
  residuals, invalid null branch, invalid masks, or insufficient legal labels.
- `LCG_STAGE_0_NO_USABLE_HEADROOM`: CAG proxy or simple baselines leave no
  measurable headroom.
- `LCG_STAGE_0_DESIGN_FAILURE`: language contrast does not predict residual
  usefulness, LCG equals CAG coefficient tuning, no-language ablation explains
  the effect, or standard LoRA explains the effect.
- `LCG_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE`: identity reload,
  gradient, shape, objective-scale, or action-validity failures.
- `LCG_STAGE_0_PASS_TO_BOUNDED_VALIDATION`: all Stage 0 gates pass.

Stage 0 is development-only and is not a closed-loop scientific result.

## Required Ablations

1. `counterfactual_action_guidance_proxy`
2. `lcg_no_language_contrast_ablation`
3. `standard_lora`
4. contrast-magnitude-only gate diagnostic
5. task/phase residual diagnostic
6. inactive-gate exact-Base diagnostic

## Validation Search Envelope

If Stage 0 passes, bounded validation search may consider at most six
configurations total. The only tunable factors allowed are:

- `beta` for CAG proxy from `{0.25, 0.5, 1.0}`;
- one LCG clean-retention coefficient from `{0.5, 1.0, 2.0}`;
- one residual/gate capacity choice from `{small, medium}`.

No combinatorial grid over all factors is allowed. One final configuration must
be selected on validation only before confirmatory testing.

## Current Status

No LCG implementation, training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened before this audit.

Immediate next stage: preregistration before prototype protocol,
implementation, validation search, training, or rollout.
