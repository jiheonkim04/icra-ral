# PCAV-VLA Researcher A Rebuttal

Date: 2026-07-15 KST

Decision: `PCAV_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

The frozen proposal remains unchanged at
`E8B23C755C6D4E450FD193101CC0B15F88AAFE20E137A0F86830ED6D421E12AA`.
This rebuttal narrows the executable interpretation in response to Reviewer B.

## 1. Candidate-Oracle Headroom Before Any Head Training

Stage 0A generates exactly four fixed-noise Base candidates on `24` discovery
rows, `8` per task and balanced across episode thirds. Candidate index zero is
the direct deterministic Base output.

For each row, compare the first `10` postprocessed actions with the aligned
demonstration prefix. Report translation, rotation, and gripper error
separately and a fixed standardized aggregate using discovery action standard
deviations computed before candidate generation.

Headroom gate:

- `PASS`: at least `25%` of rows have an alternative with at least `5%` lower
  aggregate error than Base, and median oracle relative reduction over those
  improvable rows is at least `5%`;
- `EXACT_NO_HEADROOM`: no row has any strictly better unique alternative;
- `UNDERPOWERED_OR_UNRESOLVED`: some rows improve but the pass threshold is not
  met.

One preregistered discovery-only expansion from `24` to `96` rows is allowed
for `UNDERPOWERED_OR_UNRESOLVED`. It is an evidence expansion, not a
hyperparameter search. If the expanded audit still lacks the pass threshold,
classification is `NO_USABLE_HEADROOM`; it is not a result about all candidate
verification methods.

The FAMR endpoint is never loaded.

## 2. Explicit Future Representation

PCAV will not add a learned delta to arbitrary Transformer context.

For each camera, frozen SmolVLA `embed_image` produces `64 x 960` tokens. Mean
pool each stream and concatenate:

`v_t in R^1920`.

The episode-initial representation is `v_0 in R^1920`. Frozen masked-mean
SmolVLA language-token embeddings produce:

`e_ell in R^960`.

Fixed Gaussian projections with seed `1801` and scale `1/sqrt(128)` map visual
and language vectors to `128` dimensions. These projections are serialized and
never trained.

The consequence model predicts the future projected visual representation
directly:

`F_omega(r(v_t), s_t, a_i[0:10], r(e_ell)) -> r(v_t+10)`.

No residual addition appears in the executable method. The primary baseline
is persistence, `r(v_hat_t+10) = r(v_t)`. An action-shuffled consequence model
is a diagnostic only, not a sixth policy.

## 3. Initial-State Progress Anchor

The progress head receives:

- episode-initial projected visual representation `r(v_0)`;
- current or predicted-future projected visual representation;
- projected language representation `r(e_ell)`;
- normalized 8D proprioception.

The initial observation is captured at reset and retained in policy memory. It
is deployment-observable, contains no simulator identity, and is reset exactly
when the environment resets.

The required ablation inside Stage 0 reporting removes `r(v_0)` while holding
all other inputs fixed. It is a mechanism diagnostic, not an additional
closed-loop policy.

## 4. Temporal Label Health And Shortcut Audit

Before progress training, report by task:

- episode count and length distribution;
- normalized-time histogram;
- terminal padding and repeated-frame frequency;
- repeated-action and near-zero-motion frequency;
- early/middle/late pair counts;
- translation/rotation/gripper path-length distributions.

Exclude terminal padding, exact repeated frames, and pairs separated by fewer
than `10` frames. Pair labels are symmetrized.

Compare progress ordering against:

1. task-only constant;
2. proprioception-only MLP;
3. frame-difference norm;
4. normalized proprioceptive path length.

The full progress model must beat the strongest trivial baseline on held-out
discovery episodes before validation search. Failure after adequate training is
`DATA_OR_SUPERVISION_FAILURE` or `UNDERPOWERED_OR_UNRESOLVED`, depending on
label and capacity diagnostics, not a scientific kill.

## 5. Frozen Consequence Horizon

The future offset is exactly `10` dataset frames at `10 Hz`, or one second.
The model consumes exactly the first `10` actions of the `50 x 7` SmolVLA
candidate chunk. The target is the two-camera visual representation at frame
`t+10` from the same episode.

Rows without a valid `t+10` target are masked before split construction. No
other horizon is searched. Native and postprocessed action semantics are
audited; the consequence model uses postprocessed action units because those
are the executed units and match the raw demonstration audit.

## 6. Transparent TACO Proxy

The closest-prior arm is explicitly a faithful transparent proxy, not an
official SmolVLA checkpoint reproduction.

Frozen specification:

- feature: final denoising-step action-expert suffix hidden state immediately
  before `action_out_proj`, expected shape `50 x 720`, mean pooled to `720`;
- high-fidelity noising levels: `{0.25, 0.50, 0.75, 1.00}`;
- expert endpoint estimate at level `t`: `x_t - t * v_theta(x_t, t)`;
- retained demonstration feature: the level with minimum standardized
  endpoint error to the demonstration action;
- Rademacher target dimension: `32`;
- target seed: `1801`, deterministic by row identity;
- CFN: `Linear(720,256)`, GELU, `Linear(256,32)`;
- objective: mean squared error to fixed Rademacher targets;
- optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`;
- pseudo-count: `1 / (||f_psi(h)||_2^2 + 1e-6)`;
- selection: maximum pseudo-count, stable tie break to lowest candidate index;
- support percentile population: all finite non-Base candidate scores on the
  validation rows, with Base always eligible in PCAV.

Official TACO code and equations are the mechanism source. Any unavoidable
SmolVLA hook difference is recorded in a proxy-fidelity manifest.

## 7. Capacity And Training Boundary

Stage 0B's `20` steps test only gradients, loss direction, serialization, and
noncollapse. It cannot produce a scientific rejection.

The fixed full development architecture is:

- CFN: approximately `193k` parameters as above;
- consequence MLP: input `334`, hidden widths `256,256`, output `128`;
- progress MLP: input `392`, hidden widths `256,128`, scalar sigmoid output;
- total learned parameters: less than `600k`, verified from the implementation.

The full schedule is frozen before it runs:

- support head: `2,000` optimizer steps;
- consequence and progress heads: `2,000` joint optimizer steps;
- physical batch `16`, gradient accumulation `1`;
- AdamW, learning rate `3e-4`, weight decay `1e-4`;
- seed `1801`;
- no architecture or learning-rate search.

A second seed is allowed only for the selected validation configuration as
already budgeted. Parameter counts, objective magnitudes, gradient norms,
gradient conflicts, peak memory, and checkpoint hashes are persisted.

## 8. Numerical Action-Validity Gates

Apply to every candidate and selected chunk in postprocessed 7D units:

- finite fraction exactly `1.0`;
- absolute maximum at most `1.25`;
- fraction outside `[-1,1]` no greater than direct Base fraction plus `0.01`;
- p99 exceedance beyond `[-1,1]` no greater than direct Base p99 exceedance plus
  `0.02`;
- simulator action-space acceptance must be `1.0` before rollout.

Report first-10-step candidate-versus-Base translation L2, rotation L2, and
gripper disagreement. These are disruption diagnostics, not silently tuned
gates. Candidates failing any hard validity gate are ineligible. No
post-result clipping or threshold change is allowed.

## 9. Exact Initial Base Behavior

The serialized policy state contains a `heads_trained` boolean initialized
`false`. While false, selection returns candidate zero without evaluating a
learned score. After training, strict `>` comparisons and stable lowest-index
ties preserve Base on equal scores.

Before and after disk reload, report:

- direct Base action;
- candidate-zero action;
- selected action;
- maximum absolute identity error;
- selected index;
- support score;
- current, candidate, and advantage progress scores;
- intervention or fallback reason.

The required initial maximum identity error is `0.0`.

## 10. Clean Retention

Offline clean rows come from the original 40-task stable artifact with all
target and confirmatory identities excluded. Validation selection reports
intervention rate, action delta, validity, and exact Base fallback on these
rows.

Before confirmatory evaluation, freeze a small disjoint closed-loop clean
manifest shared by all five policies. Clean success is part of the validation
score and paper gate.

## 11. Standard LoRA As A Reviewer-Killer Only

Standard LoRA receives the same target discovery demonstrations and its own
reported compute. It is not PCAV's generator, does not provide PCAV features,
and cannot be substituted after a failed Base candidate-oracle audit.

If untouched Base has no candidate headroom, PCAV stops as
`NO_USABLE_HEADROOM`. Building an adapted generator would require a new frozen
method cycle.

## 12. Activation Without Rewarding Intervention

Validation reports intervention frequency but does not maximize it. The score
requires:

- nonzero acting rows;
- no catastrophic or global intervention;
- better paired validation success or the closest feasible proxy;
- clean retention and action validity.

Tie breaking chooses lower intervention, then lower overhead, then fewer
candidates. Zero intervention after demonstrated headroom and adequate
optimization is exact equivalence; zero intervention with weak capacity is
underpowered.

## 13. Narrow Novelty Claim

PCAV claims none of its components individually. The contribution is exactly:

> hard support eligibility, followed by action-conditioned progress preference,
> followed by Base-relative abstention for a frozen VLA candidate set.

TACO owns coupled pseudo-count support. ProgressVLA owns progress-conditioned
future modeling and action guidance. VLA-ATTC owns adaptive candidate
deliberation. RoboMonkey owns its sampling and VLM verification pipeline.

PCAV fails its contribution if support-only or progress-only explains the
closed-loop gain.

## 14. Durable Execution And Resource Quarantine

Every job longer than a unit test writes:

- PID;
- heartbeat;
- status;
- atomic partial JSON;
- final JSON;
- stdout/stderr log;
- exit code.

Resume only missing row or optimizer-step keys. Never duplicate a living
worker. Validate partial JSON before resume. Every rollout key is
`(policy, task, reset_identity)` and must be unique and present in the frozen
manifest.

The recorded Windows Efficiency Mode interval remains a resource-contention
interval. No overlapping timing, throughput, latency, VRAM-utilization, or
wall-clock result enters paper evidence. Closed-loop success rows remain valid
only under synchronous simulation, unchanged action semantics and identities,
zero timeout/exception, and zero duplicate rows.

## Rebuttal Decision

The method is ready for a mathematical mechanism audit and executable
preregistration. No training, validation search, confirmatory decoding, or
rollout is authorized by this rebuttal alone.
