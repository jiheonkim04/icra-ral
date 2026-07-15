# SPARC-VLA Researcher A Rebuttal

Date: 2026-07-15 KST

Decision: `SPARC_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

The frozen proposal remains unchanged at
`CC2F9ACCE2A26EC438C58F2854ADC95134354C245CAD8ED961D29A895DBC697D`.
This rebuttal accepts every Reviewer B constraint and narrows the executable
method.

## 1. Narrow Novelty And Claim Boundary

SPARC claims only the following construction and evidence:

> Under a frozen no-target-failure-activation operator fit, combine a target-
> success conceptor with an equal-task-weight multi-source failure conceptor;
> this should beat complete single-source COAST transfer and failure-only
> suppression on the same frozen VLA.

The contribution fails if any of these occurs in an adequately powered matched
test:

- the COAST proxy matches or beats SPARC;
- `sparc_source_failure_only` matches or beats SPARC;
- standard LoRA explains the gain;
- clean retention fails;
- the proposed geometry is absent.

No claim of generic conceptor novelty, generic negative guidance, zero-failure
adaptation, new-task generalization, or first frozen-policy failure use is
allowed.

## 2. Faithful SmolVLA Post-Residual Site

The MLP-output hook named in the proposal will be used only as an optional
debug capture. It is not the SPARC intervention site.

For selected residual layer `l in {0, 5, 11, 14}`, the scientific tensor is
the full action-expert hidden state after both residual additions at layer `l`.
In the installed implementation this is exactly the tensor passed into
`lm_expert.layers[l+1].input_layernorm` on the next layer.

The adapter registers an inference-only forward pre-hook on that action-expert
`input_layernorm`:

1. assert `torch.is_grad_enabled() == false`;
2. clone the incoming `[B, 50, 720]` tensor for capture and diagnostics;
3. when unconfigured or `beta = 0`, do not write to it;
4. when configured, compute the steered tensor and copy it in-place into the
   incoming tensor under `torch.no_grad()`;
5. allow the original forward to continue.

The in-place copy is required because SmolVLA later reuses the same tensor in
the next residual addition. Returning a transformed layernorm input alone
would steer attention but leave the skip path unsteered and is forbidden.

The hook is on the action expert only. Prefix-only VLM cache construction has
no action-expert tensor and must not fire. The installed package is not edited.

Stage 0 requires:

- exactly one hook firing per denoising step in each action-generation pass;
- shape `[1, 50, 720]`;
- `10` ordered denoising-step captures;
- capture-only and `beta = 0` action identity error exactly `0.0`;
- hook removal restoring direct Base exactly;
- disk-reloaded adapter behavior matching in-memory behavior within `1e-6` for
  configured float32 operators and exactly for unconfigured Base.

Failure is `IMPLEMENTATION_FAILURE` and blocks labeled collection.

## 3. Token-Level Construction And Application

For fitting, each replan/denoising record retains both:

- full `H in R^(50 x 720)` post-residual token tensor;
- token mean `x = mean_token(H) in R^720`.

Conceptor covariances use the token means, matching COAST. At inference, the
operator is applied independently to every token:

`H_steered[b, t, :] = H[b, t, :] M^T`.

It is never applied only to the pooled vector and broadcast as an additive
shift.

Diagnostics report per-token input norm, output norm, delta norm, token index,
action-index grouping, maximum token delta, and the postprocessed `50 x 7`
action consequence. A token-axis collapse or a constant broadcast
implementation is `IMPLEMENTATION_FAILURE`.

## 4. Episode-Balanced Failure Aggregation

The full `12`-reset discovery manifest runs for every source and target task.
It does not stop when a favorable class quota is reached.

For each completed episode, retain at most `16` replan indices selected by
uniform quantiles over the ordered replan list, including first and last.
When fewer than `16` unique replans exist, retain all. Every retained replan
contributes all `10` denoising steps.

For task `j`, let episode `e` contain `n_e` retained replan-step vectors. The
equal-episode task mean and covariance are:

`mu_j = (1 / E_j) sum_e [(1 / n_e) sum_i x_ei]`

and

`R_j = (1 / E_j) sum_e [(1 / n_e) sum_i (x_ei-mu_j)(x_ei-mu_j)^T]`.

Thus tasks, episodes, replans, and denoising steps cannot dominate merely by
length. Each source task then receives equal weight in `R_f^src`.

No step-level failure label is invented. Every retained vector inherits only
the episode outcome.

Every source task must provide at least `3` successful and `3` failed episodes:
failures are required by SPARC and both classes are required to instantiate
every candidate complete-source COAST operator. A collapsed source class is
`DATA_FAILURE` after the full manifest.

Prefix sensitivity is diagnostic-only:

- recompute geometry from the first eight retained replans;
- recompute geometry from the last eight retained replans;
- compare each to the all-16 scientific operator by normalized Frobenius inner
  product, effective rank, and target-failure containment.

The all-16 operator remains the method regardless of this diagnostic. Median
operator similarity below `0.80` across prefix halves is classified
`UNDERPOWERED_OR_UNRESOLVED` because episode-level failure geometry is not
stable enough for a decisive fit.

## 5. Aggregate Covariance Justification

The scientific object is a task-balanced mixture distribution over failure
activations. Covariance aggregation is therefore defined before conceptor
regularization:

`R_f^src = (1 / J) sum_j R_f^j`.

Applying one common aperture after aggregation ensures every eigen-direction
is regularized according to its variance in that mixture. Averaging already-
regularized conceptors would make task contribution depend nonlinearly on each
task's spectrum before the tasks are combined and is not SPARC.

The audit computes a diagnostic-only alternative:

`C_f_mean = (1 / J) sum_j C(R_f^j, alpha)`.

It reports normalized Frobenius similarity to `C(R_f^src, alpha)`, eigenvalue
range, trace quota, effective rank at eigenvalue `>= 0.1`, condition number of
the regularized solve, and each source task's leave-one-task-out operator
change. It cannot select the scientific aggregation or enter validation as a
seventh configuration.

If `C_f^src` has quota above `0.95`, failure complement quota below `0.01`, or
condition number above `1e12` at every frozen aperture, classify
`DESIGN_FAILURE` or `IMPLEMENTATION_FAILURE` according to whether the inputs
or solve are responsible. Do not steer.

## 6. Target-Success Coverage And Stability

Each target requires at least `3` successful discovery episodes, and adequate
coverage additionally requires:

- at least `30` retained replan-step vectors per target episode after the
  fixed cap;
- at least `10%` of target-success records in each early, middle, and late
  normalized replan third;
- nonzero activation variance at every candidate layer;
- leave-one-success-episode-out operator similarity median `>= 0.85` and
  minimum `>= 0.70` at the selected layer/aperture.

Similarity is normalized Frobenius inner product after symmetrization. Fewer
than three episodes or collapsed phase coverage is `DATA_FAILURE`. Passing the
minimum count but failing stability is `UNDERPOWERED_OR_UNRESOLVED`, not a
scientific kill.

## 7. Exact Target-Failure Disclosure

The fixed wording is:

`No target failure activations enter the SPARC operator fit, layer selection,
aperture selection, or inference.`

Target failure episodes may be used only for:

- a privileged discovery headroom diagnostic;
- validation success measurement shared by every policy;
- validation-only source selection for the COAST proxy.

Every report counts target failed episodes and retained failure activations in
each use. SPARC is not called failure-blind or zero-failure adaptation.

## 8. Frozen Layer, Aperture, And Numerical Rules

Candidate post-residual sites are exactly `{0, 5, 11, 14}`. Site `14` is
captured and applied before layer `15`; no final-layer special case exists.

Apertures are exactly `{0.1, 0.5, 1.0, 2.0, 10.0}`.

For each site at `alpha = 10.0`, construct each target's SPARC conceptor from
`C_s^T` and `C_f^src`, using no target failures. Define quota:

`q(C) = trace(C) / 720`.

Select the site with maximum mean quota across the three targets. Ties within
`1e-12` choose the lower site.

At that site define success/failure overlap:

`overlap(C_s, C_f) = trace(C_s C_f) / (||C_s||_F ||C_f||_F)`.

Compute mean overlap across targets for every aperture. Select the aperture in
the closed interval `[0.80, 0.90]` nearest `0.85`; if none lies in the band,
select the globally nearest to `0.85`. Ties within `1e-12` choose the smaller
aperture.

All covariance, inverse, and pseudoinverse operations use float64. Symmetrize
with `(C + C^T)/2` for diagnostics and storage. Use NumPy's default SVD-based
Moore-Penrose pseudoinverse with explicit `rcond = 1e-12`. No eigenvalue
truncation or scientific PSD projection is applied. Eigenvalues outside
`[-1e-8, 1+1e-8]` are `IMPLEMENTATION_FAILURE`; values within that tolerance
are reported and the symmetric matrix is stored unchanged in float32, matching
the prior's no-truncation rule.

Target failures are absent from every selection above.

## 9. Known-Task Claim Limit

The current prototype is explicitly an in-distribution known-task identity
test with fresh reset identities. It does not test new-task generalization.

If SPARC reaches paper-candidate status, the immediate second condition is a
frozen LIBERO-90 held-out-task set with zero normalized task-identity overlap
with the Base checkpoint's 40 tasks. Quantized OpenVLA-OFT INT4 verification
also follows under the campaign paper gate. Neither is added before a positive
prototype result.

The frozen target confirmatory range contains only `60` paired target cases.
Stage B therefore uses `40` and may expand once to all `60` under the unresolved
rule. No ad hoc identities are added to reach `80`; unresolved evidence at
`60` is reported as unresolved rather than rescued.

## 10. Fixed Manifest And Prior Budget Accounting

All `84` discovery Base episodes are run because the proposal froze seven
tasks and twelve resets. Historical five-reset rows motivated task selection
but provide no activation and are never merged into the SPARC fit.

Both SPARC and the COAST proxy receive access to the same four source task
datasets. The COAST proxy must preserve its prior mechanism and therefore uses
one complete source success/failure operator selected on validation. SPARC's
mechanism uses all source failure datasets and target successes. The primary
comparison is honestly labeled `same available source pool, different retained
operator data`, not equal fitted episode count.

Reports include available episodes, retained episodes, activation count, and
effective episode weight for every arm. A diagnostic-only SPARC operator is
also built after deterministic episode subsampling to the selected COAST
operator's retained episode count, stratified over source tasks. It reports
geometry and offline action consequences but is not a rollout policy or
configuration selector.

This diagnostic tests data-volume sensitivity without changing the prior or
adding a sixth policy.

## 11. Matched Filtered-BC LoRA

`standard_lora_target_success` receives exactly the successful target
discovery observation/action pairs at the same capped replan indices used for
`C_s^T`. The targets are the successful Base-emitted postprocessed action
chunks, not expert demonstrations, failed target rows, or extra frames.

Frozen specification:

- PEFT rank `4`, alpha `4`, dropout `0.0`;
- official SmolVLA default target-module regex;
- seed `1919`;
- flow-matching training objective already implemented by SmolVLA;
- AdamW, learning rate `1e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`, weight
  decay `1e-10`;
- physical batch `8`, gradient accumulation `1`;
- `2,000` optimizer steps;
- final-step checkpoint, with no best-checkpoint selection;
- persisted adapter disk reload before validation;
- no SPARC activation or conceptor input.

Adequacy requires finite nonzero target gradients, final discovery flow loss
at least `20%` below the first-50-step median, action validity, and exact
checkpoint reload within `1e-6`. Failure is
`LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT` or `IMPLEMENTATION_FAILURE`, not a
scientific SPARC win.

## 12. Numerical Headroom Rules

For PSD operator `A` and covariance `R`, define retained energy:

`ret(A, R) = trace(A R A^T) / max(trace(R), 1e-12)`.

For each target, use discovery target successes and diagnostic target failures
to compute:

`m_T = ret(C_sparc, R_s^T) - ret(C_sparc, R_f^T)`.

Headroom requires:

- at least `3` diagnostic target failures per target;
- `m_T >= 0.05` on at least two targets;
- mean `m_T >= 0.05`;
- source-failure containment
  `ret(C_f^src, R_f^T)` above the `95th` percentile of `256` deterministic
  random orthogonal rotations of `C_f^src`, seed `1919`, with absolute margin
  at least `0.02` on at least two targets;
- target-success retention at least `0.02` higher for SPARC than the selected
  complete source COAST gate on at least two targets, with no target worse by
  more than `0.02`.

Fewer than three target failures after the full manifest is
`UNDERPOWERED_OR_UNRESOLVED`; target failure labels are diagnostic, not a
SPARC fit requirement. Adequately powered failure of the margins is
`NO_HEADROOM`. It is not a closed-loop scientific result.

## 13. Numerical Action-Safety Rules

On postprocessed `50 x 7` chunks require:

- finite fraction exactly `1.0`;
- maximum absolute value `<= 1.25`;
- fraction outside `[-1,1] <= Base + 0.01`;
- p99 exceedance beyond `[-1,1] <= Base + 0.02`;
- first-10-step translation delta L2 p95 `<= 0.20`;
- first-10-step rotation delta L2 p95 `<= 0.20`;
- first-10-step absolute gripper delta p95 `<= 0.20`;
- full-chunk per-step 7D delta L2 p95 `<= 0.30`;
- simulator action acceptance `= 1.0` before rollout.

At `beta = 0.1`, acting additionally requires mean full-chunk action delta
`> 1e-6`. Failure of every beta to act after geometry passes is
`DESIGN_FAILURE`; destructive action at every beta is also `DESIGN_FAILURE`.
A hook or matrix error is `IMPLEMENTATION_FAILURE`.

No clipping, renormalization, threshold rescue, or post-result beta insertion
is allowed.

## 14. Identity, Resume, And Hash Rules

Episode key:

`(partition, policy, suite, task_id, reset_seed)`.

Activation key:

`(episode_key, replan_index, residual_site, denoising_step)`.

Token index is an axis inside one activation record, not an append key. Each
activation record stores tensor shape and SHA256 over canonical float32 bytes.

An episode is complete only after its episode row, ordered replan list, all
expected activation records, action hashes, outcome, and exception status are
atomically persisted. Resume operates only on missing episode keys. A partial
episode is deleted and recomputed as the same key before acceptance; a second
complete row for a key is forbidden.

Before accepting any result require JSON parse, manifest exactness, episode
duplicate count zero, activation duplicate count zero, missing/extra count
zero, exceptions zero, and recomputed hashes.

## 15. False-Negative Classification

- `DATA_FAILURE`: required source failures, target successes, or phase coverage
  collapse after the full frozen discovery manifest.
- `IMPLEMENTATION_FAILURE`: hook, shape, identity, serialization, numerical,
  manifest, hash, or action-semantics error.
- `LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT`: the required LoRA control cannot
  be trained adequately under its frozen local implementation.
- `UNDERPOWERED_OR_UNRESOLVED`: minimum labels exist but LOO/prefix stability,
  target-failure diagnostic count, or uncertainty is inadequate.
- `NO_HEADROOM`: adequately powered frozen geometry lacks reusable failure
  containment or target-success/failure separation.
- `DESIGN_FAILURE`: valid geometry cannot produce a bounded acting policy, or
  the scientific aggregate is intrinsically degenerate.
- `GENUINE_METHOD_KILL`: only a valid matched closed-loop Stage A/B result in
  which SPARC is catastrophically worse, prior/ablation/simple control explains
  the result, or the frozen useful-improvement threshold is excluded.

Data, implementation, low-compute, underpowered, no-headroom, and design
classifications do not become evidence that conceptor steering generally
fails. None authorizes same-cycle task, label, threshold, aggregation, or hook
rescue.

## Rebuttal Decision

SPARC passes to the mathematical mechanism audit. The proposal hash remains
unchanged. No labeled activation collection, validation search, confirmatory
test, or steered rollout is authorized by this rebuttal alone.
