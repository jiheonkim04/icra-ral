# URF-VLA Researcher A Proposal

Date: 2026-07-16 KST

Decision: `URF_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

Method: `URF-VLA`, Uncertainty-Routed Residual Flow for Base-preserving
SmolVLA chunks.

Closest positive prior: SUREFlow, `https://arxiv.org/abs/2607.10504`, official
repository `https://github.com/tanvirnwu/SUREFlow`.

## Boundary

URF is not generic uncertainty reporting, not a confidence head, not ordinary
LoRA, not TS-Mask-style masked completion, not CFR-style full-chunk iterative
refinement, not AMP action-manifold projection, not RAP retrieval anchoring,
and not CCIF coarse-intent conditioning.

The scientific method is uncertainty-routed bounded residual transport around
a pretrained SmolVLA Base action chunk. Any LoRA or small adapter is only the
low-compute parameterization for residual and uncertainty heads.

## Scientific Method

For a legal deployment input `x_t` and frozen SmolVLA decoded chunk
`B_t in R^[50,7]`, URF predicts:

- residual mean `mu_theta(x_t, B_t) in R^[50,7]`;
- log residual variance `ell_theta(x_t, B_t) in R^[50,7]`;
- bounded route gate `g_theta(x_t, B_t) in [0, g_max]^[50,7]`.

The URF action chunk is:

`A_URF = B_t + g_theta * clip_group(mu_theta, rho_translate, rho_rotate, rho_gripper)`.

The route gate must be a function of the predicted residual uncertainty and
the residual magnitude. Cells whose residual is predicted to be unreliable or
trivial default to Base passthrough.

Training labels use only discovery/validation demonstrations:

`R_t = A_expert_t - B_t`.

The primary loss is heteroscedastic residual regression:

`L_res = mean(0.5 * exp(-ell_theta) * Huber(R_t - mu_theta) + 0.5 * ell_theta)`.

An action loss may train the routed chunk:

`L_act = Huber(A_URF, A_expert_t)`.

A clean-retention term is required if training proceeds:

`L_clean = Huber(A_URF, B_t)` on clean-retention records.

No KL is used between deterministic 7D actions.

## Mechanism Hypothesis

Problem condition -> Base SmolVLA is strong overall, but some time/action cells
in a chunk contain correctable residual errors while many other cells are
already good or unsafe to alter.

Intermediate failure mechanism -> ordinary residual adaptation changes too
many cells or applies corrections in regions where the residual model is
uncertain, damaging clean behavior or gripper timing.

Policy representation/action behavior -> global residuals can leave the
pretrained action support or blur translation, rotation, and gripper changes.

Proposed method -> learn a heteroscedastic residual-flow field and route
bounded corrections only where the residual estimate is useful and reliable.

Intended action behavior -> targeted bounded residual changes with Base
passthrough under high uncertainty.

Expected closed-loop improvement -> higher success on targeted tasks than
Base, a SUREFlow-style proxy, a no-uncertainty residual ablation, and standard
LoRA, while retaining clean behavior.

## Evidence Partitions

`DISCOVERY`: inspect Base residuals, construct residual targets, estimate
residual uncertainty, debug implementation, and run cheap diagnostics.

`VALIDATION`: select one bounded configuration, route threshold, residual cap,
and clean-retention coefficient from at most six configurations.

`CONFIRMATORY_TEST`: used once only after the final method, checkpoint,
baseline list, ablation, tasks, reset identities, metrics, and thresholds are
frozen.

No confirmatory-test identity may be used for URF training, threshold
selection, proxy construction, or failure inspection.

## Stage 0 Development Audit

Before bounded validation search or rollout, Stage 0 must verify:

1. discovery/validation/test identity separation and zero overlap;
2. Base residual targets are noncollapsed across tasks and phases;
3. residual headroom exists beyond task/phase mean residuals;
4. a deployment-input heteroscedastic probe beats a homoscedastic residual
   baseline on validation negative log-likelihood or calibrated Huber proxy;
5. uncertainty strata are noncollapsed and have monotonic residual-error
   ordering on validation data;
6. the full URF routed chunk differs from both Base and the no-uncertainty
   residual ablation after a small fit;
7. the SUREFlow transparent proxy is not a strawman and uses the same legal
   inputs, labels, split, optimizer budget, and action postprocessing;
8. residual deltas are bounded by action group;
9. action chunks remain finite and postprocessor-valid;
10. initialized and disk-reloaded URF reproduces Base within the frozen
    tolerance;
11. expected URF parameters receive finite nonzero gradients and frozen Base
    parameters receive no gradients;
12. no simulator reward, success, done flag, object pose, privileged future
    observation, confirmatory reset identity, or hidden test label is read.

Stage 0 stop classes:

- `URF_STAGE_0_DATA_OR_SUPERVISION_FAILURE`
- `URF_STAGE_0_NO_USABLE_HEADROOM`
- `URF_STAGE_0_DESIGN_FAILURE`
- `URF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`
- `URF_STAGE_0_PASS_TO_BOUNDED_VALIDATION`

## Bounded Validation Search

Maximum six total configurations. Candidate factors may include exactly:

- uncertainty route threshold or temperature;
- residual cap `g_max`;
- clean-retention coefficient.

No broad rank, module-target, seed, task, or threshold grid is allowed.

The validation score must combine:

- validation routed-action proxy improvement against Base;
- margin over the SUREFlow proxy;
- margin over no-uncertainty route ablation;
- clean-retention/action-validity term;
- route sparsity so URF does not globally alter every action cell.

Offline action L2 alone cannot select the configuration or support a paper
claim.

## First Serious Comparison

After Stage 0 and bounded validation, the first serious comparison is:

1. `smolvla_base`
2. `sureflow_uncertainty_residual_proxy` or official `sureflow` if locally
   installed and verified
3. `urf_full`
4. `urf_no_uncertainty_route_ablation`
5. `standard_lora`

The key ablation removes uncertainty routing while keeping residual capacity
matched. Standard LoRA remains required because URF trains on the same
demonstrations and ordinary adaptation is a plausible alternative explanation.

## Paper-Candidate Gate

URF becomes a serious paper candidate only if frozen SmolVLA comparisons show
that URF beats Base, the closest prior/proxy, the no-uncertainty ablation, and
standard LoRA on the matched claim axis; clean behavior is retained; the
uncertainty route is active and noncollapsed; and the mechanism evidence
supports uncertainty-routed residual transport rather than generic adaptation.

After a SmolVLA prototype GO, evaluate Quantized OpenVLA-OFT INT4 versus
Quantized OpenVLA-OFT INT4 plus URF under the same-backbone requirement and
add one claim-specific second condition.

## Next Stage

Reviewer B must independently attack URF novelty, the SUREFlow comparison,
the no-uncertainty ablation, standard LoRA rationale, uncertainty calibration,
identity preservation, and the risk that URF is merely another residual method
before mathematical audit, preregistration, implementation, or training.
