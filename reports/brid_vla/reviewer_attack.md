# BRID-VLA Reviewer B Attack

Date: 2026-07-16 KST

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Method: `BRID-VLA`, Base-Residual Implicit Diffusion for SmolVLA action
chunks.

Proposal: `reports/brid_vla/researcher_proposal.md`

Proposal SHA-256:
`2D4769CF126DF0580029486F7D64EF3C09D435571589F87C569F60A71CBC5CA2`

## Summary Judgment

Conditional pass to rebuttal.

BRID is the strongest currently selected candidate because Diffusion Policy is
a direct positive prior for action-sequence denoising, and existing LIBERO
demonstrations plus cached SmolVLA Base chunks plausibly contain the residual
supervision required to test the mechanism locally. The proposal also obeys
the current design constraint: LoRA is only implementation infrastructure, and
the closest prior enters the first serious comparison.

The method may continue only if Researcher A accepts the conditions below.
The novelty is real only if BRID remains a Base-residual diffusion mechanism
with exact Base passthrough and bounded residual caps. It must not become raw
Diffusion Policy, ordinary LoRA imitation, a generic residual MLP, or a
post-hoc action-smoothing module.

## Closest Prior Boundary

Closest prior: Diffusion Policy.

Primary sources:

- `https://diffusion-policy.cs.columbia.edu/`
- `https://github.com/real-stanford/diffusion_policy`

Diffusion Policy is the closest prior because it already demonstrates the
central positive result: visuomotor policies can generate robot action
sequences through conditional denoising and outperform prior behavior-cloning
methods across 12 manipulation tasks and four benchmarks, with a reported
average success-rate improvement of `46.9%`.

BRID's allowed novelty is not the broad claim that diffusion can model robot
actions. That is Diffusion Policy's prior result.

Permitted novelty boundary:

`A frozen-SmolVLA, Base-conditioned residual diffusion score field that learns
bounded corrections around the Base action chunk, initializes to exact
zero-residual Base passthrough, and only applies residual edits when
development-validated score/confidence/action-validity rules permit them.`

BRID novelty is not:

- training a raw diffusion policy on LIBERO demonstrations;
- sampling a replacement action chunk beside SmolVLA;
- adding a residual MLP or denoiser without exact Base identity;
- choosing the best sampled action after seeing test outcomes;
- using demonstration actions, future observations, success labels, or object
  poses at inference;
- or claiming Diffusion Policy's action-denoising result under a new acronym.

## Closest Three Primary Anchors

1. Diffusion Policy:
   `https://diffusion-policy.cs.columbia.edu/` and
   `https://github.com/real-stanford/diffusion_policy`.
   This is policy 2 and the direct prior.

2. FAST / FAST+:
   `https://arxiv.org/abs/2501.09747`.
   FAST is a strong current prior for action-sequence representation through
   frequency-space tokenization. It is not the closest prior because BRID is
   diffusion/residual-score based rather than tokenization based, but it
   challenges whether action representation, not residual diffusion, is the
   useful mechanism.

3. ACT / action chunking:
   `https://tonyzhaozh.github.io/aloha/` and the LeRobot ACT documentation.
   ACT is a prior for chunk-level imitation and temporal ensembling. It is not
   the closest prior because it is not a VLA diffusion mechanism, but it
   challenges whether a simple chunk predictor or smoothing baseline explains
   BRID.

OpenVLA-OFT remains a later backbone and optimized fine-tuning reference, not
the closest prior for this SmolVLA-local prototype. It becomes mandatory only
after prototype GO.

## Required Policy Order

The first serious comparison must keep:

1. `smolvla_base`
2. `diffusion_policy_action_chunk_proxy`
3. `brid_full`
4. `brid_no_base_residual_ablation`
5. `standard_lora`

Policy 2 must be a fair closest-prior proxy. If official Diffusion Policy code
cannot be run directly in the existing SmolVLA/LIBERO scaffold, the proxy must
be transparent: train a raw action-chunk diffusion denoiser on the same
development rows, same legal observations/proprio/language inputs, same action
semantics, same discovery/validation split, and comparable compute budget, but
without Base-residual conditioning or exact Base passthrough.

`brid_no_base_residual_ablation` must preserve the denoising objective,
training rows, optimizer budget, parameter budget, inference step count, and
action caps while removing Base-residual conditioning and zero-residual
identity integration.

`standard_lora` must use matched demonstrations, optimizer budget, action
postprocessing, and checkpoint-selection rules. It is required because BRID
uses demonstration supervision and additional trainable capacity.

## Major Risks

### Risk 1: BRID May Be Raw Diffusion Policy In Disguise

Conditioning on `B_t` is not enough if the model learns to ignore Base and
generate raw actions. In that case the contribution collapses to a weaker
local Diffusion Policy proxy.

Required rebuttal: Stage 0 must measure whether BRID uses Base residuals by
comparing to a raw action diffusion proxy and no-Base-residual ablation under
matched budget. BRID cannot advance if either explains the effect.

### Risk 2: Residual Targets May Be Collapsed Or Too Small

If `R_t = E_t - B_t` is near zero everywhere, there is no useful residual
headroom. If residuals are huge or dominated by action-scale mismatch, BRID
will damage actions rather than improve control.

Required Stage 0 gate: report residual L2/Huber distributions by task, phase,
time index, and action group; show noncollapse and useful residual headroom
relative to Base and the raw diffusion proxy.

### Risk 3: Score Prediction May Not Be Observable From Deployment Inputs

The diffusion target is created from demonstration actions. At inference, BRID
has current RGB/proprio/language/Base chunks only. If the noisy residual score
cannot be predicted above trivial baselines, the denoiser is not an
observable deployment mechanism.

Required Stage 0 gate: validation score/noise prediction must beat trivial
zero-noise, mean-noise, and task/phase baselines on the residual rows used by
the policy.

### Risk 4: Identity Preservation Can Fail Quietly

Diffusion sampling can introduce small global changes everywhere. That would
violate the core Base-preserving claim even if action L2 improves offline.

Required diagnostics: initialized and disk-reloaded BRID must equal Base
within tolerance; inactive/low-confidence rows must be exact Base; report
Base action, BRID action, residual norm, gate value, dimensions changed, and
activation context for sampled active and inactive rows.

### Risk 5: Validation Search Can Become Diffusion Hyperparameter Tuning

Diffusion models expose many knobs: step count, schedule, noise scale, seed
rule, residual cap, gate threshold, loss weights, adapter capacity. A broad
search would violate the bounded validation policy.

Required rebuttal: freeze a maximum of six configurations, at most two
architecture choices, and one primary coefficient/rule family before any
training. Save negative configs and do not tune on confirmatory test.

### Risk 6: Offline Action Loss May Not Predict Closed Loop

The campaign has repeatedly shown offline action metrics can mislead. A
denoiser can reduce residual loss while harming contact timing, gripper
events, or task completion.

Required rebuttal: Stage 0 may use offline diagnostics only as development
gates. The first serious comparison must use closed-loop success or the
closest feasible preregistered proxy, plus action validity and clean
retention. Do not select purely by action L2.

### Risk 7: Standard LoRA Or A Simple Residual Baseline Can Explain The Gain

BRID trains on demonstrations. A reviewer will ask whether ordinary adapter
training, L1 residual regression, temporal smoothing, or a simple residual MLP
does the same thing.

Required Stage 0 gate: standard LoRA remains policy 5; no-Base-residual
ablation remains policy 4. A simple residual-regression diagnostic may be
included in Stage 0, but it must not replace the five-policy first serious
comparison.

### Risk 8: Clean Behavior Can Be Damaged

The method is allowed to improve targeted failures only if it preserves Base
where residual intervention is not justified.

Required Stage 0 gate: clean-retention rows must preserve Base behavior under
the frozen delta tolerances, action postprocessing must remain valid, and
intervention frequency must be bounded and state-dependent rather than global.

## Mathematical Audit Requirements

The mathematical audit must freeze:

- `x_t`, `B_t`, `E_t`, `R_t`, noisy residual `r_k`, diffusion step `k`,
  predicted noise `epsilon_theta`, denoised residual `Delta_t`, gate
  `g_theta`, and deployed chunk `A_t` with tensor shapes;
- diffusion schedule, noise identity construction, and deterministic replay
  rule;
- residual caps for translation, rotation, and gripper groups;
- score-confidence or denoising-consistency rule for intervention;
- zero-residual identity initialization and disk-reload identity tolerance;
- all objective terms, scales, units, coefficients, and gradient paths;
- frozen-Base no-gradient checks;
- validation-only coefficient, cap, and step-count search budget;
- raw Diffusion Policy proxy construction;
- no-Base-residual ablation construction;
- standard LoRA matching rules;
- action postprocessor and validity contract;
- no deterministic-action KL.

If any KL term is proposed, reject it unless valid probability distributions,
supports, direction, estimator, gradient flow, and alternatives are justified.
Do not compute KL directly between deterministic 7D action vectors.

## Required Ablations And Diagnostics

1. `diffusion_policy_action_chunk_proxy`
2. `brid_no_base_residual_ablation`
3. matched `standard_lora`
4. zero-noise, mean-noise, and task/phase score-prediction baselines
5. residual noncollapse and residual-headroom diagnostics
6. clean-retention and inactive-gate exact-Base diagnostics
7. action-bound and intervention-frequency diagnostics

## Stage 0 Must Stop For

- collapsed residual targets;
- score prediction not above trivial baselines;
- insufficient task/phase/action-group residual coverage;
- no residual headroom relative to Base;
- raw Diffusion Policy proxy dominates or makes BRID redundant;
- no-Base-residual ablation explains the effect;
- standard LoRA explains the effect;
- intervention everywhere or nowhere;
- global action changes rather than bounded residual edits;
- clean retention failure;
- action-bound violations;
- identity or checkpoint reload failure;
- use of reward/success/done/object pose/future observation;
- any confirmatory-test task, reset identity, label, or outcome read.

## Conditions For Researcher A

Researcher A must accept:

1. Diffusion Policy remains the closest prior and policy 2.
2. The raw action-chunk diffusion baseline is a transparent local proxy unless
   official Diffusion Policy assets are installed and verified under the same
   scaffold.
3. BRID novelty is narrowed to Base-residual diffusion with zero-residual
   identity integration and bounded residual caps.
4. Residual target construction, noise identity construction, and action caps
   must be frozen before Stage 0.
5. Residual targets and score targets must be noncollapsed across tasks and
   phases.
6. Score prediction must beat trivial validation baselines before denoising can
   be treated as an observable mechanism.
7. The raw diffusion proxy, no-Base-residual ablation, and standard LoRA remain
   required.
8. Clean retention and exact Base passthrough are mandatory.
9. No deterministic-action KL is allowed.
10. AFID and all previous methods remain closed.

If these conditions are accepted, proceed to Researcher A rebuttal. If not,
BRID must be rejected before implementation.
