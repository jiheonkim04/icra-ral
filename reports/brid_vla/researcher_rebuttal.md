# BRID-VLA Researcher A Rebuttal

Date: 2026-07-16 KST

Decision: `BRID_REBUTTAL_PASS_TO_MATHEMATICAL_AUDIT`

Method: `BRID-VLA`, Base-Residual Implicit Diffusion for SmolVLA action
chunks.

Proposal: `reports/brid_vla/researcher_proposal.md`

Reviewer attack: `reports/brid_vla/reviewer_attack.md`

Proposal SHA-256:
`2D4769CF126DF0580029486F7D64EF3C09D435571589F87C569F60A71CBC5CA2`

## Response Summary

Researcher A accepts all Reviewer B conditions.

BRID will proceed only as a frozen-SmolVLA, Base-residual diffusion mechanism
with zero-residual identity integration, bounded residual caps, and exact Base
passthrough whenever the residual score/intervention rule is inactive. The
method will not claim raw action diffusion, ordinary LoRA, action smoothing, or
generic residual imitation as novelty.

No BRID implementation, training, validation search, rollout, simulator access,
or confirmatory-test tuning has happened before this rebuttal.

## Accepted Novelty Boundary

Accepted boundary:

`A frozen-SmolVLA, Base-conditioned residual diffusion score field that learns
bounded corrections around the Base action chunk, initializes to exact
zero-residual Base passthrough, and only applies residual edits when
development-validated score/confidence/action-validity rules permit them.`

This excludes:

- raw Diffusion Policy as the method;
- raw action replacement beside SmolVLA;
- a generic residual MLP without diffusion score training;
- a tuned action smoother;
- standard LoRA or PEFT as the contribution;
- privileged simulator, future-action, reward, success, done, object-pose, or
  confirmatory-test information at inference.

LoRA or a lightweight adapter may parameterize the residual score network only
as low-compute infrastructure.

## Closest Prior And Policy Order

Accepted closest prior: Diffusion Policy.

Accepted primary sources:

- `https://diffusion-policy.cs.columbia.edu/`
- `https://github.com/real-stanford/diffusion_policy`

The first serious comparison remains exactly:

1. `smolvla_base`
2. `diffusion_policy_action_chunk_proxy`
3. `brid_full`
4. `brid_no_base_residual_ablation`
5. `standard_lora`

Policy 2 will be a transparent raw action-chunk diffusion proxy unless
official Diffusion Policy assets are installed and verified under the same
SmolVLA/LIBERO scaffold. The proxy must share the development rows, legal
inputs, action semantics, split, and comparable compute budget, but it cannot
use Base-residual conditioning or exact Base passthrough.

## Accepted Reviewer Conditions

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

## Stage 0 Commitments

Stage 0 remains a development-only audit, not a closed-loop scientific result.

It must stop before bounded validation if any of the following occurs:

- residual targets are collapsed;
- score/noise prediction does not beat zero-noise, mean-noise, and task/phase
  baselines;
- task/phase/action-group residual coverage is insufficient;
- no residual headroom exists relative to Base;
- the raw Diffusion Policy proxy dominates or makes BRID redundant;
- the no-Base-residual ablation explains BRID;
- standard LoRA explains BRID;
- intervention activates everywhere or nowhere;
- the module globally changes actions rather than applying bounded residual
  edits;
- clean retention fails;
- action bounds or official postprocessing validity fails;
- identity initialization or checkpoint reload fails;
- any privileged inference input or confirmatory-test identity is used.

## Mathematical Audit Requirements Accepted

The mathematical audit must freeze:

- variables and tensor shapes for `x_t`, `B_t`, `E_t`, `R_t`, `r_k`, `k`,
  `epsilon_theta`, `Delta_t`, `g_theta`, and `A_t`;
- diffusion schedule, noise identities, and deterministic replay rule;
- residual caps for translation, rotation, and gripper groups;
- score-confidence or denoising-consistency intervention rule;
- zero-residual identity initialization and disk-reload tolerance;
- objective terms, coefficient scales, and gradient paths;
- frozen-Base no-gradient checks;
- validation-only search budget;
- raw Diffusion Policy proxy construction;
- no-Base-residual ablation construction;
- standard LoRA matching rules;
- action postprocessor validity contract;
- no deterministic-action KL.

## Immediate Next Stage

Proceed to the BRID mathematical mechanism audit before preregistration,
implementation, validation search, rollout, or confirmatory-test access.
