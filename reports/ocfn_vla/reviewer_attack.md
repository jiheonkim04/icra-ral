# OCFN-VLA Reviewer B Attack

Date: 2026-07-12 KST

Proposal hash: `F60B9B7BB2640A073AC16EAB6284A68D41569A6A4D67A54462DEF81F06F3F8EA`

Reviewer B read the frozen proposal after the hash above was saved.

## Closest Primary Sources

1. Guided Action Flow / Q-Guided Inference for Flow-Matching VLA Policies: https://arxiv.org/html/2607.02092v1
2. CF-VLA, Efficient Coarse-to-Fine Action Generation for VLA Policies: https://arxiv.org/html/2604.24622v1
3. VLS, Steering Pretrained Robot Policies via VLMs: https://arxiv.org/html/2602.03973v1
4. ACG, Action Coherence Guidance for Flow-based VLA models: https://arxiv.org/html/2510.22201v2
5. VOTE, Vision-Language-Action Optimization with Trajectory Ensemble Voting: https://arxiv.org/html/2507.05116v3

Secondary pressure:

- FASTER and real-time flow execution papers already study flow-sampling schedules and action chunk inference.
- HiPolicy and AAC occupy uncertainty/adaptive execution routes.
- VLA-OPD and VITA-VLA occupy teacher/distillation routes, which is why CAD-VLA was not selected.

## Novelty Attack

OCFN is near the crowded family of inference-time flow manipulation. The proposal's strongest distinction is narrow: it learns a prior over initial noise identities from closed-loop outcomes and does not score action proposals online. This is not an exact duplicate of QGF, VLS, ACG, CF-VLA, or VOTE, but it is adjacent enough that the paper claim must be framed as latent-prior calibration for frozen flow VLAs, not generic VLA guidance.

Reviewer B does not reject before implementation because exact duplication is not proven across all axes:

- QGF uses a learned action-chunk critic and gradients through the reverse flow; OCFN uses no critic at inference.
- CF-VLA constructs action-aware starts and local refinement; OCFN uses outcome-selected noise identities without an action coarse generator.
- VLS synthesizes VLM rewards, keypoints, stage switching, gradients, and particle resampling; OCFN uses none of those.
- ACG targets action coherence; OCFN targets closed-loop outcome-conditioned latent priors.
- VOTE uses trajectory ensemble voting; OCFN fixes one prior before held-out evaluation.

## Trivial Baseline Attack

The obvious way OCFN could collapse is as follows:

1. A fixed zero-noise start is as good as the learned prior.
2. One globally lucky noise identity is as good as the task-conditioned prior.
3. Task-conditioned selection is just train-reset overfitting.
4. Default random/noise-free frozen SmolVLA already dominates.
5. The selected noise changes actions but only as uncontrolled stochastic perturbation, not a reproducible mechanism.

Therefore Stage A must include:

- `frozen_smolvla`
- `zero_noise_smolvla`
- `global_success_noise_prior`
- `task_shuffled_noise_prior`
- `ocfn_full`

The key ablation is `task_shuffled_noise_prior`. The simple killer is `global_success_noise_prior`.

## Leakage And Protocol Attack

The implementation must not use held-out reset identities to choose the noise prior. It must not tune the noise bank size, tie-break rules, task set, or identities after seeing Stage A results.

Reset identities:

- train: `20260711`, `20260712`
- held-out Stage A: `20260713` to `20260717`

Any partial result file must clearly separate train acquisition from held-out Stage A.

The implementation must log:

- noise seed/index selected per variant and task;
- train table success/count by task/noise;
- exact selection rule;
- whether full equals global or shuffled;
- mean action delta between `ocfn_full` and `global_success_noise_prior` on held-out episodes.

If full and global choose the same noise identity for both tasks, the full method is trivial unless a predeclared secondary task split shows a task-conditioned difference before held-out inspection. No secondary split is allowed in Stage A unless the primary acquisition is measurement-invalid.

## Resource Attack

OCFN is locally feasible with SmolVLA because the current runner can pass `noise` into `select_action`. It has a final-paper risk: Quantized OpenVLA-OFT INT4 uses an L1 regression action head and may not expose a flow-noise analogue. A Stage A success would not yet satisfy second-backbone governance. The Stage B/scale-up plan must either:

- find a second locally feasible flow-action VLA with an exposed latent/noise interface; or
- explicitly narrow the claim to flow-matching VLAs and use Quantized OpenVLA-OFT INT4 as a non-applicability boundary, which may be insufficient for `READY_TO_DRAFT_RAL_PAPER_PACKAGE`.

This risk is not a hard implementation blocker for Stage A.

## Decision

Decision before implementation: `IMPLEMENT_STAGE_A_PROTOTYPE`.

Rationale: OCFN is not exact prior-art duplication, not mathematically equivalent to a simple baseline before measurement, and not blocked by unavailable resources. The decisive experiment is cheap and should kill it cleanly if it is just fixed-noise luck.
