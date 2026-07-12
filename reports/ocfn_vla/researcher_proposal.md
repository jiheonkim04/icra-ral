# OCFN-VLA Researcher Proposal

Date: 2026-07-12 KST

Method: `OCFN-VLA`, Outcome-Conditioned Flow-Noise Prior VLA

Governance: `reports/current_research_governance.md`

Researcher A freezes this proposal before Reviewer B begins.

## Claim

Frozen flow-matching VLAs do not only fail because their action head lacks capacity or because their action output needs post-hoc correction. They may also fail because deployment samples from an uncalibrated initial noise prior. OCFN tests whether closed-loop outcome labels can learn a small task-conditioned prior over SmolVLA's initial flow noise, improving held-out task success while leaving the frozen denoising field and official action postprocessor intact.

## Mechanism

SmolVLA predicts an action chunk by denoising an initial noise tensor. OCFN constructs a fixed deterministic bank:

`Z = {z_j in R^{chunk_size x action_dim}}`.

For each train task and train reset identity, OCFN executes frozen SmolVLA with each `z_j` supplied to `policy.select_action(batch, noise=z_j)`. It records only the closed-loop success label and basic episode metadata. A task-conditioned prior then chooses one noise identity for held-out inference:

`j*(task) = argmax_j q_phi(j | task)`.

At deployment, OCFN performs exactly one frozen SmolVLA action call per control step:

`a_t = postprocess(first(F_theta(o_t, l, z_{j*(task)})))`.

The action is not smoothed, filtered, damped, residual-corrected, ranked, verified, or replaced by a learned action head.

## Training Signal

Training uses only self-generated closed-loop outcome labels on train reset identities:

- task success: `1` if the official environment reports success by termination, else `0`;
- elapsed steps and reward sum for diagnostics only;
- no simulator state, object pose, reward, success, or reset identity at inference.

Stage A uses the simplest discrete prior estimator:

- `task_success_noise_prior`: choose the highest training-success noise identity per task, with deterministic tie-break by lower mean episode steps and then lower noise index;
- `global_success_noise_prior`: choose the highest training-success noise identity pooled across tasks;
- `task_shuffled_noise_prior`: fit the same rule after shuffling task labels in the training table.

If Stage A reaches GO, Stage B may replace the rule with a small regularized classifier over task/family features, but that is not part of Stage A.

## Baselines

Stage A compares:

1. `frozen_smolvla`
2. `zero_noise_smolvla`
3. `global_success_noise_prior`
4. `task_shuffled_noise_prior`
5. `ocfn_full`

The key ablation is `task_shuffled_noise_prior`, which preserves the same training table and selection rule but destroys the task-to-noise relationship.

The simple reviewer-killer baseline is `global_success_noise_prior`, which tests whether OCFN is merely choosing one globally lucky latent.

The direct deterministic baseline is `zero_noise_smolvla`, which tests whether any fixed denoising start is enough.

## Prototype Tasks

Train tasks:

- `libero_spatial/task_4`
- `libero_10/task_4`

Train reset identities:

- `20260711`
- `20260712`

Held-out Stage A reset identities:

- `20260713`
- `20260714`
- `20260715`
- `20260716`
- `20260717`

Stage A held-out total:

- `5 variants * 2 tasks * 5 identities = 50 episodes`

Noise-bank train acquisition:

- `4 noise identities * 2 tasks * 2 train identities = 16 train-label episodes`

## GO And Kill Rules

Stage A is directional and may permanently kill only under `reports/current_research_governance.md`.

Permanent kill in Stage A if:

- implementation/data mechanism is invalid;
- `ocfn_full` is at least 30 absolute task-balanced points below the strongest baseline or key ablation;
- `ocfn_full` has `0 / 10` while a paired baseline has at least `4 / 10`;
- exact trivial equivalence to `zero_noise_smolvla`, `global_success_noise_prior`, or `task_shuffled_noise_prior` is demonstrated;
- the train acquisition proves no latent-sampling headroom, meaning every noise identity has identical train success, identical held-out action tensors under a synthetic deterministic check, and the selected full prior collapses exactly to the global baseline.

Advance to Stage B if:

- `ocfn_full` beats frozen SmolVLA and the key ablation; or
- the result is tied/noisy/small-negative but mechanism activation is valid and no Stage A permanent kill holds.

Stage B would use at least 40 paired episodes per key policy.

## Resource Plan

- no downloads;
- no full VLA fine-tuning;
- no OpenVLA-OFT work before prototype GO;
- one frozen SmolVLA policy resident during train acquisition or held-out evaluation;
- deterministic noise tensors stored by seed, not as large checkpoints;
- train acquisition and Stage A are resumable through partial JSON files.

## Expected Failure Modes

- latent noise has little or no closed-loop effect;
- one global fixed noise explains all performance;
- task-conditioned selection overfits two train identities and fails held-out identities;
- deterministic fixed-noise behavior is worse than frozen default sampling;
- the method cannot satisfy the final second-backbone requirement unless another flow-action VLA or an OpenVLA-compatible analogue is available later.
