# RAC-VLA Reviewer B Attack

Date: 2026-07-14 KST

Researcher proposal hash: `71ABA93E37FC725C1A2E5EAE6E1461BC77AACDAFF9B0711C37F17D5C0AB0902F`

Note: this attack is against the proposal after the Stage 0 synthetic-command semantics correction. The correction makes labels represent hidden transforms `S_k` via inverse commands `S_k^{-1}(a)`.

## Closest Primary Sources

1. Reflective VLA, https://arxiv.org/abs/2606.25215
2. ReactVLA, https://arxiv.org/abs/2606.14255
3. PDF, https://arxiv.org/abs/2604.18107
4. GEAR-VLA, https://arxiv.org/abs/2606.08530
5. ProgressVLA, https://arxiv.org/abs/2603.27670

Additional local threats:

- FEDO-VLA and SCVC-style feedback or affine correction failures in this repository.
- RCV-VLA, where no-context/stateless variants beat the full mechanism.
- EvoState-VLA, where action-conditioned transition prediction did not beat actionless prediction enough.

## Main Novelty Threat

Reflective VLA already claims that observation-action-consequence triplets help VLAs infer deployment-specific factors. RAC-VLA is therefore not a new problem. It is a frozen-policy, low-dimensional, resource-bounded extension of that prior.

RAC may proceed only if it is framed as a `PRIOR_EXTENSION`, not as a wholly new paradigm. The contribution must be the minimal frozen-policy calibration mechanism and the matched local evaluation, not the observation that consequences matter.

## Trivial Equivalence Threat

The most dangerous simple explanation is:

`recent action effects reveal a diagonal action gain; invert that gain online`.

If an online diagonal inverse-gain or affine calibration baseline matches RAC, then RAC is not a paper candidate. This baseline must be included in the first serious comparison.

The second dangerous explanation is:

`history length helps, not action consequences`.

A no-consequence history ablation must use the same horizon, task, phase, and previous-action information but remove realized state deltas. If that ablation matches RAC, the Reflective mechanism is not supported locally.

The third dangerous explanation is:

`the synthetic perturbation labels are too easy because the action transform is visible directly from the command`.

Stage 0 must ensure the target cannot be solved by transformed action features alone. The audit must compare:

- action-only features;
- history-only/no-consequence features;
- full action-consequence features.

## Mathematical Threats

No KL divergence is justified. The proposal currently uses cross-entropy over explicitly defined synthetic perturbation classes and Huber/L2 action retention. That is acceptable if:

- perturbation classes are predeclared;
- class priors are balanced;
- labels are not generated from confirmatory identities;
- calibration residuals are bounded in action units;
- gradients flow only through the RAC calibration model.

If regression is used, report units and scale for each action group. Do not mix translation, rotation, and gripper residuals without normalization or explicit caps.

## Data And Leakage Threats

The development traces include terminal success labels, reset identities, and future states in the JSONL rows. RAC may use next-state deltas during training/audit, but not future state or identity at inference.

Stage 0 must prove:

- no identity `>= 20260917` is used before confirmatory freezing;
- no terminal success or reward enters inference features;
- duplicate `(task, identity, step, perturbation)` keys are zero;
- perturbation label counts are noncollapsed;
- train and validation identities are disjoint.

## Headroom Threat

If the controlled action-channel shift is too mild, Base may not fail meaningfully. If it is too severe, no policy can recover. Stage 0 or Stage A must include a diagnostic headroom check:

- Base under clean condition;
- Base under shifted condition;
- a diagnostic oracle inverse transform, labeled as oracle only.

The oracle is not an inference baseline and may not be used as a method.

## Prior Proxy Threat

The local Reflective proxy cannot be a straw baseline. It must use action-consequence history, not merely task/phase history. A fair proxy is allowed to choose among the same predeclared inverse templates using recent consequence evidence, but it must not use the RAC learned residual head.

If the Reflective proxy beats RAC, the extension fails.

## Resource Threat

The method can be developed cheaply. It should not launch hundreds of rollouts until:

- Stage 0 label and mechanism audits pass;
- the selected config differs from Base and ablation but remains bounded;
- clean validation behavior is retained.

## Reviewer B Decision

Decision: `ALLOW_WITH_REQUIRED_AUDITS`.

Reason: RAC is close to Reflective VLA and generic online calibration, but it is not an exact duplicate across architecture, policy component, supervision, inference, and resource setting. The proposal is testable and includes the simple killer baselines that can defeat it. Unknown empirical performance is not a rejection reason under current governance.

Required before Stage A:

1. mathematical mechanism audit with exact shapes, units, losses, and action caps;
2. Stage 0 data/headroom/mechanism audit;
3. validation search capped at six configs;
4. Reflective proxy and online inverse-gain baseline specified before rollout.
