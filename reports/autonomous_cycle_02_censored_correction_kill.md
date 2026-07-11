# Autonomous Cycle 02 - Intervention-Censored Correction Credit

Date: 2026-07-11 KST

Cycle branch target: `codex/ral-cycle-02-censored-correction`

Final cycle decision: `KILL_RECENT_WORK_AND_FEASIBILITY_COLLAPSE`

## Researcher A Proposal

Cycle 02 changed at least two axes relative to Cycle 01:

- core problem: temporal credit under deployment corrections rather than action representation;
- training signal: intervention-censored success credit rather than latent/effect action codes;
- closed-loop intervention: correction or truncation around failure-prone chunks.

Proposed method family:

`CENSOR-VLA`: train a lightweight recoverability or value head from corrected/perturbed rollouts, but censor success credit across intervention boundaries so bad pre-intervention actions are not rewarded just because a later recovery succeeds.

Three concrete variants were considered:

1. `CensoredValue`: value/recoverability head trained with post-intervention success blocked from pre-intervention actions.
2. `SetCorrection-Censored`: set-valued positive/negative action chunks from paired failed/corrected attempts, with intervention-censored labels.
3. `ResidualCensor`: small residual correction policy trained only near intervention boundaries, evaluated under controlled execution perturbations.

## Reviewer B Search

Closest current papers:

- TORL-VLA, https://arxiv.org/html/2606.09337v3, introduces an intervention-censored critic for contact-rich VLA online RL with human interventions and tactile/wrench feedback.
- Set-Supervised Diffusion Policy, https://arxiv.org/abs/2606.01865, trains action-chunk diffusion policies from paired undesired and corrective actions using set-valued supervision.
- AFIL, recorded in `reports/latest_vla_method_landscape_2026.md`, already uses online VLA failure rollouts as adaptive negative guidance.
- BORA, https://arxiv.org/html/2605.30226, bridges offline RL and online residual adaptation for dexterous VLA action diversity and physically reliable execution.
- VLA-Corrector, https://www.alphaxiv.org/abs/2607.01804, directly occupies lightweight detect-and-correct inference for action-chunked VLA policies.
- Pre-VLA, https://arxiv.org/abs/2605.22446, directly occupies pre-execution action validity and advantage verification.

Local feasibility blockers:

- no physical robot or human-intervention data;
- no tactile/force/wrench hardware;
- no authorized online RL on a real system;
- local failures are not a stable cross-backbone mechanism after Quantized OpenVLA-OFT INT4 succeeded `20/20` on the hard slice;
- ECHO showed that frozen SmolVLA stochastic candidates do not contain recoverable downstream headroom under same-state action interventions.

## Rebuttal

Researcher A could attempt a simulator-only intervention-censored prototype under artificial action disturbances. Reviewer B rejects this as a paper route because TORL-VLA already owns the intervention-censored critic idea in the harder real contact-rich setting, SDP owns paired positive/negative correction chunks, and VLA-Corrector owns lightweight correction/truncation at inference. A local simulator-only version would be weaker and likely viewed as a combination of known pieces.

## Kill Reason

The method family is killed before implementation:

- novelty collapses under TORL-VLA, SDP, AFIL, BORA, VLA-Corrector, and Pre-VLA;
- the essential data source for the strongest version is unavailable locally;
- a controlled-disturbance simulator prototype would not be review-resistant because it would compare against direct recent correction and set-supervision methods without a distinct mechanism.

Implementation is not authorized.

