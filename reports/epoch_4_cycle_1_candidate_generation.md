# Epoch 4 Cycle 1 Candidate Generation

Date: 2026-07-13 KST

Decision: `SELECT_RCV_VLA`

## Candidate 1: RCV-VLA

Name: `RCV-VLA`, Receding-Chunk Verification for Frozen VLAs.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Closest external prior: SV-VLA, https://arxiv.org/abs/2604.02965.

Unresolved problem:

Action chunks are generated from stale observations, but executing only first actions every step was a strong local diagnostic in PSE. The unresolved question is whether a cheap verifier can recover most of the receding-horizon benefit without blindly invoking the heavy VLA every control step.

Technical mechanism:

Train a lightweight verifier from frozen-policy self-disagreement. At each rollout step, compare the queued action to the stateless fresh first action from the same frozen policy. Train a small head to predict whether that discrepancy exceeds a training-only threshold using current proprioception, queued action, chunk index, and short action history. At inference, continue the queued chunk when predicted valid, otherwise reset/replan.

Falsifiable mechanism chain:

stale chunk suffix -> high queued-vs-fresh action disagreement -> verifier-triggered replan -> fewer stale actions -> higher closed-loop success or lower heavy-policy calls at equal success.

Mathematical formulation:

`d_t = ||a_t^queue - a_t^fresh||_1 / 7`.

`y_t = 1[d_t > tau_train]`.

`h_phi(z_t) = P(y_t = 1 | q_t, a_t^queue, a_{t-1}, j_t/H)`.

Replan when `h_phi(z_t) > theta_train`.

First experiment:

Stage 0 compares normal queued SmolVLA, stateless first-action replanning, and queued-vs-fresh disagreement on a small paired manifest.

External-prior experiment:

Compare against an SV-VLA faithful proxy: direct deviation-based replanning using fresh first-action reference at every step. Label it clearly as a local proxy, not an official reproduction.

Simple killer:

Always use stateless first-action replanning.

Key ablation:

Verifier without current proprioception/action-history context.

Failure criteria:

Kill if RCV does not beat queued base, does not beat the key ablation, is matched by stateless first-action, or is worse than the SV-VLA proxy without meaningful inference savings.

Scores:

- provisional method novelty: `24 / 30`
- unresolved problem importance: `18 / 20`
- mechanism quality: `17 / 20`
- external-prior comparison feasibility: `14 / 15`
- decisive local experiment feasibility: `14 / 15`
- total: `87 / 100`

## Candidate 2: SPM-VLA

Name: `SPM-VLA`, Success-Peak Memory for Frozen VLAs.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: Retrieve-then-Steer / Online Success Memory, https://arxiv.org/abs/2605.10094.

Unresolved problem:

Independent zero-shot evaluation ignores persistent deployment, where the robot may repeat similar tasks and exploit prior successes.

Technical mechanism:

Store prefixes from successful frozen-policy episodes, retrieve nearest state-action prefixes by proprioceptive and visual-summary features, and blend retrieved actions into the frozen action chunk only when a high-precision progress proxy identifies a matching task stage.

Falsifiable mechanism chain:

repeated local deployment -> reliable successful segment memory -> retrieved stage-matched action prior -> corrected multi-stage action chunk -> higher repeated-task success.

Mathematical formulation:

`m_i = (e_i, a_i, s_i)` memory tuples from successful prefixes.

`w_i proportional exp(-D(e_t, e_i) / sigma)`.

`a_t = (1 - alpha_t) a_t^base + alpha_t sum_i w_i a_i`.

First experiment:

Build memory from successful training identities and evaluate held-out identities on the same two hard tasks.

External-prior experiment:

Compare to a faithful Retrieve-then-Steer proxy using the same memory and confidence-adaptive action blending.

Simple killer:

Nearest-neighbor action replay without progress filtering.

Key ablation:

No progress/stage confidence.

Failure criteria:

Kill if nearest-neighbor replay matches full, if memory contamination causes clean degradation, or if no held-out improvement appears.

Scores:

- provisional method novelty: `17 / 30`
- unresolved problem importance: `15 / 20`
- mechanism quality: `13 / 20`
- external-prior comparison feasibility: `12 / 15`
- decisive local experiment feasibility: `12 / 15`
- total: `69 / 100`

## Candidate 3: CMSS-VLA

Name: `CMSS-VLA`, Critical-Moment Speed Shaping for Frozen VLAs.

Contribution type: `NEW_DEPLOYMENT_PROBLEM`.

Closest external prior: TempoVLA, https://arxiv.org/abs/2606.06491.

Unresolved problem:

Manipulation alternates between transit phases and contact-sensitive phases, but frozen VLAs inherit a fixed execution speed and cannot explicitly decelerate near risk.

Technical mechanism:

Estimate critical moments from short-window action instability and proprioceptive velocity changes, then scale only translational/rotational action magnitude during high-risk windows while preserving gripper semantics.

Falsifiable mechanism chain:

contact-risk phase -> action instability spike -> dynamic speed reduction -> lower overshoot/contact error -> higher task success under speed-sensitive perturbation.

Mathematical formulation:

`r_t = max_{k in window} ||a_k - a_{k-1}||_2`.

`s_t = s_min + (1 - s_min) sigmoid(beta (eta - r_t))`.

`a_t' = [s_t a_t^{xyz,rpy}, a_t^gripper]`.

First experiment:

Evaluate under a controlled high-speed perturbation condition and compare to frozen and fixed-scale actions.

External-prior experiment:

TempoVLA official comparison is not locally compatible; use a transparent speed-control proxy only if selected.

Simple killer:

Fixed global slow scale.

Key ablation:

No critical-moment gating.

Failure criteria:

Kill if fixed scaling matches full, if clean task success drops, or if the method becomes another retiming/global-scaling rescue.

Scores:

- provisional method novelty: `14 / 30`
- unresolved problem importance: `14 / 20`
- mechanism quality: `10 / 20`
- external-prior comparison feasibility: `6 / 15`
- decisive local experiment feasibility: `13 / 15`
- total: `57 / 100`

## Selection

Selected method: `RCV-VLA`.

Reason:

RCV has the strongest concrete local problem evidence, the clearest closest external prior, and the most decisive local experiment. It is not allowed to claim novelty as generic replanning or adaptive chunk size. Its only defensible claim is a self-supervised, frozen-policy disagreement verifier that approximates receding-horizon replanning while comparing directly to SV-VLA-style deviation replanning and stateless first-action replanning.
