# Epoch 4 Cycle 2 Candidate Generation

Date: 2026-07-13 KST

Decision: `SELECT_CAVM_VLA`

## Candidate 1: CAVM-VLA

Name: `CAVM-VLA`, Contrastive Action-Value Memory for Frozen VLAs.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Closest external prior: Retrieve-then-Steer / Online Success Memory, https://arxiv.org/html/2605.10094v1.

Strongest adjacent priors: HELM, LaMem-VLA, Harness VLA, MemoryVLA/ReMem-VLA.

Unresolved problem:

Recent memory methods show that successful experience can steer frozen or generative VLAs, but success-only retrieval may replay actions that are common to both successes and failures. RCV's Stage 2B kill also showed that cheap action interventions can look useful until a no-context or stateless baseline explains them. The unresolved question is whether success and failure traces from the same frozen policy define a local action-advantage field that is more useful than success-only memory or nearest-success replay.

Technical mechanism:

Collect frozen SmolVLA traces on training identities and split them by terminal success. At inference, retrieve nearby success and failure trace states for the same task using non-privileged state/action-history features. Estimate a local contrastive action prior:

`v_t = mu_success(z_t) - mu_failure(z_t)`.

Intervene only when both neighborhoods are sufficiently populated and their action means are separated. Adjust the frozen action toward the local success mean and away from the local failure mean with a clipped confidence weight.

Falsifiable mechanism chain:

partial frozen competence -> nearby success/failure traces diverge in action space -> contrastive memory estimates a local successful action-value direction -> action chunk is nudged away from failure-like behavior -> higher held-out closed-loop success than success-only memory, nearest-success replay, and frozen SmolVLA.

Mathematical formulation:

For trace key `z_t = [q_t, a_t, a_{t-1}, rho_t, task_one_hot]`, retrieve same-task memories within a standardized distance kernel.

`w_i^+ proportional exp(-D(z_t, z_i^+) / sigma)`.

`w_j^- proportional exp(-D(z_t, z_j^-) / sigma)`.

`mu^+(z_t) = sum_i w_i^+ a_i^+`.

`mu^-(z_t) = sum_j w_j^- a_j^-`.

`m_t = ||mu^+(z_t) - mu^-(z_t)||_2`.

`c_t = clip((m_t - eta) / gamma, 0, 1) * density_gate(z_t)`.

`a'_t = clip_action((1 - alpha c_t) a_t + alpha c_t (mu^+(z_t) + beta (mu^+(z_t) - mu^-(z_t))))`.

First experiment:

Stage 0 runs a frozen SmolVLA trace acquisition on fresh training identities and measures whether each task has both successful and failed traces and whether local success/failure action means separate beyond a preregistered margin without using test identities.

External-prior experiment:

Compare against a transparent Retrieve-then-Steer proxy using success-only memory, same features, same retrieval kernel, and no failure memory. Label it as a local proxy, not an official reproduction, because the official progress critic and flow-sampler prior injection are unavailable here.

Simple killer:

Nearest-success action replay/blending using the same memory.

Key ablation:

No failure memory: use the success mean only with the same density gate and confidence mapping.

Failure criteria:

Kill if Stage 0 shows no local success/failure action separation, if the success-only proxy or nearest-success baseline matches full CAVM, if CAVM degrades below frozen SmolVLA, or if the method requires privileged simulator state at inference.

Scores:

- provisional method novelty: `25 / 30`
- unresolved problem importance: `18 / 20`
- mechanism quality: `18 / 20`
- external-prior comparison feasibility: `13 / 15`
- decisive local experiment feasibility: `13 / 15`
- total: `87 / 100`

## Candidate 2: EPHM-VLA

Name: `EPHM-VLA`, Episode-Phase History Memory for Frozen VLAs.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: HELM, https://arxiv.org/abs/2604.18791.

Unresolved problem:

Long-horizon tasks require retaining task phase and cross-phase context, but frozen SmolVLA execution observes only the current step and internal action queue.

Technical mechanism:

Maintain a compact phase-history vector from previous observations and action chunks, then feed it into a lightweight action residual head trained from successful frozen traces. Unlike HELM, this does not perform rollback or subgoal-aware verification.

Falsifiable mechanism chain:

lost phase context -> wrong action mode in repeated subgoals -> phase memory shifts action residual -> improved long-horizon held-out success.

Mathematical formulation:

`h_t = GRU(h_{t-1}, [q_t, a_t, task_one_hot])`.

`delta a_t = f_phi(h_t, q_t, a_t)`.

`a'_t = a_t + lambda delta a_t`.

First experiment:

Train on successful frozen traces and test held-out hard long-horizon identities.

External-prior experiment:

Compare to a HELM-inspired local proxy: phase memory plus failure verifier/reset, if implementable without privileged subgoals.

Simple killer:

Chunk-index-only phase residual.

Key ablation:

No recurrent history.

Failure criteria:

Kill if recurrent history does not beat chunk-index-only residual, if it collapses to a known action residual route, or if it needs subgoal labels unavailable locally.

Scores:

- provisional method novelty: `15 / 30`
- unresolved problem importance: `17 / 20`
- mechanism quality: `12 / 20`
- external-prior comparison feasibility: `8 / 15`
- decisive local experiment feasibility: `10 / 15`
- total: `62 / 100`

## Candidate 3: SAF-Lite-VLA

Name: `SAF-Lite-VLA`, Lightweight Spatial-Affordance Trap Escape for Frozen VLAs.

Contribution type: `NEW_DEPLOYMENT_PROBLEM`.

Closest external prior: Affordance Field Intervention, https://arxiv.org/abs/2512.07472.

Unresolved problem:

VLA memory traps under object displacement may require explicit spatial affordance cues, but the local setup lacks a 3D affordance model.

Technical mechanism:

Use only non-privileged image features and proprioception to detect repeated end-effector movement toward obsolete regions, then apply a small waypoint-like translation correction toward recent visually changed regions.

Falsifiable mechanism chain:

object/layout shift -> repeated movement toward obsolete visual region -> image-change waypoint prior -> improved OOD placement or grasp robustness.

Mathematical formulation:

`u_t = image_change_centroid(o_t, o_{t-k})`.

`a'_t^{xyz} = a_t^{xyz} + lambda gate_t (u_t - eef_projected_t)`.

First experiment:

Create a controlled visual/object-position perturbation manifest and compare frozen, fixed image-centroid correction, and SAF-Lite.

External-prior experiment:

AFI official reproduction is infeasible without SAF assets; use a clearly labeled image-affordance proxy only if selected.

Simple killer:

Fixed translation toward image-change centroid.

Key ablation:

No trap detector.

Failure criteria:

Kill if the image-change centroid baseline matches full, if clean performance drops, or if any object/affordance signal requires simulator state at inference.

Scores:

- provisional method novelty: `12 / 30`
- unresolved problem importance: `16 / 20`
- mechanism quality: `9 / 20`
- external-prior comparison feasibility: `5 / 15`
- decisive local experiment feasibility: `9 / 15`
- total: `51 / 100`

## Selection

Selected method: `CAVM-VLA`.

Reason:

CAVM is the only candidate with a strong current-prior comparison, a feasible local prototype, and a mechanism that changes at least two core dimensions relative to RCV. It is not allowed to claim novelty as generic memory, retrieval, nearest-neighbor action replay, or Retrieve-then-Steer reproduction. Its defensible claim is outcome-contrastive action memory: using both successful and failed frozen-policy traces to estimate a local action-advantage prior that success-only memory and nearest-success replay cannot explain.
