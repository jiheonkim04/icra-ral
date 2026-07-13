# Researcher A Proposal: CAVM-VLA

Date: 2026-07-13 KST

Method name: `CAVM-VLA`, Contrastive Action-Value Memory for Frozen VLAs.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Closest external prior: Retrieve-then-Steer / Online Success Memory, https://arxiv.org/html/2605.10094v1.

## Core Claim

Frozen VLA rollouts contain two kinds of reusable deployment evidence:

1. successful traces showing action patterns that can complete the task;
2. failed traces showing action patterns that looked locally plausible but led to terminal failure.

Success-only memory methods retrieve and replay or steer toward successful action segments. CAVM tests the stricter hypothesis that, on partially competent frozen VLAs, the useful local intervention is not the success mean alone but the contrast between nearby successful and failed action neighborhoods.

The method estimates a local outcome-contrastive action prior and applies it as a bounded action-vector intervention during closed-loop execution.

## Problem Evidence

RCV-VLA Stage 2B showed that action interventions can beat queued execution and an SV-style deviation proxy while still being explained by simpler no-context or stateless baselines. Therefore the next method must not be another verifier or replanning gate.

The frozen SmolVLA hard tasks used throughout the campaign have mixed success/failure outcomes under fixed task/reset manifests. This creates an opportunity to ask whether outcome-labeled traces contain local action differences that can improve held-out closed-loop success.

## Mechanism

Collect frozen SmolVLA traces on training identities for the selected hard tasks. For each step record:

- task key;
- 8D proprioceptive state `q_t`;
- executed 7D action `a_t`;
- previous 7D action `a_{t-1}`;
- chunk-index fraction `rho_t`;
- terminal episode success label `y_episode`.

At inference, build a non-privileged key:

`z_t = [q_t, a_t, a_{t-1}, rho_t, task_one_hot]`.

Retrieve same-task memories using standardized Euclidean distance over `z_t`.

Let `M+` be successful trace records and `M-` be failed trace records. For a current key `z_t`:

`w_i+ = exp(-D(z_t, z_i+) / sigma) / sum_k exp(-D(z_t, z_k+) / sigma)`.

`w_j- = exp(-D(z_t, z_j-) / sigma) / sum_k exp(-D(z_t, z_k-) / sigma)`.

`mu+(z_t) = sum_i w_i+ a_i+`.

`mu-(z_t) = sum_j w_j- a_j-`.

`v(z_t) = mu+(z_t) - mu-(z_t)`.

`m(z_t) = ||v(z_t)||_2`.

The confidence gate is:

`c_t = density_gate(z_t) * clip((m(z_t) - eta) / gamma, 0, 1)`.

The full CAVM action is:

`a'_t = clip_action((1 - alpha c_t) a_t + alpha c_t (mu+(z_t) + beta v(z_t)))`.

## Falsifiable Chain

Observed failure:

Frozen SmolVLA succeeds and fails on similar tasks/resets, and action interventions can be confounded by stateless or no-context baselines.

Intermediate mechanism:

Nearby successful and failed traces separate in action space at some states.

Policy behavior:

CAVM shifts the current action toward the local successful action mean and away from the local failed action mean only when the contrastive margin is high.

Closed-loop outcome:

CAVM improves task-balanced success over frozen SmolVLA, success-only memory, nearest-success replay, and a no-failure-memory ablation on held-out identities.

## Required Baselines

1. `frozen_smolvla`: unmodified queued SmolVLA.
2. `success_only_memory_proxy`: local Retrieve-then-Steer proxy using the same retrieval kernel but only successful traces.
3. `nearest_success_replay`: strongest simple reviewer-killer baseline.
4. `cavm_no_failure_ablation`: key ablation that removes failed-trace contrast.
5. `cavm_full`: full outcome-contrastive memory.

## Stage Plan

Stage 0: problem diagnostic. Acquire frozen SmolVLA traces on training identities. Proceed only if every selected task has both successes and failures and local success/failure action means separate beyond the preregistered margin.

Stage 1: memory calibration. Fit standardization, retrieval bandwidth, contrast margin, and clipping constants using only training/calibration identities. Save memory and config artifacts.

Stage 2A: early closed-loop comparison over `10` paired episodes per variant.

Stage 2B: confirmatory comparison over `40` paired episodes per variant when Stage 2A is non-catastrophic.

## GO Criteria

Stage 2B reaches prototype GO only if:

- `cavm_full` beats the strongest baseline and `cavm_no_failure_ablation`;
- the absolute gain is at least `10` task-balanced points at prototype scale or paired evidence is consistently positive with meaningful failure-rate reduction;
- the contrastive memory gate activates on held-out episodes;
- no privileged simulator signal is used at inference;
- heavy policy call count does not exceed frozen queued SmolVLA, because CAVM modifies actions after the normal policy call rather than replanning.

## Kill Criteria

Permanently kill the current formulation if:

- Stage 0 finds no success/failure mixture or no local action separation;
- `success_only_memory_proxy`, `nearest_success_replay`, or `cavm_no_failure_ablation` matches or beats `cavm_full`;
- `cavm_full` is clearly worse than frozen SmolVLA;
- the effect requires privileged simulator state at inference;
- the intervention collapses to global action scaling, mean-action replay, or task-level memorization.

## Non-Claims

CAVM is not an official Retrieve-then-Steer reproduction.

CAVM is not generic memory retrieval.

CAVM is not a verifier, confidence head, value head, or replanning method.

CAVM does not fine-tune SmolVLA and does not claim that memory alone is novel. The claim lives in outcome-contrastive action memory and must be proven against success-only and nearest-success baselines.
