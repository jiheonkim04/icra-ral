# FEDO-VLA Researcher Proposal

Date: `2026-07-12 KST`

Cycle: `2`

Method: `FEDO-VLA`, Feedback Execution-Disturbance Observer for VLA Policies

## Problem

Action-chunked VLAs emit policy-level end-effector references. In deployment, the low-level controller or robot hardware may realize a damped, biased, or phase-dependent version of those references. This execution gap is distinct from the Cycle 1 delay problem: the command is timely, but the realized motion is not the intended motion.

## Claim

FEDO-VLA learns a lightweight disturbance observer over recent command/realized-action feedback and task phase, then emits a residual command so that the realized low-level action better matches the VLA's intended action under controlled actuator faults.

## Technical Novelty

The method transfers disturbance-observer feedback into VLA action deployment:

- the backbone VLA is unchanged;
- the compensator is conditioned on measured execution error, not only the current image or action;
- task/phase features determine which action dimensions should be compensated;
- no simulator state, success predicate, or privileged object label is used at inference.

This is not DICD-VLA: DICD compensates stale observations with a delay-indexed chunk adapter. FEDO compensates action-realization error with feedback from the previous command and measured realized action.

## Closest Current Work

- APEX: `https://arxiv.org/html/2606.16504`
- RobustVLA: `https://arxiv.org/html/2510.00037v4`
- VLA-Corrector: `https://arxiv.org/abs/2607.01804`
- Real-time Correction for VLA Action Chunks: `https://arxiv.org/html/2509.23224v1`
- TIC-VLA: `https://arxiv.org/html/2602.02459v2`

## Exact Difference

APEX is the closest direct prior: it inserts a policy/controller execution adapter and adapts from tracking error. FEDO-VLA differs by testing a VLA-specific, phase-aware disturbance observer learned from VLA action traces under controlled action-realization faults, with APEX-style feedback as a required baseline. If an APEX proxy or static inverse-gain baseline matches FEDO, the method is killed.

RobustVLA trains for multi-modal perturbation robustness with adversarial action noise. FEDO is not broad robust post-training; it is a narrow test-time execution-disturbance observer with explicit command/realized-action feedback.

VLA-Corrector detects visual dynamics drift and replans/truncates chunks. FEDO does not use candidate ranking, visual drift detection, or replanning; it continuously changes the executed action reference.

## Mathematical Formulation

Let the frozen VLA emit intended action `a_t`. The low-level execution channel realizes `e_t = F_t(u_t)` for sent command `u_t`, with unknown phase-dependent disturbance. FEDO predicts a residual

`r_t = g_theta(a_t, a_{t-1}, u_{t-1} - e_{t-1}, phi_t, c)`

and sends

`u_t = clip(a_t + r_t)`.

Training minimizes

`L(theta) = ||F_t(a_t + g_theta(.)) - a_t||_1 + lambda ||g_theta(.)||_2^2`

using supervised labels from controlled action-realization faults over frozen SmolVLA action traces. At inference the only feedback is measured realized action from the low-level interface.

## Prototype

First backbone: official SmolVLA-LIBERO.

Controlled condition: deterministic phase-dependent action-realization fault in the rollout wrapper. The frozen policy sees normal observations from the faulted environment. The wrapper logs sent command and measured realized action.

Tasks:

- `libero_spatial/task_4`
- `libero_10/task_4`

Evaluation identities: `20260713` through `20260717`.

Variants:

1. faulted frozen SmolVLA
2. static inverse-gain baseline
3. APEX-style feedback proxy
4. FEDO without phase/error feedback ablation
5. FEDO full

## GO Gate

FEDO reaches prototype GO only if:

- full FEDO improves task-balanced closed-loop success by at least 5 points over the strongest prototype baseline;
- full FEDO beats the APEX-style feedback proxy;
- full FEDO beats the no-phase/error ablation;
- clean no-fault retention does not materially degrade in a small check;
- no simulator state or success predicate is used at inference.

## Kill Rules

Kill if:

- static inverse gain matches or beats full;
- APEX-style feedback proxy matches or beats full;
- no-phase/error ablation matches or beats full;
- faulted success does not improve over frozen;
- clean retention degrades materially;
- the result requires privileged simulator feedback unavailable on a real robot.
