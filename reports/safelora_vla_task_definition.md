# SafeLoRA-VLA Task Definition

Date: 2026-07-08

Status: bounded feasibility gate only. No training, rollout, large download,
GPU job, OpenVLA-OFT execution, or STATE 2 work was performed.

## Problem

Recent VLA safety benchmarks show that task success is not the same as safe
execution. SafeManip focuses on temporal manipulation properties, LIBERO-Safety
focuses on physical and semantic safety across procedurally varied LIBERO tasks,
ForesightSafety-VLA focuses on process-level safety diagnostics, and
SafeVLA-Bench exposes successful-but-unsafe rollouts under native LIBERO and
RoboCasa protocols.

SafeLoRA-VLA is only viable if it uses those official safety signals to improve
the policy, not merely to diagnose failures or build another local proxy.

## Candidate Thesis

SafeLoRA-VLA would convert official temporal or process-level safety violations
into parameter-efficient VLA adaptation signals:

- property-specific safe-over-unsafe trajectory or action-chunk preferences,
- property-conditioned LoRA routing, gating, or adapter mixtures,
- utility-retention losses that prevent the model from becoming safe only by
  stopping,
- evaluation on official safe-success and process-safety metrics.

The method target is not "lower violation rate" alone. The target is higher
utility-retained safe success.

## Required Method Shape

The minimum method must include:

- a temporal/process safety monitor from an official benchmark source,
- property-specific labels for categories such as grasp stability, release
  stability, containment, no transport with open gripper, collision/contact,
  cross-contamination, action onset, or mechanism ordering,
- one of property-token-conditioned LoRA, property embedding-routed LoRA, a small
  property adapter mixture, or shared LoRA with property-conditioned gating,
- imitation/L1 retention on successful safe actions,
- no-op/stop penalty and action-magnitude preservation,
- explicit anti-filtering comparisons.

## Required SOTA Axis

SafeLoRA may only claim progress on axes that the official benchmark supports:

- safe-success rate,
- temporal violation rate,
- cumulative safety cost,
- risk exposure time,
- utility retention/task success retention,
- property-wise safety improvement.

## Difference From SafeTrace-VLA

SafeTrace-VLA was killed because monitor-derived preferences collapsed to
safety-only/risk-only scoring and generic DPO. SafeLoRA must therefore prove an
additional mechanism-level contribution:

- property conditioning must change the trainable update, not only the text
  label;
- retention must be optimized jointly with safety, not measured after the fact;
- generic DPO/ORPO with the same pair set must be a required baseline;
- safety-only or stop-on-risk must be scored on safe-success, not just
  violation rate.

## Current Gate Result

The concept has a plausible method gap, but this run did not find a green
official benchmark plus real LoRA/QLoRA execution path under the current
constraints. Continue only after the blockers in
`reports/safelora_vla_state1_decision.md` are resolved.
