# SafeLoRA-VLA Experiment Plan

Date: 2026-07-08

This is a plan-only artifact. This run is not allowed to perform training,
rollouts, large downloads, GPU jobs, OpenVLA-OFT execution, or STATE 2.

## Stage Boundary

STATE 0 and STATE 1 are allowed:

- read prior project reports,
- inspect official papers, code pages, dataset metadata, requirements, and
  licenses,
- estimate dataset/model/compute needs,
- define baselines and kill criteria,
- write go/no-go reports.

STATE 2 is disabled until a later explicit user approval.

## Candidate Future Stack

Preferred future stack if blockers are resolved:

- benchmark: LIBERO-Safety only if a bounded official subset can be used without
  monolithic asset/download pressure,
- model: local SmolVLA first,
- adaptation: property-conditioned LoRA or small property adapter mixture,
- fallback model: OpenVLA LoRA only if memory and setup become explicitly green,
- excluded locally: OpenVLA-OFT heavy training, full VLA fine-tuning, A100/H100
  multi-GPU jobs.

## Required Arms For A Future Smoke

Any approved future LoRA smoke must include:

| Arm | Purpose |
| --- | --- |
| Base policy or base adapter path | Measures unadapted safety/utility. |
| Standard imitation LoRA or L1 LoRA | Tests whether ordinary adaptation solves the gap. |
| Safety-only filter or stop-on-risk | Tests whether simple conservatism explains gains. |
| Generic DPO/ORPO LoRA | Tests whether generic preference tuning explains gains. |
| SafeLoRA property-conditioned LoRA | Tests the proposed method. |

## Required Metrics

The future smoke must report:

- task success or official task proxy,
- safe success,
- unsafe success or successful-but-unsafe rate,
- temporal/process violation rate,
- cumulative safety cost if available,
- risk exposure time if available,
- no-op/stop/abort rate,
- utility retention relative to base and standard LoRA,
- property-wise improvement,
- trainable parameter count,
- GPU/CPU memory and runtime,
- loss curves if training runs.

## Candidate SafeLoRA Objective

The future loss should be explicit before implementation:

```text
L = L_pref_property
  + lambda_retention * L_imitation_safe
  + lambda_noop * L_noop_or_abort
  + lambda_magnitude * L_action_preservation
  + lambda_route * L_property_router_regularization
```

Where `L_pref_property` must use official safe-over-unsafe pairs grouped by
safety property, not synthetic local proxy labels alone.

## Execution Gate

No command is authorized by this report. A later run may only proceed if
`reports/safelora_vla_state1_decision.md` changes to
`READY_FOR_USER_APPROVAL_TO_RUN_LORA` after blockers are resolved.

Current decision: `NO_CLEAR_LORA_PATH`.
