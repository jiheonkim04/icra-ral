# SafeLoRA-VLA Benchmark Selection

Date: 2026-07-08

## Priority Order Result

| Priority | Candidate | Selection status | Reason |
| ---: | --- | --- | --- |
| 1 | LIBERO-Safety official subset | Candidate, not green | Public code/data/assets/model exist, and it is closest to a low-compute SmolVLA path. But no explicit official tiny split, property-conditioned unsafe labels, or real LoRA recipe is green. |
| 2 | SafeManip official subset | Cloud-only, not local | Best temporal-property benchmark, but official reproduction needs simulator setup, large checkpoint assets, and GPU rollout generation. |
| 3 | ForesightSafety-VLA | Source blocked | Strong CC/RET/process metrics on paper, but no official code/data found. |
| 4 | Local standard LIBERO fallback | Rejected as evidence | Useful only for engineering smoke; not official safety benchmark evidence and previous proxy route was killed. |

## Selected Benchmark For A Future Gate

No benchmark is selected for immediate training.

Best future candidate: `LIBERO-Safety`, conditional on resolving:

- bounded official subset or streaming/sample policy,
- license due diligence,
- property-level unsafe labels or rollout-derived official violations,
- real SmolVLA/OpenVLA LoRA implementation path,
- baseline table that includes standard LoRA, generic DPO/ORPO, and
  safety-only/stop-on-risk.

## Why Not READY

`READY_FOR_USER_APPROVAL_TO_RUN_LORA` would require:

- an official accessible benchmark/source,
- a small or clearly bounded dataset path,
- no license/token/payment blocker,
- a technically feasible LoRA/QLoRA path,
- estimated memory/runtime,
- defined baselines,
- acceptable failure risk.

This run has estimated memory/runtime for local SmolVLA engineering checks, but
the official property-label and LoRA execution path is not clear enough.

Current benchmark decision: no immediate benchmark run.
