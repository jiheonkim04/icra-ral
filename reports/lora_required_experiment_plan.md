# LoRA/QLoRA Required Experiment Plan

## Decision

LoRA and QLoRA are no longer optional nice-to-have items. They are required experimental tracks after the frozen/head-only path is validated.

They are not the main novelty. The main novelty remains:

- target-conditioned action heatmaps,
- counterfactual target/action consistency,
- Distributional TCA-Select.

LoRA and QLoRA are required compute-efficient adaptation arms. They test whether TCA-Map and Distributional TCA-Select still help when the model is allowed a small parameter-efficient adaptation budget.

Full fine-tuning remains forbidden locally.

## Existing Config Roles

The repository already contains:

- `configs/lora_adapter_lowcompute.yaml`,
- `configs/qlora_adapter_lowcompute.yaml`.

These configs are planning/guard configs for required low-compute adaptation tracks. They keep:

- SmolVLA-first,
- frozen backbone,
- no OpenVLA-OFT,
- no full fine-tuning,
- no rollout,
- no dataset download,
- batch size 1 style execution,
- low trainable-parameter estimates.

They do not authorize training by themselves. Any tiny LoRA smoke must be bounded by the same local safety budget used elsewhere.

## Required Experiment Matrix

| Stage | Track | Purpose | Local status |
| --- | --- | --- | --- |
| 0 | Native SmolVLA / frozen baseline | Establish frozen native behavior. | Load/interface smoke only so far. |
| 1 | ActionMap head-only | Required heatmap baseline. | Tiny smoke path exists. |
| 2 | TCA-Map head-only | Test target-conditioned head value. | Tiny smoke path exists. |
| 3 | TCA-Map head-only + Distributional TCA-Select | Test inference-time selection value. | Scaffolded; no paper result. |
| 4 | ActionMap + LoRA | Required PEFT baseline. | Planning/config guard only. |
| 5 | TCA-Map + LoRA | Required PEFT target-conditioned arm. | Planning/config guard only. |
| 6 | TCA-Map + LoRA + Distributional TCA-Select | Required combined PEFT + inference-selection arm. | Planning/config guard only. |
| 7 | TCA-Map + QLoRA + Distributional TCA-Select | Required feasibility arm if memory/tooling allows. | Planning/config guard only. |

## Required Comparisons

The minimum analysis must include:

- TCA-Map head-only vs ActionMap head-only,
- TCA-Map + TCA-Select vs TCA-Map without TCA-Select,
- TCA-Map + LoRA vs ActionMap + LoRA,
- TCA-Map + LoRA + TCA-Select vs TCA-Map + LoRA only,
- QLoRA variant if feasible under the local compute budget.

These comparisons separate:

- target-conditioning gain,
- inference-time Distributional TCA-Select gain,
- parameter-efficient adaptation gain,
- QLoRA memory/tooling tradeoff.

## LoRA Policy

Standing-approved after SmolVLA load-only and single-sample interface smoke pass:

- LoRA adapter construction,
- LoRA config validation,
- tiny LoRA smoke,
- tiny TCA-Map + LoRA smoke,
- tiny TCA-Map + LoRA + Distributional TCA-Select smoke,
- frozen backbone except LoRA adapter weights,
- batch size 1,
- max 100 steps for tiny smoke,
- max 200 samples for bounded local pilot diagnostics,
- max runtime 30 minutes for smoke,
- max VRAM target 14GB,
- no rollout,
- no simulator,
- no OpenVLA-OFT,
- no full fine-tuning.

Hard-stop if:

- LoRA requires full backbone fine-tuning,
- memory estimate exceeds 14GB,
- training would exceed 100 smoke steps,
- runtime is expected to exceed 30 minutes,
- CUDA/PyTorch major changes are required,
- OpenVLA-OFT is required,
- dataset download is required.

## Interpretation Rules

LoRA/QLoRA results must never be presented as the main novelty. If LoRA improves the numbers, report it as parameter-efficient adaptation. The method claim remains about target-conditioned action decoding, counterfactual consistency, and Distributional TCA-Select.

To avoid attribution confusion, always compare:

```text
ActionMap + LoRA
TCA-Map + LoRA
TCA-Map + LoRA + Distributional TCA-Select
```

If TCA-Map only improves without LoRA but disappears under LoRA, report that honestly. If LoRA dominates all gains, the method claim weakens and the paper should narrow its claim.

## Current Local Status

The current local status is ready for bounded local SmolVLA pilot work inside the risk-assessed pilot envelope and remains no-go for larger paper-grade experimental stages. The safe smoke stack has validated interface paths, not paper-grade results.

Next bounded local pilot work may continue autonomously through tiny LoRA smoke, TCA-Map + LoRA smoke, and TCA-Map + LoRA + Distributional TCA-Select diagnostics if each run stays within the documented 100-step, 200-sample, 30-minute, 14GB budget and does not cross hard-stop gates.
