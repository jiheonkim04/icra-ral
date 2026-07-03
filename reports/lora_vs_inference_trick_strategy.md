# LoRA/QLoRA vs Inference-Time Selection Strategy

## Bottom Line

LoRA and QLoRA are required experimental tracks after the head-only path is validated, but they are not enough for novelty. TCA-Select is the main inference-time trick in the low-compute method.

## Role Of Each Component

Head-only TCA-Map is the default. It keeps the backbone frozen and trains only the target/action heads.

TCA-Select is the main method extension. It turns target-conditioned heatmaps into a stronger inference procedure by selecting among candidate actions using internal target/action consistency.

LoRA is required as a compute-efficient adaptation arm after head-only validation. It must stay within the low-compute budget and must not become full-backbone fine-tuning.

QLoRA is required as a feasibility track if memory/tooling allows. It must require explicit config and should not be used to sneak in full-backbone fine-tuning.

## Evaluation Separation

Reports must separate:

- head-only gain,
- LoRA gain,
- QLoRA gain if used,
- inference-time TCA-Select gain.

Minimum ablation table:

| Variant | Backbone | Trainable params | Selection | Purpose |
| --- | --- | --- | --- | --- |
| Native head | SmolVLA frozen | native/head only | none | base VLA action head |
| ActionMap | SmolVLA frozen | head only | argmax/expected action | heatmap baseline |
| ActionMap + CF aug | SmolVLA frozen | head only | argmax/expected action | data augmentation baseline |
| TCA-Map | SmolVLA frozen | head only | argmax/expected action | target-conditioned heatmap |
| TCA-Map + TCA-Select | SmolVLA frozen | head only | TCA-Select | main low-compute method |
| ActionMap + LoRA | SmolVLA frozen + small adapters | head + LoRA | none | required PEFT baseline |
| TCA-Map + LoRA | SmolVLA frozen + small adapters | head + LoRA | none | required PEFT target-conditioned arm |
| TCA-Map + LoRA + TCA-Select | SmolVLA frozen + small adapters | head + LoRA | TCA-Select | separate adapter and selection effects |

## Reporting Rule

If LoRA improves results, report it as an efficiency/supporting adaptation result. Do not present LoRA as the core novelty. The core novelty is target-conditioned heatmap decoding plus inference-time TCA-Select. Always compare ActionMap + LoRA against TCA-Map + LoRA so LoRA gains do not get mistaken for TCA-Map gains.
