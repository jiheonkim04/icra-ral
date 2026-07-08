# TG-VLA Model Feasibility

Date: 2026-07-09 KST

## SmolVLA Runtime Status

Current checker output:

- `ready_for_smolvla_adapter_smoke=true`
- `smolvla_load_only_smoke_passed=true`
- `smolvla_single_sample_interface_passed=true`
- `feature_cache_eval_smoke_passed=true`
- `tiny_head_only_smoke_passed=true`
- runtime dependencies ready for load-only runtime: true

Present runtime packages include:

- `torch`
- `transformers`
- `lerobot`
- `safetensors`
- `accelerate`
- `huggingface_hub`
- `h5py`
- `pandas`

Missing low-resource adapter packages:

- `peft`
- `bitsandbytes`

## Known Local SmolVLA Interface

The local SmolVLA config exposes:

- state feature: `(6,)`
- image features: three 3x256x256 camera tensors
- output action feature: `(6,)`
- max action dim: 32
- chunk size: 50
- tokenizer max length: 48

Existing single-sample smoke produced finite action output with shape `[1, 6]` on CPU. Local LIBERO actions are 7D, so any real smoke must use an explicit 6D-to-7D action adapter or evaluate the first 6 action dimensions plus gripper separately.

## Adapter Injection Feasibility

Technically plausible:

- a small target-conditioned residual adapter after the action output,
- a small action-head conditioning module around SmolVLA action expert internals,
- a FiLM/AdaLN-style gate if the relevant action-head conditioning path is exposed cleanly,
- a custom LoRA wrapper on selected action-path linear layers.

Not currently ready:

- no repo-integrated TG-VLA SmolVLA action-path adapter runner,
- no off-the-shelf PEFT LoRA package in the active environment,
- no QLoRA path because `bitsandbytes` is absent,
- no leakage-safe target resolver implemented for visible object names and instruction text.

## Feasible First Real Smoke Design

If this route were reopened, the first acceptable real smoke would:

1. Load local SmolVLA once with offline Hugging Face settings.
2. Build real batches from local LIBERO HDF5 first-frame observations and original/paraphrased instructions.
3. Freeze SmolVLA.
4. Train only a tiny action-path adapter or LoRA wrapper.
5. Compare against frozen/base, standard adapter, canonicalization-only, and paraphrase augmentation.
6. Report action metrics, held-out paraphrase metrics, object lexical metrics, counterfactual sensitivity, clean retention, loss curve, trainable parameters, runtime, and VRAM.

## Model Decision

Model feasibility is yellow, not green.

SmolVLA can be loaded and queried locally, and official model documentation supports training-step APIs. However, the exact TG-VLA action-path adapter is not implemented, PEFT/QLoRA tooling is absent, and a naive residual adapter would not yet clear the novelty/baseline gate.
