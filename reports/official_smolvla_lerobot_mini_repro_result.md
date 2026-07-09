# Official SmolVLA / LeRobot Mini-Repro Result

Date: 2026-07-09 KST

## Execution Boundary

- Experiments happened: yes, one bounded synthetic official-loader mini-repro.
- Training happened: no.
- Loss computed: no.
- GPU model execution happened: no.
- CPU-only diagnostic: yes, intentionally.
- Downloads happened: no.
- Rollout/simulator happened: no.
- OpenVLA-OFT happened: no.
- Custom LIBERO 7D adapter route used: no.
- Paper claims: no.

## What Ran

The mini-repro loaded the local SmolVLA base checkpoint using:

- `SmolVLAPolicy.from_pretrained`
- `make_pre_post_processors`
- local cached tokenizer/processor files for `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`

The synthetic input matched the local checkpoint schema:

- batch size `1`
- state `[1, 6]`
- three images `[1, 3, 256, 256]`
- one task string

## Result

- Model loaded: yes
- Processor/preprocessor loaded: yes, with local tokenizer override
- Action normalizer status: present, SO100 6D
- Official recipe status: official SmolVLA base loader/processor smoke passed
- Official LIBERO status: not reproduced
- LoRA smoke status: no LoRA training; tiny bitsandbytes CUDA optimizer smoke passed separately
- Action output shape and convention: `[1, 6]`, SO100-style action normalizer
- One-sample forward result: finite
- VRAM peak: `0.0 MB` because the smoke was intentionally CPU-only
- Runtime: `30.922 sec` end-to-end, `1.735 sec` single-sample inference

## CUDA And CPU-Fallback Check

CUDA was verified:

```text
torch.cuda.is_available(): True
torch.cuda.get_device_name(0): NVIDIA GeForce RTX 5080
```

This mini-repro intentionally used CPU and was not a training run. Therefore the CPU parameter/input devices are not a CPU fallback bug.

For any future LoRA training, the training script must log model parameter device, input tensor devices, CUDA allocated/max memory, and autocast state. If CUDA is available but model or inputs remain on CPU during a GPU training run, stop with `CPU_FALLBACK_BUG`.

## Limitation

The local checkpoint is 6D/SO100-style. LeRobot LIBERO is 8D state / 7D action. The mini-repro is valid as a SmolVLA base loader/processor feasibility check, but it is not an official LIBERO baseline or method evidence.

