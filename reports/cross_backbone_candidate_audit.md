# Cross-Backbone Candidate Audit

Date: 2026-07-11 KST

Objective: determine whether a second VLA backbone can test whether the observed SmolVLA failures are model-general execution problems. No model download, training, or rollout was performed.

## Candidate Order

The requested order was followed:

1. OpenVLA-OFT
2. A1 or another current fully open VLA only if OpenVLA-OFT is infeasible

OpenVLA-OFT is not ruled infeasible, so no A1-style fallback is selected.

## OpenVLA-OFT

| Field | Audit |
| --- | --- |
| Official source repository | `https://github.com/moojink/openvla-oft` |
| Paper | `https://arxiv.org/abs/2502.19645` |
| License/access | GitHub repo MIT; selected Hugging Face checkpoint MIT, public, non-gated |
| LIBERO support | Official `LIBERO.md` provides LIBERO evaluation scripts under `experiments/robot/libero/` and released LIBERO checkpoints |
| Selected checkpoint | `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` |
| Checkpoint SHA from Hugging Face API | `638918f3d1c2e43a39a8a20772bdb8b91835e4b7` |
| Expected checkpoint size | `15,939,168,050` bytes, `14.845` GiB |
| Action/state convention | LIBERO relative control; OpenVLA-OFT uses two images plus proprio state in its official LIBERO recipe |
| Frozen inference without fine-tuning | Yes, official released checkpoints can be evaluated directly |
| Expected local setup | Separate Linux/WSL conda env, OpenVLA-OFT repo, LIBERO repo, custom transformers fork, PyTorch/torchvision, robosuite/MuJoCo |
| Expected rollout time | At least a small integration pass plus 48-96 bounded episodes; no full default 500-trial suite |
| RTX 5080 16GB feasibility | Not safe to claim ready. The checkpoint alone is ~14.845 GiB, leaving too little margin for activations, images, action head, and simulator overhead unless quantization/offload is used, which would depart from the official path. |
| 8x RTX 3090 lab feasibility | Feasible for inference with a 24GB card likely available per process; official fine-tuning is not needed and remains forbidden. |
| Download status | Not downloaded |

Official OpenVLA-OFT notes that the released LIBERO evaluation commands automatically download the checkpoints and that the paper results used Python 3.10.14, PyTorch 2.2.0, a custom transformers v4.40.1 fork, and an NVIDIA A100 GPU. It also reports that training with batch size 8 requires about 62GB VRAM, while batch size 1 requires about 25GB VRAM; this audit does not approve any fine-tuning.

## Fallback Candidate Status

A1 or another fully open VLA was not selected because OpenVLA-OFT has unambiguous official source, released checkpoints, LIBERO evaluation support, and public access. The blocker is download/hardware approval, not candidate ambiguity.

## State 1 Decision

Decision: `SECOND_BACKBONE_DOWNLOAD_APPROVAL_REQUIRED`

Rationale: OpenVLA-OFT is the selected second backbone, but the selected checkpoint is a large 14.845 GiB download and local 16GB inference feasibility is not proven. No download or rollout should happen without explicit approval and a hardware path.
