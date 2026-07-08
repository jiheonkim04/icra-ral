# LIBERO-Safety Asset Matrix

Date: 2026-07-09

Decision: `TOO_HEAVY_LOCAL`

## Official Assets

| Asset | URL | Public/gated metadata | Size/status | Local status |
| --- | --- | --- | --- | --- |
| Paper | https://arxiv.org/abs/2606.23686 | Public | ArXiv page available; accepted to ECCV 2026. | Not applicable |
| Project page | https://libero-safety.github.io/ | Public | Paper, dataset, and code links available. | Not applicable |
| Code | https://github.com/LIBERO-SAFETY/LIBERO-Safety | Public | GitHub page showed official code, benchmark scripts, LIBERO fork, and 34 commits. | Missing at `C:\assets\repos\LIBERO-Safety` |
| Training dataset | https://huggingface.co/datasets/LIBERO-Safety/libero_safety | Public, ungated in API metadata | HF page reports about 19.1 GB. `meta/info.json` reports 19,664 episodes, 3,443,735 frames, 20 chunks, 7D actions, and two 256x256 video streams. | Missing at `C:\assets\data\libero_safety` |
| Assets archive | https://huggingface.co/datasets/LIBERO-Safety/libero_safety_assets/tree/main | Public, ungated in API metadata | `assets.zip`, 10,670,353,443 bytes by API tree metadata. | Missing |
| pi0.5 model | https://huggingface.co/LIBERO-Safety/pi05_libero_safety/tree/main | Public, ungated in API metadata | 29 files totaling 12,440,507,736 bytes by API tree metadata. | Missing at `C:\assets\checkpoints\libero-safety-pi05` |

## Existing Local Assets

| Local path | Present? | Notes |
| --- | --- | --- |
| `C:\assets\repos\LIBERO` | Yes | Standard LIBERO repo exists. |
| `C:\assets\repos\robosuite` | Yes | Robosuite repo exists, but not the official LIBERO-Safety vendored path. |
| `C:\assets\data\libero` | Yes | Standard LIBERO data exists: `libero_10`, `libero_90`, `libero_goal`, `libero_object`, `libero_spatial`. |
| `C:\assets\data\libero_safety` | No | Official LIBERO-Safety data/assets are not present. |
| `C:\assets\checkpoints\smolvla` | Yes | Not an official LIBERO-Safety checkpoint. |
| `C:\assets\checkpoints\openvla-oft` | No | Missing, and OpenVLA-OFT is forbidden in this run. |
| `C:\assets\checkpoints\libero-safety-pi05` | No | Official pi0.5 checkpoint is not local. |

## Access and License Notes

- The official GitHub, Hugging Face dataset, Hugging Face assets, and Hugging
  Face model pages are publicly viewable.
- API metadata reported `private=false` and `gated=false` for the dataset,
  assets, and model entries.
- No explicit GitHub license file or Hugging Face card license was confirmed
  during this bounded scout.
- No authentication or click-through gate was observed for metadata access.
  Actual artifact download would still need to respect Hugging Face terms and
  any repository-specific license updates.

## Disk Floor

Official assets plus dataset plus pi0.5 model imply a lower-bound footprint of
about 42 GB before code, conda environments, simulator caches, rendered videos,
logs, or additional model baselines.
