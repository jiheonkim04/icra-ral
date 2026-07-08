# LIBERO-Safety Official Feasibility

Date: 2026-07-09

Branch: `codex/libero-safety-official-feasibility`

Decision: `TOO_HEAVY_LOCAL`

This scout audited official LIBERO-Safety sources only. No experiments, training,
rollouts, large downloads, GPU use, OpenVLA-OFT runs, or local proxy benchmarks
were performed.

## Official Sources

| Item | Official source | Finding |
| --- | --- | --- |
| Paper | https://arxiv.org/abs/2606.23686 | Accepted to ECCV 2026. The abstract reports 19,664 strictly collision-free demonstrations and evaluation of 8 VLA plus 2 embodied foundation models. |
| Project page | https://libero-safety.github.io/ | Links paper, dataset, and code. Reports physical safety, semantic safety, data scaling, robustness, and failure cases. |
| Code | https://github.com/LIBERO-SAFETY/LIBERO-Safety | Public official codebase built on LIBERO. Repo page showed 34 commits, benchmark scripts, LIBERO fork, third-party robosuite 1.4, requirements, and setup files. No release tag was observed. |
| Training dataset | https://huggingface.co/datasets/LIBERO-Safety/libero_safety | Public and ungated in metadata. `meta/info.json` reports 19,664 episodes, 3,443,735 frames, 15 training tasks, 20 chunks, 7D actions, and two 256x256 video streams. HF page reports about 19.1 GB. |
| Assets | https://huggingface.co/datasets/LIBERO-Safety/libero_safety_assets/tree/main | Public and ungated in metadata. Contains `assets.zip`; API metadata reports 10,670,353,443 bytes. |
| Model weights | https://huggingface.co/LIBERO-Safety/pi05_libero_safety/tree/main | Public and ungated in metadata. API tree metadata totals 12,440,507,736 bytes for 29 files. |

## Reproduction Feasibility

| Scope | Feasible under current constraints? | Reason |
| --- | --- | --- |
| Full official reproduction | No | Requires official repo install, simulator setup, assets, checkpoints, rollouts, and likely GPU-backed model evaluation/training. |
| Small official subset | No | Still requires the 10.67 GB assets archive, official simulator setup, and at least one official policy/evaluation path. This violates the no-large-download and no-rollout boundary. |
| Metric-only from released demos/logs | No | No raw official rollout logs or standalone metric artifacts were found during this scout. Project figures are not enough for reproduction. |
| Code-level feasibility only | Yes | The source tree and setup instructions are inspectable, but code-level inspection is not a benchmark reproduction. |
| Not feasible | Partly | The source is not blocked, but meaningful local reproduction is too heavy for this run. |

## Local Compatibility

| Local item | Status | Implication |
| --- | --- | --- |
| `C:\assets\repos\LIBERO` | Present | Existing LIBERO assets may help later, but official LIBERO-Safety asks to replace/use the LIBERO-Safety fork. |
| `C:\assets\repos\robosuite` | Present | A robosuite checkout exists, but official repo vendors/uses `third_party/robosuite-1.4`. |
| `C:\assets\repos\LIBERO-Safety` | Missing | Official repo is not locally installed. |
| `C:\assets\data\libero` | Present | Standard LIBERO HDF5 assets exist, but they are not LIBERO-Safety official assets. |
| `C:\assets\data\libero_safety` | Missing | No local official LIBERO-Safety dataset/assets. |
| `C:\assets\checkpoints\smolvla` | Present | Useful for separate local VLA work, not an official LIBERO-Safety baseline by itself. |
| `C:\assets\checkpoints\openvla-oft` | Missing | OpenVLA-OFT cannot be used locally, and this run forbids it. |
| `C:\assets\checkpoints\libero-safety-pi05` | Missing | Official pi0.5 checkpoint is not local and is about 12.44 GB by API metadata. |
| Conda env `tca_map` | Present | Python 3.10.20 exists, but official simulator install was not attempted. |
| Windows host | Present | Official instructions include Linux-style system packages for rendering; WSL/Linux is likely safer for future reproduction. |

## Method-Gap Notes

LIBERO-Safety is not merely a collision-free demonstration dataset. The official
material includes physical safety, semantic safety, success rate, collision rate,
refusal behavior, robustness, and failure cases such as collision-free
incompletion and semantic misalignment.

A method gap may exist, but it is not validated by this scout. Any future method
work must first pass an official benchmark reproduction gate and must beat
safety-only data scaling, stop-on-risk, action clipping/no-op, generic DPO/SFT,
and simple adapter or LoRA tooling. This report does not propose or implement a
method.

## Verdict

`TOO_HEAVY_LOCAL`

The official source is available, so this is not `SOURCE_BLOCKED`. However, a
meaningful official mini reproduction would require large assets/checkpoints and
simulator rollouts, which are explicitly out of scope for this run.
