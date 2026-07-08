# Official ActionMap Feasibility

Date: 2026-07-09

Branch: `codex/official-actionmap-feasibility`

## Decision

`SOURCE_BLOCKED`

The official ActionMap source is public but currently pre-release. The repository states that the full code is coming soon and exposes only the core heatmap action head preview. That is enough for code-level reading, but not enough for exact official reproduction, small official subset reproduction, released-log metric reproduction, or any paper-grade claim.

## Sources Audited

- Paper: https://arxiv.org/abs/2606.06904
- Project page: https://showlab.github.io/ActionMap/
- GitHub: https://github.com/showlab/ActionMap
- Core preview file: https://github.com/showlab/ActionMap/blob/main/heatmap_action_head.py
- License: https://github.com/showlab/ActionMap/blob/main/LICENCE

## Official Source Audit

| Item | Finding |
| --- | --- |
| Paper URL | `https://arxiv.org/abs/2606.06904`, submitted 2026-06-05 and revised 2026-06-10. |
| Project page | `https://showlab.github.io/ActionMap/`. |
| Code availability | Pre-release only. GitHub contains README, license, project-page assets/docs, and `heatmap_action_head.py`; full training/evaluation code is not released. |
| License | CC BY-NC-SA 4.0. |
| Dataset requirements | LIBERO four-suite simulation for main evaluation; real Franka demonstrations for physical robot studies. |
| Checkpoint requirements | OpenVLA-OFT / Prismatic-7B path for the main backbone; pi0.5/JAX path for cross-backbone verification. Official checkpoint links are not provided by ActionMap. |
| Expected download size | Not specified by official ActionMap. A faithful reproduction would require at least multi-GB VLA checkpoints plus LIBERO assets; exact size cannot be audited without official manifests. |
| Expected GPU/CPU requirements | Paper reports OpenVLA-OFT LIBERO finetuning on 2 H200 GPUs and pi0.5/JAX LIBERO runs on 8 H200 GPUs. CPU-only reproduction is not described. |
| Simulator requirements | LIBERO / RoboSuite for simulation evaluation; real Franka setup for physical-robot results. |
| Authentication/token/click-through | Public paper, project page, and GitHub need no authentication. Any model/dataset token or click-through requirement is not specified by ActionMap. |
| Instruction sufficiency | Insufficient. The official repo does not provide install, data, checkpoint, training, or evaluation commands. |

## Reproduction Scope

Current possible scope: code-level feasibility only.

Not currently possible:

- exact official reproduction;
- small official subset reproduction;
- metric-only reproduction from released logs/artifacts.

Reason: there are no official scripts, configs, checkpoints, dataset manifests, metric logs, or evaluation commands in the released source.

## Local Compatibility

| Local surface | Status |
| --- | --- |
| LIBERO/RoboSuite source | Present locally at `C:\assets\repos\LIBERO` and `C:\assets\repos\robosuite`. |
| LIBERO HDF5 demos | Present locally at `C:\assets\data\libero`, including `libero_10`, `libero_90`, `libero_goal`, `libero_object`, and `libero_spatial`. |
| Existing conda env | `C:\Users\jiheo\miniconda3\envs\tca_map` exists and reports Python `3.10.20`. |
| SmolVLA path | Present at `C:\assets\checkpoints\smolvla`, but this is not an official ActionMap backbone. |
| OpenVLA-OFT path | Configured as `C:\assets\checkpoints\openvla-oft`, but the path is missing locally. |
| Windows/WSL | Existing reports show WSL simulator import-readiness work, but official ActionMap does not provide a Windows/WSL install path. |
| GPU constraints | This run is GPU-forbidden. The paper's reported recipes require H200-class multi-GPU training, so local official reproduction is blocked even after source release unless a bounded official subset is published. |

## Minimum Reproduction Plan

No executable official reproduction command is available today.

Conditional smallest official plan if the authors release full instructions:

1. Clone the official repo at a tagged release or commit.
2. Install the official environment exactly as specified by the release.
3. Download only the official minimum dataset/checkpoint subset named by that release.
4. Run the official minimum LIBERO evaluation or finetuning command without modification.
5. Compare the released success metric against the official baseline in the same command set.

Expected runtime, VRAM, disk, and output metrics cannot be responsibly specified until official commands and asset manifests exist. Based on the paper, a faithful LIBERO finetuning reproduction is expected to require multi-GPU H200-class compute, large VLA checkpoints, and LIBERO simulator evaluation.

## Method Gap Boundary

No new method is proposed.

Because official reproduction is not currently feasible, no failure-mining targets are activated or approved. Failure mining can be scoped only after an official ActionMap reproduction is feasible and green.

## Exact Next Step

Monitor the official ActionMap repository for a full release with install, data, checkpoint, training, and evaluation instructions. Do not run a local proxy or Target-Grounded ActionMap work while the source remains pre-release.
