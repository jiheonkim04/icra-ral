# Official ActionMap Asset Matrix

Date: 2026-07-09

## Decision-Relevant Summary

The official ActionMap repository does not currently publish the assets needed for reproduction. Local LIBERO assets exist, but they are not enough because the official ActionMap integration scripts, baseline checkpoints, ActionMap checkpoints, and evaluation commands are missing.

## Official Assets

| Asset | Official status | Local status | Feasibility impact |
| --- | --- | --- | --- |
| Paper | Public at arXiv. | Not downloaded to repo. | Sufficient for reading only. |
| Project page | Public. | Not downloaded to repo. | Confirms code is coming soon and summarizes results. |
| GitHub repo | Public. | Not cloned for this scout. | Pre-release only; no reproduction scripts. |
| Core heatmap head | Released as `heatmap_action_head.py`. | Not imported or copied. | Useful for code-level review only; not an official reproduction. |
| License | CC BY-NC-SA 4.0. | Audited by URL. | Noncommercial/share-alike constraints apply. |
| Training scripts | Not released. | Not available. | Blocks exact and subset reproduction. |
| Evaluation scripts | Not released. | Not available. | Blocks metric reproduction. |
| Config files | Not released. | Not available. | Blocks controlled reproduction. |
| Official checkpoints | Not linked. | OpenVLA-OFT path missing; SmolVLA exists but is not official ActionMap. | Blocks local official reproduction. |
| LIBERO datasets | Required by paper. | Present at `C:\assets\data\libero`. | Useful only after official scripts exist. |
| LIBERO/RoboSuite source | Required for simulation evaluation. | Present at `C:\assets\repos\LIBERO` and `C:\assets\repos\robosuite`. | Local source exists, but official environment compatibility is unknown. |
| Real Franka data/setup | Used by paper. | Not available. | Real-world reproduction not feasible locally. |
| Released logs/artifacts | Not linked. | Not available. | Blocks metric-only reproduction. |

## Expected Size / Compute

Official ActionMap does not publish an asset manifest or download-size table.

Conservative expectation:

- repository source: small;
- LIBERO assets: already local;
- VLA backbone checkpoints: multi-GB;
- H200-class training/evaluation compute: required for faithful paper reproduction according to the reported recipes.

Because the official release is incomplete, exact disk, VRAM, and runtime cannot be audited.

## Authentication / Licenses

- Paper, project page, and GitHub are public.
- GitHub/project license is CC BY-NC-SA 4.0.
- No ActionMap-specific authentication or click-through is visible.
- Any model checkpoint, dataset, or third-party backbone access requirement is unspecified by ActionMap.

## Local Compatibility Notes

- Windows has local data/checkpoint paths configured through `configs/paths.local.yaml`.
- WSL simulator readiness exists in prior reports, but official ActionMap does not provide WSL instructions.
- Local `tca_map` conda Python exists; `conda` itself was not on PATH, but `C:\Users\jiheo\miniconda3\Scripts\conda.exe` lists the `tca_map` env.
- The configured OpenVLA-OFT checkpoint path is absent, which blocks the paper's main backbone locally even if code were released.

## Asset Decision

`SOURCE_BLOCKED`

The missing official scripts, configs, checkpoints, logs, and asset manifest are the primary blockers.
