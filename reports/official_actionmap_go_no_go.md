# Official ActionMap Go/No-Go

Date: 2026-07-09

## Final Decision

`SOURCE_BLOCKED`

## Decision Rule

Choose exactly one:

- `GO_OFFICIAL_ACTIONMAP_MINI_REPRO`
- `NEEDS_CLOUD_OR_GPU`
- `SOURCE_BLOCKED`
- `TOO_HEAVY_LOCAL`
- `NO_OFFICIAL_CODE_OR_ASSETS`
- `STOP_VLA_METHOD_SEARCH_UNDER_CURRENT_CONSTRAINTS`

## Why Not GO

No official mini reproduction can be run now because the official source is incomplete:

- full code is marked as coming soon;
- no official install instructions are present;
- no official training or evaluation commands are present;
- no official checkpoint links are present;
- no official dataset manifest or released logs are present;
- local OpenVLA-OFT checkpoint path is missing;
- the paper's reported compute is H200-class multi-GPU, while this scout forbids GPU use and training.

## Reproduction Scope Chosen

Code-level feasibility only.

The released `heatmap_action_head.py` can be read as a preview of the action head, but using it locally would become another proxy approximation rather than official reproduction. That is disallowed by the current project state.

## Secondary Blockers

If the full official code appears later, the next blockers will likely be:

- cloud or H200-class GPU access;
- large VLA checkpoint downloads;
- official LIBERO/RoboSuite environment matching;
- model/dataset access requirements if the authors depend on third-party gated assets.

These are secondary today because the official source itself is not yet sufficient.

## Success Criteria For Future GO

Switch to `GO_OFFICIAL_ACTIONMAP_MINI_REPRO` only when all of the following are true:

- official repo releases complete install and reproduction instructions;
- a tagged release or pinned commit is available;
- a bounded official subset is documented;
- required datasets/checkpoints and sizes are known before download;
- required GPU/VRAM/runtime are compatible with the approved environment;
- expected output metrics and pass/fail tolerance are specified.

## Exact Next Command

No reproduction command is valid under `SOURCE_BLOCKED`.

The only safe next check, later, is a source-availability check against the official repo:

```powershell
git ls-remote https://github.com/showlab/ActionMap.git
```

Do not run training, rollout, OpenVLA-OFT, GPU jobs, downloads, Target-Grounded ActionMap, or local proxy approximations from this scout.
