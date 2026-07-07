# ResetSpec-Retarget Reusable Artifacts

Keep these pieces:
- bounded WSL replay script: `scripts\170_resetspec_retarget_diagnostic.ps1`,
- diagnostic module: `tca_map.resetspec.retarget`,
- exact-init/default-reset replay comparison,
- object and EEF pose extraction from simulator observations,
- object-shifted EEF trajectory drift metric,
- EEF-object distance change and object-movement metrics,
- translation/rotation/gripper timing error metrics,
- controller-valid action and clip-rate metrics,
- non-leaking target resolver based on instruction text plus visible object keys,
- concise route result report: `reports\resetspec_state1_result.md`.

Use these only as diagnostic infrastructure. Do not reuse ResetSpec-Retarget as a main method claim unless a future task beats global scale, diagonal affine, clipping, nearest-demo, and raw replay under predeclared criteria.
