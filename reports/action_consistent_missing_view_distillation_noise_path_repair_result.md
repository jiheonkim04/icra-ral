# Action-Consistent Missing-View Noise Preflight: One Path Repair

Decision: `PREFLIGHT_OFFICIAL_READER_IMPORT_INITIALIZATION_ORDER_REPAIRED_ONCE`

The first Python noise worker failed during fixed-row materialization because
the official `datasets.dataset.InfiniteDataReader` was called before the pinned
local X-VLA source root was on `sys.path`. The X-VLA model was not loaded, CUDA
forward count was zero, optimizer steps were zero, and no confirmatory outcome
was accessed.

The complete failed run remains at
`runs/action_consistent_missing_view_distillation/noise_calibration_20260719T024502KST`;
its result SHA-256 is
`f57d7a88b5cbc99040ba54a48a49a7bbd0e9b977d0ff2140482552d8bef617c5`.
An earlier shell-redirection miss occurred before any Python worker existed and
is preserved as `prelaunch_attempt1.json`; it is an orchestration prelaunch
event, not a second scientific/preflight execution attempt.

Registering the source root exposed the second layer of the same import-order
defect: the unchanged rerun at `noise_calibration_20260719T024907KST` reached
the official reader, which then imported `mmengine.fileio` before the
repository's existing optional shim had run. That run also stopped before
model load, CUDA forward, or any optimizer step; its result SHA-256 is
`3e2bd5be48350416795584f52379b5822df56dceb2737f9e5780ef60d079631f`.

The single permitted repair therefore completed one narrow official-reader
import boundary: add the already-frozen `spec.xvla.source_root` and invoke the
already-existing optional shim immediately before the reader import. It adds
no package, download, data, objective, threshold, task, identity, repetition,
optimizer step, or output reinterpretation.

Repair count is now `1 / 1`. No additional implementation repair is
authorized.

The unchanged post-repair run at `noise_calibration_20260719T025602KST`
materialized all 12 frozen rows, so the reader-import repair passed its stated
boundary. It then encountered a distinct `RuntimeError: Invalid device
argument` at `torch.cuda.reset_peak_memory_stats(device)`, before model load,
CUDA forward, or optimizer execution. Its result SHA-256 is
`d6b82e257ba01639ab79565d4995757dadf066d8cd5644b92920e8b828c0d76f`.
Because the repair budget is exhausted, that device-runtime defect is
preserved rather than modified, and no additional rerun is authorized.
