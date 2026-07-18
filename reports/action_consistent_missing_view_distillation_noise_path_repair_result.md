# Action-Consistent Missing-View Noise Preflight: One Path Repair

Decision: `PREFLIGHT_OFFICIAL_READER_PATH_ERROR_REPAIRED_ONCE`

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

The single permitted implementation repair adds the already-frozen
`spec.xvla.source_root` to `sys.path` immediately before the official reader
import. It adds no package, download, data, objective, threshold, task,
identity, repetition, optimizer step, or output reinterpretation. The same
12-row, three-repeat, zero-optimizer calibration will be rerun unchanged.

Repair count is now `1 / 1`. No additional implementation repair is
authorized.
