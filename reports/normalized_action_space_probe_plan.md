# Normalized Action-Space Probe Plan

This is a planning-only gate after the LIBERO action-stat subset audit.

Current evidence:

- local LIBERO HDF5 actions are 7D and unit-scale,
- the local SmolVLA checkpoint processor action stats are 6D and SO100-prefixed,
- checkpoint action mean/std magnitudes are far outside the local LIBERO action range,
- VLM-enabled offline decoding improved action-distance metrics but remained weak.

Decision policy:

- Do not scale learned-policy rollout from the current bridge.
- Do not bypass or replace action postprocessing yet.
- First resolve checkpoint/task provenance in a report-only audit.
- Only after provenance is resolved should a bounded normalized-action-space probe be implemented.

A future normalized-action-space probe, if justified, must be separately gated, offline, CPU-first, capped to a tiny local HDF5 subset, and labeled as diagnostic evidence only. It must not train, rollout, download, use GPU jobs, execute OpenVLA-OFT, alter model weights, or make paper-grade claims.

Command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\121_plan_normalized_action_space_probe.ps1
```

Expected interpretation:

- `decision=reduce_scope` with `selected_next_step=checkpoint_task_provenance_resolution` means the current checkpoint/action provenance mismatch is strong enough that rollout scaling and postprocessor changes remain blocked.
- `ready_for_bounded_normalized_action_space_probe_runner=false` means a future runner still needs a separate plan and task-local gate.
