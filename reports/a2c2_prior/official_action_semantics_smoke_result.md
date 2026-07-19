# A2C2 Official Action-Semantics Smoke Result

Date: 2026-07-19 KST

Exact decision: `CORRECTED_A2C2_OFFICIAL_SEMANTICS_SMOKE_PASS`.

All four preregistered 80-step technical traces completed. The run created
zero scientific episode rows and did not inspect, persist, or count task
success, done, or reward. Exact Base/Prior checkpoints strict-loaded on CUDA,
their action normalization statistics matched exactly, Base forwards were
nonzero in every trace, and the Prior executed 160 live forwards.

Raw nominal-bound exceedance occurred only in gripper dimension 6. It is an
official-path diagnostic rather than an automatic invalidity. All raw actions
were finite 7-D values, every controller call was accepted, native arm and
gripper effective values remained within their official bounds, actuator
commands and torques remained within limits, and simulator states stayed
finite. No external action clip was added. There were zero native arm clip
steps. Native gripper saturation operated as specified.

The Prior did not show reproducible practical action instability. Relative to
matched Base, its maximum raw exceedance and exceedance fraction were lower on
both development identities; neither identity met the frozen substantial-
instability rule.

Peak allocated VRAM was `1532.542 MiB`, peak process RSS was `4131.066 MiB`,
and host physical use peaked at `68.71%`. WSL swap was zero, pagefile current
usage did not grow, page writes remained zero, no offload occurred, memory was
released after shutdown, and no exception occurred.

The full raw result, all 122 nominal-exceedance events, per-step raw values,
and host samples remain under
`runs/a2c2_fidelity_corrected/official_semantics_smoke_20260719T205835KST/`;
their hashes are recorded in the JSON report. This valid smoke opens only the
unchanged 45-row corrected panel.
