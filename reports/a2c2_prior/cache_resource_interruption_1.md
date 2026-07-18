# A2C2 Cache Resource Interruption 1

Date: `2026-07-19 KST`

Classification: `RESOURCE_COMPATIBILITY_DEFECT`

At the 270-anchor heartbeat, the runner's WSL-local view reported 23.3% RAM
and 6.7% reserved VRAM, but the Windows host reported 87.93% physical RAM,
above the frozen 82% ceiling. The worker was stopped. Its durable HDF5 cache
was valid at 384 unique anchors, 1,525 rows, and 26 episodes.

No training, rollout, comparator outcome, or decision was exposed. With no
research worker left in WSL, a clean WSL shutdown returned retained host
memory and reduced Windows RAM use to 65.34%. The identical cache command may
resume only the missing frozen anchors; no scientific field changes.
