# A2C2 Base Rollout OOM Failed Attempt 1

Date: `2026-07-19 KST`

Classification: `RESOURCE_COMPATIBILITY_DEFECT`

The 3,584 MiB cache-path WSL cap was too small for the distinct simulator
actual path. During the first LIBERO environment construction, before an
episode row or model forward, the kernel OOM-killed Python at about 2.77 GiB
anonymous RSS plus 256 MiB WSLg shared memory. Swap remained zero. No policy
outcome, success value, training, Prior rollout, or Ours result was exposed.

The one root-bounded simulator-memory correction increases the WSL cap to
4,096 MiB, retains zero swap, and disables unused WSL GUI support to remove
the observed WSLg shared-memory cost. These are documented WSL2 VM settings;
the simulator uses EGL and needs no GUI display. The same frozen Base stage is
rerun once with no scientific-field change.

Official configuration reference:
<https://learn.microsoft.com/en-us/windows/wsl/wsl-config>
