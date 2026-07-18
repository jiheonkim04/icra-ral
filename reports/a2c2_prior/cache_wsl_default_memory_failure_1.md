# A2C2 Cache WSL Default-Memory Failure 1

Date: `2026-07-19 KST`

Classification: `RESOURCE_COMPATIBILITY_DEFECT`

After a clean WSL restart, Windows RAM began at 65.78%. The minimum live cache
path raised it to 88.93%, while WSL's local view was only 22.8% of its default
11.266 GiB limit. This is distinct from stale retained memory: the default
WSL2 VM allocation itself exceeded the host ceiling during a real model path.
The worker stopped with a valid durable cache at 533 anchors, 2,115 rows, and
35 episodes. No training, rollout, comparator result, or decision was exposed.

The previously absent global `.wslconfig` is set to a 3,584 MiB VM cap, zero
swap, and immediate cache reclaim. This follows Microsoft's documented WSL2
configuration interface and leaves about 1 GiB above the observed 2.57 GiB
minimum model RSS. The frozen method and all scientific fields are unchanged.

Official configuration reference:
<https://learn.microsoft.com/en-us/windows/wsl/wsl-config>
