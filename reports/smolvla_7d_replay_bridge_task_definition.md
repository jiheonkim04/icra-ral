# SmolVLA 7D Adapter Replay Bridge Task Definition

Objective: determine whether the fixed SmolVLA/LIBERO 7D adapter baseline is executable enough for exact-init LIBERO/RoboSuite replay.

This is not a new method, not paper novelty, and not a benchmark claim.

Allowed scope:
- fixed `LIBERO_7D` adapter path only;
- deterministic adapter export/reload if the baseline did not persist weights;
- offline replay-demo action sanity;
- one-task, one-demo exact-init replay only when the bounded replay gate is set and the simulator stack imports.

Disallowed scope: TG-7D, TCA, PRISM, SafeLoRA, PatchGuard, OpenVLA-OFT, downloads, full benchmark, or new method invention.
