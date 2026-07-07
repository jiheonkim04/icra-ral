# TL-ChunkRepair Reusable Artifacts

Keep these pieces as diagnostic infrastructure:
- temporal perturbation runner for early release, delayed close, lift-before-grasp, open-gripper transport, premature release, truncation, phase skip, and unsafe-contact insertion,
- exact-init LIBERO/RoboSuite replay diagnostic wrapper: `scripts\181_tl_chunkrepair_state1_diagnostic.ps1`,
- diagnostic module: `tca_map.tl_chunkrepair.state1_diagnostic`,
- temporal property monitor for grasp/lift/release/transport/contact/onset properties,
- violation metrics and safe-success aggregation,
- baseline suite covering no-repair, clipping-only, safety-only one-step filter, gripper-only timing fix, fixed delay shift, linear time warp, abort-to-stop, repeat-last/hold, and TL repair,
- compact evidence report: `reports\tl_chunkrepair_state1_result.md` and `.json`,
- focused unit tests: `tests\test_tl_chunkrepair_state1.py`.

Do not reuse TL-ChunkRepair as a main RA-L method claim unless a future, separately predeclared route beats both the best single simple baseline and the best per-failure-mode simple baseline on real replay/control utility. Symbolic violation reduction alone is explicitly insufficient.
