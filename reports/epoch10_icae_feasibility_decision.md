# Epoch 10 ICAE-VLA feasibility decision

Date: 2026-07-21 KST

Decision: `PROCEED_TO_PROSPECTIVE_CHECKPOINT_GENERATION_AND_ASSAY_BUILD`

The exact novelty collision gate is clear. The local assets include the released SmolVLA-LIBERO base, three verified standard rank-4 LoRA adapters, the 1,693-episode/273,465-frame/40-task LeRobot LIBERO dataset, all four official LIBERO suites, exact-state restoration machinery, and a serial official route that completed 400/400 episodes with zero infrastructure failures.

The current paper-eligible checkpoint inventory is insufficient by itself: it has four genuine identities, and all four have known development outcomes. This is a locally resolvable panel-construction gap, not a terminal blocker. Epoch 10 will train four matched standard rank-4 LoRA seeds and persist predeclared early, intermediate, and converged snapshots. Whole seeds, not adjacent snapshots, will define development versus holdout checkpoint groups. Historical identities remain development-only anchors and will not populate the prospective official panel.

The risk assessment is inside budget: about 307 GB is free on the system drive; the base checkpoint is about 907 MB; each existing rank-4 adapter is under 1 MB plus optimizer metadata; the verified training recipe uses one resident SmolVLA and batch size 1; the RTX 5080 has about 13.7 GB free VRAM; host RAM is below the 80% warning threshold; and WSL swap is unused. Execution will remain serial with one VLA and one simulator environment at a time.

No paper claim, prospective utility claim, or assay-validity claim is made at this stage.
