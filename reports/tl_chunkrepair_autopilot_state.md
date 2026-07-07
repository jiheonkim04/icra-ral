# TL-ChunkRepair Autopilot State

Current state: STATE 1 completed with kill decision.

Branch: `codex/tl-chunkrepair-state0-state1`.

Starting main commit: `1409dd1f737f9e70c4121dc01a1378ce16942a3b`.

Fresh-route rule: do not continue previous killed routes. TL-ChunkRepair inherits reusable exact-init replay infrastructure only.

STATE 0 completed:
- task definition,
- experiment plan,
- kill criteria,
- related-work matrix,
- autopilot state.

STATE 1 output:
- gated TL repair diagnostic script: `scripts\181_tl_chunkrepair_state1_diagnostic.ps1`,
- focused tests: `tests\test_tl_chunkrepair_state1.py`,
- real exact-init replay/control metrics: yes,
- report: `reports\tl_chunkrepair_state1_result.md` and `.json`,
- decision: kill.

Key result:
- exact-init expert replay succeeded,
- total simulator steps: `19803`,
- variants: `73`,
- perturbations tested: `8`,
- perturbations that degraded replay: `7 / 8`,
- TL reduced temporal property violations: `8 / 8`,
- TL safe-success count: `0 / 8`,
- TL success/reward: `0 / 0.0`,
- best single simple baseline: `no_repair`, with success/reward `1 / 1.0`,
- TL beat best simple per perturbation: `0 / 7` degraded perturbations.

Conclusion: archive or reframe TL-ChunkRepair. It improved symbolic temporal satisfaction but not replay success, reward, done index, safe-success, or progress enough to beat simple baselines.

Forbidden actions:
- downloads,
- GPU,
- training,
- OpenVLA-OFT,
- full VLA fine-tuning,
- unsupported paper-grade claim.
