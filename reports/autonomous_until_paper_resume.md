# Autonomous Until Paper Resume

Date: 2026-07-12 KST

Resume command:

```powershell
cd /d C:\Users\jiheo\tca_map
git switch codex/autonomous-until-ral-evidence-ready
type reports\autonomous_until_paper_state.json
```

Current stage: `cycle_1_real_smolvla_chunk_smoke_pending`

Next automatic stage:

1. Run real SmolVLA action-chunk smoke using `predict_action_chunk`.
2. Verify postprocessed chunk horizon, delay index, and no-privileged inference on an actual LIBERO observation.
3. If real chunk smoke passes, generate frozen training traces for identity `20260711`.
4. Train full and no-history adapters from real traces.
5. Run Stage A closed-loop evaluation on identities `20260713` through `20260717`.

No long-running command is active.

Checkpoint from synthetic smoke: `reports/dicd_vla/checkpoints/dicd_synthetic_smoke.pt`
