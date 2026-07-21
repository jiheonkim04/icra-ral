# Epoch 10 prospective checkpoint-generation log

Date: 2026-07-21 KST

Status: `FROZEN_PENDING_TRAINING`

The prospective panel uses the verified standard SmolVLA-LIBERO rank-4 LoRA recipe. Four new seeds (`101`, `202`, `303`, `404`) are trained once to 100 optimizer steps with snapshots persisted after steps `10`, `30`, and `100`. These stages were chosen before any new checkpoint action, intervention, or closed-loop outcome was observed. They represent early, intermediate, and converged training progress rather than outcome-selected checkpoints.

Seeds `101` and `202` are development lineages. Seeds `303` and `404` are the completely held-out official lineages. No adjacent snapshots cross this whole-seed boundary. The frozen base is a named reference but is not counted as a separate training run. Historical seeds `11/22/33` and their 400 known official outcomes are loader/direction anchors only.

The predeclared competitive subset contains steps `30` and `100` from every new seed. Step `10` remains in the full panel to ensure training-progress range, but cannot alone carry the competitive-subset paper result.

The runner saves each bundle atomically, records source and asset hashes, persists optimizer/RNG state, reloads every bundle from disk, and queries one common training-state sample solely to verify a finite 7-D deployment action. It performs no simulator rollout and reads no success, reward, or held-out official outcome during generation.

Planned command:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_epoch10_checkpoint_panel.py
```
