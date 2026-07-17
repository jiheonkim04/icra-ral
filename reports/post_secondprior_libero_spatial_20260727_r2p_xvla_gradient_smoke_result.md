# R2P-XVLA No-Optimizer Gradient Smoke

Decision: `R2P_XVLA_GRADIENT_SMOKE_PASS`

This was a one-batch pre-optimizer gradient gate for `R2P-XVLA`. It loaded the cached X-VLA prior from the local WSL snapshot, attached PEFT LoRA, ran one forward pass and one backward pass, and verified finite nonzero adapter gradients. It did not create an optimizer, call `optimizer.step`, write a checkpoint, run a simulator, download files, or evaluate Ours closed-loop.

## Runtime

- Result: `runs/xvla_prior/r2p_xvla_gradient_smoke_offline_20260718T0425KST/result.json`
- Result SHA-256: `c170d52cbbc01974ff51c8b3ad6e8d68136abc8cf90d9a3eb6580d72302b1f76`
- Exit code: `0`
- Elapsed: `11.967` seconds
- Offline flags: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`
- Load source: local X-VLA snapshot under `/home/jiheon/assets/checkpoints/xvla_hf_cache/...`

## Materialized clip

- Demo: `demo_0`
- Source indices: `41..137`
- Clip steps: `96`
- `abs_action_6d` shape: `[96, 10]`
- Phase counts source/transit/target: `26 / 23 / 47`
- Mean phase weight: `1.484375`

## Batch and gradient

| Item | Value |
| --- | --- |
| Batch action shape | `[1, 30, 20]` |
| Batch proprio shape | `[1, 20]` |
| Batch image shape | `[1, 3, 3, 224, 224]` |
| Weighted loss | `12.958698272705078` |
| Trainable parameters | `11868760` |
| Grad tensors finite / total | `537 / 537` |
| Nonzero grad tensors | `271` |
| Gradient global norm | `2372.1450494696983` |
| Max CUDA allocated MiB | `5260.354` |

## No-training statement

- Optimizer created: `false`
- Optimizer step happened: `false`
- Checkpoint written: `false`
- Training run happened: `false`
- Closed-loop Ours evaluation happened: `false`
- Simulator rollout happened: `false`
- Downloads performed: `false`

## Next action

Freeze the first optimizer-step authorization gate for `R2P-XVLA`. Do not start training until that gate records the exact two arms, output directories, heartbeat/status requirements, and stop conditions.
