# Epoch 5 Prior Reproduction Result

Selected prior ecosystem: OpenVLA-OFT on LIBERO.

## Result

Decision: `OPENVLA_OFT_PRIOR_REPRODUCTION_RECOVERED_AND_VALIDATED_RESIDUAL_PENDING`

This epoch validated the existing local OpenVLA-OFT INT4 prior execution artifacts instead of launching a new rollout. No new training, simulator rollout, model download, or Ours design occurred.

Focused validation command:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest tests\test_openvla_oft_int4_gate.py -q
```

Observed result: `4 passed in 0.10s`.

## Evidence

| Evidence | Value |
|---|---|
| Prior result | `reports/openvla_oft_quantized_hard_slice_result.json` |
| Prior result summary | OpenVLA-OFT INT4 completed 20/20 and succeeded 20/20 |
| Matched Base result | `runs/openvla_oft_int4/hard_slice_smolvla_exact.json` |
| Matched Base summary | SmolVLA frozen-base exact-init completed 20/20 and succeeded 11/20 |
| Matched manifest | `reports/openvla_oft_quantized_hard_slice_manifest.json` |
| Policy-load evidence | `reports/openvla_oft_int4_policy_load_result.md` |
| Memory preflight | `reports/openvla_oft_int4_memory_preflight.md` |
| Quantization caveat | INT4 is not claimed numerically identical to full-precision OpenVLA-OFT |
| Local checkpoint | `/home/jiheon/assets/checkpoints/openvla-oft/moojink_openvla-7b-oft-finetuned-libero-spatial-object-goal-10` |
| Local checkpoint size | 15G by WSL `du -sh`; result metadata visible size 14.845 GiB |
| Local official repo | `C:\assets\repos\openvla-oft`, HEAD `e4287e94541f459edc4feabc4e181f537cd569a8`, dirty from prior local compatibility changes |

## Matched Base/Prior Interpretation

The selected prior produces a reproducible positive effect on the recovered hard-slice condition:

- OpenVLA-OFT INT4: 20/20.
- SmolVLA frozen-base exact-init: 11/20.
- Peak OpenVLA hard-slice CUDA allocation in the existing artifact: 5539.458 MiB.
- No CPU/disk offload detected in the existing artifact.

However, this condition is not usable for Ours design because the prior leaves no measured residual failure on the 20-episode hard slice. The "Base fails -> Prior improves -> residual remains" structure is only partially satisfied:

| Required structure | Status |
|---|---|
| Base has meaningful failure | COMPLETE: SmolVLA 11/20 |
| Prior improves | COMPLETE: OpenVLA-OFT INT4 20/20 |
| Prior leaves residual gap | MISSING on this condition |
| Condition neither floor nor saturated | MISSING for prior; saturated at 20/20 |
| OpenVLA-OFT does not fully solve it | MISSING |
| Upper bound indicates recoverable headroom | PARTIAL: expert/exact-init infrastructure exists, but residual absent |

## Next Decision

Do not design Ours yet. The next step is a bounded residual-gap diagnostic for the selected prior. If OpenVLA-OFT remains saturated on all locally feasible conditions, move to the second-ranked ecosystem, pi0.5/OpenPI, instead of creating a proxy-only method.
