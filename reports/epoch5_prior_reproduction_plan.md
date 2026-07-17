# Epoch 5 Prior Reproduction Plan

Selected ecosystem: OpenVLA-OFT on LIBERO.

## Boundary

This plan reproduces or validates the selected external prior before any Ours design. It does not create a new method candidate, does not rescue MCI-VLA, does not run training, and does not claim a full-precision paper reproduction from an INT4 run.

## Preferred Evidence Order

1. Official execution: run official OpenVLA-OFT code and checkpoint under its intended environment.
2. Mechanism-faithful local port: allowed only if official code is inspected and the computational graph, inputs, action semantics, and inference mechanism are preserved.
3. Existing validated local execution: acceptable as recovered evidence for the branch transition only if the artifact was produced by the official stack, is result-file backed, and passes focused validation now.

## Local Artifact State

| Item | Status |
|---|---|
| Official code checkout | present at `C:\assets\repos\openvla-oft`; checkout is dirty from prior local compatibility changes and must not be cleaned destructively |
| Official checkpoint | present in WSL at `/home/jiheon/assets/checkpoints/openvla-oft/moojink_openvla-7b-oft-finetuned-libero-spatial-object-goal-10` |
| Checkpoint visible size | about 15G local disk; state records 14.845 GiB visible size |
| Prior local execution | `reports/openvla_oft_quantized_hard_slice_result.json` and `runs/openvla_oft_int4/hard_slice_openvla_int4.json` |
| Matched Base artifact | `runs/openvla_oft_int4/hard_slice_smolvla_exact.json` |
| Validation test | `tests/test_openvla_oft_int4_gate.py` |

## Risk Assessment

| Risk | Decision |
|---|---|
| Download | no new download in this step |
| GPU | no new rollout or model load in this step; focused test only reads artifacts |
| Simulator | no new simulator rollout in this step |
| Full BF16 OpenVLA-OFT | not attempted; prior local preflight forbids full BF16 on this 16GB GPU |
| Quantization | INT4 result is valid as a quantized local prior diagnostic, not as numerical full-precision reproduction |
| External checkout dirtiness | record it; do not reset or clean external repo |

## Reproduction Validation Command

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest tests\test_openvla_oft_int4_gate.py -q
```

Expected result: all focused OpenVLA-OFT gate tests pass.

## Matched Base/Prior Diagnostic Already Available

Existing matched hard-slice diagnostic:

- OpenVLA-OFT INT4: 20/20 successful episodes.
- SmolVLA frozen-base exact-init: 11/20 successful episodes.
- Hard-slice failures in SmolVLA were not reproduced by OpenVLA-OFT INT4.

This satisfies prior-positive reproduction for the selected hard-slice condition but does not yet satisfy residual-gap discovery because OpenVLA-OFT saturated that condition.

## Residual-Gap Next Step

Do not design Ours yet. The next scientific step is to preregister and run a small residual-condition diagnostic for OpenVLA-OFT. Candidate residual conditions must come from the selected prior's known limits or benchmark stressors, for example:

- LIBERO-PRO-style object/instruction/environment perturbations if assets are locally accessible without risky downloads;
- official LIBERO tasks where OpenVLA-OFT is not saturated under a matched small manifest;
- language-grounding or visual-feedback cases motivated by the OpenVLA-OFT paper's own qualitative discussion.

If no residual remains, move to pi0.5/OpenPI as the second-ranked ecosystem rather than inventing a proxy-only local method.
