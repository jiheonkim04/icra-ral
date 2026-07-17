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

## Preregistered Residual Diagnostic: `epoch5_libero10_residual_v1`

Execution status: `COMPLETE`

The first residual diagnostic is a bounded LIBERO-10 long-horizon expansion
using official LIBERO tasks that were not in the saturated hard-slice result.

Rationale:

- OpenVLA-OFT reports very strong but non-perfect LIBERO performance, so a
  residual, if locally accessible, is most likely to appear on long-horizon
  LIBERO-10 tasks rather than already-saturated spatial/control tasks.
- Task IDs `8` and `9` are official LIBERO-10 tasks, share the selected
  OpenVLA-OFT checkpoint's action/observation semantics, and were not evaluated
  in the recovered hard-slice condition.
- Reset labels `20260716..20260723` map to official initial-state indices
  `5..12`, disjoint from the recovered hard-slice indices `0..4`.

Frozen manifest:

| Field | Value |
|---|---|
| Label | `epoch5_libero10_residual_v1` |
| Tasks | `libero_10/task_8` "put both moka pots on the stove"; `libero_10/task_9` "put the yellow and white mug in the microwave and close it" |
| Reset identities | `20260716,20260717,20260718,20260719,20260720,20260721,20260722,20260723` |
| Episodes per policy | `16` |
| Policies | SmolVLA frozen-base exact-init; Quantized OpenVLA-OFT INT4 |
| Matched row SHA-256 | `13642c7bed5e7d5944f7377e9848aeec1b9090be96d110362b53bc9cd9a3b3b2` |
| OpenVLA manifest | `runs/openvla_oft_int4/epoch5_libero10_residual_openvla_manifest.json` |
| SmolVLA manifest | `runs/openvla_oft_int4/epoch5_libero10_residual_smolvla_manifest.json` |

Execution boundaries:

- no new download;
- no training or fine-tuning;
- no full-BF16 OpenVLA-OFT load;
- no Ours method, method candidate, or local proxy;
- stop after two identical infrastructure failures, matching the existing
  runner safety behavior.

Decision rules:

- `RESIDUAL_FOUND_PRIOR_POSITIVE`: Base has meaningful failure, OpenVLA-OFT
  improves over Base, and OpenVLA-OFT leaves at least one measured failure.
- `PRIOR_SATURATED_NEXT_CONDITION`: OpenVLA-OFT succeeds on all 16 episodes;
  do not design Ours for this condition.
- `PRIOR_NOT_POSITIVE_ON_CONDITION`: OpenVLA-OFT does not improve over Base;
  do not design Ours from this condition.
- `INFRASTRUCTURE_BLOCKED`: simulator/model execution fails under the safety
  rules; repair only the runner or move to the next selected prior ecosystem.

If residual is found, run the smallest available upper-bound/headroom check
before any Ours design. If no residual remains, preregister the next
claim-specific condition or fall back to pi0.5/OpenPI.

## Completed Residual Outcome

The frozen `epoch5_libero10_residual_v1` condition completed after
preregistration:

| Policy | Successes | Episodes | Infrastructure failures |
|---|---:|---:|---:|
| SmolVLA frozen-base exact-init | 7 | 16 | 0 |
| Quantized OpenVLA-OFT INT4 | 14 | 16 | 0 |

OpenVLA-OFT INT4 improves over Base and leaves a residual on
`libero_10/task_8`:

- task 8: Base 3/8, OpenVLA-OFT 6/8;
- task 9: Base 4/8, OpenVLA-OFT 8/8.

The two OpenVLA-OFT residual failures are task 8 reset identities `20260721`
and `20260722`, corresponding to official initial-state indices `10` and `11`.

## Completed Upper/Headroom Diagnostic

The smallest available headroom check was a task-level HDF5 expert exact-init
teacher replay for task 8:

- artifact:
  `runs/openvla_oft_int4/epoch5_libero10_residual_expert_headroom_task8_demo10.json`;
- task: `libero_10/task_8`;
- demo: `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5::demo_10`;
- result: success, reward `1.0`, done/success at step `377`;
- exact demo init-state proof: `after_set_state_l2_to_hdf5_init = 0.0`;
- training/download/VLA load: none.

Caveat: this is not a same-reset upper bound. Local HDF5 demo init-state hashes
did not match the frozen benchmark initial-state hashes for residual reset
identities `20260721`/`20260722`. The result is therefore task-level
recoverability evidence, sufficient to avoid `CONDITION_TOO_SEVERE`, but the
identity mismatch must be carried into Ours design and claims.

## Current Gate Result

Decision: `RESIDUAL_FOUND_PRIOR_POSITIVE_TASK_LEVEL_HEADROOM_POSITIVE`.

Ours design is now permitted only for the exact task-8 residual limitation and
must generate at most two candidates. Do not broaden into a generic method
search.
