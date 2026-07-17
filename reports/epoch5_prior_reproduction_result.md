# Epoch 5 Prior Reproduction Result

Selected prior ecosystem: OpenVLA-OFT on LIBERO.

## Result

Decision: `RESIDUAL_FOUND_PRIOR_POSITIVE_TASK_LEVEL_HEADROOM_POSITIVE`

Epoch 5 has now completed the selected-prior-first diagnostic sequence without
designing Ours, training, downloading assets, or attempting full-BF16
OpenVLA-OFT.

The recovered hard-slice condition established that the selected prior is
locally runnable and positive, but saturated. The preregistered
`epoch5_libero10_residual_v1` condition then produced the required matched
Base/Prior residual structure:

- SmolVLA frozen-base exact-init: 7/16.
- Quantized OpenVLA-OFT INT4: 14/16.
- OpenVLA-OFT still fails 2/16, both on `libero_10/task_8`.
- No infrastructure failures occurred in either matched run.

The smallest available upper/headroom check was then run as a task-level HDF5
expert replay on `libero_10/task_8`. It succeeded, but it is not a same-reset
upper bound because local HDF5 demo init-state hashes do not match the frozen
benchmark initial-state hashes used in the Base/Prior diagnostic.

## Validation Commands

Focused OpenVLA artifact validation:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe -m pytest tests\test_openvla_oft_int4_gate.py -q
```

Observed after manifest-control patches: `5 passed`.

## Recovered Hard-Slice Evidence

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

The recovered hard slice is prior-positive but unusable for Ours design because
OpenVLA-OFT INT4 saturates it at 20/20.

## Matched Residual Diagnostic: `epoch5_libero10_residual_v1`

Frozen condition:

- tasks: `libero_10/task_8` and `libero_10/task_9`;
- reset identities: `20260716..20260723`, mapping to official initial-state
  indices `5..12`;
- planned episodes: `16` SmolVLA frozen-base exact-init and `16` Quantized
  OpenVLA-OFT INT4;
- matched row SHA-256:
  `13642c7bed5e7d5944f7377e9848aeec1b9090be96d110362b53bc9cd9a3b3b2`.

| Policy | Completed | Successes | Failures | Infrastructure failures | Result |
|---|---:|---:|---:|---:|---|
| SmolVLA frozen-base exact-init | 16 | 7 | 9 | 0 | `runs/openvla_oft_int4/epoch5_libero10_residual_smolvla_exact.json` |
| Quantized OpenVLA-OFT INT4 | 16 | 14 | 2 | 0 | `runs/openvla_oft_int4/epoch5_libero10_residual_openvla_int4.json` |

Per-task result:

| Task | SmolVLA Base | OpenVLA-OFT INT4 | Interpretation |
|---|---:|---:|---|
| `libero_10/task_8` put both moka pots on the stove | 3/8 | 6/8 | prior improves but leaves residual |
| `libero_10/task_9` put the yellow and white mug in the microwave and close it | 4/8 | 8/8 | prior improves and saturates this task |

The two OpenVLA-OFT residual failures are:

| Task | Reset identity | Initial-state index | Initial-state SHA-256 |
|---|---:|---:|---|
| `libero_10/task_8` | `20260721` | 10 | `098c331d6cad1772de3e8ee22a7f983b4c109493f657735e7e7e78319ac1f455` |
| `libero_10/task_8` | `20260722` | 11 | `7753c014bd3caf96ff9694b20b5ea40358f64730fa10607312183377f69fb305` |

Manifest/result integrity:

| Artifact | SHA-256 |
|---|---|
| OpenVLA residual manifest | `b2de1d683d7ab0c5aff7462857f0366bd72c9208c98b2e8566e6a42a296b5adf` |
| SmolVLA residual manifest | `6defb7769a75b595bc8456e6938254d7185d2b03fd94a4bda4fd0a95464a837c` |
| OpenVLA residual result | `29cddfb319df9f3ffa19bd34f8b571e69199118783338423eca25e94ee16f1e9` |
| SmolVLA residual result | `24569154c305ef2dbfe25d71ba2ea8d9c5de5b7c1d85851596ba93671a1e38c1` |

The manifests have identical `(suite, task_id, reset_identity,
initial_state_index, initial_state_sha256)` rows. This satisfies the matched
Base/Prior part of the required structure.

## Headroom Diagnostic

Artifact:
`runs/openvla_oft_int4/epoch5_libero10_residual_expert_headroom_task8_demo10.json`.

| Field | Value |
|---|---|
| Diagnostic | task-level HDF5 expert exact-init replay |
| Task | `libero_10/task_8` |
| Demo | `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5::demo_10` |
| Same benchmark reset as residual failure? | No |
| HDF5 demo init-state SHA-256 | `3ebe5ab024c6896e57dee59422f47ef631355a9e20f10a082fae7ad7f533f81a` |
| Nearest residual reset index represented by label | OpenVLA residual failure index 10, but hash differs |
| Expert replay result | success, reward 1.0, done at step 377 |
| Exact demo init-state set proof | `after_set_state_l2_to_hdf5_init = 0.0` |
| Training/download/VLA load | none |

Interpretation: task-level recoverable headroom is positive for the residual
task, but this is weaker than a same-reset oracle. It is sufficient to avoid
classifying the condition as floor/too-severe at this stage, while the exact
identity mismatch must remain visible in any Ours design.

## Required Structure

| Required structure | Status |
|---|---|
| Base has meaningful failure | COMPLETE: SmolVLA 7/16 |
| Prior improves | COMPLETE: OpenVLA-OFT INT4 14/16 |
| Prior leaves residual gap | COMPLETE: OpenVLA-OFT fails 2/16 |
| Condition neither floor nor saturated | COMPLETE for task 8; prior saturated task 9 only |
| OpenVLA-OFT does not fully solve it | COMPLETE on task 8 |
| Upper/headroom indicates recoverability | PARTIAL-COMPLETE: task-level expert replay positive; same-reset expert unavailable |

## Next Decision

Ours design is now allowed only around the exact task-8 residual limitation and
must preserve the caveat that the current upper bound is task-level, not
same-reset. Generate at most two method candidates, select one, and keep LoRA or
QLoRA strictly as implementation infrastructure.
