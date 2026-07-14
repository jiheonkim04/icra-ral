# Epoch 4 Cycle 2 CAVM-VLA Adjudication

Date: 2026-07-14 KST

Decision: `STAGE_2B_EXPANDED_NON_GO_NO_THIRD_EXPANSION`

This is not a terminal campaign decision. Continue to Epoch 4 Cycle 3.

## Fixed Protocol

CAVM-VLA was evaluated exactly under the frozen preregistered protocol and the one preregistered Stage 2B expansion. The expansion did not change the method, memory construction, tasks, reset identities, variants, hyperparameters, thresholds, or adjudication rules.

Key artifacts:

- proposal: `reports/cavm_vla/researcher_proposal.md`
- proposal hash: `849A98B2F137FC43EAA68C7B7D7DB246FEF58DD2EDBBD1F8869C4BA092DE68F2`
- expansion preregistration: `reports/cavm_vla/stage_2b_expansion_preregistration.md`
- Stage 2B result: `reports/cavm_vla/stage_2b_result.json`
- expanded result: `reports/cavm_vla/stage_2b_expansion_result.json`
- expanded report: `reports/cavm_vla/stage_2b_expansion_result.md`

## Manifest Validation

- planned rows: `290`
- completed rows: `290`
- unique `(variant, task_key, identity)` rows: `290`
- duplicate `(variant, task_key, identity)` rows: `0`
- paired `(task_key, identity)` cases: `58`
- bad paired cases: `0`
- rows per variant: `58`
- original Stage 2B keys preserved: `200 / 200`
- preregistered expansion keys added: `90`
- exceptions: `0`

The combined manifest contains identities `20260922` through `20260950` across the two preregistered task keys, with all five variants present for each paired case.

## Expanded Result

| Variant | Successes | Total | Task-Balanced Success |
| --- | ---: | ---: | ---: |
| `frozen_smolvla` | 22 | 58 | 0.379310 |
| `success_only_memory_proxy` | 20 | 58 | 0.344828 |
| `nearest_success_replay` | 23 | 58 | 0.396552 |
| `cavm_no_contrast_ablation` | 21 | 58 | 0.362069 |
| `cavm_full` | 24 | 58 | 0.413793 |

Strongest baseline: `nearest_success_replay`.

Paired comparisons against `cavm_full`:

| Comparator | Delta | Wins | Losses | Ties | Bootstrap CI |
| --- | ---: | ---: | ---: | ---: | --- |
| `frozen_smolvla` | 0.034483 | 7 | 5 | 46 | `[-0.086207, 0.155172]` |
| `success_only_memory_proxy` | 0.068966 | 5 | 1 | 52 | `[0.000000, 0.155172]` |
| `nearest_success_replay` | 0.017241 | 4 | 3 | 51 | `[-0.068966, 0.103448]` |
| `cavm_no_contrast_ablation` | 0.051724 | 5 | 2 | 51 | `[-0.034483, 0.137931]` |

Mechanism evidence:

- `cavm_full` mean gate: `0.004846`
- `cavm_full` mean gate activation rate: `0.633522`
- `cavm_full` mean action delta L2: `0.006137`
- `cavm_full` mean heavy policy calls per step: `0.021712`
- peak CUDA allocated: `926.645 MB`

## Scientific Ruling

CAVM full numerically beats the frozen backbone, success-only memory proxy, nearest-success replay, and no-contrast ablation. The mechanism is active and the implementation is valid.

However, after the maximum preregistered expansion, the improvement remains too small for a prototype GO:

- full versus strongest baseline is only `+1 / 58` success;
- paired delta versus strongest baseline is `0.017241`;
- the paired CI versus strongest baseline is `[-0.068966, 0.103448]`;
- the effect does not satisfy the preregistered useful-improvement criterion.

Therefore the current CAVM formulation is archived as a valid non-GO result. No third expansion is allowed. Do not rescue CAVM by threshold retuning, memory reconstruction changes, new CAVM hyperparameter variants, altered tasks, altered identities, or post-hoc reinterpretation of partial results.

Next action: begin Epoch 4 Cycle 3 under the post-CAVM performance-oriented research-design governance.
