# DICD-VLA Preregistration

Date: 2026-07-12 KST
Branch: `codex/auto-method-20260712-01-dicd-vla`

## Hypothesis

Under controlled action delay `d=2`, a learned delay-indexed adapter over frozen SmolVLA action chunks and executed-action history improves task-balanced closed-loop success over frozen delayed execution and over direct chunk-index execution, while preserving clean `d=0` behavior.

## Method

Train a small adapter `g_theta` over postprocessed SmolVLA action chunks, recent executed actions, delay index, and timing features.

The adapter predicts one executable 7D relative action.

## Data Source

Training trace:

- official SmolVLA-LIBERO;
- tasks: `libero_spatial/task_4`, `libero_10/task_4`;
- training identity: `20260711`;
- generated locally from frozen-policy chunks and executed/future actions;
- no evaluation reset identity in training.

Validation/smoke trace:

- identity: `20260712`;
- used only for mechanism smoke and checkpoint identity, not for GO/KILL claims.

Stage A evaluation:

- identities: `20260713`, `20260714`, `20260715`, `20260716`, `20260717`;
- tasks: `libero_spatial/task_4`, `libero_10/task_4`;
- variants: `frozen_smolvla_clean`, `frozen_smolvla_delay`, `direct_chunk_index_delay`, `dicd_no_history_ablation`, `dicd_full`;
- planned episodes: `10` per variant, `50` total.

## Comparisons

Required first-prototype comparisons:

1. unmodified backbone: `frozen_smolvla_clean`
2. delayed backbone: `frozen_smolvla_delay`
3. closest direct local proxy: `direct_chunk_index_delay`
4. simple reviewer-killer baseline: `direct_chunk_index_delay`
5. key ablation: `dicd_no_history_ablation`
6. full method: `dicd_full`

## Metrics

Primary metric:

- task-balanced official closed-loop task success.

Secondary metrics:

- paired win/loss/tie versus `direct_chunk_index_delay`;
- mean action delta versus delayed frozen execution;
- mean action delta versus no-history ablation;
- shaped/delayed action frequency;
- clean retention success and action delta;
- peak VRAM and latency.

## Compute Budget

Mechanism smoke must complete before Stage A.

Stage A maximum:

- `50` closed-loop episodes;
- no more than `2` hours wall time expected;
- no new downloads.

## GO Criteria

Prototype GO if all are true:

- mechanism smoke passes;
- full DICD improves task-balanced delayed success by at least `5` absolute percentage points over the strongest delayed baseline;
- full DICD beats `direct_chunk_index_delay`;
- full DICD beats `dicd_no_history_ablation`;
- full changes actions relative to frozen delayed execution and direct chunk-index execution;
- no privileged inference signal;
- clean `d=0` retention shows no obvious degradation in the small retention check.

## KILL Criteria

Classify after valid Stage A as:

- `SIMPLE_BASELINE_EXPLAINS_METHOD` if direct chunk indexing matches or beats full;
- `KEY_COMPONENT_NOT_USEFUL` if no-history ablation matches or beats full;
- `GENUINE_METHOD_KILL` if full is active, valid, and loses to delayed frozen execution or strongest baseline;
- `UNDERPOWERED_ONE_EXPANSION_ALLOWED` if full is positive but below confidence and not beaten by baseline/ablation;
- `IMPLEMENTATION_OR_DATA_FAILURE` if mechanism smoke fails or the adapter does not act as intended;
- `EXACT_PRIOR_ART_DUPLICATE` only if implementation collapses to DEFLECT/TIC-VLA equivalence.

## One Permitted Repair

Exactly one narrow repair is allowed only for a concrete defect:

- wrong chunk postprocessing;
- wrong delay index;
- detached gradients;
- wrong checkpoint reload;
- train/eval feature mismatch;
- silent broadcasting/indexing bug.

The repair may not introduce a new architecture, new loss family, new supervision concept, new tasks, or multiple hyperparameter variants.
