# Epoch 2 Cycle 3 OCFN-VLA Adjudication

Date: 2026-07-12 KST

Decision: `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`

This is a valid current-formulation kill for `OCFN-VLA`, not a terminal campaign decision.

## Method

`OCFN-VLA` supplied a task-conditioned initial noise tensor to the frozen SmolVLA flow sampler. It did not replace, filter, damp, rank, verify, or residual-correct emitted actions.

Proposal hash: `F60B9B7BB2640A073AC16EAB6284A68D41569A6A4D67A54462DEF81F06F3F8EA`

## Evidence

Train acquisition completed `16 / 16` official SmolVLA-LIBERO episodes with zero exceptions and decision `TRAIN_ACQUISITION_PASS`.

Stage A completed `50 / 50` held-out episodes with zero exceptions and decision `STAGE_A_NON_GO_TO_STAGE_B_REQUIRED`.

Stage B first block completed `200 / 200` episodes with zero exceptions and decision `STAGE_B_UNRESOLVED_EXPAND_TO_80_REQUIRED`.

The one allowed Stage B expansion completed `400 / 400` episodes with zero exceptions and decision `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`.

Expanded Stage B success:

| Variant | Success | Task-balanced |
| --- | ---: | ---: |
| `frozen_smolvla` | `23 / 80` | `0.2875` |
| `zero_noise_smolvla` | `27 / 80` | `0.3375` |
| `global_success_noise_prior` | `23 / 80` | `0.2875` |
| `task_shuffled_noise_prior` | `25 / 80` | `0.3125` |
| `ocfn_full` | `26 / 80` | `0.3250` |

Mechanism activation was nonzero:

- mean initial action delta, full versus global prior: `0.020219`
- mean initial action delta, full versus shuffled prior: `0.032354`

Paired full-minus-baseline evidence:

| Baseline | Full wins | Full losses | Ties | Delta | Bootstrap CI |
| --- | ---: | ---: | ---: | ---: | --- |
| `frozen_smolvla` | `8` | `5` | `67` | `0.0375` | `[-0.05, 0.125]` |
| `zero_noise_smolvla` | `5` | `6` | `69` | `-0.0125` | `[-0.10, 0.0625]` |
| `global_success_noise_prior` | `5` | `2` | `73` | `0.0375` | `[-0.025, 0.10]` |
| `task_shuffled_noise_prior` | `7` | `6` | `67` | `0.0125` | `[-0.075, 0.10]` |

## Ruling

The implementation was valid, the mechanism acted, Stage B completed at the maximum allowed expansion, and the strongest simple baseline was `zero_noise_smolvla` at `27 / 80` versus `ocfn_full` at `26 / 80`.

The paired bootstrap upper confidence bound for `ocfn_full - zero_noise_smolvla` was `0.0625`, which excludes the preregistered useful `+0.10` prototype improvement. Under `reports/current_research_governance.md`, this is a permanent scientific kill for the current OCFN formulation.

Do not rescue OCFN by changing noise count, task selection, train identities, reset identities, tie-breaks, or selection rules.

Next action: synthesize Epoch 2 related failures and pivot to Epoch 3 with at least two changed core dimensions.
