# OCFN-VLA Preregistration

Date: 2026-07-12 KST

Proposal hash: `F60B9B7BB2640A073AC16EAB6284A68D41569A6A4D67A54462DEF81F06F3F8EA`

Decision before implementation: `IMPLEMENT_STAGE_A_PROTOTYPE`

## Fixed Variants

1. `frozen_smolvla`
2. `zero_noise_smolvla`
3. `global_success_noise_prior`
4. `task_shuffled_noise_prior`
5. `ocfn_full`

## Fixed Train Acquisition

Tasks:

- `libero_spatial/task_4`
- `libero_10/task_4`

Train reset identities:

- `20260711`
- `20260712`

Noise identities:

- `0`
- `1`
- `2`
- `3`

Total train-label episodes:

- `4 noise identities * 2 tasks * 2 identities = 16 episodes`

## Fixed Held-Out Stage A

Tasks:

- `libero_spatial/task_4`
- `libero_10/task_4`

Held-out reset identities:

- `20260713`
- `20260714`
- `20260715`
- `20260716`
- `20260717`

Total held-out Stage A episodes:

- `5 variants * 2 tasks * 5 identities = 50 episodes`

## Fixed Stage B If Required

Stage B is required for tied, noisy, small-negative, or one/two episode Stage A differences that do not satisfy a permanent-kill rule.

Stage B uses the same five variants, the same two tasks, and the frozen train-acquisition selection table. It does not refit, retune, reselect, or inspect simulator state at inference.

Stage B reset identities:

- first block: `20260718` through `20260737`
- one allowed expansion if unresolved: extend the same manifest prefix through `20260757`

Identity mapping rule:

- identity `20260711 + k` maps to official LIBERO initial-state index `k`
- both fixed Stage B tasks expose 50 official initial states, so the expansion stays inside the available state range

First Stage B total episodes:

- `5 variants * 2 tasks * 20 identities = 200 episodes`
- this equals `40` paired task-reset episodes per key policy

Expanded Stage B total episodes:

- `5 variants * 2 tasks * 40 identities = 400 episodes`
- this equals `80` paired task-reset episodes per key policy

Stage B reports:

- successes/counts and task-balanced success by variant;
- per-task results;
- `ocfn_full` paired wins/losses/ties versus each key baseline;
- deterministic paired bootstrap confidence intervals for the full-minus-baseline paired success delta;
- McNemar-style exact discordant-pair p-values;
- effect sizes, failure-rate reduction, mechanism activation, latency, and VRAM.

## Fixed Selection Rules

For each task and each noise identity, compute:

- train success count;
- train total count;
- mean episode steps among train episodes;
- mean reward sum.

`ocfn_full` selects the best noise identity separately per task:

1. higher train success count;
2. lower mean episode steps;
3. higher mean reward sum;
4. lower noise identity index.

`global_success_noise_prior` selects one noise identity after pooling across both tasks with the same tie-break order.

`task_shuffled_noise_prior` uses the same train table after deterministically swapping the two task labels before applying the `ocfn_full` selection rule.

`zero_noise_smolvla` uses an all-zero noise tensor.

`frozen_smolvla` calls the official policy without passing a custom noise tensor.

## Fixed Hyperparameters

- noise identities: `4`
- noise seed base: `2026071203`
- task-shuffle seed: `2026071204`
- max train identities: `2`
- max held-out identities: `5`
- no video by default unless the official runner already records it cheaply
- max eval steps: official task max (`0` override)

No hyperparameter may be changed after Stage A result inspection.

## Metrics

Primary:

- task-balanced held-out closed-loop success rate.

Secondary:

- successes/counts per variant;
- per-task success rate;
- train acquisition success/count by task/noise;
- selected noise identity per task and variant;
- full-vs-global selected-noise equality;
- full-vs-shuffled selected-noise equality;
- mean initial same-observation action delta between `ocfn_full` and each fixed-noise baseline;
- latency and CUDA memory;
- exception count.

## GO / KILL

Use `reports/current_research_governance.md`.

Stage A permanent kill if:

- implementation or data mechanism invalid;
- `ocfn_full` is at least 30 absolute task-balanced points below the strongest baseline or key ablation;
- `ocfn_full` has `0 / 10` while a paired baseline has at least `4 / 10`;
- exact trivial equivalence to `zero_noise_smolvla`, `global_success_noise_prior`, or `task_shuffled_noise_prior` is demonstrated.

Advance to Stage B if:

- `ocfn_full` beats frozen and `task_shuffled_noise_prior`; or
- result is noisy/tied/small-negative but mechanism activation is valid and no permanent-kill condition holds.

Stage B prototype GO requires:

- `ocfn_full` beats the strongest Stage B baseline and `task_shuffled_noise_prior`;
- absolute task-balanced gain is at least `10` points, or paired evidence is consistently positive with meaningful failure-rate reduction;
- mechanism activation remains nonzero;
- no privileged inference signal is used.

Stage B permanent kill or non-GO archive:

- permanent kill if the implementation is valid, mechanism activation is nonzero, Stage B is complete, and the paired upper confidence bound excludes a useful `10` point improvement or the shuffled/direct baseline explains the method;
- if the first 40 paired episodes per key policy are unresolved, run the one fixed expansion to 80 paired episodes per key policy;
- if the 80-pair expansion is still unresolved and no GO rule holds, archive this OCFN formulation as non-GO and pivot under `reports/current_research_governance.md`.

## One Allowed Measurement Repair

If synthetic or train acquisition shows the runner is ignoring the provided noise tensor, one repair is allowed before held-out Stage A:

- document the failed noise-effect check;
- fix the noise plumbing once;
- rerun the synthetic check and train acquisition;
- do not inspect held-out Stage A before the repair.
