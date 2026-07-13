# PSE-VLA Researcher Proposal

Date: 2026-07-12 KST

Method: `PSE-VLA`, Photometric Sensor-Ensemble VLA.

## Claim

Frozen VLA policies may produce unstable actions under nuisance photometric variation. `PSE-VLA` uses a fixed bank of photometric observation transforms and averages the frozen policy's first action-chunk prediction across the bank. It does not train, score, rank, route, use success labels, use a teacher, or canonicalize images to a target statistic.

## Method

For transform bank `T = {identity, bright-low-contrast, dark-high-contrast}`:

`a_t = (1 / |T|) sum_{T_k in T} first(pi_S(T_k(o_t), q_t, l))`.

All actions are postprocessed into official 7D LIBERO action space before averaging.

Fixed transform definitions on preprocessed image tensors:

- `identity`: `x`
- `bright_low_contrast`: `clip(0.42 * x + 0.28, 0, 1)`
- `dark_high_contrast`: `clip(1.25 * x - 0.10, 0, 1)`

The method calls a stateless first-action-chunk path when available, using `predict_action_chunk(batch)[:, 0]` before official postprocessing. This avoids corrupting the policy's internal action queue when several transformed observations are evaluated at the same environment step.

## Baselines

1. `clean_frozen_smolvla`
2. `bright_single`
3. `dark_single`
4. `pse_duplicate_clean`
5. `pse_full`

`pse_duplicate_clean` averages three stateless clean predictions and must behave like the clean policy except for numerical noise. It is the implementation and aggregation killer baseline.

## Prototype

Held-out identities:

- `20260741..20260745`

Tasks:

- `libero_spatial/task_4`
- `libero_10/task_4`

Stage A has `50` episodes.

If Stage A is not a valid permanent kill, Stage B expands to the predeclared PSE held-out identity range `20260741..20260760`, for `40` paired episodes per policy and `200` total episodes. The first five identities are the Stage A identities; Stage B is analyzed as an expanded paired set, not as a cherry-picked rerun.

Primary metric:

- task-balanced closed-loop success.

Mechanism metrics:

- mean postprocessed action delta of `pse_full` versus clean;
- mean postprocessed action delta of `pse_full` versus the strongest single transform;
- per-step transform count;
- duplicate-clean action delta versus clean.

## Kill Conditions

Kill in Stage A only under `reports/current_research_governance.md`: invalid implementation/mechanism, full at least `0.30` below the strongest baseline, `0 / 10` full while a paired baseline is at least `4 / 10`, oracle/no-headroom proof, or exact trivial equivalence. A single-transform or duplicate-clean tie in Stage A is diagnostic unless it proves exact equivalence.

Kill in Stage B if the implementation is valid, the mechanism acts, and `pse_full` is clearly worse than the strongest baseline, or the paired upper confidence bound excludes a useful improvement, or `bright_single`, `dark_single`, or `pse_duplicate_clean` explains the result.

## Paper Path If Positive

If `pse_full` reaches prototype GO, scale-up must add:

- controlled photometric-shift second condition;
- latency and forward-pass cost;
- Quantized OpenVLA-OFT INT4 same-backbone comparison if its action wrapper can safely support stateless transformed observations;
- recent direct test-time adaptation and verification baselines, not broad unrelated VLA SOTA.
