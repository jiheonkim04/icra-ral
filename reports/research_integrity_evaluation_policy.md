# Research Integrity Evaluation Policy

This policy applies before any confirmatory ActionMap vs TCA-Map, TCA-Select,
LoRA, or QLoRA evaluation. Its purpose is to make the conclusion trustworthy,
not to force a positive TCA-Map result.

## Core Rule

Do not optimize experiments to make TCA-Map look good. Optimize experiments to
test whether TCA-Map is actually valuable.

Exploratory debugging is allowed, but it must be labeled exploratory and kept
separate from confirmatory evaluation. Confirmatory metrics, baselines,
ablations, splits, sample policy, and tuning budget must be fixed before seeing
confirmatory results.

## Fixed Primary Metrics

Offline proxy metrics are not standard success and must not be reported as
paper-grade manipulation success.

Confirmatory offline proxy metrics:

- `action_l1_to_expert`,
- `action_mse_to_expert`,
- `distance_to_expert_voxel` or action voxel hit rate,
- `target_heatmap_top1_accuracy`,
- `target_heatmap_topk_accuracy`,
- `wrong_target_proxy_rate`,
- `counterfactual_target_action_separation_margin`,
- `nuisance_stability_score`,
- latency,
- max memory when measurable.

Rollout metrics, only after the simulator path is valid:

- LIBERO rollout success rate,
- task success,
- counterfactual rollout success,
- wrong-target rollout rate when measurable,
- latency,
- max memory when measurable.

The primary robustness decision should use wrong-target rate and
counterfactual success or their offline proxy equivalents. Standard performance
must be reported alongside robustness so robustness gains cannot be purchased by
collapsing standard behavior.

## Fixed Baseline List

The required comparison set is:

- native SmolVLA or frozen baseline,
- ActionMap head-only,
- ActionMap head-only with counterfactual augmentation,
- TCA-Map head-only,
- TCA-Map head-only with Distributional TCA-Select,
- ActionMap with LoRA,
- TCA-Map with LoRA,
- TCA-Map with LoRA and Distributional TCA-Select,
- TCA-Map with QLoRA and Distributional TCA-Select if feasible under the
  documented compute budget,
- zero-action or simple sanity baseline when useful for diagnosing rollouts.

If ActionMap with LoRA or ActionMap with counterfactual augmentation matches
TCA-Map, report that the TCA-Map novelty is weak under that evidence level.

## Fixed Ablation List

Required ablations:

- TCA-Map without TCA-Select,
- TCA-Map with heuristic target/action selection instead of Distributional
  TCA-Select,
- top-heatmap candidate without TCA-Select,
- target-conditioned head without counterfactual augmentation,
- target-conditioned head with counterfactual augmentation,
- LoRA gain separated from TCA-Select gain,
- QLoRA feasibility separated from method contribution,
- no-privileged-inference audit.

If TCA-Select adds no measurable gain, report that directly. If LoRA accounts
for the gain while TCA-Select adds little, report that attribution risk.

## Fixed Split And Sample Policy

Before confirmatory evaluation, write the exact split manifest or deterministic
selection rule to a report or config. The manifest must include:

- dataset root and subset name,
- task list,
- sample count,
- sample ordering rule,
- random seed list when seeds are used,
- counterfactual target-swap construction rule,
- excluded samples and the reason for each exclusion,
- whether the run is exploratory or confirmatory.

Do not cherry-pick tasks, samples, seeds, metrics, baselines, visualizations, or
rollout episodes. Failed runs and weak results must be logged. If a sample is
excluded for corruption, missing files, unsupported action shape, or simulator
failure, the exclusion must be recorded before aggregate metrics are computed.

## Fixed Tuning Budget

Before confirmatory evaluation, fix:

- model arms to run,
- maximum samples,
- maximum steps,
- batch size,
- learning rate search space,
- LoRA rank and target modules,
- heatmap resolution,
- candidate count `K`,
- TCA-Select temperature,
- maximum runtime,
- maximum VRAM or RAM target,
- allowed number of exploratory debugging reruns.

Do not change primary metrics, evaluation split, or tuning budget after seeing
results unless the change is logged as exploratory and the previous result is
kept. Confirmatory comparison must use the same tuning budget for ActionMap and
TCA-Map arms unless a difference is predeclared and justified.

## Kill And Pivot Criteria

Produce a kill or pivot report instead of forcing a positive result if:

- ActionMap with LoRA or ActionMap with counterfactual augmentation matches or
  beats TCA-Map under the fixed metrics,
- TCA-Select adds no measurable gain over TCA-Map without TCA-Select,
- offline gains disappear in rollout,
- wrong-target rate does not improve meaningfully,
- counterfactual success or proxy robustness does not improve,
- standard performance drops more than the predeclared tolerance,
- gains require privileged simulator state at inference,
- gains depend on cherry-picked tasks, samples, seeds, visualizations, or
  rollout episodes,
- compute cost or latency violates the documented low-compute claim.

Allowed pivot outcomes include:

- publish only a negative/diagnostic report,
- narrow the claim to an engineering artifact,
- pivot to ActionMap plus counterfactual augmentation,
- pivot to data/interface diagnosis,
- stop local experiments and require a cloud or simulator benchmark before any
  paper-grade claim.

## Reporting Requirements

Every confirmatory report must state:

- whether the run was exploratory or confirmatory,
- which fixed split and tuning budget were used,
- all failed runs and exclusions,
- all required baselines that were skipped and why,
- whether TCA-Map won, tied, or lost under each primary metric,
- whether the result supports, weakens, or kills the current novelty claim.
