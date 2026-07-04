# Publishability Criteria

All ActionMap vs TCA-Map, TCA-Select, LoRA, or QLoRA claims must first satisfy
`reports/research_integrity_evaluation_policy.md`. Metrics, baselines,
ablations, split/sample policy, tuning budget, and kill/pivot criteria must be
fixed before confirmatory results are inspected.

## RA-L+ High Target

RA-L+ is high only if all of the following hold:

- At least one real simulator rollout benchmark runs.
- ActionMap baseline is included.
- TCA-Map + TCA-Select beats ActionMap + counterfactual augmentation.
- Wrong-target rate drops by at least 20 percent relative.
- Counterfactual success improves by at least +10 percentage points.
- Standard performance drops by no more than 1-2 percentage points.
- Default inference uses no privileged simulator state or oracle target labels.
- Compute table is included.

The compute table must include:

- GPU type,
- VRAM peak,
- latency,
- trainable parameters,
- batch size,
- heatmap grid size,
- whether features were cached,
- whether the required LoRA/QLoRA tracks were run or explicitly ruled infeasible.

## SOTA High Target

SOTA potential is high only if the claim is restricted to low-compute target-conditioned action decoding and counterfactual robustness.

Do not claim full standard OpenVLA-OFT leaderboard SOTA unless OpenVLA-OFT is directly reproduced under comparable benchmark conditions.

Required comparisons:

- native SmolVLA head,
- ActionMap,
- ActionMap + counterfactual augmentation,
- TCA-Map,
- TCA-Map + TCA-Select,
- ActionMap + LoRA,
- TCA-Map + LoRA,
- TCA-Map + LoRA + TCA-Select,
- QLoRA variant if feasible under the compute budget.

Required reporting:

- latency,
- VRAM,
- trainable parameters,
- offline proxy metrics clearly labeled as proxy metrics,
- simulator rollout metrics before claiming standard manipulation success,
- no-privileged-inference audit.

## Kill Conditions

Kill or pivot if:

- TCA-Map + TCA-Select does not beat ActionMap + counterfactual augmentation.
- Robust gains appear only in offline proxy metrics and do not transfer to a tiny rollout.
- Wrong-target rate does not improve meaningfully.
- TCA-Select adds unacceptable latency or brittle selection behavior.
- LoRA/QLoRA accounts for the gains while TCA-Select adds little.
- ActionMap + LoRA matches TCA-Map + LoRA under the fixed primary metrics.
- The result depends on cherry-picked tasks, samples, seeds, visualizations, or rollout episodes.
- Primary metrics, splits, or tuning budget had to be changed after seeing results without labeling the change exploratory.
