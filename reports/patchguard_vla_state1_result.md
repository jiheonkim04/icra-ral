# PatchGuard-VLA STATE 1 Result

Bounded offline VLA patch-sensitivity diagnostic only. This is not a benchmark, rollout, training result, or paper claim.

- STATE 1 decision: `TOO_HEAVY_LOCAL`
- real VLA model used: `True`
- patch effect measured: `True`
- max attacked policy L1 vs clean: `0.181765`
- max attacked translation L2 vs clean: `0.213965`
- kinematic signal available: `True`
- cutout baseline dominated fixed patch: `False`
- local adapter path feasible now: `False`
- training happened: `False`
- rollouts happened: `False`
- downloads happened: `False`
- GPU jobs happened: `False`

## Variant Metrics

| variant | mean action L1 to expert | mean policy L1 vs clean | max policy L1 vs clean | mean translation L2 vs clean | expert alignment delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 0.453773 | 0.0 | 0.0 | 0.0 | 0.0 |
| random_patch | 0.41589 | 0.117395 | 0.163939 | 0.145187 | -0.037883 |
| fixed_visible_patch | 0.430836 | 0.157706 | 0.181765 | 0.152733 | -0.022937 |
| fixed_patch_cutout_defense | 0.445735 | 0.049337 | 0.060118 | 0.108373 | -0.008038 |
| fixed_patch_visual_aug_proxy | 0.457785 | 0.233666 | 0.254161 | 0.2776 | 0.004012 |

## Interpretation

Do not train in this run. Resolve real local LoRA/adapter tooling without unapproved installs or move the adapter smoke to approved WSL/Linux/cloud before STATE 2.
