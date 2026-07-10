# Official LIBERO Closed-Loop Failure Summary

Date: 2026-07-10 KST

## Result Snapshot

The bounded official pilot completed all `48/48` planned episodes with no rollout exceptions.

| Policy | Overall success | `libero_spatial` | `libero_object` | `libero_goal` | `libero_10` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `frozen_base` | `75.0%` | `66.7%` | `100.0%` | `100.0%` | `33.3%` |
| `rank4_lora_seed_11` | `83.3%` | `66.7%` | `100.0%` | `100.0%` | `66.7%` |
| `rank4_lora_seed_22` | `66.7%` | `100.0%` | `33.3%` | `100.0%` | `33.3%` |
| `rank4_lora_seed_33` | `75.0%` | `66.7%` | `100.0%` | `100.0%` | `33.3%` |

Per-task success vectors, in the three fixed reset episodes:

| Policy | Spatial | Object | Goal | LIBERO-10 |
| --- | --- | --- | --- | --- |
| `frozen_base` | `101` | `111` | `111` | `010` |
| `rank4_lora_seed_11` | `101` | `111` | `111` | `011` |
| `rank4_lora_seed_22` | `111` | `100` | `111` | `010` |
| `rank4_lora_seed_33` | `101` | `111` | `111` | `010` |

## Offline L2 Versus Closed-Loop Success

Canonical offline action L2 order, best to worst:

1. `frozen_base`: `0.085579125`
2. `rank4_lora_seed_22`: `0.086474081`
3. `rank4_lora_seed_11`: `0.086743582`
4. `rank4_lora_seed_33`: `0.086918872`

Closed-loop success order in this pilot:

1. `rank4_lora_seed_11`: `83.3%`
2. `frozen_base`: `75.0%`
3. `rank4_lora_seed_33`: `75.0%`
4. `rank4_lora_seed_22`: `66.7%`

Lower offline L2 did not correspond to higher closed-loop success in the pilot. Seed 11 had worse offline L2 than seed 22 but higher closed-loop success; seed 22 had the second-best offline L2 but the lowest pilot success.

## Failure Concentration

The official metrics did not include videos or semantic failure labels, so categories below are inferred from task groups and success vectors, not from frame-level inspection.

- `libero_goal/task_0` was solved by every policy on all three resets.
- `libero_10/task_0` was the hardest and shows long-horizon compounding sensitivity. Seed 11 improved this suite from `33.3%` to `66.7%`; the other LoRA seeds matched frozen base.
- `libero_spatial/task_0` had a repeatable reset-2 failure for frozen base, seed 11, and seed 33; seed 22 solved all three resets.
- `libero_object/task_0` was solved by frozen base, seed 11, and seed 33, but seed 22 failed resets 2 and 3.
- Without videos, the pilot cannot reliably separate perception/target selection, initial reach, rotation/orientation, gripper timing/contact, transport, and placement failures.

## Diagnosis

- Any LoRA seed improve over frozen base? Yes, seed 11 improved overall success by `+8.3` percentage points and `libero_10` by `+33.3` percentage points in this bounded pilot.
- Do LoRA seeds consistently help? No. Seed 22 hurt `libero_object` and overall success; seed 33 matched frozen base overall.
- Are failures consistent across reset seeds? Partly. The same reset positions fail for several policies in `libero_spatial` and `libero_10`, but seed-specific differences are visible.
- Is there a structured method-worthy gap? Not yet. The pilot is enough to validate the official rollout baseline path, but not enough to justify a new method-design gate.
- Is the pilot large enough for method selection? No. It is a readiness and failure-mining pilot, not a statistically adequate method-selection run.
