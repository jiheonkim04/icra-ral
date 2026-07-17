# R2P-XVLA Training Gate Result

Decision: `R2P_XVLA_OFFLINE_SELECTION_NOT_PASSED_NO_CLOSED_LOOP`

Both frozen 64-step arms trained successfully and offline validation ran. No closed-loop Ours rollout, simulator episode, download, or residual-reward checkpoint selection happened.

Frozen protocol decision: `R2P_XVLA_OFFLINE_SELECTION_NOT_PASSED`.

Calibrated scientific interpretation: Primary R2P-XVLA and Uniform LoRA were effectively tied on the frozen phase-weighted offline metric. Primary was worse by `6.08464082452187e-08`, so the frozen rule blocks rollout; the result does not support the R2P component claim, but the microscopic margin should be reported as a no-pass/tie rather than proof that phase weighting is broadly impossible.

Comparator-role summary:

| Comparator | Scientific question | Matched result | Uncertainty | Blocks claim? | Reason |
| --- | --- | --- | --- | --- | --- |
| Uniform LoRA ablation | Is R2P phase weighting responsible for the offline gain? | Primary `0.9418842308`, Uniform `0.9418841700` weighted loss | Not estimated; point difference `6.08e-08` | Yes, for rollout gate/component claim | Frozen rule required Primary to beat Uniform |
| X-VLA prior | Does adapted policy stay close to prior on fixed chunks? | mean action delta `0.08167`, max `0.53001` | Not estimated | No | Within frozen delta bounds |
| Base/Prior closed loop | Does method improve task success? | Not run | Not applicable | Not evaluated | Offline gate blocked rollout |

Primary arm: `64/64` optimizer steps, checkpoint written, final step weighted loss `1.8720510005950928`, max CUDA allocated `5350.398` MiB.

Uniform arm: `64/64` optimizer steps, checkpoint written, final step weighted loss `1.1197317838668823`, max CUDA allocated `5351.867` MiB.

Offline selection: `24` validation chunks; source/transit/target counts `5/10/9`; source degradation vs uniform `0.0`; CUDA peak `3573.601` MiB.

Next action: do not run closed-loop R2P-XVLA. Archive this method at the frozen offline-selection gate and resume official-prior-first residual search/candidate generation under the comparator-role calibration rules.
