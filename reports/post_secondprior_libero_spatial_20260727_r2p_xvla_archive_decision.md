# R2P-XVLA Archive Decision

Decision: `R2P_XVLA_ARCHIVED_AT_FROZEN_OFFLINE_SELECTION_GATE`

Source result: `reports/post_secondprior_libero_spatial_20260727_r2p_xvla_training_gate_result.json`

Target: `libero_spatial/task_5`, reset identity `20260727`, initial-state index `16`.

Frozen protocol decision: `R2P_XVLA_OFFLINE_SELECTION_NOT_PASSED_NO_CLOSED_LOOP`.

Calibrated scientific interpretation: the selected R2P phase-weighted arm and the uniform LoRA ablation were effectively tied on the frozen offline metric. Primary was worse by `6.08464082452187e-08`, so the frozen selector blocks closed-loop rollout and any R2P component/paper claim. The microscopic margin should be reported as a no-pass/tie, not as broad evidence that every phase-weighting idea is impossible.

| Comparator | Scientific question | Matched result | Uncertainty | Blocks claim? | Reason |
| --- | --- | --- | --- | --- | --- |
| Base | Does R2P improve the backbone in closed-loop task success? | Not evaluated; offline gate failed before Ours rollout. | N/A | Yes | Closed-loop Base-vs-Ours evidence is missing by frozen design. |
| Closest Prior | Does R2P improve over X-VLA on the residual protocol? | Not evaluated in closed loop. | N/A | Yes | No prior-advance claim without passing the selector and running the matched rollout gate. |
| Uniform LoRA ablation | Is phase balancing better than generic task adaptation? | Primary `0.9418842308223248`; uniform `0.9418841699759165`; delta `6.08464082452187e-08`. | Single frozen validation split; no paired uncertainty. | Yes | The selected component did not beat the preregistered ablation. |
| Simple control | Can a simpler explanation account for the offline result? | Uniform task5 LoRA matched/slightly exceeded primary. | Microscopic no-pass/tie. | Yes | Generic adaptation explains the offline outcome enough to block this claim. |

No new training, optimizer step, checkpoint write, simulator episode, or closed-loop Ours evaluation happened while creating this archive.

Immediate next action: do not run closed-loop R2P-XVLA; resume official-prior-first residual search under comparator-role calibration.
