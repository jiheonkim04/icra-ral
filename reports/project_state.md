# Project State

Date: 2026-07-09 KST

Branch:

`codex/smolvla-libero-7d-baseline-reproduction`

Current branch base:

`ebcc466 Fix SmolVLA LIBERO 7D action interface`

Current decision:

`READY_FOR_RA_L_METHOD_ON_SMOLVLA_7D`

## Current Bounded Run Boundary

- Experiments happened: yes, bounded fixed-interface baseline reproduction only.
- Training happened: yes, small CPU supervised 7D adapters and LoRA-on-`state_proj` baselines only.
- Loss computed: yes.
- GPU training happened: no.
- Downloads happened: no.
- Rollout/replay happened: no.
- OpenVLA-OFT happened: no.
- Full benchmark happened: no.
- PatchGuard continued: no.
- New method implementation happened: no.
- Paper claims happened: no.
- Old broken 6D/SO100 action path used: no.
- Hard-coded gripper fill used: no.

## Dataset And Split Status

Primary split used for the baseline suite:

- split: `same_task_demo_holdout`
- task: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo`
- train/eval records: `300 / 100`
- train demos: `demo_0` through `demo_29`
- eval demos: `demo_30` through `demo_39`
- raw timesteps for task: `13298`
- exact record leakage: no
- demo leakage: no
- task overlap: yes, by same-task design

Additional split audits:

- `same_task_time_holdout`: `160 / 80` records, exact record leakage no, same-demo overlap yes, 50-step chunk temporal-overlap risk yes
- `multi_task_demo_holdout`: `150 / 60` records over 3 local tasks, exact record leakage no, demo leakage no

## Baseline Suite

All learned baselines use the fixed `LIBERO_7D` label path with train-split-only 7D normalization and learned gripper output.

Primary held-out action L2:

- global mean-action: `1.082453`
- per-task mean-action: `1.082453`
- previous-action persistence diagnostic: `0.181765`
- ridge: `0.890603`
- small MLP: `0.518738`
- frozen/base SmolVLA 7D linear adapter: `0.890604`
- SmolVLA 7D adapter without LoRA: `0.561651`
- SmolVLA `state_proj` LoRA rank 4 + 7D adapter: `0.504675`
- SmolVLA `state_proj` LoRA rank 8 + 7D adapter: `0.494959`

Important persistence caveat:

The previous-action baseline uses the previous expert action from the held-out HDF5 sequence. It is recorded as a diagnostic persistence oracle, not treated as the learned-action decision gate.

## LoRA And Target Module Status

- LoRA ranks tested: `[4, 8]`
- rank 16: not run; optional and skipped to keep the reproduction bounded
- executable target modules: `libero_7d_adapter_head_only`, `frozen_state_proj_plus_7d_adapter`, `state_proj_lora_plus_7d_adapter`
- audited but not executed in fixed 7D path: `action_in_proj`, `action_out_proj`, `action_time_mlp_in`, `action_time_mlp_out`
- reason: native action projection modules require `max_action_dim` / native flow actions and would re-enter the old 6D/SO100 action path

Trainable params:

- small MLP: `487`
- frozen/base SmolVLA 7D linear adapter: `6734`
- SmolVLA 7D adapter without LoRA: `124039`
- rank 4 `state_proj` LoRA + 7D adapter: `128007`
- rank 8 `state_proj` LoRA + 7D adapter: `131975`

## Best Fixed-Interface Result

Best learned variant:

`smolvla_state_proj_lora_rank8_7d_adapter`

Metrics:

- train action L2: `0.28441`
- eval action L2: `0.494959`
- eval translation L2: `0.230133`
- eval rotation L2: `0.064995`
- eval gripper error: `0.365736`
- eval gripper accuracy: `0.88`
- eval per-dim MAE: `[0.100928, 0.103205, 0.138828, 0.017847, 0.040066, 0.034002, 0.365735]`
- train/eval action-L2 gap: about `0.210549`
- VRAM peak: `0.0` MB
- total runtime: `9.438` sec

Best LoRA beats:

- mean-action: yes
- ridge/MLP: yes
- frozen/base 7D adapter: yes

## Optional Replay/Progress

Optional replay/progress was eligible by action metrics but was not run.

Reason: no bounded executable LIBERO environment bridge for the learned 7D adapter is part of this baseline runner. No rollout or benchmark claim is made.

## Conclusion

The fixed-interface 7D baseline is strong enough to allow future method planning, provided the baseline table is preserved and simple baselines are predeclared. This is not a paper claim. Any future method must compare against the rank-8 fixed-interface SmolVLA 7D LoRA/adapter baseline, mean-action, ridge/MLP, and the diagnostic persistence baseline where appropriate.
