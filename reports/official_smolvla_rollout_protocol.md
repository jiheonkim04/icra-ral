# Official SmolVLA Rollout Protocol

Date: 2026-07-10 KST

Status: `FROZEN_NOT_EXECUTABLE_YET`

This protocol defines the next official closed-loop LIBERO rollout path. It does not claim that rollout has already been run.

## Preconditions

Before any official rollout:

- model revision must be locked: `31d453f7edd78c839a8bbc39744a292686daf0de`
- dataset revision must be locked: `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`
- split manifest hash must remain `1279F939648CF13E2F599084E42631681E1DFA5606B5D9B0851FFEB32710934B`
- metric protocol hash must remain `64430225940C5168B3734BB40F9F48AD02877E0BA04DC804367AFBB214AE486E`
- LoRA adapter checkpoints for rollout seeds must be complete and immutable
- official `lerobot-eval --env.type=libero` must import and construct the official LIBERO env
- no custom `LIBERO_7D` adapter, local normalization, hard-coded gripper conversion, or replay bridge may be used

## Policies

Stage A and Stage B use these policy names:

- `frozen_base`
- `rank4_lora`
- `validation_selected_action_space_static_mix`

The following may be reported only as analysis or bounds:

- `task_or_instruction_router_proxy`
- `frame_oracle_upper_bound`
- `task_oracle_upper_bound`

## Stage A: Bounded Readiness Pilot

Purpose: confirm official closed-loop execution, logging, and ranking signal before scaling.

Required inputs:

- persisted LoRA adapter checkpoints for seeds `11`, `22`, and `33`, or a single predeclared validation seed if the pilot is explicitly scoped that way before launch
- fixed task list
- fixed reset seeds
- fixed maximum steps from official LIBERO env config
- official preprocessing/postprocessing
- validation-selected static-mix alpha, selected before any test rollout

Policies:

- `frozen_base`
- `rank4_lora`
- `validation_selected_action_space_static_mix`

Primary metric:

- official LIBERO success rate

Required logs:

- policy name
- seed
- task
- reset seed
- episode index
- success/failure
- steps
- exceptions
- wall-clock time
- latency per action or action chunk
- forward passes per env step
- peak VRAM
- package versions
- model/dataset revisions
- adapter checkpoint hash, where applicable

Stage A pass condition must be declared before execution. No Stage B scaleup is allowed until Stage A completes without protocol violations.

## Stage B: Scaleup

Purpose: estimate more stable task success with confidence intervals after Stage A confirms the execution path.

Stage B must:

- reuse the same locked source revisions or explicitly create a superseding revision lock
- reuse complete adapter checkpoint bundles
- expand tasks and reset seeds according to a predeclared plan
- report confidence intervals
- report failures and exceptions, not only successful runs
- preserve all raw rollout logs

## Fairness Rule

Static mix consumes both base and LoRA outputs. Its compute cost must not be hidden. Report its additional forward-pass, latency, wall-clock, and VRAM cost next to success rate.

## Current Protocol Status

The protocol is frozen, but it is not executable yet because:

- official seed LoRA adapter checkpoints are missing;
- local official LIBERO eval dependencies are missing (`libero`, `robosuite`);
- native Windows official rollout support is unproven and WSL/Linux is recommended.
