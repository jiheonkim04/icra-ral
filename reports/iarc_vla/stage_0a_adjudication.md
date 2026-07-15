# IARC-VLA Stage 0A Adjudication

Date: 2026-07-15 KST

Decision: `IARC_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`

Proposal hash:
`A1B0CF8BCBCF6A88F27B31EF5E38BAF408A3E62BB34206A1AC9F051EA6B57408`.

## Result Integrity

The detached worker completed normally and is not rerun:

- Linux wrapper PID `294` and Python child PID `357` are dead;
- child and wrapper exit codes are both `0`;
- status is `completed`;
- gradient partial JSON parses as `40 / 40` with zero exceptions;
- final result JSON parses;
- gradient, partial, and validation duplicate-key counts are all `0`;
- missing and extra gradient-manifest key counts are both `0`;
- confirmatory observations and actions are both `0`.

Two launcher-only failures occurred before any model load or parameter update.
They are preserved under `runs/iarc_vla/stage0a/attempt_1_launcher_failure/`
and `runs/iarc_vla/stage0a/attempt_2_no_process/`. They are implementation
history, not experimental rows.

## Gates That Passed

- partition, perturbation, target, and shared-draw health passed;
- zero-effect identity error was exactly `0.0` before micro-fit;
- only rank-4 LoRA parameters were trainable;
- fixed-subset loss decreased from `0.0959141970379278` to
  `0.09489042789209634` after the frozen `20` steps;
- Base parameter hash remained unchanged;
- checkpoint files hash-verify, disk reload passed, and reload output error was
  exactly `0.0`;
- `18 / 40` rows crossed cosine `< -0.01`, with all four perturbation families
  active;
- all `18 / 18` projected rows satisfied the constraint;
- all `22 / 22` agreeing rows were bitwise unchanged;
- no robust gradient was below the frozen floor;
- all actions were finite with semantic max-absolute validity;
- peak CUDA allocation was `1.0882797241210938 GiB`, below `15.5 GiB`.

These observations establish that the projected-gradient mechanism acts. They
do not override a failed hard integration gate or establish closed-loop
performance.

## Failing Hard Gate

The frozen protocol requires every postprocessed validation action to remain
within the dataset action bounds. Only `12 / 40` clean/perturbed pairs passed,
for validity `0.30` against the required `1.0`.

There were `40` scalar violations across `28` pair rows:

- `38` gripper-dimension violations;
- `2` z-translation violations;
- `27` violations on clean actions and `13` on perturbed actions;
- the largest gripper low/high values were `-1.0234497785568235` and
  `1.0540539026260376` against `[-1,1]`.

The failure appears across all four families. It is not a duplicate, timeout,
exception, stale-worker, or confirmatory-leak artifact.

## Scientific Ruling

This is an implementation/action-validity failure, not a scientific kill of
conflict-aware robustness consolidation. The mechanism evidence is positive,
but Stage 0A is ineligible to advance because the frozen action-validity gate
failed.

Do not clip actions, widen bounds, change postprocessing, reduce learning rate,
change rank, change rows, add retention loss, run the one-check, run Stage 0B,
or launch validation search. Any such change would rescue the formulation
after a valid frozen stop.

Cycle 16 closes without confirmatory testing. Continue automatically to Epoch
4 Cycle 17 candidate generation under the active governance.
