# RL4IL Prior Module Training Launcher Repair

- Execution classification: `PRIOR_MODULE_TRAINING`
- Implementation label: `MECHANISM_FAITHFUL_RL4IL_LOCAL_PORT`
- Decision: `RL4IL_PRIOR_MODULE_TRAINING_NARROW_LAUNCHER_REPAIR_USED`

One narrow launcher/path repair was used before the successful training run.

Failed attempt 1: PowerShell timestamp formatting inserted `+09:00` into the run directory path, creating an invalid Windows path.

Failed attempt 2: Bash background-launch quoting did not write `launcher_pid.txt`; process inspection confirmed no training process was running.

Repair applied: switched to Windows `Start-Process` with explicit WSL arguments and Windows-side logs. No RL4IL mechanism, action oracle, panel, identity list, epoch budget, or decision rule changed.

Successful rerun: `runs/rl4il_prior/action_oracle_port_20260718T194750KST`; result copied to `reports/rl4il_action_oracle_prior_module_training_result.json`.
