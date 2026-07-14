# RAC-VLA Prototype Protocol

Date: 2026-07-14 KST

Proposal hash: `71ABA93E37FC725C1A2E5EAE6E1461BC77AACDAFF9B0711C37F17D5C0AB0902F`

## Prototype Policies

1. `base_smolvla_shifted`
   - Frozen SmolVLA under the predeclared action-channel shift.

2. `reflective_history_proxy`
   - Local transparent Reflective VLA proxy.
   - Uses recent action-consequence history to select among predeclared inverse templates.
   - Not an official Reflective VLA reproduction.

3. `rac_full`
   - Learned consequence-history calibration context plus bounded residual calibration.

4. `rac_no_consequence_ablation`
   - Same horizon, state, action, task, and phase features as RAC but removes `delta_state`.

5. `online_diagonal_inverse_gain`
   - Simple reviewer-killer baseline.
   - Estimates a diagonal correction from recent observed state/action deltas without the learned RAC context.

## Shift Condition

The first closed-loop condition is a controlled action-channel shift. The exact transform must be written into the runner config before Stage A. It may not be tuned after observing Stage A/B outcomes.

Clean-retention episodes use the same task/reset manifest without the shift.

## Stage 0 Command

Planned command:

```powershell
C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_rac_vla_development.py
```

## Stage A Command

Planned command:

```powershell
wsl -d Ubuntu-22.04 bash -lc "cd /mnt/c/Users/jiheo/tca_map && /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_rac_vla_prototype.py --mode stage-a"
```

## Stage B Command

Planned command:

```powershell
wsl -d Ubuntu-22.04 bash -lc "cd /mnt/c/Users/jiheo/tca_map && /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_rac_vla_prototype.py --mode stage-b"
```

## Nonretroactivity

RAC cannot change EvoState, FANG, CAVM, RCV, PSE, or earlier results. All prior kills and non-GO decisions remain fixed.
