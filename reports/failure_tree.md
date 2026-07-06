# Failure Tree

## 1. Offline Proxy Success

Fixed-prior TCA succeeded on offline proxy metrics when the target prior was correct. This established a real offline hypothesis: target-prior-conditioned action decoding can reduce wrong-target behavior in proxy space.

Status: passed as offline proxy only.

## 2. Rollout Bridge Success

The project validated the LIBERO/RoboSuite action bridge and found that raw 7D HDF5 actions are the correct control format for the relevant environment path.

Status: passed.

## 3. Expert Replay Success

Matched-init HDF5 expert replay succeeded, while zero-action and default-reset controls exposed important reset/action provenance issues.

Status: passed.

## 4. Online Action Source Creation

A non-leaking online 7D diagnostic head was created. It generated 7D actions from current observation/instruction features and did not use same-timestep or future HDF5 expert actions at inference.

Status: passed as an engineering milestone.

## 5. Online 7D Action-Quality Failure

The online 7D diagnostic head produced weak action quality. Fixed-prior TCA was only marginally better than ActionMap in the first online 7D action-quality diagnosis.

Status: failed as a rollout-readiness signal.

## 6. Mean-Action Baseline Failure

The simple train-split mean-action baseline beat all current and redesigned learned 7D heads on the key held-out action-quality gate.

Key final numbers:

- Mean-action baseline eval 7D L2: `0.57299313`
- Best redesigned eval 7D L2: `0.669078005`

Status: failed.

## 7. Rollout Gate Red

The final rollout gate required the best ActionMap/TCA head to beat the mean-action baseline by the documented threshold, fixed-prior TCA to beat ActionMap, and ActionMap/TCA actions to differ meaningfully.

Fixed-prior TCA beat ActionMap and differed meaningfully, but no non-mean head beat the mean-action baseline. The rollout gate stayed red.

Status: red.

## 8. Final Kill Decision

Because the current low-compute route cannot produce a rollout-ready online action source, the RA-L route is killed.

Status: final kill/archive.

