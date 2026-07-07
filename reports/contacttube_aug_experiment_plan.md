# ContactTube-Aug Experiment Plan

## STATE 1: Extraction And Replay Smoke

Use one local LIBERO HDF5 expert demo from `reports/libero_offline_counterfactual_split_scaled_report.json`.

Extract:

- EEF trajectory,
- object pose trajectory if available in HDF5 or bounded replay observations,
- EEF-object distance profile,
- gripper close/release timing,
- object motion onset,
- lift and place/release indices,
- contact/proximity window.

Generate/replay small variants:

- exact-init no-op upper bound,
- raw demo replay under default reset,
- random pose jitter,
- simple object-relative translation retargeting,
- random action jitter,
- ContactTube-Aug.

Metrics:

- replay validity and reward/success,
- controller-valid action rate and clip rate,
- contact-tube preservation error,
- gripper timing error,
- EEF-object distance profile error,
- object motion onset error,
- lift/place phase error.

STATE 2 is allowed only if STATE 1 is green: train the smallest BC/action-head diagnostic on original versus augmented demos and evaluate held-out pose/reset perturbations.

## STATE 1 Outcome

STATE 1 was not green. The bounded replay smoke produced a real replay/control metric, but ContactTube-Aug failed the controller-validity gate and did not beat simple object-relative translation retargeting.

Do not start STATE 2 for this branch without a new predeclared method change and a fresh simple-baseline gate.
