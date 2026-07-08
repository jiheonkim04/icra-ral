# ContactTube-Aug Kill Summary

Decision: kill ContactTube-Aug before STATE 2 training.

## Original Hypothesis

Successful robot demonstrations contain contact tubes: EEF-object relative trajectories, gripper timing, object motion onset, lift/place phases, and contact/proximity windows. Preserving those tubes while retargeting object pose, reset state, or distractor conditions should generate useful failure-driven demonstrations without new teleoperation.

## Strongest Positive Evidence

- Contact-tube extraction succeeded using HDF5 EEF/gripper traces plus bounded runtime object traces.
- A real bounded LIBERO/RoboSuite replay/control diagnostic ran: `1621` simulator steps across `6` variants.
- Exact-init no-op upper bound succeeded with reward/success `1.0 / true`.
- Runtime object pose was available even though HDF5 object pose was unavailable.
- ContactTube-Aug beat random action jitter and random pose jitter on contact-tube preservation.

## Decisive Negative Evidence

- ContactTube-Aug augmentation validity failed: controller-valid action rate `0.849265`.
- ContactTube-Aug clip-step rate was `0.150735`.
- Simple object-relative translation retargeting beat ContactTube-Aug on the predeclared tube metric: `0.009154` versus `0.015226`.
- HDF5 object pose was unavailable, making offline contact-tube construction depend on runtime traces for object state.
- ContactTube-Aug reward/success was `0.0 / false`.

## Kill Criteria Triggered

- Augmented actions were not controller-valid enough.
- Simple object-relative retargeting matched or beat ContactTube-Aug.
- HDF5 object/contact state was incomplete.

## Why Training Is Not Allowed

Training on invalid augmented actions would test whether a learner can absorb bad supervision, not whether ContactTube-Aug creates useful demonstrations. Since the simple retarget baseline preserves the contact trajectory better and ContactTube-Aug clips often, BC/action-head or VLA training would only add compute and interpretation noise after the replay-validity gate already failed.

## Baseline That Killed The Route

The decisive baseline is simple object-relative translation retargeting. Random jitter baselines were weaker, but the route required a win over simple retargeting before STATE 2.

## Why Not RA-L-Stable

The route has useful extraction/replay infrastructure but no method-level evidence beyond the strongest simple baseline. A RA-L-stable data-augmentation claim needs physically valid augmented demonstrations and a baseline gap before training; ContactTube-Aug failed both the controller-validity and simple-retarget gates.

Execution boundary for this archive: documentation only. No new experiment, replay, rollout, training, loss computation, GPU job, download, OpenVLA-OFT execution, model loading, or paper claim occurred.

