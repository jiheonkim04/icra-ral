# ContactTube-Aug Failure Tree

Root failure: ContactTube-Aug is not a valid continuation route before training.

## Branch 1: Augmentation Validity Failure

- Generated ContactTube-Aug actions were not controller-valid enough.
- Valid action rate: `0.849265`.
- Clip-step rate: `0.150735`.
- Triggered kill criterion: augmented trajectories are not controller-valid or replay-valid.

## Branch 2: Simple Baseline Dominance

- Simple object-relative translation retargeting preserved the contact trajectory better.
- Simple retarget tube score: `0.009154`.
- ContactTube-Aug tube score: `0.015226`.
- Triggered kill criterion: simple object-relative retargeting matches or beats ContactTube-Aug.

## Branch 3: Incomplete HDF5 Object State

- HDF5 EEF/gripper traces were available.
- HDF5 object pose was unavailable.
- Runtime object pose was available during bounded replay.
- Triggered risk: offline augmentation cannot rely on complete object trajectories from the local HDF5 file without simulator replay traces.

## Branch 4: Training Would Not Rescue The Claim

- No BC/action-head training happened.
- No loss was computed.
- Training after invalid augmentation would confound augmentation quality with learner robustness to clipped/bad actions.
- Required gate before training was not met.

## Surviving Positive Signal

Contact-tube extraction and replay diagnostics are reusable. ContactTube-Aug beat random action jitter and random pose jitter, but this only shows that random noise is a weak baseline; it does not support continuation against object-relative retargeting.

