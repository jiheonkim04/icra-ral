# ContactTube-Aug Task Definition

Long title: Failure-Driven Contact-Preserving Demonstration Augmentation for Vision-Language-Action Robot Manipulation.

Core hypothesis: successful demonstrations contain contact tubes: EEF-object relative motion, gripper timing, object motion onset, lift/place phases, and contact/proximity windows. Preserving these tubes while retargeting object/reset/distractor conditions should produce useful additional demonstrations without new teleoperation.

STATE 1 evidence target: before training, show that a local LIBERO HDF5 demo exposes enough tube structure to build controller-valid augmented 7D action trajectories and replay them under a bounded exact/default-reset diagnostic.

Non-goals for this branch:

- no OpenVLA-OFT,
- no large VLA fine-tuning,
- no GPU job,
- no downloads,
- no paper-grade claim,
- no hiding simple-baseline ties.

Primary STATE 1 question: does ContactTube-Aug preserve contact/gripper/object-motion structure better than random action jitter, random pose jitter, and simple object-relative translation retargeting while remaining replay-valid?

STATE 1 answer: no. Replay and extraction were feasible, but ContactTube-Aug did not remain controller-valid under the generated retargeting action stream and simple object-relative translation retargeting produced a lower contact-tube preservation error.
