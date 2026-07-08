# Rejected Topic Patterns

Do not select topics with these shapes:

- Method-first action-head variants before the anchor beats mean-action, linear/L1, and cheap MLP baselines.
- Richer geometry injection when active single-point or destination-only baselines are stronger.
- Language robustness that only improves paraphrase consistency but loses to canonicalization or weakens counterfactual object/target sensitivity.
- Repair methods that improve symbolic violations but lose replay/control utility to no-repair, safety-only, clipping, or diagonal affine controls.
- Timing methods that can be matched by gripper-only correction, fixed shift, or linear time warp.
- Retargeting methods that lose to global scale or object-relative retargeting.
- Data-augmentation methods whose generated actions are not controller-valid before training.
- Latency-only VLA topics after one-step high-noise schedules already show strong results.
- Heavy pretraining or full VLA fine-tuning as first evidence.
- Topics chosen primarily because local HDF5 code is easy to write.

Current stop rule: ActionMap implementation, diagnostics, reproduction, extension design, and failure mining are stopped. The local ActionMap mini-gate was already merged into local `main` before the stop steer arrived, but it has not been pushed.

