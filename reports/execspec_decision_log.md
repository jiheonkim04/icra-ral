# ExecSpec-Repair Decision Log

## STATE 0 Initialization

Decision: start ExecSpec-Repair as a fresh rollout/replay-first direction and archive Target-Prior TCA-Map and CSS-Shield as negative evidence.

Reason: the killed routes exposed a repeated deployment failure mode: action dimensions, gripper conventions, metadata scales, and controller interfaces can collapse execution even when checkpoints or offline proxies look plausible.

Consequence: the first branch must produce a real mismatch or replay metric, not only planning documents.

## STATE 1 HDF5 Mismatch and Exact-Init Replay Result

Decision: continue ExecSpec-Repair to a calibrated repair replay state.

Reason: local LIBERO HDF5 action metrics reproduced substantial executable-spec mismatch, and bounded exact-init replay showed degradation from plausible mismatches. Correct 7D expert replay reached reward/success `1.0 / true` at observed first reward/done index `260`. Gripper sign flip replay and translation-scale mismatch replay both degraded to `0.0 / false`.

Key HDF5 metric: `gripper_sign_flip` had action L2 mean `2.0` and gripper mismatch rate `1.0`. Translation-scale mismatch had action L2 mean `0.363970119`.

Repair metric: supervised diagonal calibration, labeled as HDF5 calibration/evaluation rather than rollout policy action generation, beat identity, clipping-only, and naive global affine baselines on global scale, per-dimension scale, gripper sign flip, gripper threshold, translation scale, rotation scale, and 6D/7D zero-gripper bridge variants.

Consequence: STATE 2 should replay a minimal calibrated repair for the strongest degraded mismatch under the same exact-init boundary. Do not claim paper-grade results or deployable policy repair until calibrated replay evidence exists without using future expert actions as rollout actions.

Training happened: false. LoRA training happened: false. Loss was computed: false. Replay/rollout happened: true, bounded exact-init diagnostic only. GPU/download/OpenVLA-OFT happened: false. Paper-grade claim: false.
