# SmolVLA-LIBERO 7D Interface Fix

Final decision: `READY_FOR_REAL_METHOD_AFTER_INTERFACE_FIX`

This is an infrastructure fix, not a new research method or paper claim.

## Summary

- action schema before: `SMOLVLA_NATIVE_SO100_6D`
- action schema after: `LIBERO_7D adapter path with native SmolVLA 6D schema preserved separately`
- normalization before: `SO100 6D checkpoint action normalizer`
- normalization after: `train-split-only LIBERO 7D mean/std`
- gripper before: `hard-coded 6D-to-7D bridge fill`
- gripper after: `learned 7th adapter output with separate MSE loss`
- one-sample overfit passed: `True`
- one-demo overfit passed: `True`
- mean-action metric: `1.082453`
- frozen/base metric: `1.6029`
- best 7D adapter metric: `0.573503`
- best MLP/ridge metric: `0.518738`
- 7D interface fixed: `True`
- exact next step: Run a standard fixed-interface SmolVLA/LIBERO 7D baseline reproduction on an official or standard split before proposing any new method.
