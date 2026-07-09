# Official SmolVLA / LeRobot Kill Criteria

Date: 2026-07-09 KST

Kill or block the route if any criterion triggers:

1. Only the archived custom LIBERO 7D adapter route can run.
2. The action or state convention is unclear.
3. SO100 6D normalizer statistics are used on LIBERO 7D labels without an official recipe.
4. A local HDF5-to-LeRobot conversion silently changes action semantics.
5. A supposed official LIBERO baseline does not use LeRobot's official LIBERO processor/eval stack.
6. Required official assets exceed the bounded local budget or require token/login/license click-through.
7. GPU training is requested but model parameters or input tensors remain on CPU.
8. Mini-repro evidence is used as paper-grade success or benchmark evidence.
9. The recipe requires OpenVLA-OFT, full VLA fine-tuning, full benchmark, or long GPU training on this machine.

## Current Trigger Status

- Criterion 1: not triggered for the SmolVLA base mini-repro; triggered if attempting LIBERO through the old custom adapter route.
- Criterion 2: not triggered for local base checkpoint; action convention is 6D SO100-style.
- Criterion 3: active risk for LIBERO; do not proceed that way.
- Criterion 5: active risk; official LIBERO baseline has not been reproduced yet.
- Criterion 7: not triggered because the mini-repro was intentionally CPU-only and no LoRA training ran.

