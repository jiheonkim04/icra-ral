# SmolVLA LoRA Baseline Diagnosis

Final decision: `ACTION_INTERFACE_BUG`

This is a baseline diagnosis, not a new method or paper claim.

## Key Findings

- raw HDF5 timesteps: `13298`
- previous split records: `9 / 6`
- larger split possible: `300 / 100`
- interface audit result: `ACTION_INTERFACE_BUG`
- one-sample overfit passed: `False`
- one-demo overfit passed: `False`
- mean-action metric: `0.486561`
- frozen/base metric: `1.6029`
- best LoRA metric: `0.912258`
- best small MLP/ridge metric: `0.401848`
- LoRA beats mean-action: `False`
- exact next step: Fix the SmolVLA/LIBERO action interface before any method work: the local data is 7D LIBERO action space while the checkpoint action head and normalizer are 6D SO100-style, and overfit sanity did not clear the action metric gate.
