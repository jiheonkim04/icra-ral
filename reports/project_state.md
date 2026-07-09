# Project State

Date: 2026-07-09 KST

Branch:

`codex/smolvla-libero-7d-action-interface-fix`

Current branch base:

`bc6ad03 Diagnose SmolVLA LoRA action interface`

Current decision:

`READY_FOR_REAL_METHOD_AFTER_INTERFACE_FIX`

## Current Bounded Run Boundary

- Experiments happened: yes, bounded interface-fix diagnosis only.
- Training happened: yes, small supervised LIBERO_7D adapter and simple baselines only.
- Loss computed: yes.
- GPU training happened: no.
- Downloads happened: no.
- Rollout/replay happened: no.
- OpenVLA-OFT happened: no.
- Full benchmark happened: no.
- PatchGuard continued: no.
- New method implementation happened: no.
- Paper claims happened: no.

## Dataset And Split Status

- raw HDF5 demos: `50`
- raw HDF5 timesteps: `13298`
- previous deterministic demo-holdout split: `9 / 6`
- larger deterministic demo-holdout split: `300 / 100`
- records are sampled observation/action-window records, not full demonstrations
- task holdout was not trained in this run

## Action Schema Before And After

Before:

- native SmolVLA action schema: `SMOLVLA_NATIVE_SO100_6D`
- model action shape: `[6]`
- preprocessor action shape: `[6]`
- postprocessor action shape: `[6]`
- native action normalizer: SO100-style `ACTION: MEAN_STD`
- gripper handling: hard-coded 6D-to-7D bridge fill

After:

- fixed baseline schema: `LIBERO_7D`
- adapter output action shape: `[7]`
- label action shape: `[7]`
- label normalizer: train-split-only LIBERO 7D mean/std
- unnormalize path: guarded `Libero7DNormalizer.unnormalize`
- gripper handling: learned 7th output dimension with separate normalized MSE loss
- SO100 action normalizer used for LIBERO labels: no
- eval labels used for training/normalization: no

## Schema Audit Evidence

- LIBERO labels are 7D throughout all demos.
- translation dims: `[0, 1, 2]`
- rotation dims: `[3, 4, 5]`
- gripper dim: `6`
- action min: `[-0.774107, -0.886607, -0.9375, -0.147857, -0.233571, -0.110357, -1.0]`
- action max: `[0.932143, 0.875893, 0.9375, 0.235714, 0.290357, 0.375, 1.0]`
- action mean: `[0.039349, 0.056619, -0.091883, 0.012416, -0.005815, 0.063391, -0.178072]`
- action std: `[0.262207, 0.308574, 0.370461, 0.035068, 0.05897, 0.102793, 0.984038]`
- action variance: `[0.068752, 0.095218, 0.137241, 0.00123, 0.003477, 0.010566, 0.968331]`
- gripper observed values: `[-1.0, 1.0]`
- action semantics audit: controller-delta-like, not absolute pose; no simulator/controller was instantiated

## Alignment Evidence

- 7D chunk shape: `[50, 7]`
- chunk first action matches HDF5 action at observation timestep: yes
- chunk second action matches HDF5 action at next timestep: yes
- off-by-one detected: no
- action chunks reduced to 6D in fixed path: no

## Overfit And Capacity Status

- one-sample overfit passed: yes
- one-sample action L2: `0.0`
- one-sample gripper accuracy: `1.0`
- one-demo overfit passed: yes
- one-demo action L2: `0.002593`
- one-demo gripper accuracy: `1.0`
- previous split mean-action L2: `0.486561`
- previous split fixed 7D adapter L2: `0.353069`
- larger split mean-action L2: `1.082453`
- larger split fixed 7D adapter L2: `0.573503`
- larger split best MLP/ridge L2: `0.518738`
- frozen/base SmolVLA L2 from previous 6D run: `1.6029`
- fixed 7D adapter beats mean-action on larger split: yes
- fixed 7D adapter beats frozen/base metric: yes
- small MLP baseline is still slightly stronger than the 128-hidden adapter on larger held-out L2

## Conclusion

The 7D action-interface blocker is cleared for baseline work: labels remain 7D, train-only normalization is used, the gripper is learned, and one-sample/one-demo overfit pass. This is not a paper result. The next valid step is standard fixed-interface SmolVLA/LIBERO 7D baseline reproduction on an official or standard split before any new RA-L method is proposed.
