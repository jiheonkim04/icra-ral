# Project State

Date: 2026-07-09 KST

Branch:

`codex/smolvla-lora-baseline-diagnosis`

Current branch base:

`9faa030 Run SmolVLA LoRA baseline gate`

Current decision:

`ACTION_INTERFACE_BUG`

## Current Bounded Run Boundary

- Experiments happened: yes, bounded diagnosis only.
- Training happened: yes, bounded LoRA overfit/capacity sanity checks only.
- Loss computed: yes.
- GPU happened: yes, RTX 5080 CUDA.
- Downloads happened: no.
- Rollout/replay happened: no.
- OpenVLA-OFT happened: no.
- Full benchmark happened: no.
- PatchGuard continued: no.
- New method implementation happened: no.
- Paper claims happened: no.

## Dataset And Split Status

The previous `9 / 6` split came from sampling three timesteps per demo over three train demos and two eval demos.

Diagnosis found:

- records are sampled observation/action-window records, not full demos,
- raw HDF5 demos: `50`,
- raw HDF5 timesteps: `13298`,
- larger deterministic demo-holdout split possible: `300 / 100`,
- same-demo time-holdout split possible: `80 / 40`,
- task holdout is feasible without download within the local suite.

## Action Variance And Mean-Action Strength

Raw action variance:

- per-dim variance: `[0.068752, 0.095218, 0.137241, 0.00123, 0.003477, 0.010566, 0.968331]`
- translation variance mean: `0.100404`
- rotation variance mean: `0.005091`
- gripper variance: `0.96829`

The mean-action baseline is strong on the previous split:

- mean-action action L2: `0.486561`
- previous-action action L2: `0.188748`

## Interface Audit Result

The local action interface is not proven correct. It is contradicted by the diagnosis:

- HDF5 action dim: `7`
- SmolVLA model action shape: `[6]`
- policy preprocessor action shape: `[6]`
- policy postprocessor action shape: `[6]`
- checkpoint action normalization: `ACTION: MEAN_STD`
- local LIBERO action first-six mean: `[0.039349, 0.056619, -0.091883, 0.012416, -0.005815, 0.063391]`
- local LIBERO action first-six std: `[0.262207, 0.308574, 0.370461, 0.035068, 0.05897, 0.102793]`
- checkpoint normalizer is SO100-style and differs by up to `6.881818` mean-std units.
- gripper is synthesized by `ACTION_STRATEGY_GRIPPER_CLOSE` instead of learned by the 6D action head.

Label reconstruction and chunk alignment passed:

- chunk first action matches HDF5 action at observation timestep,
- chunk second action matches the next HDF5 action,
- no off-by-one was detected in the local chunk builder.

## Overfit And Capacity Status

- one-sample overfit passed: no
- one-demo overfit passed: no
- mean-action metric: `0.486561`
- frozen/base metric: `1.6029`
- best LoRA metric: `0.912258`
- best LoRA variant: `current_projection_lora`
- best small MLP/ridge metric: `0.401848`
- best small MLP/ridge: `state_time_mlp`
- LoRA beats mean-action: no
- LoRA beats small MLP/ridge: no
- VRAM peak MB: `1189.167`
- runtime sec: `112.797`

## Conclusion

Do not start a new RA-L method. Fix the SmolVLA/LIBERO action interface first. The current evidence points to an action dimension, normalization, and gripper-interface bug rather than a valid method substrate.
