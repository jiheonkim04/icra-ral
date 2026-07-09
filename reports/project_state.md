# Project State

Date: 2026-07-09 KST

Branch:

`codex/tg7d-adapter-state0-state1`

Current branch base:

`bb0c372 Reproduce SmolVLA LIBERO 7D baseline`

Current decision:

`KILL_CANONICALIZATION_DOMINATED`

## Baseline Foundation

The fixed SmolVLA/LIBERO 7D baseline remains the reusable foundation:

- fixed `LIBERO_7D` labels,
- train-split-only 7D normalization,
- learned gripper output,
- no old 6D/SO100 action label path,
- no hard-coded gripper fill,
- best baseline: rank-8 `state_proj` LoRA + 7D adapter action L2 `0.494959`,
- mean-action action L2 `1.082453`,
- ridge / MLP action L2 `0.890603 / 0.518738`,
- frozen/base SmolVLA 7D adapter action L2 `0.890604`.

This baseline still authorizes method planning in general, but each candidate must survive its own strong-baseline gate.

## TG-7D Adapter Gate

Candidate:

`Target-Grounded 7D Adapter for SmolVLA-LIBERO`

Boundary:

- Experiments happened: yes, bounded local method gate only.
- Training happened: yes, tiny CPU rank-4 fixed-7D adapter variants only.
- Loss computed: yes.
- GPU training happened: no.
- Downloads happened: no.
- Rollout/replay happened: no.
- OpenVLA-OFT happened: no.
- Full benchmark happened: no.
- Old TCA-Select used: no.
- Old broken 6D/SO100 path used: no.
- Hard-coded gripper fill used: no.
- Paper claims happened: no.

## STATE 1 Feasibility

STATE 1 found a meaningful local evaluation path:

- local LIBERO-Para metadata rows: `4092`,
- original instruction count: `10`,
- matched local LIBERO-Goal HDF5 task count: `10`,
- selected task count: `10`,
- clean train/eval records: `120 / 60`,
- train paraphrase records: `480`,
- held-out paraphrase records: `360`,
- held-out object lexical records: `60`,
- counterfactual records: `30`,
- paraphrase group leakage: no,
- clean train/eval exact record overlap: `0`,
- target prior source: instruction text plus HDF5 model-XML visible object-candidate names,
- BDDL/eval labels/task IDs/filenames used as inference labels: no.

## STATE 2 Method Gate Results

Dataset/split:

`local_libero_goal_libero_para_group_holdout`

LoRA rank:

`4`

Held-out paraphrase action L2:

- mean-action: `0.903848`,
- MLP: `0.619985`,
- standard SmolVLA 7D LoRA/adapter: `0.600887`,
- canonicalization-only: `0.587661`,
- simple paraphrase augmentation: `0.739425`,
- TG-7D Adapter: `0.740922`,
- oracle target upper bound: `0.724674`.

TG-7D details:

- clean action L2: `0.735738`,
- object lexical action L2: `0.744749`,
- target consistency same-target prediction L2: `0.017795`,
- counterfactual prediction delta L2: `0.06286`,
- counterfactual collapse rate below `0.05`: `0.5`,
- trainable params: `295623`,
- VRAM peak: `0.0` MB,
- runtime: `14.093` sec.

## Post-Canonicalization Residual Mining

The residual mining run used the archived TG-7D metrics and reconstructed only split/group metadata. It did not train, download, run GPU, run OpenVLA-OFT, or run rollout.

Residual findings:

- canonicalization residual size: `0.587661`,
- canonicalization residual versus best non-oracle target/language arm: `-0.013226`,
- canonicalization clean-to-paraphrase delta: `-0.000748`,
- largest absolute residual subgroup: gripper error `0.389255`,
- residual structured as method-worthy target/language failure: no,
- standard LoRA or MLP already solves the residual within margin: yes,
- oracle/headroom exists: no (`oracle_headroom_l2 = -0.137013`).

## Conclusion

TG-7D Adapter is killed as formulated, and the local language/target robustness family has no method-worthy post-canonicalization residual in this artifact.

Exact route-level decision:

`KILL_CANONICALIZATION_DOMINATED`

Do not scale TG-7D or start another language-target method from this evidence. The reusable artifact is the leakage-safe LIBERO-Para/fixed-7D split and target-prior audit, not the method.
