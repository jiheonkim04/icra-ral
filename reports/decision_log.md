# Decision Log

## 2026-07-08: Research Reset And Target-Grounded ActionMap Scout

Decision: `NEED_ACTIONMAP_ANCHOR_REPRO_FIRST`

Reason: the only salvageable family was Target-Prior TCA reframed as Target-Grounded ActionMap / Language-Grounded Action Heatmap, but the local ActionMap substrate had not cleared mean-action, linear/L1, and cheap-MLP gates.

## 2026-07-08: ActionMap Mini-Anchor Gate

Decision: `KILL_ACTIONMAP_ANCHOR`

Reason: the bounded LIBERO/HDF5 mini-anchor produced real metrics but failed the hard gate.

Key metrics:

- dataset/split: `8` local LIBERO HDF5 demos, `deterministic_per_demo_time_holdout`
- train/eval records: `1008 / 432`
- mean-action action L2: `0.466767673`
- linear/L1 action L2: `0.812610317`
- simple MLP action L2: `0.501926707`
- ActionMap-style action L2: `0.529931357`
- oracle candidate action L2: `0.065653208`
- candidate top1: `0.018518519`
- candidate collapse: yes, unique translation/rotation/gripper bins `5 / 1 / 2`

Triggered kill criteria:

- mean-action baseline matched or beat the ActionMap-style heatmap head;
- cheap MLP action head matched or beat the ActionMap-style heatmap head;
- ActionMap-style head collapsed to too few candidates.

Consequence: do not proceed to Target-Grounded ActionMap from this local anchor result.

Interpretation: this kills the local minimal ActionMap approximation, not the official ActionMap paper. The low oracle candidate upper bound suggests candidate-space headroom, but the learned local head collapsed and did not exploit it.

## 2026-07-08: Official Anchor Required

Decision: `OFFICIAL_ANCHOR_REQUIRED`

Reason: local proxy and minimal approximations have repeatedly produced plausible auxiliary evidence while failing simple-baseline gates. No new VLA method should be started without an official anchor reproduction.

Only viable next steps:

A. Official ActionMap reproduction with official code/assets.

B. Official LIBERO-Safety/SafeManip benchmark reproduction.

C. Stop VLA method search under current constraints.

Execution boundary for this archive pass:

- experiments happened: no;
- training happened: no;
- rollout/replay happened: no;
- downloads/GPU/OpenVLA-OFT happened: no / no / no;
- Target-Grounded ActionMap implementation happened: no;
- new method implementation happened: no.

## 2026-07-09: PatchGuard-VLA STATE 1B Decision

Decision: `KILL_BASELINE_DOMINATED`

Reason: STATE 1B resolved the prior installable environment blocker and proved that local PEFT/bitsandbytes/CUDA/SmolVLA LoRA can run, but PatchGuard did not beat the predeclared baselines.

Key positive evidence:

- patch effect measured in STATE 1: max attacked policy-action L1 `0.181765`;
- max attacked translation-action L2 `0.213965`;
- kinematic/proprioceptive signal available;
- PEFT `0.19.1` installed and worked;
- bitsandbytes `0.49.2` installed and 4-bit/8-bit CUDA smokes passed;
- CUDA/PyTorch on RTX 5080 worked;
- SmolVLA LoRA injection worked;
- tiny training smoke ran;
- loss computed;
- VRAM peak `2224.845` MB;
- runtime `57.438` sec.

Decisive negative evidence:

- standard LoRA metric `0.144186`;
- generic adversarial LoRA metric `0.142803`;
- PatchGuard metric `0.13356`;
- cutout/random-erasing metric `0.02973`;
- PatchGuard did not beat generic adversarial LoRA under the archive decision criterion;
- PatchGuard did not beat cutout/random-erasing;
- PatchGuard did not beat both generic adversarial LoRA and cutout/random-erasing.

Consequence: kill PatchGuard-VLA as the current RA-L method route. Do not proceed to PatchGuard STATE 2 or more PatchGuard training.

Interpretation: this kills the PatchGuard method claim, not the LoRA environment. The next valid step is standard SmolVLA LoRA baseline reproduction on an official or standard task split.

Execution boundary for this archive pass:

- experiments happened: no;
- training happened: no;
- GPU job happened: no;
- rollout/replay happened: no;
- downloads happened: no;
- OpenVLA-OFT happened: no;
- new method implementation happened: no;
- paper claims happened: no.

## 2026-07-09: SmolVLA LoRA Baseline STATE 1

Decision: `KILL_MEAN_BASELINE_DOMINATED`

Reason: standard PEFT LoRA training ran and loss decreased, but held-out eval action L2 did not beat the mean-action baseline.

Execution boundary:

- experiments happened: yes, one bounded standard LoRA baseline;
- training happened: yes, rank-4 LoRA only;
- GPU happened: yes, RTX 5080 CUDA;
- downloads happened: no;
- rollout/replay happened: no;
- OpenVLA-OFT happened: no;
- new method implementation happened: no;
- paper claims happened: no.

Dataset and split:

- model: `C:\assets\checkpoints\smolvla`;
- HDF5: `C:\assets\data\libero\libero_10\KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5`;
- split: `deterministic_demo_holdout`;
- train demos: `demo_0`, `demo_1`, `demo_2`;
- eval demos: `demo_3`, `demo_4`;
- train/eval records: `9 / 6`.

Key metrics:

- LoRA rank: `4`;
- trainable params: `9984`;
- loss start/end: `0.06359 / 0.008743`;
- loss decreased meaningfully: yes;
- VRAM peak MB: `1190.228`;
- runtime sec: `43.765`;
- mean-action eval action L2: `0.486561`;
- frozen/base SmolVLA eval action L2: `1.6029`;
- standard LoRA eval action L2: `0.940196`;
- LoRA beats frozen/base: yes;
- LoRA beats mean-action: no.

Consequence: do not start a method on top of this LoRA setup. First diagnose the baseline/action-interface issue or reproduce a stronger official-style standard LoRA baseline.

## 2026-07-09: SmolVLA LoRA Baseline Diagnosis

Decision: `ACTION_INTERFACE_BUG`

Reason: the mean-action dominance is not yet a valid LoRA-capacity conclusion. The diagnosis found an action-interface mismatch between local LIBERO labels and the SmolVLA checkpoint interface, and overfit sanity checks failed in action space.

Execution boundary:

- experiments happened: yes, bounded diagnosis only;
- training happened: yes, bounded LoRA overfit/capacity sanity checks only;
- loss computed: yes;
- GPU happened: yes, RTX 5080 CUDA;
- downloads happened: no;
- rollout/replay happened: no;
- OpenVLA-OFT happened: no;
- full benchmark happened: no;
- PatchGuard continued: no;
- new method implementation happened: no;
- paper claims happened: no.

Dataset and split evidence:

- previous split was `9 / 6` because it sampled three timesteps per demo over three train demos and two eval demos;
- records are sampled observation/action-window records;
- raw HDF5 demos: `50`;
- raw HDF5 timesteps: `13298`;
- larger deterministic demo-holdout split possible: `300 / 100`;
- same-demo time-holdout split possible: `80 / 40`;
- task holdout is feasible locally.

Action-interface evidence:

- HDF5 action dim: `7`;
- SmolVLA model action shape: `[6]`;
- policy preprocessor/postprocessor action shape: `[6]`;
- checkpoint action normalizer is SO100-style `MEAN_STD`;
- local LIBERO first-six action mean/std are small, roughly centered near zero;
- local/checkpoint mean mismatch reaches `6.881818` checkpoint standard deviations;
- gripper is synthesized by `ACTION_STRATEGY_GRIPPER_CLOSE`.

Sanity checks:

- label reconstruction sanity: passed;
- action chunk horizon alignment: passed;
- off-by-one in chunk builder: not detected;
- one-sample overfit: failed;
- one-demo overfit: failed.

Capacity metrics:

- mean-action action L2: `0.486561`;
- frozen/base action L2: `1.6029`;
- best LoRA action L2: `0.912258`;
- best small MLP/ridge action L2: `0.401848`;
- LoRA beats mean-action: no;
- LoRA beats small MLP/ridge: no;
- VRAM peak MB: `1189.167`;
- runtime sec: `112.797`.

Consequence: fix the SmolVLA/LIBERO action interface before any method work. No new paper method is allowed from this state.

## 2026-07-09: SmolVLA-LIBERO 7D Action Interface Fix

Decision: `READY_FOR_REAL_METHOD_AFTER_INTERFACE_FIX`

Reason: the local SmolVLA/LIBERO action-interface blocker was repaired for baseline work by adding an explicit LIBERO_7D adapter path with train-split-only 7D normalization and learned gripper output. Native SmolVLA remains a separate SO100-style 6D action schema; the fix does not pretend the native head is 7D.

Execution boundary:

- experiments happened: yes, bounded interface-fix diagnosis only;
- training happened: yes, small supervised 7D adapter and simple baselines only;
- loss computed: yes;
- GPU training happened: no;
- downloads happened: no;
- rollout/replay happened: no;
- OpenVLA-OFT happened: no;
- full benchmark happened: no;
- PatchGuard continued: no;
- new method implementation happened: no;
- paper claims happened: no.

Schema evidence:

- LIBERO labels are 7D throughout all demos;
- translation dims: `[0, 1, 2]`;
- rotation dims: `[3, 4, 5]`;
- gripper dim: `6`;
- SmolVLA native model/preprocessor/postprocessor action shape: `[6]`;
- fixed path output action shape: `[7]`;
- fixed path label action shape: `[7]`;
- fixed path normalization: train-split-only LIBERO 7D mean/std;
- fixed path gripper handling: learned 7th adapter output with separate normalized MSE loss;
- SO100 action normalizer used for LIBERO labels: no;
- eval labels used for training/normalization: no.

Alignment evidence:

- 7D chunk shape: `[50, 7]`;
- chunk first action matches HDF5 action at observation timestep;
- chunk second action matches HDF5 action at next timestep;
- off-by-one detected: no;
- action chunks reduced to 6D in fixed path: no.

Sanity checks:

- one-sample overfit: passed, action L2 `0.0`, gripper accuracy `1.0`;
- one-demo overfit: passed, action L2 `0.002593`, gripper accuracy `1.0`.

Capacity metrics:

- previous split mean-action action L2: `0.486561`;
- previous split fixed 7D adapter action L2: `0.353069`;
- larger split mean-action action L2: `1.082453`;
- larger split fixed 7D adapter action L2: `0.573503`;
- larger split best MLP/ridge action L2: `0.518738`;
- frozen/base SmolVLA action L2 from prior 6D run: `1.6029`;
- fixed 7D adapter beats larger-split mean-action: yes;
- fixed 7D adapter beats frozen/base: yes;
- small MLP baseline remains slightly stronger than the 128-hidden fixed adapter on larger held-out L2.

Consequence: the action interface is fixed enough for standard fixed-interface baseline reproduction. The next valid step is not a new paper method; it is an official or standard split SmolVLA/LIBERO 7D baseline with mean-action, ridge/MLP, frozen/base, and fixed-interface adapter comparisons preserved.

## 2026-07-09: SmolVLA-LIBERO 7D Standard Baseline Reproduction

Decision: `READY_FOR_RA_L_METHOD_ON_SMOLVLA_7D`

Reason: the fixed-interface rank-8 SmolVLA `state_proj` LoRA + learned LIBERO_7D adapter beat mean-action and the best ridge/MLP baseline on the primary held-out same-task demo split.

Execution boundary:

- experiments happened: yes, bounded fixed-interface baseline reproduction only;
- training happened: yes, small CPU supervised 7D adapters and LoRA-on-`state_proj` baselines only;
- loss computed: yes;
- GPU training happened: no;
- downloads happened: no;
- rollout/replay happened: no;
- OpenVLA-OFT happened: no;
- full benchmark happened: no;
- PatchGuard continued: no;
- new method implementation happened: no;
- paper claims happened: no;
- old broken 6D/SO100 action path used: no;
- hard-coded gripper fill used: no.

Split evidence:

- primary split: `same_task_demo_holdout`;
- task: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo`;
- train/eval records: `300 / 100`;
- train demos: `demo_0` through `demo_29`;
- eval demos: `demo_30` through `demo_39`;
- raw timesteps: `13298`;
- exact record leakage: no;
- demo leakage: no;
- task overlap: yes, by same-task design.

Additional split audits:

- same-task time holdout: `160 / 80`, no exact record leakage, same-demo temporal overlap risk from 50-step chunks;
- multi-task demo holdout: `150 / 60` across 3 local tasks, no exact record leakage, no demo leakage.

Baseline metrics on primary held-out split:

- global mean-action action L2: `1.082453`;
- per-task mean-action action L2: `1.082453`;
- previous-action persistence diagnostic action L2: `0.181765`;
- ridge action L2: `0.890603`;
- small MLP action L2: `0.518738`;
- frozen/base SmolVLA 7D linear adapter action L2: `0.890604`;
- SmolVLA 7D adapter without LoRA action L2: `0.561651`;
- SmolVLA `state_proj` LoRA rank 4 + 7D adapter action L2: `0.504675`;
- SmolVLA `state_proj` LoRA rank 8 + 7D adapter action L2: `0.494959`.

Best learned variant:

- name: `smolvla_state_proj_lora_rank8_7d_adapter`;
- train action L2: `0.28441`;
- eval action L2: `0.494959`;
- eval translation L2: `0.230133`;
- eval rotation L2: `0.064995`;
- eval gripper error: `0.365736`;
- eval gripper accuracy: `0.88`;
- eval per-dim MAE: `[0.100928, 0.103205, 0.138828, 0.017847, 0.040066, 0.034002, 0.365735]`;
- trainable params: `131975`;
- LoRA rank: `8`;
- VRAM peak: `0.0` MB;
- runtime sec: `9.438`.

Target-module audit:

- executable fixed-7D target modules: `libero_7d_adapter_head_only`, `frozen_state_proj_plus_7d_adapter`, `state_proj_lora_plus_7d_adapter`;
- audited but not executed: `action_in_proj`, `action_out_proj`, `action_time_mlp_in`, `action_time_mlp_out`;
- reason: native action projection modules require `max_action_dim` / native flow actions and would re-enter the old 6D/SO100 action path.

Caveats:

- previous-action persistence is a diagnostic oracle because it uses the previous expert action from the held-out HDF5 sequence;
- optional replay/progress was eligible by action metrics but not run because no bounded executable LIBERO bridge for the learned 7D adapter is part of this runner;
- this is not a paper result or benchmark claim.

Consequence: future method planning may start only after preserving this fixed-interface baseline table and predeclaring comparisons against rank-8 fixed-interface SmolVLA 7D LoRA/adapter, mean-action, ridge/MLP, frozen/base 7D adapter, no-LoRA 7D adapter, and persistence diagnostics where appropriate.

## 2026-07-09: TG-7D Adapter STATE 0-2 Gate

Decision: `KILL_CANONICALIZATION_DOMINATED`

Candidate: Target-Grounded 7D Adapter for SmolVLA-LIBERO.

STATE 1 feasibility was green:

- local LIBERO-Para metadata rows: `4092`;
- matched local LIBERO-Goal HDF5 tasks: `10`;
- clean train/eval records: `120 / 60`;
- train paraphrase records: `480`;
- held-out paraphrase records: `360`;
- held-out object lexical records: `60`;
- counterfactual records: `30`;
- paraphrase group leakage: no;
- target prior source: instruction text plus HDF5 model-XML visible object-candidate names;
- BDDL/eval labels/task IDs/filenames used as inference labels: no.

STATE 2 bounded fixed-7D method gate:

- LoRA rank: `4`;
- mean-action held-out paraphrase action L2: `0.903848`;
- MLP held-out paraphrase action L2: `0.619985`;
- standard SmolVLA 7D LoRA/adapter held-out paraphrase action L2: `0.600887`;
- canonicalization-only held-out paraphrase action L2: `0.587661`;
- simple paraphrase augmentation held-out paraphrase action L2: `0.739425`;
- TG-7D Adapter held-out paraphrase action L2: `0.740922`;
- oracle target upper bound held-out paraphrase action L2: `0.724674`;
- TG-7D clean action L2: `0.735738`;
- standard LoRA clean action L2: `0.600887`;
- TG-7D counterfactual prediction delta L2: `0.06286`;
- TG-7D counterfactual collapse rate below `0.05`: `0.5`;
- TG-7D trainable params: `295623`;
- VRAM peak: `0.0` MB;
- runtime: `14.093` sec.

Exact kill criterion triggered: canonicalization-only matched or beat TG-7D on the target/paraphrase metric. Standard LoRA and MLP also beat TG-7D, and clean action quality was worse than standard LoRA.

Consequence: do not scale TG-7D Adapter as formulated. Preserve the leakage-safe LIBERO-Para/fixed-7D split and target-prior audit as reusable artifacts.

## 2026-07-09: Post-Canonicalization Residual Mining

Decision: `KILL_CANONICALIZATION_DOMINATED`

Scope:

- used existing TG-7D gate metrics and reconstructed split/group metadata only;
- experiments happened: no new experiment;
- training happened: no;
- downloads/GPU/OpenVLA-OFT/rollout happened: no / no / no / no.

Residual findings:

- canonicalization held-out paraphrase L2: `0.587661`;
- canonicalization residual versus best non-oracle target/language arm: `-0.013226`;
- canonicalization clean-to-paraphrase delta: `-0.000748`;
- object lexical canonicalization L2: `0.587388`;
- largest absolute residual subgroup: gripper error `0.389255`;
- residual structured as method-worthy target/language failure: no;
- standard LoRA/MLP already solve within margin: yes;
- oracle/headroom evidence: no (`oracle_headroom_l2 = -0.137013`).

Consequence: stop the local language/target route. Do not start TG-7D v2 or another language-target method without a new official or clearly named benchmark slice where canonicalization-only still has a large structured residual, simple baselines do not solve it, and oracle/headroom evidence is positive.
