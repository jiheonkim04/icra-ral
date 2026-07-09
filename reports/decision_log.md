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

## 2026-07-09: SmolVLA 7D Adapter Replay Bridge

Decision: `READY_FOR_METHOD_AFTER_REPLAY_BRIDGE`

- experiments happened: `True`
- training happened: `False`
- loss computed: `False`
- replay/control happened: `True`
- model/adapter used: `smolvla_state_proj_lora_rank8_7d_adapter`
- dataset/demo used: `C:\assets\data\libero\libero_10\KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5::demo_30`
- env acceptance status: `accepted_by_env_step`
- expert replay: reward_sum `1.0`, success `True`, first_done_index `250`
- mean/ridge/adapter progress proxy: `0.106222` / `0.167573` / `0.234297`
- adapter clip/controller-valid proxy: `0.429119` / `0.570881`
- exact next step: reproduce a real SmolVLA LoRA baseline on an official or standard task split before starting any new method.
## 2026-07-09: SmolVLA 7D Standard Replay Baseline

Decision: `EXPERT_REPLAY_UNSTABLE`

- experiments happened: `True`
- training happened: `True`
- loss computed: `True`
- replay/control happened: `True`
- model/adapter used: `smolvla_state_proj_lora_rank4_7d_adapter`
- tasks/demos used: `{'tasks': ['KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo'], 'replay_cases': [{'hdf5_path': 'C:\\assets\\data\\libero\\libero_10\\KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5', 'demo_name': 'demo_7', 'task_name': 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo'}, {'hdf5_path': 'C:\\assets\\data\\libero\\libero_10\\KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5', 'demo_name': 'demo_5', 'task_name': 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo'}]}`
- expert aggregate: `{'case_count': 2, 'success_count': 1, 'success_rate': 0.5, 'reward_sum_mean': 0.5, 'first_done_indices': [None, 225], 'progress_proxy_mean': 0.21065, 'object_movement_mean': 0.170842, 'runtime_case_steps': [272, 226]}`
- mean/ridge/MLP/adapter progress: `{'mean_action': 0.068504, 'ridge': 0.154496, 'small_mlp': 0.060931, 'smolvla_7d_adapter': -0.055137}`
- action validity: `{'action_shape': [32, 7], 'expected_action_shape': ['T', 7], 'shape_exactly_7d': True, 'finite': True, 'action_low_high': {'low': [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], 'high': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'min': [-0.451046, -0.205794, -0.466078, -0.035002, -0.07305, -0.255751, -1.161312], 'max': [0.317361, 0.439485, 0.508266, 0.061366, 0.062319, 0.171873, 1.099052]}, 'clip_rate_element': 0.026786, 'clip_rate_step': 0.1875, 'controller_valid_rate_proxy': 0.8125, 'silent_broadcast_or_truncation_detected': False, 'note': 'Proxy validity uses LIBERO HDF5/controller action convention [-1, 1]; env acceptance is reported separately when replay runs.', 'per_dim_clip_rate': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1875], 'dominant_clip_dim': 6, 'gripper_clip_rate': 0.1875}`
- exact next step: Fix or narrow exact-init replay until expert succeeds on every evaluated replay case.
## 2026-07-09: Exact-Init Expert Replay Stabilization

Decision: `OFFLINE_TO_CONTROL_GAP`

- experiments happened: `True`
- training happened: `False`
- replay/control happened: `True`
- learned policy replay happened: `True`
- candidate demos tested: `['KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_7', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_8', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_30', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_31', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_5', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_6', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_7', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_8']`
- expert-success eligible cases: `['KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_8', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_30', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_31', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_5', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_7', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_8']`
- expert-failed cases and reasons: `[{'task_name': 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo', 'demo_name': 'demo_7', 'hdf5_path': 'C:\\assets\\data\\libero\\libero_10\\KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5', 'failure_reason': 'controller_or_action_convention_mismatch_suspected', 'eligibility_reasons': ['expert replay did not reach reward, done, or final success', 'reward did not reach 1.0 and final success was not true', 'finite done index missing'], 'hdf5_first_signal_index': 271, 'steps_performed': 272, 'reward_sum': 0.0, 'first_done_index': None}, {'task_name': 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo', 'demo_name': 'demo_6', 'hdf5_path': 'C:\\assets\\data\\libero\\libero_10\\KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5', 'failure_reason': 'controller_or_action_convention_mismatch_suspected', 'eligibility_reasons': ['expert replay did not reach reward, done, or final success', 'reward did not reach 1.0 and final success was not true', 'finite done index missing'], 'hdf5_first_signal_index': 240, 'steps_performed': 241, 'reward_sum': 0.0, 'first_done_index': None}]`
- mean/ridge/MLP/adapter replay: `{'case_count': 6, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None, None, None, None, None, None], 'progress_proxy_mean': 0.038336, 'object_movement_mean': 0.000125, 'runtime_case_steps': [275, 261, 228, 237, 234, 258]}` / `{'case_count': 6, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None, None, None, None, None, None], 'progress_proxy_mean': 0.040788, 'object_movement_mean': 0.000286, 'runtime_case_steps': [275, 261, 228, 237, 234, 258]}` / `{'case_count': 0, 'success_count': 0, 'success_rate': None, 'reward_sum_mean': None, 'first_done_indices': [], 'progress_proxy_mean': None, 'object_movement_mean': None, 'runtime_case_steps': []}` / `{'case_count': 6, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None, None, None, None, None, None], 'progress_proxy_mean': -0.059671, 'object_movement_mean': 0.000125, 'runtime_case_steps': [275, 261, 228, 237, 234, 258]}`
- action validity audit: `{'case_rows': [{'task_name': 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo', 'demo_name': 'demo_8', 'policy': 'mean_action', 'clip_rate_element': 0.0, 'clip_rate_step': 0.0, 'controller_valid_rate_proxy': 1.0, 'dominant_clip_dim': 0, 'gripper_clip_rate': 0.0}, {'task_name': 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo', 'demo_name': 'demo_8', 'policy': 'ridge', 'clip_rate_element': 0.009351, 'clip_rate_step': 0.065455, 'controller_valid_rate_proxy': 0.934545, 'dominant_clip_dim': 6, 'gripper_clip_rate': 0.065455}, {'task_name': 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo', 'demo_name': 'demo_8', 'policy': 'smolvla_7d_adapter', 'clip_rate_element': 0.025455, 'clip_rate_step': 0.178182, 'controller_valid_rate_proxy': 0.821818, 'dominant_clip_dim': 6, 'gripper_clip_rate': 0.178182}, {'task_name': 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo', 'demo_name': 'demo_30', 'policy': 'mean_action', 'clip_rate_element': 0.0, 'clip_rate_step': 0.0, 'controller_valid_rate_proxy': 1.0, 'dominant_clip_dim': 0, 'gripper_clip_rate': 0.0}, {'task_name': 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo', 'demo_name': 'demo_30', 'policy': 'ridge', 'clip_rate_element': 0.002189, 'clip_rate_step': 0.015326, 'controller_valid_rate_proxy': 0.984674, 'dominant_clip_dim': 6, 'gripper_clip_rate': 0.015326}, {'task_name': 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo', 'demo_name': 'demo_30', 'policy': 'smolvla_7d_adapter', 'clip_rate_element': 0.015873, 'clip_rate_step': 0.111111, 'controller_valid_rate_proxy': 0.888889, 'dominant_clip_dim': 6, 'gripper_clip_rate': 0.111111}, {'task_name': 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo', 'demo_name': 'demo_31', 'policy': 'mean_action', 'clip_rate_element': 0.0, 'clip_rate_step': 0.0, 'controller_valid_rate_proxy': 1.0, 'dominant_clip_dim': 0, 'gripper_clip_rate': 0.0}, {'task_name': 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo', 'demo_name': 'demo_31', 'policy': 'ridge', 'clip_rate_element': 0.009398, 'clip_rate_step': 0.065789, 'controller_valid_rate_proxy': 0.934211, 'dominant_clip_dim': 6, 'gripper_clip_rate': 0.065789}, {'task_name': 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo', 'demo_name': 'demo_31', 'policy': 'smolvla_7d_adapter', 'clip_rate_element': 0.048872, 'clip_rate_step': 0.342105, 'controller_valid_rate_proxy': 0.657895, 'dominant_clip_dim': 6, 'gripper_clip_rate': 0.342105}, {'task_name': 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo', 'demo_name': 'demo_5', 'policy': 'mean_action', 'clip_rate_element': 0.0, 'clip_rate_step': 0.0, 'controller_valid_rate_proxy': 1.0, 'dominant_clip_dim': 0, 'gripper_clip_rate': 0.0}, {'task_name': 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo', 'demo_name': 'demo_5', 'policy': 'ridge', 'clip_rate_element': 0.006631, 'clip_rate_step': 0.046414, 'controller_valid_rate_proxy': 0.953586, 'dominant_clip_dim': 6, 'gripper_clip_rate': 0.046414}, {'task_name': 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo', 'demo_name': 'demo_5', 'policy': 'smolvla_7d_adapter', 'clip_rate_element': 0.030139, 'clip_rate_step': 0.21097, 'controller_valid_rate_proxy': 0.78903, 'dominant_clip_dim': 6, 'gripper_clip_rate': 0.21097}, {'task_name': 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo', 'demo_name': 'demo_7', 'policy': 'mean_action', 'clip_rate_element': 0.0, 'clip_rate_step': 0.0, 'controller_valid_rate_proxy': 1.0, 'dominant_clip_dim': 0, 'gripper_clip_rate': 0.0}, {'task_name': 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo', 'demo_name': 'demo_7', 'policy': 'ridge', 'clip_rate_element': 0.020757, 'clip_rate_step': 0.145299, 'controller_valid_rate_proxy': 0.854701, 'dominant_clip_dim': 6, 'gripper_clip_rate': 0.145299}, {'task_name': 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo', 'demo_name': 'demo_7', 'policy': 'smolvla_7d_adapter', 'clip_rate_element': 0.056777, 'clip_rate_step': 0.397436, 'controller_valid_rate_proxy': 0.602564, 'dominant_clip_dim': 6, 'gripper_clip_rate': 0.397436}, {'task_name': 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo', 'demo_name': 'demo_8', 'policy': 'mean_action', 'clip_rate_element': 0.0, 'clip_rate_step': 0.0, 'controller_valid_rate_proxy': 1.0, 'dominant_clip_dim': 0, 'gripper_clip_rate': 0.0}, {'task_name': 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo', 'demo_name': 'demo_8', 'policy': 'ridge', 'clip_rate_element': 0.012182, 'clip_rate_step': 0.085271, 'controller_valid_rate_proxy': 0.914729, 'dominant_clip_dim': 6, 'gripper_clip_rate': 0.085271}, {'task_name': 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo', 'demo_name': 'demo_8', 'policy': 'smolvla_7d_adapter', 'clip_rate_element': 0.022702, 'clip_rate_step': 0.158915, 'controller_valid_rate_proxy': 0.841085, 'dominant_clip_dim': 6, 'gripper_clip_rate': 0.158915}], 'adapter_clip_rate_step_mean': 0.23312, 'adapter_controller_valid_rate_proxy_mean': 0.76688, 'adapter_action_validity_fix_needed': False, 'mlp_replay_executed': False, 'mlp_skip_reason': 'No persisted executable MLP artifact exists; no MLP retraining was performed.'}`
- exact next step: Stop method work; diagnose the offline-to-control gap in the fixed SmolVLA 7D baseline on the eligible set before proposing any new method.
## 2026-07-09: SmolVLA 7D Offline-To-Control Gap Diagnosis

Decision: `FEATURE_PATH_MISMATCH`

- experiments happened: `True`
- training happened: `False`
- replay/control happened: `True`
- eligible demos used: `['KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_8', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_30', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_31', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_5', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_7', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_8']`
- feature path audit result: `FEATURE_PATH_MISMATCH_FOR_TRUE_CLOSED_LOOP`
- teacher-forced sequence result: `{'mean_action': {'case_count': 6, 'action_l2_mean': 1.073821, 'translation_l2_mean': 0.512406, 'rotation_l2_mean': 0.101629, 'gripper_error_mean': 0.878721, 'first20_action_l2_mean': 0.724121, 'phase_critical_error_ratio_mean': 1.378987, 'gripper_timing_error_values': [None, None, None, None, None, None], 'gripper_sign_agreement_mean': 0.642681, 'translation_cosine_mean': 0.306502, 'translation_cosine_negative_rate_mean': 0.217218}, 'ridge': {'case_count': 6, 'action_l2_mean': 0.949596, 'translation_l2_mean': 0.495662, 'rotation_l2_mean': 0.094938, 'gripper_error_mean': 0.704616, 'first20_action_l2_mean': 0.429934, 'phase_critical_error_ratio_mean': 1.344207, 'gripper_timing_error_values': [120, None, None, 2, 48, 24], 'gripper_sign_agreement_mean': 0.721458, 'translation_cosine_mean': 0.280931, 'translation_cosine_negative_rate_mean': 0.311952}, 'smolvla_7d_adapter': {'case_count': 6, 'action_l2_mean': 0.867437, 'translation_l2_mean': 0.494638, 'rotation_l2_mean': 0.089128, 'gripper_error_mean': 0.628795, 'first20_action_l2_mean': 0.417105, 'phase_critical_error_ratio_mean': 1.35718, 'gripper_timing_error_values': [-6, -3, 3, -21, 26, 6], 'gripper_sign_agreement_mean': 0.778529, 'translation_cosine_mean': 0.321581, 'translation_cosine_negative_rate_mean': 0.265838}, 'critical_question': {'low_sparse_offline_l2_hides_full_sequence_or_phase_error': True, 'adapter_teacher_forced_l2_worse_than_ridge': False}}`
- open-loop action replay result: `{'eligible_case_count': 6, 'expert': {'case_count': 6, 'first_done_indices': [259, 250, 215, 225, 222, 245], 'object_movement_mean': 0.272579, 'progress_proxy_mean': 0.246324, 'reward_sum_mean': 1.0, 'runtime_case_steps': [260, 251, 216, 226, 223, 246], 'success_count': 6, 'success_rate': 1.0}, 'learned_aggregate_uses_only_eligible_cases': True, 'mean_action': {'case_count': 6, 'first_done_indices': [None, None, None, None, None, None], 'object_movement_mean': 0.000125, 'progress_proxy_mean': 0.038336, 'reward_sum_mean': 0.0, 'runtime_case_steps': [275, 261, 228, 237, 234, 258], 'success_count': 0, 'success_rate': 0.0}, 'ridge': {'case_count': 6, 'first_done_indices': [None, None, None, None, None, None], 'object_movement_mean': 0.000286, 'progress_proxy_mean': 0.040788, 'reward_sum_mean': 0.0, 'runtime_case_steps': [275, 261, 228, 237, 234, 258], 'success_count': 0, 'success_rate': 0.0}, 'small_mlp': {'case_count': 0, 'first_done_indices': [], 'object_movement_mean': None, 'progress_proxy_mean': None, 'reward_sum_mean': None, 'runtime_case_steps': [], 'success_count': 0, 'success_rate': None}, 'smolvla_7d_adapter': {'case_count': 6, 'first_done_indices': [None, None, None, None, None, None], 'object_movement_mean': 0.000125, 'progress_proxy_mean': -0.059671, 'reward_sum_mean': 0.0, 'runtime_case_steps': [275, 261, 228, 237, 234, 258], 'success_count': 0, 'success_rate': 0.0}}`
- closed-loop divergence result: `{'executed': False, 'reason': 'Skipped as a model-quality measurement because STATE 1 found live closed-loop feature mismatch.', 'blocked_by': 'FEATURE_PATH_MISMATCH_FOR_TRUE_CLOSED_LOOP', 'required_before_rerun': 'Provide live env features matching HDF5 ee_states (ee_pos + ee_ori) or retrain/evaluate with the live observation schema.'}`
- oracle diagnostic result: `{'adapter_motion_error_first6_mean': 0.494638, 'adapter_rotation_l2_mean': 0.089128, 'adapter_gripper_error_mean': 0.628795, 'gripper_oracle_alone_unlikely_to_fix_motion': True}`
- failure category: `FEATURE_PATH_MISMATCH`
- exact next step: Fix the live closed-loop feature schema so replay uses HDF5-compatible ee_states features, then rerun teacher-forced and replay diagnostics before any method work.
## 2026-07-09: SmolVLA 7D Live Feature Schema Fix

Decision: `ACTION_VALIDITY_RANGE_FAILURE`

- experiments happened: `True`
- training happened: `False`
- replay/control happened: `True`
- eligible demos used: `['KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_8', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_30', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo::demo_31', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_5', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_7', 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo::demo_8']`
- selected orientation conversion: `xyzw_quaternion_axis_angle_0_to_2pi`
- feature L2 before/after: `2.248343` / `0.033195`
- teacher-forced before/after: `{'case_count': 6, 'action_l2_mean': 2.285072, 'translation_l2_mean': 0.943406, 'rotation_l2_mean': 0.24099, 'gripper_error_mean': 1.95543, 'first20_action_l2_mean': 1.505044, 'phase_critical_error_ratio_mean': 1.557651, 'top_error_timestep_examples': [{'timestep': 87, 'action_l2': 4.844927, 'translation_l2': 2.068367, 'rotation_l2': 0.663889, 'gripper_abs_error': 4.330638, 'pred_gripper': -3.330638, 'expert_gripper': 1.0}, {'timestep': 88, 'action_l2': 4.796395, 'translation_l2': 1.964477, 'rotation_l2': 0.683836, 'gripper_abs_error': 4.321875, 'pred_gripper': -3.321875, 'expert_gripper': 1.0}, {'timestep': 97, 'action_l2': 4.811155, 'translation_l2': 1.776217, 'rotation_l2': 0.675985, 'gripper_abs_error': 4.419877, 'pred_gripper': -3.419877, 'expert_gripper': 1.0}, {'timestep': 135, 'action_l2': 3.994623, 'translation_l2': 1.141749, 'rotation_l2': 0.303345, 'gripper_abs_error': 3.81594, 'pred_gripper': -2.81594, 'expert_gripper': 1.0}, {'timestep': 144, 'action_l2': 3.911969, 'translation_l2': 0.931691, 'rotation_l2': 0.285715, 'gripper_abs_error': 3.788644, 'pred_gripper': -2.788644, 'expert_gripper': 1.0}, {'timestep': 157, 'action_l2': 4.115587, 'translation_l2': 1.05012, 'rotation_l2': 0.324395, 'gripper_abs_error': 3.966115, 'pred_gripper': -2.966115, 'expert_gripper': 1.0}], 'per_dim_mae_mean': [0.222851, 0.355751, 0.749697, 0.046065, 0.118613, 0.190425, 1.95543]}` / `{'case_count': 6, 'action_l2_mean': 0.843733, 'translation_l2_mean': 0.48049, 'rotation_l2_mean': 0.086464, 'gripper_error_mean': 0.616686, 'first20_action_l2_mean': 0.409785, 'phase_critical_error_ratio_mean': 1.42216, 'top_error_timestep_examples': [{'timestep': 89, 'action_l2': 1.893353, 'translation_l2': 1.199903, 'rotation_l2': 0.272044, 'gripper_abs_error': 1.439101, 'pred_gripper': -0.439101, 'expert_gripper': 1.0}, {'timestep': 251, 'action_l2': 2.113338, 'translation_l2': 0.496581, 'rotation_l2': 0.031235, 'gripper_abs_error': 2.05393, 'pred_gripper': -1.05393, 'expert_gripper': 1.0}, {'timestep': 216, 'action_l2': 2.456689, 'translation_l2': 1.047604, 'rotation_l2': 0.104703, 'gripper_abs_error': 2.219658, 'pred_gripper': -1.219658, 'expert_gripper': 1.0}, {'timestep': 53, 'action_l2': 1.36209, 'translation_l2': 0.610235, 'rotation_l2': 0.061259, 'gripper_abs_error': 1.216202, 'pred_gripper': 0.216202, 'expert_gripper': -1.0}, {'timestep': 70, 'action_l2': 1.314397, 'translation_l2': 0.270466, 'rotation_l2': 0.006591, 'gripper_abs_error': 1.286251, 'pred_gripper': -0.286251, 'expert_gripper': 1.0}, {'timestep': 154, 'action_l2': 1.573364, 'translation_l2': 0.665714, 'rotation_l2': 0.054659, 'gripper_abs_error': 1.424539, 'pred_gripper': -0.424539, 'expert_gripper': 1.0}], 'per_dim_mae_mean': [0.178238, 0.2239, 0.294855, 0.027278, 0.043426, 0.051627, 0.616686]}`
- replay result after fix: `{'expert': {'case_count': 6, 'success_count': 6, 'success_rate': 1.0, 'reward_sum_mean': 1.0, 'first_done_indices': [259, 250, 215, 225, 222, 245], 'progress_proxy_mean': 0.246324, 'object_movement_mean': 0.272579, 'runtime_case_steps': [260, 251, 216, 226, 223, 246]}, 'mean_action': {'case_count': 6, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None, None, None, None, None, None], 'progress_proxy_mean': 0.038336, 'object_movement_mean': 0.000125, 'runtime_case_steps': [275, 261, 228, 237, 234, 258]}, 'ridge': {'case_count': 6, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None, None, None, None, None, None], 'progress_proxy_mean': -0.173863, 'object_movement_mean': 0.000125, 'runtime_case_steps': [275, 261, 228, 237, 234, 258]}, 'smolvla_7d_adapter_fixed_live': {'case_count': 6, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None, None, None, None, None, None], 'progress_proxy_mean': -0.041091, 'object_movement_mean': 0.022539, 'runtime_case_steps': [275, 261, 228, 237, 234, 258]}}`
- exact next step: Feature fix works, but adapter action clipping/controller-validity remains too weak.
## 2026-07-09: SmolVLA 7D Action Range Fix

Decision: `CLIP_ONLY_BASELINE_DOMINATES`

- experiments happened: `True`
- training happened: `True`
- replay/control happened: `True`
- action range before/after: `{'low': [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], 'high': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'min': [-0.451046, -0.217334, -0.466079, -0.035002, -0.118721, -0.255753, -1.569224], 'max': [0.299245, 0.439486, 0.734157, 0.070848, 0.062318, 0.177517, 1.099053]}` / `{'low': [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], 'high': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'min': [-0.283834, -0.546853, -0.97641, -0.419788, -0.268929, -0.428587, -1.0], 'max': [0.617245, 0.721099, 0.789849, 0.267825, 0.291906, 0.27366, 1.0]}`
- clip rate before/after: `0.15625` / `0.0`
- controller-valid proxy before/after: `0.84375` / `1.0`
- offline metrics before/after: `{'sample_count': 32, 'action_l2': 0.795274, 'action_l2_first6': 0.493013, 'translation_l2': 0.479167, 'rotation_l2': 0.098805, 'gripper_error': 0.567381, 'gripper_accuracy': 0.84375, 'per_dim_mae': [0.165448, 0.224278, 0.319078, 0.032909, 0.051444, 0.053925, 0.567381], 'worst_action_dimensions': [{'dim': 6, 'mae': 0.567381}, {'dim': 2, 'mae': 0.319078}, {'dim': 1, 'mae': 0.224278}]}` / `{'sample_count': 32, 'action_l2': 0.976681, 'action_l2_first6': 0.608732, 'translation_l2': 0.550087, 'rotation_l2': 0.229188, 'gripper_error': 0.560213, 'gripper_accuracy': 0.71875, 'per_dim_mae': [0.213978, 0.244234, 0.352273, 0.099675, 0.119066, 0.126034, 0.560213], 'worst_action_dimensions': [{'dim': 6, 'mae': 0.560213}, {'dim': 2, 'mae': 0.352273}, {'dim': 1, 'mae': 0.244234}]}`
- offline baseline comparison: `{'mean_action': {'action_l2': 0.972739, 'translation_l2': 0.442767, 'rotation_l2': 0.098776, 'gripper_error': 0.814062, 'gripper_accuracy': 0.71875, 'clip_rate_step': 0.0, 'controller_valid_rate_proxy': 1.0, 'train_eval_gap': None}, 'ridge': {'action_l2': 0.854115, 'translation_l2': 0.417876, 'rotation_l2': 0.106551, 'gripper_error': 0.680647, 'gripper_accuracy': 0.84375, 'clip_rate_step': 0.125, 'controller_valid_rate_proxy': 0.875, 'train_eval_gap': None}, 'small_mlp': {'action_l2': 0.792409, 'translation_l2': 0.449542, 'rotation_l2': 0.104076, 'gripper_error': 0.580501, 'gripper_accuracy': 0.78125, 'clip_rate_step': 0.1875, 'controller_valid_rate_proxy': 0.8125, 'train_eval_gap': 0.164114}, 'previous_unfixed_adapter': {'action_l2': 0.795274, 'translation_l2': 0.479167, 'rotation_l2': 0.098805, 'gripper_error': 0.567381, 'gripper_accuracy': 0.84375, 'clip_rate_step': 0.15625, 'controller_valid_rate_proxy': 0.84375, 'train_eval_gap': 0.206301}, 'previous_unfixed_adapter_clip_only': {'action_l2': 0.785721, 'translation_l2': 0.479167, 'rotation_l2': 0.098805, 'gripper_error': 0.5381, 'gripper_accuracy': 0.84375, 'clip_rate_step': 0.0, 'controller_valid_rate_proxy': 1.0, 'train_eval_gap': 0.214859}, 'train_split_affine_range_calibrated_adapter_diagnostic': {'action_l2': 0.80622, 'translation_l2': 0.474716, 'rotation_l2': 0.100196, 'gripper_error': 0.5569, 'gripper_accuracy': 0.84375, 'clip_rate_step': 0.0, 'controller_valid_rate_proxy': 1.0, 'train_eval_gap': 0.254521}, 'range_fixed_smolvla_7d_adapter': {'action_l2': 0.976681, 'translation_l2': 0.550087, 'rotation_l2': 0.229188, 'gripper_error': 0.560213, 'gripper_accuracy': 0.71875, 'clip_rate_step': 0.0, 'controller_valid_rate_proxy': 1.0, 'train_eval_gap': 0.844643}}`
- replay metrics before/after: `{'case_count': 6, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None, None, None, None, None, None], 'progress_proxy_mean': -0.041091, 'object_movement_mean': 0.022539, 'runtime_case_steps': [275, 261, 228, 237, 234, 258], 'clip_rate_step_mean': 0.523375, 'controller_valid_rate_proxy_mean': 0.476625}` / `{'case_count': 6, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None, None, None, None, None, None], 'progress_proxy_mean': -0.902509, 'object_movement_mean': 0.018666, 'runtime_case_steps': [275, 261, 228, 237, 234, 258], 'clip_rate_step_mean': 0.0, 'controller_valid_rate_proxy_mean': 1.0}`
- simple baseline comparison: `{'mean_action': {'case_count': 6, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None, None, None, None, None, None], 'progress_proxy_mean': 0.038336, 'object_movement_mean': 0.000125, 'runtime_case_steps': [275, 261, 228, 237, 234, 258], 'clip_rate_step_mean': 0.0, 'controller_valid_rate_proxy_mean': 1.0}, 'ridge': {'case_count': 6, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None, None, None, None, None, None], 'progress_proxy_mean': -0.173863, 'object_movement_mean': 0.000125, 'runtime_case_steps': [275, 261, 228, 237, 234, 258], 'clip_rate_step_mean': 0.08159, 'controller_valid_rate_proxy_mean': 0.91841}, 'small_mlp': {'case_count': 6, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None, None, None, None, None, None], 'progress_proxy_mean': 0.163687, 'object_movement_mean': 0.019114, 'runtime_case_steps': [275, 261, 228, 237, 234, 258], 'clip_rate_step_mean': 0.086629, 'controller_valid_rate_proxy_mean': 0.913371}, 'clip_only': {'case_count': 6, 'success_count': 0, 'success_rate': 0.0, 'reward_sum_mean': 0.0, 'first_done_indices': [None, None, None, None, None, None], 'progress_proxy_mean': -0.041091, 'object_movement_mean': 0.022539, 'runtime_case_steps': [275, 261, 228, 237, 234, 258], 'clip_rate_step_mean': 0.0, 'controller_valid_rate_proxy_mean': 1.0}, 'affine_diagnostic_offline': {'sample_count': 32, 'action_l2': 0.80622, 'action_l2_first6': 0.489753, 'translation_l2': 0.474716, 'rotation_l2': 0.100196, 'gripper_error': 0.5569, 'gripper_accuracy': 0.84375, 'per_dim_mae': [0.165012, 0.222609, 0.317036, 0.032273, 0.05341, 0.054373, 0.5569], 'worst_action_dimensions': [{'dim': 6, 'mae': 0.5569}, {'dim': 2, 'mae': 0.317036}, {'dim': 1, 'mae': 0.222609}]}}`
- exact next step: Clip-only postprocessing matched or beat the range-fixed adapter; do not count the range fix as method success.
## 2026-07-09: Archive Custom SmolVLA Adapter Route

Decision: `OFFICIAL_VLA_RECIPE_REPRODUCTION_REQUIRED`

- experiments happened: `False`
- training happened: `False`
- replay/control happened: `False`
- downloads/GPU/OpenVLA-OFT happened: `False`
- custom adapter route archived: `True`
- stop criterion: `CLIP_ONLY_BASELINE_DOMINATES`
- positive evidence retained: PEFT/bitsandbytes/RTX 5080 LoRA path works; LIBERO 7D interface fixed; expert replay stable set exists; live/HDF5 feature schema mismatch fixed.
- decisive negative evidence: learned adapter failed replay/progress; offline improvement did not transfer to control; range fix worsened replay; clip-only baseline dominated range fix.
- final strategic decision: `OFFICIAL_VLA_RECIPE_REPRODUCTION_REQUIRED`
- exact next step: Reproduce an official SmolVLA/LeRobot/OpenVLA-style baseline recipe before any method work; otherwise stop VLA method search under this setup.

## 2026-07-09: Official SmolVLA / LeRobot Feasibility And Mini-Repro

Decision: `NEEDS_OFFICIAL_DATASET_CONVERSION`

- experiments happened: `True`
- training happened: `False`
- loss computed: `False`
- GPU model execution happened: `False`
- CPU-only diagnostic happened: `True`
- downloads/GPU training/OpenVLA-OFT happened: `False` / `False` / `False`
- custom LIBERO 7D adapter route used: `False`
- model loaded: `True`
- processor/preprocessor loaded: `True`
- action normalizer status: SO100 6D normalizer tensors present
- official recipe status: SmolVLA base loader/processor mini-repro passed; official LIBERO baseline not reproduced
- LoRA feasibility status: official PEFT target regex exists; rank-4 count-only wrap has `185,664` trainable params; rank-16 has `742,656`; no LoRA training ran
- VRAM peak: `0.0 MB` for the CPU-only mini-repro
- runtime: `30.922 sec` end-to-end, `1.735 sec` single-sample inference

Key evidence:

- local checkpoint `C:\assets\checkpoints\smolvla` declares 6D state and 6D action;
- LeRobot LIBERO docs and installed `LiberoEnv` use 8D state and 7D action;
- official LeRobot `SmolVLAPolicy.from_pretrained` and `make_pre_post_processors` loaded the local checkpoint with a local tokenizer override;
- one synthetic forward returned finite action shape `[1, 6]`;
- CUDA is available on `NVIDIA GeForce RTX 5080`, and bitsandbytes CUDA smoke passed.

Consequence: SmolVLA is locally feasible as an official base loader/processor backbone, but the target LIBERO baseline still needs official-compatible data/checkpoint alignment. Do not start method work yet.

Exact next step: produce a bounded official-compatible LIBERO alignment plan using either official `lerobot/smolvla_libero` plus `lerobot/libero`, or a tiny local HDF5-to-LeRobot conversion that preserves the 8D state / 7D action convention.

## 2026-07-09: Official SmolVLA / LIBERO Dataset Alignment

Decision: `READY_FOR_OFFICIAL_ASSET_APPROVAL`

- experiments happened: `False`
- training happened: `False`
- loss computed: `False`
- downloads/GPU/OpenVLA-OFT happened: `False` / `False` / `False`
- custom LIBERO 7D adapter route used: `False`
- official model asset: `lerobot/smolvla_libero`, public/not gated, Apache-2.0, `0.844 GiB`
- official dataset asset: `lerobot/libero`, public/not gated, Apache-2.0, `1.803 GiB`
- selected official asset size: `2.647 GiB`, above the objective's `2GB` no-approval threshold
- alternate official docs dataset: `HuggingFaceVLA/libero`, public/not gated, Apache-2.0, `32.528 GiB`
- official LIBERO dataset schema: 8D `observation.state`, 7D `action`, two `256x256` image/video keys, fps `10.0`, robot type `panda`
- `smolvla_libero` action schema: 7D action with LIBERO action normalizer stats
- unresolved official checkpoint wrinkle: config lists 6D state and three cameras, while official dataset/stats are 8D state and two image keys
- local conversion feasibility: 1-demo conversion is feasible in principle using `obs/ee_pos`, `obs/ee_ori`, `obs/gripper_states`, two RGB streams, and 7D `actions`

Consequence: do not train. Official asset acquisition is ready for explicit approval; otherwise implement the no-download tiny HDF5-to-LeRobot conversion as the next milestone.

Exact next command after explicit approval:

```powershell
$env:HF_HOME='C:\assets\hf_home'
huggingface-cli download lerobot/smolvla_libero --local-dir C:\assets\checkpoints\smolvla_libero
huggingface-cli download lerobot/libero --repo-type dataset --local-dir C:\assets\datasets\lerobot_libero
```

## 2026-07-09: Official SmolVLA-LIBERO Asset Download And Mini-Repro

Decision: `READY_FOR_OFFICIAL_BASELINE_SCALEUP`

- downloads happened: `True`
- downloaded assets: `lerobot/smolvla_libero`, `lerobot/libero`
- visible downloaded size: `2,842,253,889` bytes, about `2.647 GiB`
- model loaded: `True`
- dataset loaded: `True`
- processor/preprocessor loaded: `True`
- GPU used: `True`
- OpenVLA-OFT happened: `False`
- custom `LIBERO_7D` adapter route used: `False`
- full benchmark / simulator rollout happened: `False`
- official dataset schema: 8D state, 7D action, two 256x256 video image keys
- official checkpoint output action schema: 7D
- video backend used locally: `pyav`
- one-sample forward: action shape `[1, 7]`, finite
- five-sample offline smoke: action L2 mean `0.072885`, translation L2 mean `0.071989`, rotation L2 mean `0.006936`, gripper abs mean `0.007376`, gripper sign accuracy `1.0`
- tiny LoRA smoke: rank `4`, batch size `1`, steps `5`, trainable params `185,664`
- LoRA loss before/after: `0.003114` / `0.003007`
- LoRA peak VRAM: `1102.960 MB`
- LoRA gradients flowed: `74` trainable grad tensors, nonzero for all `74` after step 1

Consequence: official SmolVLA-LIBERO baseline infrastructure is green for bounded scaleup. Official simulator eval remains a separate WSL/Linux/MuJoCo readiness milestone.

Exact next step: create a bounded official baseline scaleup run using the downloaded assets, standard rank-4 LoRA, batch size 1, fixed small step count, runtime under 30 minutes, and full CUDA/device/autocast/action-validity logging.

## 2026-07-09: Official SmolVLA-LIBERO Baseline Scaleup

Decision: `READY_FOR_METHOD_DESIGN_ON_OFFICIAL_SMOLVLA`

- experiments happened: `True`
- training happened: `True`
- loss computed: `True`
- GPU used: `True`
- CPU fallback: `False`
- downloads happened: `False`
- OpenVLA-OFT happened: `False`
- full benchmark / simulator rollout happened: `False`
- custom `LIBERO_7D` route used: `False`
- model path: `C:\assets\checkpoints\smolvla_libero`
- dataset path: `C:\assets\datasets\lerobot_libero`
- official split/sample count: train split `0:1693`, `273465` frames
- schema: 8D state, 7D action, two 256x256 image streams
- data loading deterministic: `True`
- labels/action stats loaded: `True`
- LoRA rank: `4`
- batch size: `1`
- requested/completed steps: `100` / `100`
- trainable params: `185,664`
- train loss before/after: `0.005532921` / `0.003888785`
- frozen/base mini-holdout action L2: `0.081655363`
- rank-4 LoRA mini-holdout action L2: `0.072837438`
- frozen/base mini-holdout eval loss: `0.008015549`
- rank-4 LoRA mini-holdout eval loss: `0.020719278`
- peak CUDA allocation: `1104.506 MB`
- total runtime: `40.813 sec`

Failed attempt log: the first runner version failed before optimizer step with exit code `31` because `policy.forward()` returned a tuple-shaped loss field. The runner was fixed to extract scalar loss from tuple/list outputs, then the same bounded rank-4 LoRA run completed successfully.

Consequence: official SmolVLA/LeRobot baseline path is stable enough for method-design planning. Future method work must keep frozen/base official SmolVLA and this rank-4 LoRA baseline as anchors, and must retain the mixed signal that LoRA improved action L2 but worsened mini-holdout eval loss.

Exact next step: create the first official-path method-design plan with predeclared metrics, baselines, ablations, split/sample policy, tuning budget, and kill/pivot criteria before running any new method.

## 2026-07-09: Official SmolVLA-LIBERO Failure Mining

Decision: `GO_METHOD_DESIGN_TASK_ADAPTER_ROUTING`

- experiments happened: `True`
- training happened: `True`, standard rank-4 LoRA only
- loss computed: `True`
- GPU used: `True`
- downloads happened: `False`
- OpenVLA-OFT happened: `False`
- full benchmark / simulator rollout happened: `False`
- official dataset/model used: `True`
- custom `LIBERO_7D` route used: `False`
- method implemented: `False`
- model path: `C:\assets\checkpoints\smolvla_libero`
- dataset path: `C:\assets\datasets\lerobot_libero`
- held-out diagnostic scope: `200` frames, `5` task groups, `10` episodes, train episode excluded
- rank-4 LoRA train loss before/after: `0.008108919` / `0.011093494`
- frozen/base held-out action L2 / eval loss: `0.106514960` / `0.011978370`
- rank-4 LoRA held-out action L2 / eval loss: `0.118024259` / `0.012148290`
- mean-action prior held-out action L2: `1.144859722`
- LoRA help/hurt count: `98` / `102`
- task-mean LoRA help/hurt count: `2` / `3`
- mean prior better than LoRA: `4` / `200`
- strongest method-worthy gap: task/frame-level adapter interference
- estimated kill risk: high, because frozen/base is strong and MoIRA-style low-rank adapter routing is a close recent-paper baseline

Consequence: do not implement a method yet. A task-conditional adapter-routing design plan is now allowed, but it must explicitly compare against frozen/base official SmolVLA, standard rank-4 LoRA, mean-action prior, and MoIRA-style routing.

Exact next step: design a task-conditional adapter-routing plan only; predeclare metrics, baselines, ablations, split/sample policy, tuning budget, and kill/pivot criteria before any method run.
