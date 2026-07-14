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

## 2026-07-09: Official SmolVLA-LIBERO Routing Design Gate

Decision: `GO_DESIGN_FRAME_CONDITIONAL_ROUTING`

- experiments happened: `True`
- training happened: `True`, only standard rank-4 LoRA regenerated because saved adapter/per-frame rows were unavailable
- loss computed: `True`
- GPU used: `True`
- downloads happened: `False`
- OpenVLA-OFT happened: `False`
- full benchmark / simulator rollout happened: `False`
- official dataset/model used: `True`
- custom `LIBERO_7D` route used: `False`
- method implemented: `False`
- held-out diagnostic scope: `200` frames, `5` task groups, `10` episodes
- frozen/base action L2: `0.106514960`
- rank-4 LoRA action L2: `0.118024259`
- mean-action prior action L2: `1.144859722`
- frame oracle action L2: `0.084582188`
- task oracle action L2: `0.106079976`
- instruction/task-id oracle action L2: `0.106079976`
- eval-loss oracle action L2 / eval loss: `0.117064321` / `0.006147801`
- action-dimension oracle action L2: `0.075210683` diagnostic only
- frame oracle headroom over frozen/base: `0.021932772` absolute, `0.205912597` relative
- task oracle headroom over frozen/base: `0.000434984` absolute, `0.004083783` relative
- frame oracle selector counts: frozen/base `102`, rank-4 LoRA `98`
- task oracle selector counts: frozen/base `120`, rank-4 LoRA `80`

Consequence: pure task/instruction adapter routing is not enough. Task-oracle headroom is below the predeclared `0.005` absolute / `5%` relative gate and is also close to MoIRA-style routing. The only surviving design direction is frame/state/action-disagreement-aware routing with explicit frozen/base retention.

Exact next step: create a Frame-Conditional Adapter Retention first-experiment plan, including frozen/base, standard rank-4 LoRA, mean-action prior, frame oracle, task oracle, MoIRA-style instruction router, task-specific LoRA experts, adapter soup/weighted merge, metrics, ablations, tuning budget, and kill criteria. Do not implement the method until that plan is fixed.

## 2026-07-09: FCAR First-Experiment Plan

Decision: `READY_TO_IMPLEMENT_FCAR_TINY_GATE`

- experiments happened: `False`
- training happened: `False`
- loss computed: `False`
- GPU/download/OpenVLA-OFT happened: `False` / `False` / `False`
- full benchmark / simulator rollout happened: `False`
- official dataset/model route retained: `True`
- custom `LIBERO_7D` route used: `False`
- method implemented: `False`
- paper claim made: `False`

Problem fixed: official SmolVLA-LIBERO low-data rank-4 LoRA creates frame-level negative transfer; frozen/base is stronger on aggregate, task oracle has tiny headroom, and frame oracle has meaningful headroom.

FCAR spec fixed:

- frozen/base expert plus rank-4 LoRA expert;
- frame-level gate with alpha in `[0, 1]`;
- mixed action `a_mix = alpha * a_lora + (1 - alpha) * a_base`;
- retention objective to preserve base when LoRA is harmful;
- no ground-truth action, reward, future frame, simulator success, or custom metadata at inference.

Baselines fixed: frozen/base, standard rank-4 LoRA, mean-action prior, frame oracle, task oracle, MoIRA-style instruction/task router, adapter soup/static merge, optional rank-8 LoRA, and action-dim oracle as diagnostic only.

Metrics fixed: action L2 primary, normalized eval loss secondary, translation/rotation/gripper breakdown, per-task/per-phase breakdown, help/hurt counts, route fraction, calibration, action range validity, train/eval gap, and runtime.

Kill criteria fixed: FCAR must beat frozen/base by `0.005` absolute or `5%` relative action L2 and must beat standard LoRA, MoIRA-style router, adapter soup/static merge, and mean-action prior. It should recover at least `30%` of frame-oracle gain.

Known caveat: saved per-frame base/LoRA prediction artifacts are missing, so the implementation run must regenerate and save compact official predictions before tiny-gate training.

Exact next step: implement the FCAR tiny-gate experiment exactly as specified in `reports/fcar_implementation_todo.md`, without changing baselines, metrics, split policy, or kill criteria after seeing results.

## 2026-07-10: FCAR Tiny-Gate Implementation

Decision: `FCAR_KILLED_BY_STATIC_BASELINE`

- experiments happened: `True`
- training happened: `True`
- trained components: fixed rank-4 LoRA baseline regenerated for prediction artifact, and FCAR tiny CPU gate
- SmolVLA backbone training happened: `False`
- GPU/download/OpenVLA-OFT happened: `True` / `False` / `False`
- full benchmark / simulator rollout happened: `False`
- official dataset/model used: `True`
- custom `LIBERO_7D` route used: `False`
- method implemented: `True`
- paper claim made: `False`
- prediction artifact saved: `reports/fcar_prediction_artifact.json`
- result reports: `reports/fcar_tiny_gate_result.json`, `reports/fcar_tiny_gate_result.md`, `reports/fcar_tiny_gate_decision.md`
- split: train `120` frames, val `40` frames, test `40` frames; episode-disjoint leakage checks passed
- frozen/base test action L2: `0.123998278`
- rank-4 LoRA test action L2: `0.076191123`
- mean-action prior test action L2: `1.148631734`
- frame oracle test action L2: `0.066124022`
- task oracle test action L2: `0.076191123`
- MoIRA-style task/instruction router test action L2: `0.123998278`
- adapter soup/static merge test action L2: `0.091179973` with val-selected `w=0.5`
- FCAR tiny-gate test action L2: `0.100144625`
- FCAR gain over frozen/base: `0.023853653` absolute, `0.192370841` relative
- FCAR recovered frame-oracle headroom fraction: `0.41216345`
- alpha mean/std/min/max on test: `0.443432957` / `0.02648654` / `0.320281953` / `0.493465692`
- routed-to-LoRA fraction at alpha >= 0.5: `0.0`
- train/eval gap test minus train: `-0.014645646`
- CUDA audit: model parameter device `cuda:0`, input tensors on `cuda:0`, peak CUDA allocation `1104.506 MB`, autocast cpu/cuda `False` / `False`

Consequence: FCAR should not be scaled from this result. It beats frozen/base on the gate-test split but fails the fixed success gate because it loses to rank-4 LoRA and to the adapter-soup/static-merge baseline.

Exact next step: none, because `GO_FCAR_SCALEUP` was not reached.

## 2026-07-10: Official SmolVLA Robust Baseline Sweep After FCAR Kill

Decision: `METRIC_OR_SPLIT_INSTABILITY_BLOCKS_METHOD`

- experiments happened: `True`
- training happened: `False`
- trained components: none
- GPU/download/OpenVLA-OFT happened: `False` / `False` / `False`
- full benchmark / simulator rollout happened: `False`
- official dataset/model used: `True`, via the saved official prediction artifact
- custom `LIBERO_7D` route used: `False`
- new method implemented: `False`
- FCAR tuned: `False`
- paper claim made: `False`
- prediction artifact used: `reports/fcar_prediction_artifact.json`
- result reports: `reports/fcar_tiny_gate_postmortem.md`, `reports/official_smolvla_robust_baseline_sweep_plan.md`, `reports/official_smolvla_robust_baseline_sweep_result.json`, `reports/official_smolvla_robust_baseline_sweep_result.md`, `reports/official_smolvla_post_fcar_decision.md`
- sweep scope: `5` deterministic episode-disjoint folds, `40` test frames per fold
- frozen/base action L2 mean/std: `0.106514933` / `0.030256808`
- rank-4 LoRA action L2 mean/std: `0.118024225` / `0.023707422`
- mean-action prior action L2 mean/std: `1.144859705` / `0.018515874`
- frame oracle action L2 mean/std: `0.084582167` / `0.027591676`
- task oracle action L2 mean/std: `0.106079936` / `0.029986441`
- MoIRA-style task/instruction router action L2 mean/std: `0.106514933` / `0.030256808`
- val-selected static mix action L2 mean/std: `0.105142674` / `0.026514373`
- realistic win counts: frozen/base `2`, val-selected static mix `3`
- frame oracle win count with oracles: `5`
- rank-4 LoRA wins over frozen/base: `2` / `5`
- frame oracle mean headroom over frozen/base: `0.021932766`
- task oracle mean headroom over frozen/base: `0.000434997`
- static gap to frame oracle mean: `0.020560507`

Consequence: FCAR remains killed and must not be tuned. Frame-oracle headroom remains, but base/static/LoRA ranking is too split-dependent for a stable new method design under this current offline protocol.

Exact next step: do not design a method yet; first build a more stable official split/metric protocol.

## 2026-07-10: Official SmolVLA Stable Split/Metric Protocol

Decision: `NEEDS_LARGER_PREDICTION_ARTIFACT`

- experiments happened: `False`
- training happened: `False`
- trained components: none
- GPU/download/OpenVLA-OFT happened: `False` / `False` / `False`
- full benchmark / simulator rollout happened: `False`
- official route used: `True`
- official dataset metadata used: `True`, from `C:\assets\datasets\lerobot_libero`
- official model execution happened: `False`
- custom `LIBERO_7D` route used: `False`
- new method implemented: `False`
- FCAR tuned: `False`
- paper claim made: `False`
- result reports: `reports/official_smolvla_stable_protocol_plan.md`, `reports/official_smolvla_split_manifest.md`, `reports/official_smolvla_split_manifest.json`, `reports/official_smolvla_metric_protocol.md`, `reports/official_smolvla_prediction_artifact_plan.md`, `reports/official_smolvla_stable_protocol_result.json`, `reports/official_smolvla_stable_protocol_result.md`, `reports/official_smolvla_stable_protocol_decision.md`

Split manifest:

- status: created
- task coverage: `40` official tasks
- train: `80` episodes / `1200` frames
- validation: `40` episodes / `400` frames
- test: `80` episodes / `1200` frames
- leakage checks: train/validation, train/test, and validation/test episode sets are disjoint
- planned prediction records: `2800`

Metric protocol:

- primary metric: aggregate raw 7D action L2 after official SmolVLA postprocessing
- component reporting: translation dims `0-2`, rotation dims `3-5`, gripper dim `6`
- aggregate reporting: frame-weighted and task-balanced means
- uncertainty reporting: episode and task bootstrap intervals
- static mixture policy: select alpha on validation only, then freeze before test

Instability diagnosis:

- FCAR remains killed by rank-4 LoRA and static merge baselines on the tiny-gate split.
- The post-FCAR robust sweep shows base/static/LoRA ranking instability from too-small held-out prediction coverage, task imbalance, tiny validation slices, missing bootstrap intervals, and missing task-balanced reporting.
- Frame-oracle headroom remains meaningful, but task-oracle headroom remains tiny; this is not enough to justify a new method until the fixed protocol is populated.

Consequence: do not design or tune a method yet. The fixed manifest and metric protocol are ready, but the larger official prediction artifact has not been generated.

Exact next step:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\248_official_smolvla_prediction_artifact_from_manifest.ps1 -SplitManifest reports\official_smolvla_split_manifest.json -Output reports\official_smolvla_stable_prediction_artifact.json
```

## 2026-07-10: Official SmolVLA Stable Artifact Evaluation

Decision: `NEEDS_LONGER_LORA_BASELINE_REPRO`

- experiments happened: `True`
- training happened: `True`
- trained components: standard rank-4 LoRA baseline only
- SmolVLA backbone trained: `False`
- GPU/download/OpenVLA-OFT happened: `True` / `False` / `False`
- full benchmark / simulator rollout happened: `False`
- official model/dataset used: `True`
- custom `LIBERO_7D` route used: `False`
- new method implemented: `False`
- FCAR tuned: `False`
- paper claim made: `False`
- result reports: `reports/official_smolvla_stable_prediction_artifact_status.md`, `reports/official_smolvla_stable_artifact_eval_result.json`, `reports/official_smolvla_stable_artifact_eval_result.md`, `reports/official_smolvla_stable_baseline_table.md`, `reports/official_smolvla_stable_artifact_decision.md`
- generated artifact: `reports/official_smolvla_stable_prediction_artifact.json`
- artifact size bytes: `7,219,361`
- artifact records: `2800`

Manifest:

- tasks: `40`
- train: `80` episodes / `1200` frames
- validation: `40` episodes / `400` frames
- test: `80` episodes / `1200` frames
- leakage checks: train/validation/test episode-disjoint checks passed

CUDA and training:

- model parameter device: `cuda:0`
- input tensor devices: `cuda:0`
- model dtype: `torch.bfloat16`
- peak CUDA allocation: `1104.506 MB`
- autocast cpu/cuda: `False` / `False`
- rank-4 LoRA steps: `100`
- trainable params: `185,664`
- train loss before/after: `0.008257858` / `0.002369085`
- final nonzero grad tensors: `74`
- training elapsed: `17.953 sec`

Stable test metrics, raw 7D action L2:

- frozen/base: `0.085558433`
- rank-4 LoRA: `0.091230140`
- mean-action prior: `1.197255124`
- frame oracle: `0.068470215`
- task oracle: `0.079386015`
- MoIRA-style task/instruction router: `0.092209764`
- validation-selected static mix: `0.081135060`, selected alpha `0.5` on validation only

Additional evidence:

- frozen/base eval loss mean: `0.009544804`
- rank-4 LoRA eval loss mean: `0.009790267`
- realistic task win counts: static mix `29`, frozen/base `7`, rank-4 LoRA `4`
- rank-4 LoRA helped `599` frames and hurt `601` frames versus frozen/base
- static mix helped `769` frames and hurt `431` frames versus frozen/base
- frame oracle headroom over frozen/base: `0.017088218`
- frame oracle headroom after static mix: `0.012664845`
- task oracle headroom over frozen/base: `0.006172418`
- MoIRA-style task router remains weak
- task oracle no longer looks weak under the larger stable artifact

Consequence: the larger artifact resolved the previous split/coverage blocker enough to move the blocker to rank-4 LoRA seed robustness. Do not design a method yet, and do not revive FCAR. The next step is independent standard rank-4 LoRA seeds under the fixed manifest.

Exact next step: run independent standard rank-4 LoRA seeds under `reports/official_smolvla_split_manifest.json`, with the same metric protocol and full CUDA/device/VRAM logging.

## 2026-07-10: Official SmolVLA Rank-4 LoRA Seed Reproduction

Decision: `STATIC_MERGE_ROBUST_BASELINE_READY`

- experiments happened: `True`
- training happened: `True`
- trained components: standard rank-4 LoRA baseline seeds only
- SmolVLA backbone trained: `False`
- GPU/download/OpenVLA-OFT happened: `True` / `False` / `False`
- full benchmark / simulator rollout happened: `False`
- official model/dataset used: `True`
- custom `LIBERO_7D` route used: `False`
- new method implemented: `False`
- FCAR tuned: `False`
- paper claim made: `False`
- seeds run: `11`, `22`, `33`
- stable artifact reused: `reports/official_smolvla_stable_prediction_artifact.json`
- seed artifacts: `reports/official_smolvla_lora_seed_11_prediction_artifact.json`, `reports/official_smolvla_lora_seed_22_prediction_artifact.json`, `reports/official_smolvla_lora_seed_33_prediction_artifact.json`
- result reports: `reports/official_smolvla_lora_seed_repro_plan.md`, `reports/official_smolvla_lora_seed_repro_result.json`, `reports/official_smolvla_lora_seed_repro_result.md`, `reports/official_smolvla_lora_seed_repro_table.md`, `reports/official_smolvla_lora_seed_repro_decision.md`

CUDA and training:

- model parameter device: `cuda:0`
- input tensors: `cuda:0`
- peak CUDA allocation: about `1105.569 MB`
- CPU fallback: `False`
- rank-4 trainable params: `185,664`
- seed `11` loss before/after: `0.000215661` / `0.004648810`
- seed `22` loss before/after: `0.005131230` / `0.005719855`
- seed `33` loss before/after: `0.003007774` / `0.002006668`

Mean/std action L2 across seeds:

- frozen/base: `0.085558433` / `0.0`
- rank-4 LoRA: `0.088239344` / `0.002908670`
- mean-action prior: `1.197255124` / `0.0`
- validation-selected static mix: `0.080616431` / `0.002595356`
- task oracle: `0.081138309` / `0.001955707`
- frame oracle: `0.069117204` / `0.002049401`
- MoIRA-style task/instruction router: `0.088145305` / `0.001719703`

Per-seed primary action L2:

- seed `11`: frozen/base `0.085558433`, rank-4 LoRA `0.084128699`, static mix `0.077354597`, frame oracle `0.066234143`, task oracle `0.078372683`, MoIRA-style router `0.085719423`
- seed `22`: frozen/base `0.085558433`, rank-4 LoRA `0.090162398`, static mix `0.080789904`, frame oracle `0.070815707`, task oracle `0.082495298`, MoIRA-style router `0.089507871`
- seed `33`: frozen/base `0.085558433`, rank-4 LoRA `0.090426934`, static mix `0.083704791`, frame oracle `0.070301761`, task oracle `0.082546947`, MoIRA-style router `0.089208622`

Robustness answers:

- rank-4 LoRA robustly beats frozen/base: `False`
- rank-4 LoRA robustly beats static merge: `False`
- static merge remains strongest realistic baseline: `True`
- static merge seed win count: `3` / `3`
- realistic task win counts summed over seeds: static mix `93`, frozen/base `20`, rank-4 LoRA `7`
- LoRA seed variance action L2 std/range: `0.002908670` / `0.006298235`
- frame oracle headroom after static remains: mean `0.011499227`
- task oracle is not consistently meaningful across seeds
- MoIRA-style task router remains weak

Consequence: validation-selected static merge is now the main realistic baseline for any later planning gate. Any future method plan must beat static merge under the fixed manifest and metric protocol. FCAR remains killed and must not be revived.

Exact next step: treat validation-selected static merge as the main realistic baseline for any later planning gate; do not implement a method until a new explicit planning objective is given.

## 2026-07-10: Official SmolVLA Execution Ledger Audit

Decision: `AUDIT_FOUND_PROTOCOL_GAPS_FIX_BEFORE_ROLLOUT`

- audit type: repository audit only
- experiments/training/GPU/download happened in this audit: `False` / `False` / `False` / `False`
- audited commit range: `72ed23e` through `5d48b1e`
- audited ledger entries: `13`
- runner-backed historical executions in scope: `8`
- exact duplicate runs found: `0`
- possible exact duplicates found: `0`
- avoidable regenerations found: `2`
- artifact inconsistencies found: `0`
- test leakage found: `0`
- old custom `LIBERO_7D` route used in final official runs: `False`
- current offline results remain valid: `True`

Reports:

- `reports/official_smolvla_execution_ledger.md`
- `reports/official_smolvla_execution_ledger.json`
- `reports/official_smolvla_duplicate_run_audit.md`
- `reports/official_smolvla_skipped_stage_audit.md`
- `reports/official_smolvla_artifact_integrity_audit.md`
- `reports/official_smolvla_baseline_naming_audit.md`
- `reports/official_smolvla_protocol_compliance_audit.md`
- `reports/official_smolvla_audit_decision.md`

Artifact integrity:

- fixed manifest SHA256: `1279F939648CF13E2F599084E42631681E1DFA5606B5D9B0851FFEB32710934B`
- stable prediction artifact SHA256: `88DCA06AA05D69E8BC4FB3F1C5A7C7D22B1DC4438C65103EFD2389F24D35D59C`
- seed `11` artifact SHA256: `F40298ACB449FFCBB8FBDFA341B65FDB6120259F7986559E644BB7771CD5A331`
- seed `22` artifact SHA256: `913CB7A3D228002BB73D059D23F5112AC537B5A560CEED19DFB8A2C976A5EF86`
- seed `33` artifact SHA256: `14568E506D0D5FCC9FABA8EDF7C5D3CDE628F9321AE3CC071CBC4537F41D9363`
- manifest split counts: train `1200`, validation `400`, test `1200`
- episode intersections: train/validation `0`, train/test `0`, validation/test `0`

Duplicate/regeneration finding:

- No exact duplicate was confirmed.
- Routing design gate and FCAR tiny gate both regenerated rank-4 LoRA prediction evidence because earlier per-frame artifacts were not reusable.
- These regenerations were avoidable compute/protocol gaps, not result invalidators.

Naming corrections:

- use `task_or_instruction_router_proxy`, not official MoIRA, for the current local task/instruction router proxy
- use `validation_selected_action_space_static_mix`, not adapter soup or adapter-weight merge, for current static base/LoRA action interpolation
- use `frame_oracle_upper_bound` and `task_oracle_upper_bound` for oracle results

Protocol gaps before official rollout:

- Hugging Face model/dataset revisions are not pinned.
- Seed-specific LoRA adapter checkpoint persistence policy is not settled.
- Future baseline naming must be corrected before paper-facing summaries.
- Official closed-loop LIBERO rollout and task success-rate evaluation are still not done.

Consequence: the current offline result remains valid, and `validation_selected_action_space_static_mix` remains the strongest realistic offline baseline. The audit blocks immediate rollout only until revision pinning, naming, and checkpoint-persistence policy are fixed in a no-experiment protocol step.

Exact next step: before official rollout, create a no-experiment protocol-fix branch that records Hugging Face model/dataset revision pins, enforces the future baseline naming glossary, and decides whether LoRA adapter checkpoints must be persisted alongside prediction artifacts.

## 2026-07-10: Official SmolVLA Rollout Protocol Fix

Decision: `LORA_CHECKPOINTS_MISSING_REGENERATION_REQUIRED`

- protocol-fix type: repository metadata and source inspection only
- experiments/training/GPU/download happened in this pass: `False` / `False` / `False` / `False`
- model inference/simulator rollout happened in this pass: `False` / `False`
- historical metrics modified: `False`
- LoRA seeds regenerated: `False`

Fixed:

- model revision locked: `lerobot/smolvla_libero` at `31d453f7edd78c839a8bbc39744a292686daf0de`
- dataset revision locked: `lerobot/libero` at `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`
- package versions recorded, with package source commits marked unavailable from local wheel metadata
- canonical baseline names frozen
- LoRA adapter checkpoint persistence policy frozen
- official rollout action semantics frozen
- static-mix compute accounting requirement frozen
- Stage A and Stage B closed-loop rollout protocol frozen
- official eval readiness classified as `MISSING_OFFICIAL_EVAL_DEPENDENCY`

Canonical future names:

- `frozen_base`
- `rank4_lora`
- `validation_selected_action_space_static_mix`
- `task_or_instruction_router_proxy`
- `frame_oracle_upper_bound`
- `task_oracle_upper_bound`

Seed checkpoint audit:

- seed `11`: `CHECKPOINT_MISSING`
- seed `22`: `CHECKPOINT_MISSING`
- seed `33`: `CHECKPOINT_MISSING`

Official eval readiness:

- `lerobot-eval` entrypoint exists and lists `--env.type=libero`
- local source imports official LIBERO env code
- local env lacks `libero`
- local env lacks `robosuite`
- native Windows official rollout remains unproven; WSL/Linux or dependency repair is required before rollout

Reports:

- `configs/official_smolvla_repro_lock.yaml`
- `reports/official_smolvla_revision_lock.md`
- `reports/official_smolvla_baseline_naming_policy.md`
- `reports/official_smolvla_lora_checkpoint_policy.md`
- `reports/official_smolvla_rollout_action_semantics.md`
- `reports/official_smolvla_rollout_protocol.md`
- `reports/official_smolvla_rollout_readiness.md`
- `reports/official_smolvla_protocol_fix_decision.md`

Consequence: model/dataset revision gaps and protocol ambiguity are closed, but official rollout remains blocked. Prediction JSON artifacts cannot replace persisted adapter checkpoints, so any official LoRA rollout or final LoRA result requires a future explicit training/regeneration pass.

Exact next step: no training, regeneration, GPU, download, or rollout command is safe under the current no-experiment boundary. Future adapter regeneration must be separately approved.

## 2026-07-10: Official SmolVLA LoRA Checkpoint Regeneration

Decision: `LORA_REGEN_METRIC_DRIFT_BLOCKS_ROLLOUT`

- objective: regenerate immutable official SmolVLA-LIBERO rank-4 LoRA adapter checkpoints for seeds `11`, `22`, and `33`
- experiments/training/GPU happened: `True` / `True` / `True`
- downloads/OpenVLA-OFT/FCAR/rollout happened: `False` / `False` / `False` / `False`
- simulator dependency installation happened: `False`
- model revision changed: `False`
- dataset revision changed: `False`
- split manifest changed: `False`
- metric protocol changed: `False`
- historical metrics rewritten: `False`

Locked inputs:

- model: `lerobot/smolvla_libero` at `31d453f7edd78c839a8bbc39744a292686daf0de`
- dataset: `lerobot/libero` at `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`
- split manifest SHA256: `1279F939648CF13E2F599084E42631681E1DFA5606B5D9B0851FFEB32710934B`
- metric protocol SHA256: `64430225940C5168B3734BB40F9F48AD02877E0BA04DC804367AFBB214AE486E`

Checkpoint results:

- seed `11`: `CHECKPOINT_COMPLETE_VERIFIED`, `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_11`
- seed `22`: `CHECKPOINT_COMPLETE_VERIFIED`, `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_22`
- seed `33`: `CHECKPOINT_COMPLETE_VERIFIED`, `C:\assets\checkpoints\smolvla_libero_lora\rank4\seed_33`
- disk reload verification: passed for all three seeds
- checksum manifest: recorded for all three seeds
- CPU fallback: no

Reproduction comparison:

- frozen tolerance: per-seed action L2 absolute difference `<= 0.002`
- static-mix qualitative conclusion preserved: `True`
- tolerance pass: `False`
- seed `11` rank-4 LoRA diff: `0.003085157`
- seed `22` rank-4 LoRA diff: `0.001449016`
- seed `33` rank-4 LoRA diff: `0.004492506`
- seed `11` static-mix diff: `0.001556788`
- seed `22` static-mix diff: `0.000192324`
- seed `33` static-mix diff: `0.004988174`

Generated artifacts:

- `reports/official_smolvla_lora_checkpoint_regen_result.json`
- `reports/official_smolvla_lora_checkpoint_regen_result.md`
- `reports/official_smolvla_lora_checkpoint_manifest.json`
- `reports/official_smolvla_lora_checkpoint_verification.md`
- `reports/official_smolvla_lora_reproduction_comparison.md`
- `reports/official_smolvla_lora_checkpoint_regen_decision.md`
- `reports/official_smolvla_seed_11_prediction_artifact.json`
- `reports/official_smolvla_seed_22_prediction_artifact.json`
- `reports/official_smolvla_seed_33_prediction_artifact.json`

Consequence: the checkpoint persistence blocker is fixed, but official rollout remains blocked by metric drift. Do not proceed to rollout until configuration drift is diagnosed under a new explicit objective.

Exact next step: bounded configuration-drift diagnosis only; no rollout.

## 2026-07-10: Official SmolVLA LoRA Regeneration Drift Audit

Decision: `PROTOCOL_DRIFT_FOUND`

- objective: diagnose whether regenerated seed `11`/`22`/`33` persisted LoRA checkpoints can become the canonical rollout baseline set
- experiments happened: `True`
- training happened: `False`
- optional single-seed probe ran: `False`
- GPU evaluation happened: `True`
- downloads/OpenVLA-OFT/FCAR/rollout happened: `False` / `False` / `False` / `False`
- simulator dependency installation happened: `False`
- historical tolerance relaxed: `False`
- historical metrics overwritten: `False`

Alignment findings:

- split manifest: identical across `5d48b1e` and `15649d6`
- metric protocol: identical across `5d48b1e` and `15649d6`
- test frame IDs, task IDs, episode IDs, labels, split membership: identical
- frozen/base predictions: identical
- static-alpha grid and validation-only selection: identical

Checkpoint findings:

- all three persisted checkpoints remain complete and checksum verified
- disk evaluation repeatability passed for seeds `11`, `22`, and `33`
- max per-action repeat diff: `0.0` for all seeds
- rank-4 LoRA metric repeat diff: `0.0` for all seeds
- static-mix metric repeat diff: `0.0` for all seeds
- selected validation alpha identical: `True` for all seeds
- saved regenerated artifact metrics did not exactly match fixed-seed disk re-evaluation metrics, so evaluation RNG state was also unpinned protocol identity

Root cause:

- the historical `5d48b1e` run evaluated the trained in-memory policy and did not save/reload adapter weights
- the regenerated `15649d6` run assigns the PEFT wrapper return, saves adapter bundles, reloads with `PeftModel.from_pretrained`, and evaluates that disk identity
- the audit's fixed-seed disk re-evaluation is internally repeatable, but it differs from saved regenerated artifact metrics because the prior evaluation RNG state was not separately pinned
- old adapter weights, complete historical RNG state, and exact historical training sample order were not persisted
- therefore the old learned policy identity is not exactly reconstructable and the regenerated checkpoints must not be described as identical historical models

Consequence: the regenerated persisted checkpoints are internally valid under fixed-seed disk evaluation but not accepted as canonical in this audit because real LoRA prediction-protocol and evaluation RNG-state differences were found.

Exact next step: fix or explicitly adjudicate the PEFT in-memory versus persisted-reload protocol difference and evaluation RNG-state policy before canonicalizing or rolling out.

## 2026-07-10: Canonical Persisted SmolVLA-LoRA Baseline Evaluation

Decision: `NEEDS_WSL_OR_LINUX_OFFICIAL_ROLLOUT`

Intermediate decision: `CANONICAL_BASELINES_READY_FOR_ROLLOUT`

Reason: persisted disk-reloaded official SmolVLA base and rank-4 LoRA seeds 11/22/33 produced canonical val/test metrics under the fixed action-generation RNG policy, but the active native Windows environment is missing `hf-libero`, `libero`, and `robosuite`, so official LeRobot LIBERO rollout execution must move to the verified WSL/Linux stack.

Key evidence:

- canonical result: `reports/official_smolvla_canonical_baseline_result.md`
- canonical prediction manifest: `reports/official_smolvla_canonical_prediction_manifest.json`
- canonical artifacts: `reports/canonical_frozen_base_prediction_artifact.json`, `reports/canonical_seed_11_prediction_artifact.json`, `reports/canonical_seed_22_prediction_artifact.json`, `reports/canonical_seed_33_prediction_artifact.json`
- historical status: `SUPERSEDED_NONCANONICAL_PROTOCOL`

## 2026-07-10: Official WSL LeRobot LIBERO Rollout Pilot

Decision: `OFFICIAL_ROLLOUT_BASELINE_READY`

- branch: `codex/wsl-official-smolvla-libero-rollout-pilot`
- smoke: `4/4` completed, all four persisted policies executed on official `libero_spatial/task_0`
- pilot: `48/48` completed, suites `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, task `0`, three episodes per task per policy
- CUDA status: RTX 5080 visible in WSL, policy parameters and input tensors on `cuda:0`, no CPU fallback
- policy loading: frozen base through official factory; LoRA seeds 11/22/33 through minimal PEFT wrapper around the official base policy
- static mixes: skipped and classified `DEGENERATE_EQUIVALENT_TO_FROZEN_BASE`
- old custom `LIBERO_7D` route: not used

Pilot overall success:

- `frozen_base`: `75.0%`
- `rank4_lora_seed_11`: `83.3%`
- `rank4_lora_seed_22`: `66.7%`
- `rank4_lora_seed_33`: `75.0%`

Scientific consequence: lower offline action L2 did not correspond to higher closed-loop success. Seed 11 improved this bounded pilot, but the pilot is not large enough for best-seed selection or method design. The next step is larger official baseline rollout/failure mining with failed-episode videos enabled.

## 2026-07-11: Official SmolVLA Closed-Loop Scaleup

Decision: `OFFLINE_ONLINE_MISMATCH_CONFIRMED`

- branch: `codex/official-smolvla-closed-loop-failure-mining`
- objective: run a predeclared official SmolVLA-LIBERO closed-loop baseline scaleup and identify whether a structured failure exists for later method design
- rollout happened: `True`
- training happened: `False`
- method implemented: `False`
- static-mix duplicate rollout happened: `False`
- old custom `LIBERO_7D` route used: `False`
- OpenVLA-OFT used: `False`
- policies: `frozen_base`, `rank4_lora_seed_11`, `rank4_lora_seed_22`, `rank4_lora_seed_33`
- suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`
- task ids per suite: `0`, `2`, `4`, `6`, `8`
- reset seeds: `20260711` through `20260715`
- planned/completed episodes: `400/400`
- infrastructure failures: `0`

CUDA and route audit:

- RTX 5080 visible in WSL: `True`
- model parameter device: `cuda:0`
- input tensor devices: `cuda:0`
- action chunk device: `cuda:0`
- autocast fp16/bf16 active: `False`
- episode peak VRAM: `926.638` to `928.365` MB
- official relative-control schema: `state_dim=8`, `action_dim=7`
- old custom `LIBERO_7D` route: `False`

Policy success:

- `frozen_base`: `74/100`, `74.0%`, Wilson 95% `[0.646288, 0.815954]`
- `rank4_lora_seed_11`: `74/100`, `74.0%`, Wilson 95% `[0.646288, 0.815954]`
- `rank4_lora_seed_22`: `68/100`, `68.0%`, Wilson 95% `[0.583372, 0.76331]`
- `rank4_lora_seed_33`: `66/100`, `66.0%`, Wilson 95% `[0.562775, 0.745386]`

Paired outcomes versus frozen_base:

- seed `11` reset-level W/T/L: `9/82/9`; task-level W/T/L: `3/13/4`
- seed `22` reset-level W/T/L: `8/78/14`; task-level W/T/L: `3/11/6`
- seed `33` reset-level W/T/L: `4/84/12`; task-level W/T/L: `1/12/7`

Failure mining:

- total unsuccessful episodes: `118`
- automatic failure category counts: `ambiguous_or_unclassified = 118`
- weakest suite across policies: `libero_10`, `45/100`
- weakest task slice: `libero_10/task_4`, `5/20`
- repeated all-policy task/reset failures include `libero_10/task_4/seed_20260713`, `libero_10/task_4/seed_20260715`, and `libero_spatial/task_4` on seeds `20260712`, `20260713`, and `20260714`

Offline/online analysis:

- offline action L2 values: frozen `0.085579125`, seed 11 `0.086743582`, seed 22 `0.086474081`, seed 33 `0.086918872`
- closed-loop success: frozen `0.74`, seed 11 `0.74`, seed 22 `0.68`, seed 33 `0.66`
- all-policy Pearson/Spearman L2-versus-success diagnostics: `-0.569086` / `-0.632456`
- LoRA-only offline ordering is not selection-safe because seed `22` has lower offline L2 than seed `11`, while seed `11` has higher closed-loop success

Consequence: the run found task/reset-structured failures but did not identify a confident mechanism-linked phase failure. No method-worthy closed-loop intervention is approved yet. The next step is bounded official video capture/review for the repeated all-policy failures, not method design or best-seed selection.

Reports:

- `reports/official_closed_loop_scaleup_plan.md`
- `reports/official_closed_loop_task_manifest.json`
- `reports/official_closed_loop_episode_manifest.json`
- `reports/official_closed_loop_scaleup_result.md`
- `reports/official_closed_loop_scaleup_result.json`
- `reports/official_closed_loop_failure_annotations.json`
- `reports/official_closed_loop_failure_taxonomy.md`
- `reports/official_closed_loop_seed_robustness.md`
- `reports/official_closed_loop_offline_online_analysis.md`
- `reports/official_closed_loop_method_gap_decision.md`

## 2026-07-11: Closed-Loop Visual Method Gate

Decision: `NO_SAFE_RA_L_METHOD_YET`

- branch: `codex/closed-loop-failure-novelty-method-gate`
- objective: use bounded official videos to decide whether the current closed-loop failures support exactly one mechanism-specific VLA method
- video rerun happened: `True`
- selected/completed videos: `24/24`
- rerun errors: `0`
- training happened: `False`
- method implemented: `False`
- full 400-episode rerun happened: `False`
- old custom `LIBERO_7D` route used: `False`
- static-mix duplicate rollout happened: `False`
- OpenVLA-OFT used: `False`

Identity and rerun stability:

- same suite/task/policy/reset identity preserved: `True`
- original-vs-rerun success matches: `16/24`
- original-vs-rerun success flips: `8/24`
- interpretation: visual evidence is bounded same-identity rerun evidence, not exact original-frame replay

Visual findings:

- `libero_spatial/task_4`: visible drawer/bowl `stable_grasp` / `contact_transition` failure; seeds `20260713` and `20260714` fail across all four policies in rerun videos
- `libero_10/task_4`: visible multi-object `long_horizon_compounding`; seed `20260715` fails across all four policies, while seed `20260713` has rerun success flips for LoRA seeds `22` and `33`
- shared cross-task mechanism: `False`
- at least three independent rerun-failure seeds for one mechanism: `False`

Recent-work audit:

- confidence route killed by VLAConf
- verification routes killed by CoVer, VeriSpace, and Pre-VLA
- monitor/correct route killed by VLA-Corrector
- adaptive chunking and boundary routes killed by AAC, SEAM, and Legato
- progress/recovery/replanning routes killed by SPR, ProgressVLA, ProgVLA, and REMAC
- failure-negative route killed by AFIL
- prior/expert/adapter-routing routes killed by PriorVLA, CLARE, and VLA-GSE

Consequence: no method candidate is specified and no implementation prompt is authorized. Reopen only with new bounded evidence for one repeated mechanism plus second-backbone, second-benchmark, and simple-baseline kill plans.

## 2026-07-11: Cross-Backbone Cross-Benchmark Failure Gate

Decision: `SECOND_BACKBONE_OR_BENCHMARK_BLOCKED`

- branch: `codex/cross-backbone-cross-benchmark-failure-gate`
- objective: determine whether either observed failure is a cross-backbone, cross-benchmark VLA execution problem before method design
- downloads happened: `False`
- training happened: `False`
- rollout happened: `False`
- method implemented: `False`
- episodes completed: `0`
- videos recorded: `0`

Second backbone selection:

- selected backbone: `OpenVLA-OFT`
- selected checkpoint: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`
- checkpoint size: `14.845` GiB
- checkpoint access/license: public, non-gated, MIT
- State 1 decision: `SECOND_BACKBONE_DOWNLOAD_APPROVAL_REQUIRED`

Second benchmark selection:

- selected benchmark: `LIBERO-PRO`
- source: `https://github.com/Zxy-MLlab/LIBERO-PRO`
- dataset: `https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro`
- dataset metadata size: `1,090,523` bytes
- State 2 decision: `SECOND_BENCHMARK_READY_AFTER_SECOND_BACKBONE`

Protocol:

- mechanisms remain separate: `stable_grasp` and `long_horizon_compounding`
- maximum predeclared episodes after unblock: `96`
- no task or seed selection after seeing second-model outcomes
- videos required for every episode

Consequence: no cross-backbone or cross-benchmark evidence exists yet. The correct final decision is blocked, not method-ready, not SmolVLA-specific, and not prior-art killed. The next valid step is explicit OpenVLA-OFT download/hardware approval, then the frozen bounded protocol.

## 2026-07-11 RTX 5080 Quantized OpenVLA-OFT Cross-Backbone Gate

Current decision: `FAILURE_NOT_REPRODUCED_IN_SECOND_ARCHITECTURE`

The approved `14.845` GiB checkpoint `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` was downloaded exactly once, checksummed, and evaluated only as quantized INT4 OpenVLA-OFT on the local RTX 5080. No training, fine-tuning, full BF16 load, RLDS download, LIBERO-PRO download, CPU offload, or disk offload occurred.

Hard-slice outcome: OpenVLA-OFT INT4 completed `20/20` exact-init episodes with `20/20` successes and videos. The matched SmolVLA frozen-base exact-init rerun completed `20/20` with `11/20` successes, including hard-slice failures on `libero_spatial/task_4` (`1/5`) and `libero_10/task_4` (`1/5`).

Conclusion: the SmolVLA stable-grasp and long-horizon failures were not reproduced in the second architecture under this bounded quantized OpenVLA-OFT gate. LIBERO-PRO is not justified by this result.

Key reports:

- `reports/openvla_oft_int4_download_status.md`
- `reports/openvla_oft_int4_environment_lock.md`
- `reports/openvla_oft_int4_memory_preflight.md`
- `reports/openvla_oft_int4_policy_load_result.md`
- `reports/openvla_oft_int4_int8_consistency.md`
- `reports/openvla_oft_quantized_hard_slice_manifest.json`
- `reports/openvla_oft_quantized_hard_slice_result.md`
- `reports/openvla_oft_quantized_hard_slice_result.json`
- `reports/openvla_oft_quantized_visual_annotations.json`
- `reports/openvla_oft_quantized_cross_backbone_decision.md`

Exact next step: do not implement a method and do not proceed to LIBERO-PRO from this evidence. Archive the cross-backbone result as failure-not-reproduced unless a future full-precision or different second-backbone run is explicitly approved.

## 2026-07-11 Paper-First VLA Method Design

Decision: `READY_TO_IMPLEMENT_PRIMARY_VLA_METHOD`

- branch: `codex/paper-first-vla-ral-method-design`
- objective: literature-first VLA robotics method ideation and selection
- primary sources reviewed: `34`
- experiments happened: `False`
- training happened: `False`
- GPU/model inference happened: `False`
- simulator execution happened: `False`
- large download happened: `False`
- implementation happened: `False`
- selected method: `ECHO-VLA`
- technical novelty: phase-conditioned counterfactual action-effect credit for VLA action chunks
- estimated RA-L strength: strong if first prototype and second-backbone validation pass
- estimated kill probability: `0.35`

Rejected candidates:

- `BARRIER-VLA`: high simple-baseline and VeriSpace/Pre-VLA proximity risk.
- `SEMAPHORE-VLA`: novelty too close to SPR/ProgressVLA/ProgVLA.
- `IRIS-VLA`: too close to VLA-Corrector/Pre-VLA and expensive recoverability labels.

Consequence: implementation is authorized only for the bounded ECHO-VLA first prototype described in `reports/vla_primary_method_first_experiment.md`; OpenVLA-OFT INT4 and the full two-backbone matrix are held until the SmolVLA gate passes.

## 2026-07-11 ECHO-VLA First Prototype Headroom Gate

Decision: `NO_ECHO_CANDIDATE_HEADROOM`

- branch: `codex/implement-echo-vla-first-prototype`
- starting main commit: `5fcc87b93b627dbf09eb69676801e4412909bda4`
- targeted novelty adjudication: passed under strict same-state physical-effect mediator claim
- same-state intervention groups: `4`
- candidate records: `16`
- tasks: `libero_spatial/task_0`, `libero_object/task_4`
- reset identities: `20260711`, `20260712`
- candidate count: `4`
- horizon: `4`
- same-state group proofs valid: `4/4`
- non-gripper effect labels populated: `eef_delta_norm`, `target_distance_delta`
- oracle improvement: `0.0` percentage points
- default-failure recoverable rate: `0.0`
- components trained: `none`
- closed-loop ECHO evaluation: `False`
- OpenVLA-OFT used: `False`
- full benchmark run: `False`

Consequence: do not train ECHO heads or run ECHO closed-loop evaluation with the current candidate generator. The required oracle headroom prerequisite failed before training.

## 2026-07-11 - ECHO Final Candidate Headroom Gate

Decision: `NO_ECHO_HEADROOM_CONFIRMED`

Evidence: official downstream metrics `{'group_count': 12, 'candidate_count_total': 96, 'default_success_rate': 0.833333, 'random_success_rate': 0.833333, 'local_effect_oracle_success_rate': 0.833333, 'final_task_success_oracle_rate': 0.833333, 'oracle_improvement_pp': 0.0, 'default_failure_group_count': 2, 'recoverable_default_failure_count': 0, 'recoverable_default_failure_rate': 0.0, 'tasks_with_recovery': [], 'tasks_with_recovery_count': 0, 'phases_with_recovery': [], 'recovered_group_ids': [], 'passes_original_thresholds': False, 'headroom_spans_multiple_tasks': False, 'not_solely_one_phase_or_state': False, 'passes_final_gate': False, 'threshold_rule': 'non-relaxed original hard gate: oracle improvement >=10pp and recoverable default-failure rate >=15%, plus recovery across at least two tasks and not solely one phase/state'}`, structured diagnostic metrics `{'group_count': 12, 'candidate_count_total': 96, 'default_success_rate': 0.833333, 'random_success_rate': 0.833333, 'local_effect_oracle_success_rate': 0.833333, 'final_task_success_oracle_rate': 0.833333, 'oracle_improvement_pp': 0.0, 'default_failure_group_count': 2, 'recoverable_default_failure_count': 0, 'recoverable_default_failure_rate': 0.0, 'tasks_with_recovery': [], 'tasks_with_recovery_count': 0, 'phases_with_recovery': [], 'recovered_group_ids': [], 'passes_original_thresholds': False, 'headroom_spans_multiple_tasks': False, 'not_solely_one_phase_or_state': False, 'passes_final_gate': False, 'threshold_rule': 'non-relaxed original hard gate: oracle improvement >=10pp and recoverable default-failure rate >=15%, plus recovery across at least two tasks and not solely one phase/state'}`.

## 2026-07-11 - ECHO Final Candidate Headroom Gate

Decision: `NO_ECHO_HEADROOM_CONFIRMED`

Evidence: official downstream metrics `{'candidate_count_total': 96, 'default_failure_group_count': 2, 'default_success_rate': 0.833333, 'final_task_success_oracle_rate': 0.833333, 'group_count': 12, 'headroom_spans_multiple_tasks': False, 'local_effect_oracle_success_rate': 0.833333, 'not_solely_one_phase_or_state': False, 'oracle_improvement_pp': 0.0, 'passes_final_gate': False, 'passes_original_thresholds': False, 'phases_with_recovery': [], 'random_success_rate': 0.833333, 'recoverable_default_failure_count': 0, 'recoverable_default_failure_rate': 0.0, 'recovered_group_ids': [], 'tasks_with_recovery': [], 'tasks_with_recovery_count': 0, 'threshold_rule': 'non-relaxed original hard gate: oracle improvement >=10pp and recoverable default-failure rate >=15%, plus recovery across at least two tasks and not solely one phase/state'}`, structured diagnostic metrics `{'candidate_count_total': 96, 'default_failure_group_count': 2, 'default_success_rate': 0.833333, 'final_task_success_oracle_rate': 0.833333, 'group_count': 12, 'headroom_spans_multiple_tasks': False, 'local_effect_oracle_success_rate': 0.833333, 'not_solely_one_phase_or_state': False, 'oracle_improvement_pp': 0.0, 'passes_final_gate': False, 'passes_original_thresholds': False, 'phases_with_recovery': [], 'random_success_rate': 0.833333, 'recoverable_default_failure_count': 0, 'recoverable_default_failure_rate': 0.0, 'recovered_group_ids': [], 'tasks_with_recovery': [], 'tasks_with_recovery_count': 0, 'threshold_rule': 'non-relaxed original hard gate: oracle improvement >=10pp and recoverable default-failure rate >=15%, plus recovery across at least two tasks and not solely one phase/state'}`.

## 2026-07-11 - Autonomous Dual-Review RA-L Campaign

Decision: `NO_METHOD_AFTER_3_VALID_CYCLES`

Execution boundary:

- branch: `codex/autonomous-dual-review-ral-research`
- training happened: `False`
- simulator rollout happened: `False`
- downloads happened: `0 GiB`
- active GPU time: `0 h`
- main updated: `False`

Cycle kills:

- Cycle 01, action conditioning and action representation: killed by CAC-VLA, ACoT-VLA, LaRA-VLA, ActionMap, LARA/LAWM/AEM proximity plus local ECHO and ActionMap negative evidence.
- Cycle 02, intervention-censored correction credit: killed by TORL-VLA, SDP, AFIL, BORA, VLA-Corrector, Pre-VLA proximity and missing intervention/tactile/robot data.
- Cycle 03, contact barrier and irreversibility boundaries: killed by VeriSpace, Pre-VLA, VLA-Corrector, AAC, SEAM, Legato, TORL-VLA proximity plus local contact/geometry baseline kills and non-cross-backbone hard-slice evidence.

Consequence: stop the autonomous campaign. Do not implement another local VLA method without one of the explicit reopen conditions in `reports/autonomous_campaign_final_decision.md`.

## 2026-07-11 - Autonomous RA-L Research Implementation V2

Decision: `TWO_IMPLEMENTED_METHODS_KILLED`

Correction: the prior `NO_METHOD_AFTER_3_VALID_CYCLES` decision is procedurally rejected and reclassified as `PREMATURE_LITERATURE_ONLY_TERMINATION`.

Implemented cycle 1:

- method: `PhaseBarrier-VLA`
- code: `tca_map/smolvla/phase_barrier_vla.py`, `scripts/run_phase_barrier_vla_prototype.py`
- result: `reports/phase_barrier_vla_prototype_result.json`
- final decision: `PHASE_BARRIER_VALID_KILL`
- evidence: training happened and closed-loop SmolVLA-LIBERO evaluation happened; full method task-balanced success `0.0`.

Implemented cycle 2:

- method: `CensorCredit-VLA`
- code: `tca_map/smolvla/censored_credit_vla.py`, `scripts/run_censor_credit_vla_prototype.py`
- result: `reports/censor_credit_vla_prototype_result.json`
- final decision: `CENSOR_CREDIT_VALID_KILL`
- evidence: training happened and closed-loop SmolVLA-LIBERO evaluation happened; full method task-balanced success `0.5`, but uncensored recovery ablation also `0.5`.

Consequence: the implementation-v2 campaign has two genuinely distinct implemented valid kills. The valid final decision is `TWO_IMPLEMENTED_METHODS_KILLED`, not another literature-only no-method result.

## 2026-07-12 - Implementation V2 Empirical Postmortem

Decision: `PROTOTYPE_EVIDENCE_INSUFFICIENT_FOR_TERMINAL_CLAIM`

Execution boundary:

- branch: `codex/implementation-v2-empirical-postmortem`
- base implementation commit: `1ff7e4d420dddae290105b07f8cd03acc987e123`
- rollout rerun happened: `False`
- training rerun happened: `False`
- threshold changes happened: `False`
- broad literature search happened: `False`

Evidence:

- PhaseBarrier-VLA trained `20` records from `5` states and evaluated `2` held-out episodes per variant. Full PhaseBarrier shaped most rollout steps and changed actions, but all variants scored `0/2`. Classification: `UNDERPOWERED_PROTOTYPE_INCONCLUSIVE`.
- CensorCredit-VLA trained `24` records from `6` states and evaluated `2` held-out episodes per variant. Censored and uncensored labels were identical for every record, producing identical saved model weights. Classification: `IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`.

Consequence: preserve the implementation evidence as negative prototype evidence, but do not report it as two genuine method-level kills. No final method is promoted from this postmortem.

## 2026-07-12 - PhaseBarrier Bounded Adjudication

Decision: `PHASEBARRIER_COMPONENT_NOT_USEFUL`

Execution boundary:

- branch: `codex/phasebarrier-bounded-adjudication`
- bounded repair reason: original PhaseBarrier implementation acted but the first prototype was underpowered
- valid closed-loop episodes: `100/100`
- valid run reused original saved PhaseBarrier weights: `True`
- training rerun for valid result: `False`
- invalid retrained run preserved: `reports/phase_barrier_bounded_repair_invalid_retrained_result.json`
- CensorCredit repair happened: `False`

Result:

- frozen SmolVLA: `8/20`
- Pre-VLA-style halt proxy: `0/20`
- simple global damping: `0/20`
- no-phase ablation: `9/20`
- full PhaseBarrier: `0/20`

Mechanism:

- full PhaseBarrier shaped `20/20` episodes;
- mean full action delta: `0.105796`;
- mean full shaped steps: `357.45`;
- no-phase ablation mean action delta: `0.012180`.

Consequence: archive PhaseBarrier-VLA permanently under the current formulation. The phase-conditioned component is not useful because the key no-phase ablation beat it by `45` task-balanced percentage points.

## 2026-07-12 - CensorCredit One-Repair Gate and Final Method

Decision: `NO_VALID_CENSORCREDIT_REPAIR_FINAL_METHOD_KILLED`

Execution boundary:

- branch: `codex/censorcredit-one-repair-and-final-method`
- base commit: `1f29a422945350e33ba3be0cb6150054735c49f6`
- CensorCredit implementation changed: `False`
- CensorCredit repair attempted: `False`
- CensorCredit training rerun: `False`
- CensorCredit rollout rerun: `False`
- final method implementation run: `False`

CensorCredit diagnosis:

- exact classification: `LABEL_OR_DATA_FAILURE`
- allowed repair categories were only `CONCRETE_IMPLEMENTATION_BUG` and `CONCRETE_OPTIMIZATION_BUG`
- label-pair table: `(-1,-1)=20`, `(1,1)=4`
- censored/uncensored label disagreements: `0`
- censored/uncensored weights identical: `True`

Final distinct method:

- candidate: `Intervention-Set Action-Chunk Fine-Tuning (ISAC-VLA)`
- reviewer status: `FINAL_METHOD_KILLED_BEFORE_IMPLEMENTATION`
- kill grounds: `NEAR_EXACT_PRIOR_ART_DUPLICATION`, `HARD_UNAVAILABLE_RESOURCE`
- primary overlap sources: SDP, TORL-VLA, ConRFT, OpenVLA-OFT

Consequence: the current autonomous chain ends without a valid CensorCredit repair and without a valid final method implementation target.

## 2026-07-12 - Governance V2 Migration

Decision: `EPOCH_1_COMPLETED_PIVOT_REQUIRED`

Execution boundary:

- branch: `codex/autonomous-until-paper-governance-v2`
- starting pushed commit: `e24a6a11db49054aaf7a9d6787449f671b5035b3`
- active governance file: `reports/current_research_governance.md`

Reason:

The prior fixed-cycle terminal stop retained obsolete governance. The active campaign now has no finite global method-cycle limit and exactly four allowed final states: `READY_TO_DRAFT_RAL_PAPER_PACKAGE`, `AUTONOMOUS_CAMPAIGN_PAUSED_RESUMABLE`, `HARD_EXTERNAL_BLOCKER`, and `SAFETY_RESOURCE_STOP`.

Corrected Epoch 1 adjudication:

- `DICD-VLA`: `UNDERPOWERED_STAGE_A_NON_GO_ARCHIVED`
- `FEDO-VLA`: `VALID_CURRENT_FORMULATION_KILL`
- `GCAP-VLA`: `UNDERPOWERED_TARGET_AXIS_NON_GO_ARCHIVED`

Consequence: immediately continue to Epoch 2 candidate generation under current governance.

## 2026-07-12 - Epoch 2 Cycle 1 PTC-VLA

Decision: `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`

Execution boundary:

- method: `PTC-VLA`
- branch: `codex/autonomous-until-paper-governance-v2`
- synthetic mechanism smoke: repaired after one preserved failed smoke, then passed
- real trace training: passed with `210` examples
- Stage A: `50 / 50` episodes, zero exceptions

Evidence:

- frozen SmolVLA: `3 / 10`, task-balanced `0.30`
- global mean action: `0 / 10`
- phase mean action: `0 / 10`
- PTC no-transition ablation: `0 / 10`
- PTC full: `0 / 10`
- mechanism active: transition-context norm `0.065772`, action delta versus ablation `0.756346`

Consequence: archive PTC-VLA as a valid current-formulation kill and continue to Epoch 2 Cycle 2.

## 2026-07-12 - Epoch 2 Cycle 2 SACF-VLA

Decision: `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`

Execution boundary:

- method: `SACF-VLA`
- branch: `codex/autonomous-until-paper-governance-v2`
- proposal hash: `1C43D99A42AD97C29C1BDBDED1AB1326214C8FF0F514F79309266738C5FD1A20`
- synthetic mechanism smoke: initial brittle gate failed and was preserved, measurement repair recorded, repaired smoke passed
- real-demo training: passed with `4773` examples
- Stage A: `50 / 50` episodes, zero exceptions

Evidence:

- frozen SmolVLA: `7 / 10`, task-balanced `0.70`
- task-phase mean prefix: `0 / 10`
- plain BC prefix: `0 / 10`
- CAG null guidance proxy: `1 / 10`, task-balanced `0.10`
- SACF full: `0 / 10`
- mechanism active: semantic component norm `1.709826`, action delta versus plain BC `0.429388`

Consequence: archive SACF-VLA as a valid current-formulation kill and continue to Epoch 2 Cycle 3.

## 2026-07-12 - Epoch 2 Cycle 3 OCFN-VLA

Decision: `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`

Execution boundary:

- method: `OCFN-VLA`
- branch: `codex/autonomous-until-paper-governance-v2`
- proposal hash: `F60B9B7BB2640A073AC16EAB6284A68D41569A6A4D67A54462DEF81F06F3F8EA`
- train acquisition: `16 / 16` episodes, zero exceptions
- Stage A: `50 / 50` episodes, zero exceptions, non-GO requiring Stage B
- expanded Stage B: `400 / 400` total episodes, zero exceptions, `80` paired episodes per key policy

Evidence:

- frozen SmolVLA: `23 / 80`, task-balanced `0.2875`
- zero-noise SmolVLA: `27 / 80`, task-balanced `0.3375`
- global success noise prior: `23 / 80`, task-balanced `0.2875`
- task-shuffled noise prior: `25 / 80`, task-balanced `0.3125`
- OCFN full: `26 / 80`, task-balanced `0.3250`
- mechanism active: mean initial delta versus global `0.020219`, versus task-shuffled `0.032354`
- paired upper confidence bound versus strongest baseline: `0.0625`

Consequence: archive OCFN-VLA as a valid current-formulation kill. Synthesize the three related Epoch 2 kills and pivot to Epoch 3.

## 2026-07-12 - Epoch 2 Failure Synthesis

Decision: `EPOCH_2_SYNTHESIZED_KILLS_EPOCH_3_PIVOT_REQUIRED`

Evidence:

- `PTC-VLA` changed temporal transition representation and action generation, but full reached `0 / 10`.
- `SACF-VLA` changed semantic task-prefix representation and action generation, but full reached `0 / 10`.
- `OCFN-VLA` changed latent flow-noise initialization and outcome-conditioned selection, but full reached `26 / 80` versus zero-noise SmolVLA `27 / 80`.

Consequence: begin Epoch 3 Cycle 1 under `reports/current_research_governance.md`, changing at least two core dimensions relative to Epoch 2 action-surface interventions.

## 2026-07-12 - Epoch 3 Cycle 1 CBFD-VLA

Decision: `STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE`

Execution boundary:

- method: `CBFD-VLA`
- branch: `codex/autonomous-until-paper-governance-v2`
- proposal hash: `D355F0FC8C728320D448E572E3CB3D7F8D823EAE7C8C3E91078D1376CEE526E2`
- teacher acquisition: `10 / 10` successful Quantized OpenVLA-OFT INT4 episodes
- teacher trace rows: `1765`
- student training: passed with `192` retention rows
- Stage A: `50 / 50` episodes, zero exceptions

Evidence:

- frozen SmolVLA: `7 / 10`, task-balanced `0.70`
- direct distillation proxy: `0 / 10`
- teacher trace memory: `0 / 10`
- CBFD no-retention: `0 / 10`
- CBFD full: `0 / 10`
- mechanism active: full action delta versus direct distillation `1.244676`, versus memory `1.652989`

Consequence: archive CBFD-VLA as a valid current-formulation kill and continue to Epoch 3 Cycle 2.

## 2026-07-12 - Epoch 3 Cycle 2 SCVC-VLA

Decision: `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`

Execution boundary:

- method: `SCVC-VLA`
- branch: `codex/autonomous-until-paper-governance-v2`
- proposal hash: `BE52CB82140F56E84A0FDBC4D3F51ACD4E704551AC10CC72CE624801DABDE20C`
- synthetic mechanism smoke: passed
- calibration: passed with `10` clean calibration rows
- Stage A: `50 / 50` episodes, zero exceptions, non-GO requiring Stage B
- Stage B: `200 / 200` episodes, zero exceptions

Evidence:

- clean frozen SmolVLA: `10 / 40`
- shifted frozen SmolVLA: `20 / 40`
- known inverse affine: `10 / 40`
- SCVC no-temporal: `10 / 40`
- SCVC full: `11 / 40`
- paired full minus shifted frozen: wins `4`, losses `13`, ties `23`, delta `-0.225`, CI `[-0.425, -0.025]`

Consequence: archive SCVC-VLA as a valid current-formulation kill and continue to Epoch 3 Cycle 3.

## 2026-07-13 - Epoch 3 Cycle 3 PSE-VLA

Decision: `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`

Execution boundary:

- method: `PSE-VLA`
- branch: `codex/autonomous-until-paper-governance-v2`
- proposal hash: `3F15D6E3ADCF340C490FBD5656051DFD101136D592F5A6B5D773ABF0E5308CAD`
- synthetic mechanism smoke: passed
- Stage A: `50 / 50` episodes, zero exceptions, non-GO requiring Stage B
- Stage B 40-paired result: completed and archived separately
- expanded Stage B: `400 / 400` total rows, zero exceptions, `80` paired episodes per variant

Evidence:

- clean frozen SmolVLA: `48 / 80`
- bright single transform: `51 / 80`
- dark single transform: `46 / 80`
- duplicate-clean ensemble: `44 / 80`
- PSE full: `50 / 80`
- paired full minus bright single: wins `6`, losses `7`, ties `67`, delta `-0.0125`, CI `[-0.1000, 0.0750]`

Consequence: archive PSE-VLA as a valid current-formulation kill. No further PSE internal controls or expansions are allowed.

## 2026-07-13 - Epoch 3 Failure Synthesis

Decision: `EPOCH_3_SYNTHESIZED_KILLS_EPOCH_4_PIVOT_REQUIRED`

Evidence:

- `CBFD-VLA` changed supervision/data source using Quantized OpenVLA-OFT INT4 traces, but full reached `0 / 10` versus frozen SmolVLA `7 / 10`.
- `SCVC-VLA` changed sensor-statistic canonicalization under a fixed visual shift, but full reached `11 / 40` versus shifted frozen SmolVLA `20 / 40`.
- `PSE-VLA` changed inference-time action generation through photometric view ensembling, but full reached `50 / 80` versus bright-single `51 / 80`, with paired upper confidence bound `0.075`.

Consequence: begin Epoch 4 after applying the post-PSE research-design governance update. Epoch 4 must change at least two core dimensions relative to teacher distillation, sensor-statistic canonicalization, and photometric action ensembling.

## 2026-07-13 - Epoch 4 Cycle 1 RCV-VLA

Decision: `STAGE_2B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`

Execution boundary:

- method: `RCV-VLA`
- branch: `codex/autonomous-until-paper-governance-v2`
- proposal hash: `86044E841D178DB5AA485B7D12B01FF8E4274CBDFDCDAC7D427477BF0646F26F`
- Stage 0: `20 / 20` episodes, zero exceptions
- Stage 1: `7276` acquisition step records, full and no-context verifiers saved
- Stage 2A: `50 / 50` episodes, zero exceptions, positive enough to require Stage 2B
- Stage 2B: `200 / 200` episodes, zero exceptions, `40` paired episodes per key policy

Evidence:

- queued frozen SmolVLA: `14 / 40`, task-balanced `0.35`
- SV-deviation proxy: `16 / 40`, task-balanced `0.40`
- RCV full: `20 / 40`, task-balanced `0.50`
- RCV no-context ablation: `24 / 40`, task-balanced `0.60`
- stateless first-action: `24 / 40`, task-balanced `0.60`
- paired full minus no-context ablation: wins `2`, losses `6`, ties `32`, delta `-0.10`, CI `[-0.250, 0.025]`
- paired full minus stateless first-action: wins `2`, losses `6`, ties `32`, delta `-0.10`, CI `[-0.225, 0.025]`
- RCV full replan rate: `0.557293`
- RCV full heavy policy calls per step: `0.563500`
- no-context ablation heavy policy calls per step: `0.429078`

Consequence: archive RCV-VLA as a valid current-formulation kill. Do not rescue it by threshold retuning, a renamed verifier, or another receding-chunk replanning ablation. Continue to Epoch 4 Cycle 2 under the post-PSE problem-first, external-prior-early, mathematically justified research-design gate.

## 2026-07-14 - Epoch 4 Cycle 6 MTF-VLA Stage A Manifest

Decision: `MTF_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT`

Execution boundary:

- method: `MTF-VLA`
- branch: `codex/autonomous-until-paper-governance-v2`
- proposal hash: `11DC94A2B75CD8605577AB044E5743DFDA4131A4FA7F6C6A7390519B9F995B31`
- selected config: `mtf_r20_ret100`
- manifest: `reports/mtf_vla/stage_a_manifest.json`
- manifest hash: `1BB86A8060F8CD057AF984423021CA582E87661CB5157C072EF34B6F587739E3`
- planned episodes: `50`
- paired cases per policy: `10`
- reset seeds: `20261201`, `20261202`
- policies: `frozen_smolvla`, `frameskip_proxy_lora`, `uniform_retained_ratio_lora`, `mtf_no_retention_ablation`, `mtf_full`

Consequence: Stage A is frozen and ready for official WSL rollout. No closed-loop rollout has happened from this manifest, and no Stage A outcome may be used to retune MTF checkpoints, task selection, reset identities, policy list, or thresholds.

## 2026-07-14 - Epoch 4 Cycle 6 MTF-VLA Stage A Result

Decision: `MTF_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`

Execution boundary:

- method: `MTF-VLA`
- branch: `codex/autonomous-until-paper-governance-v2`
- proposal hash: `11DC94A2B75CD8605577AB044E5743DFDA4131A4FA7F6C6A7390519B9F995B31`
- selected config: `mtf_r20_ret100`
- manifest: `reports/mtf_vla/stage_a_manifest.json`
- result: `reports/mtf_vla/stage_a_result.json`
- episodes: `50 / 50`
- exceptions: `0`

Evidence:

- frozen SmolVLA: `8 / 10`, task-balanced `0.8`
- FrameSkip proxy: `8 / 10`, task-balanced `0.8`
- uniform retained-ratio LoRA: `8 / 10`, task-balanced `0.8`
- no-retention ablation: `7 / 10`, task-balanced `0.7`
- MTF full: `7 / 10`, task-balanced `0.7`
- paired full minus no-retention: wins `1`, losses `1`, ties `8`, delta `0.0`
- paired full minus FrameSkip proxy: wins `0`, losses `1`, ties `9`, delta `-0.1`

Consequence: this is a noncatastrophic Stage A directional screen, not a valid kill. Under the frozen Stage A rules, MTF-VLA must proceed to Stage B with no checkpoint, threshold, task, identity, or policy-list retuning from Stage A outcomes.

## 2026-07-14 - Epoch 4 Cycle 6 MTF-VLA Stage B Manifest

Decision: `MTF_STAGE_B_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT`

Execution boundary:

- method: `MTF-VLA`
- branch: `codex/autonomous-until-paper-governance-v2`
- proposal hash: `11DC94A2B75CD8605577AB044E5743DFDA4131A4FA7F6C6A7390519B9F995B31`
- selected config: `mtf_r20_ret100`
- Stage A result: `reports/mtf_vla/stage_a_result.json`
- Stage B manifest: `reports/mtf_vla/stage_b_manifest.json`
- manifest hash: `3C9D9CCF835A3B9753B81C320E9390EC9DA516514563E4850C1DC4F19ACC5743`
- planned episodes: `200`
- paired cases per policy: `40`
- tasks: all `20` official task-manifest entries
- reset seeds: `20261203`, `20261204`
- policies: `frozen_smolvla`, `frameskip_proxy_lora`, `uniform_retained_ratio_lora`, `mtf_no_retention_ablation`, `mtf_full`

Consequence: Stage B is frozen and ready for official WSL rollout. Stage A outcomes were used only to trigger the preregistered Stage B escalation; no checkpoint, threshold, task, reset, or policy-list retuning occurred.
