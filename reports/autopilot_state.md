# Autopilot State

- current main commit at branch start: `8934242 add action-source audit matched-init diagnostic`
- branch: `codex/online-action-generation-bridge`
- attempted: online action-source inventory plus bounded native online matched-init bridge diagnostic
- online action-source inventory happened: `true`
- valid native online action source found: `true` (`native_smolvla_policy_output`, 6D policy action mapped to 7D by explicit gripper-zero adapter)
- valid ActionMap/TCA online 7D action source found: `false`
- rollout happened: `true`, bounded diagnostic only
- valid closed-loop online rollout happened: `true` for native SmolVLA baseline only
- valid ActionMap/TCA method rollout happened: `false`
- training happened: `false`
- LoRA training happened: `false`
- loss computed: `false`
- downloads/GPU/OpenVLA-OFT happened: `false`
- heavy VLA import/model load/model inference happened: `true` for the bounded CPU native SmolVLA diagnostic only
- task/demo: `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`, `demo_0`
- horizon: `25` matched-init steps per variant
- variants: `zero_action_exact_init`, `hdf5_expert_replay_exact_init`, `native_smolvla_online_policy`
- reward/success: zero action `0.0 / false`; HDF5 expert upper bound `0.0 / false`; native online SmolVLA `0.0 / false`
- expert near-match rates: zero action `0.0`; HDF5 expert upper bound `1.0`; native online SmolVLA `0.0`
- mean L2 to same-timestep HDF5 expert: zero action `1.104051519`; HDF5 expert upper bound `0.0`; native online SmolVLA `1.802682551`
- target-directed movement score: zero action `-0.000457`; HDF5 expert upper bound `-0.004024`; native online SmolVLA `-0.244345`
- evidence type: native closed-loop baseline diagnostic plus ActionMap/TCA source-inventory blocker
- blocker classification: `no_nonleaking_online_actionmap_tca_7d_head`
- fixed-prior TCA valid rollout-level support: `false`
- exact next state decision: do not make rollout-level ActionMap/TCA method claims yet. The next execution milestone should implement or train a minimal non-leaking 7D online diagnostic head for ActionMap/TCA, or package current evidence with an honest offline + bridge caveat.

## 2026-07-06 - Online 7D Diagnostic Head

- branch: `codex/minimal-online-7d-diagnostic-head`
- attempted: train and evaluate the smallest non-leaking 7D online ActionMap/TCA diagnostic head
- training happened: `true`, CPU ridge/linear diagnostic heads only
- LoRA training happened: `false`
- loss computed: `true`
- rollout happened: `true`, bounded matched-init diagnostic only
- downloads/GPU/OpenVLA-OFT/full fine-tuning/paper claim happened: `false`
- heavy model import/model load/model inference happened: `true` for native SmolVLA baseline only inside the gated rollout
- HDF5 action provenance: training labels, expert upper bound, and expert-match reference only; method rollout actions are generated online from current observation/instruction
- rollout demo excluded from training labels: `true`
- offline ActionMap-7D loss: `0.179330387 -> 0.02388843`; 7D L2 `1.000304358`
- offline fixed-prior TCA-7D loss: `0.179330387 -> 0.0238874`; 7D L2 `0.995906943`
- offline hard learned-target TCA-7D loss: `0.179330387 -> 0.023892769`; 7D L2 `1.016720219`
- bounded rollout reward/success: all variants `0.0 / false`
- fixed-prior TCA valid rollout support: `false`
- fixed-prior TCA partial target-movement support: `true`
- blocker classification: `online_7d_head_partial_target_movement_no_success`
- exact next state decision: run action-quality/head-training diagnosis before any rollout scaling; do not claim rollout-level TCA support from partial target movement.
