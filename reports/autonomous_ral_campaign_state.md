# Autonomous RA-L Campaign State

Date: 2026-07-14 KST

Active governance: `reports/current_research_governance.md`

Current branch: `codex/autonomous-until-paper-governance-v2`

Current decision: `MTF_STAGE_B_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT`

Current epoch: `4`

Current cycle: `6`

Current stage: `epoch_4_cycle_6_mtf_stage_b_manifest_frozen_rollout_pending`

## Corrected Epoch 1 Result

Cycle 1 `DICD-VLA`:

- corrected status: `UNDERPOWERED_STAGE_A_NON_GO_ARCHIVED`
- full: `1 / 10`
- direct chunk-index delay: `2 / 10`
- no-history ablation: `1 / 10`
- ruling: do not rerun or rescue the current formulation; do not treat a one-episode difference at 10 episodes per policy as a permanent scientific family kill.

Cycle 2 `FEDO-VLA`:

- corrected status: `VALID_CURRENT_FORMULATION_KILL`
- faulted full: `1 / 10`
- static inverse gain: `2 / 10`
- APEX-style feedback proxy: `2 / 10`
- no-feedback ablation: `2 / 10`
- clean frozen: `4 / 10`
- clean FEDO: `0 / 10`
- ruling: do not revive the current formulation.

Cycle 3 `GCAP-VLA`:

- corrected status: `UNDERPOWERED_TARGET_AXIS_NON_GO_ARCHIVED`
- occluded full: `3 / 10`
- occluded frozen: `4 / 10`
- Sobel edge boost: `5 / 10`
- no-temporal ablation: `4 / 10`
- clean frozen: `1 / 10`
- clean GCAP: `5 / 10`
- ruling: do not rerun or rescue the current formulation; do not call the whole perception-repair family dead.

## Epoch 2 Result

Epoch 2 Cycle 1 `PTC-VLA` is archived as `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`.

Epoch 2 Cycle 2 `SACF-VLA` is archived as `STAGE_A_PERMANENT_KILL_CLEARLY_WORSE`.

Epoch 2 Cycle 3 `OCFN-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: expanded Stage B completed `400 / 400` total episodes with zero exceptions, `80` paired episodes per key policy, active mechanism, OCFN full `26 / 80`, zero-noise SmolVLA `27 / 80`, and paired upper confidence bound versus the strongest baseline `0.0625`.

These three related failures are synthesized in `reports/epoch_2_failure_synthesis.md`.

## Next Action

Epoch 3 Cycle 1 `CBFD-VLA` is archived as `STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE`: Stage A completed `50 / 50` held-out episodes with zero exceptions, frozen SmolVLA reached `7 / 10`, and full CBFD reached `0 / 10` with active mechanism.

Epoch 3 Cycle 2 `SCVC-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: full SCVC reached `11 / 40`, shifted frozen SmolVLA reached `20 / 40`, and the paired bootstrap CI versus shifted frozen was `[-0.425, -0.025]`.

Epoch 3 Cycle 3 `PSE-VLA` is archived as `STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: expanded Stage B completed `400 / 400` rows with zero exceptions, full PSE reached `50 / 80`, bright-single reached `51 / 80`, and the paired CI versus bright-single was `[-0.1000, 0.0750]`.

The related Epoch 3 failures are synthesized in `reports/epoch_3_failure_synthesis.md`.

Epoch 4 Cycle 1 `RCV-VLA` is archived as `STAGE_2B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED`: Stage 2B completed `200 / 200` episodes with zero exceptions, full RCV reached `20 / 40`, no-context ablation reached `24 / 40`, and stateless first-action reached `24 / 40`.

Epoch 4 Cycle 2 `CAVM-VLA` is archived as `STAGE_2B_EXPANDED_NON_GO_NO_THIRD_EXPANSION`: the expanded result completed `290 / 290` rows with zero exceptions, full CAVM reached `24 / 58`, nearest-success replay reached `23 / 58`, frozen SmolVLA reached `22 / 58`, success-only memory proxy reached `20 / 58`, and no-contrast ablation reached `21 / 58`.

## Epoch 4 Cycle 3

`FANG-VLA` is selected and preregistered as an AFIL-anchored identity-preserving failure-aware action-field guidance method.

Artifacts:

- `reports/epoch_4_cycle_3_prior_mechanism_map.md`
- `reports/epoch_4_cycle_3_candidate_generation.md`
- `reports/fang_vla/researcher_proposal.md`
- `reports/fang_vla/proposal_hash.txt`
- `reports/fang_vla/reviewer_attack.md`
- `reports/fang_vla/researcher_rebuttal.md`
- `reports/fang_vla/mathematical_mechanism_audit.md`
- `reports/fang_vla/preregistration.md`
- `reports/fang_vla/prototype_protocol.md`

Development audit passed and the calibrated validation search selected `fang_c01`. The uncalibrated gate failure is preserved as a negative validation result. Stage A completed `50 / 50` episodes with all policies tied at `3 / 10`.

Stage B completed `200 / 200` episodes with zero exceptions. Full FANG reached `11 / 40`, while frozen SmolVLA reached `16 / 40`, AFIL local proxy reached `15 / 40`, nearest-success replay reached `14 / 40`, and the no-failure ablation also reached `11 / 40`. The paired full-minus-base delta was `-0.125` with CI `[-0.250, 0.000]`; full was exactly tied with the key ablation.

Final FANG decision: `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT`. Do not rescue this formulation.

## Epoch 4 Cycle 4

`EvoState-VLA` is selected and preregistered as an EvoScene/DREAM-anchored action-evolved state guidance method.

Artifacts:

- `reports/epoch_4_cycle_4_prior_mechanism_map.md`
- `reports/epoch_4_cycle_4_candidate_generation.md`
- `reports/evostate_vla/researcher_proposal.md`
- `reports/evostate_vla/proposal_hash.txt`
- `reports/evostate_vla/reviewer_attack.md`
- `reports/evostate_vla/researcher_rebuttal.md`
- `reports/evostate_vla/mathematical_mechanism_audit.md`
- `reports/evostate_vla/preregistration.md`
- `reports/evostate_vla/prototype_protocol.md`

Stage 0 development audit completed without closed-loop rollout and stopped as `AUDIT_STOP_DESIGN_FAILURE`. The full transition model improved over a constant predictor by `0.715309`, but improved only `0.024689` over an actionless model, below the preregistered `0.05` action-input improvement threshold. This means the proposed action-conditioned state mechanism is not sufficiently supported for rollout.

## Epoch 4 Cycle 5

`RAC-VLA` is selected and preregistered as a Reflective VLA-anchored action-consequence calibration method for frozen SmolVLA under controlled deployment action-channel shift.

Stage 0 development audit passed with `10769` consequence pairs, `53685` labeled examples, zero duplicate perturbation keys, full validation accuracy `0.585745`, action-only validation accuracy `0.368496`, no-consequence validation accuracy `0.374483`, and full-vs-best-baseline margin `0.211262`.

The bounded six-config validation search selected `rac_h4_a0.05`: history horizon `4`, residual alpha `0.05`, score `0.508926`, full validation accuracy `0.603250`, and full-vs-best-baseline margin `0.244397`.

Stage A completed `50 / 50` episodes with zero exceptions under the frozen hidden `x_attenuate` action-channel shift. RAC full reached `0 / 10`, frozen shifted Base reached `0 / 10`, the no-consequence ablation reached `0 / 10`, the Reflective-history proxy reached `1 / 10`, and the online diagonal inverse-gain baseline reached `1 / 10`. RAC full tied Base and the key ablation, lost by only `1 / 10` to the strongest baseline and simple baseline, and did not satisfy any permanent Stage A kill criterion.

Stage B completed `200 / 200` episodes with zero exceptions and a valid shared task/reset manifest: `200` unique `(variant, task, identity)` keys, duplicate keys `0`, `40` episodes per variant, and identical paired manifests. RAC full reached `1 / 40`; shifted Base reached `1 / 40`; the Reflective-history proxy reached `1 / 40`; the no-consequence ablation reached `2 / 40`; and the online diagonal inverse-gain simple baseline reached `2 / 40`. RAC full tied Base and the closest-prior proxy, but lost to the key ablation and simple baseline.

Final RAC decision: `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT`. Do not rescue RAC, retune `rac_h4_a0.05`, change the hidden shift, or reinterpret the closed result.

## Post-RAC Governance

The post-RAC performance-oriented governance is installed in `reports/current_research_governance.md`, `AGENTS.md`, and `reports/codex_delegation_manual.md`. Future method cycles must maximize the probability of an honest paper-worthy positive result by using positive-prior anchors, usable-headroom audits, data/supervision health gates, identity-preserving integration, bounded development search, mathematical objective engineering, mechanism smoke, and frozen confirmatory tests.

## Epoch 4 Cycle 6

`MTF-VLA` is selected and preregistered as a FrameSkip and StructVLA anchored milestone-transition data-supervision method for identity-preserving SmolVLA adapter training.

Stage 0 development audit passed without training or closed-loop rollout using `reports/official_smolvla_stable_prediction_artifact.json`: `1600` development records, `1200` reserved test records not used, `40` task keys, duplicate sample keys `0`, duplicate frame keys `0`, high-low score gap `0.585702`, gripper-transition fraction `0.341875`, and adapter-init action delta p95 `0.0`.

The bounded six-config validation search selected `mtf_r20_ret100`: retained high-frame ratio `0.20`, retention coefficient `1.00`, validation score `0.643663`, `176` high train frames, and `391` base-retention train frames. The selected config and training manifest are frozen under `reports/mtf_vla/`.

Current stage: adapter training pending. Stage A must not start until disk-reloadable checkpoints exist for MTF full, no-retention ablation, FrameSkip proxy, and uniform retained-ratio LoRA.


The MTF adapter-training runner is implemented and dry-run validated in `scripts/run_mtf_vla_adapter_training.py`. The real selected-training manifest joins cleanly with the official split and stable prediction artifact: MTF full has `567` training events (`176` milestone, `391` frozen-base retention), no-retention ablation has `176`, the FrameSkip proxy has `176`, and uniform retained-ratio LoRA has `240`. Train/validation/test frame overlap is `0 / 0 / 0`; validation remains `400` frames; no training or closed-loop rollout happened in this dry run.

Current stage: adapter-training runner validated. Stage A must still not start until disk-reloadable checkpoints are trained and disk-reload verified for all four trainable policies.

Adapter training completed after the runner dry-run and a development-only FrameSkip proxy repair: all four trainable Stage A policies were trained with seed `101`, saved under `runs/mtf_vla_checkpoints/mtf_r20_ret100`, reloaded from disk, and evaluated on the `400` validation frames. Final decision: `MTF_ALL_ADAPTER_CHECKPOINTS_VERIFIED_STAGE_A_READY`. Validation action L2 means were `0.082590885` for MTF full, `0.082867367` for the no-retention ablation, `0.082553130` for the corrected FrameSkip proxy, and `0.082396918` for uniform retained-ratio LoRA. The corrected FrameSkip proxy uses `240` action-variation-selected train events and is distinct from the no-retention ablation. No closed-loop rollout happened and no confirmatory-test identities were used.

The MTF Stage A matched manifest is frozen in `reports/mtf_vla/stage_a_manifest.json` with canonical payload hash `1BB86A8060F8CD057AF984423021CA582E87661CB5157C072EF34B6F587739E3`. It contains exactly five policies (`frozen_smolvla`, `frameskip_proxy_lora`, `uniform_retained_ratio_lora`, `mtf_no_retention_ablation`, `mtf_full`), five deterministic task keys selected from the official 20-task manifest, fresh reset seeds `20261201` and `20261202`, `10` paired cases per policy, and `50` total planned episodes. `frameskip_proxy_lora` is labeled as a faithful local proxy, not an official FrameSkip reproduction.

Stage A completed `50 / 50` official LIBERO episodes with zero exceptions. Frozen SmolVLA, FrameSkip proxy, and uniform retained-ratio LoRA each reached `8 / 10`; the no-retention ablation and MTF full each reached `7 / 10`. Full MTF tied the key ablation and was only one episode behind the strongest baselines, so this is not a valid Stage A kill. The frozen adjudication is `MTF_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED`.

The MTF Stage B matched manifest is now frozen in `reports/mtf_vla/stage_b_manifest.json` with canonical payload hash `3C9D9CCF835A3B9753B81C320E9390EC9DA516514563E4850C1DC4F19ACC5743`. It uses all `20` official tasks, fresh reset seeds `20261203` and `20261204`, `40` paired cases per policy, and `200` total planned episodes. The five policy identities are unchanged from Stage A.

Current stage: Stage B official WSL rollout pending. Resume with `wsl -d Ubuntu-22.04 bash -lc "cd /mnt/c/Users/jiheo/tca_map && /home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python scripts/run_mtf_vla_stage_a.py --mode stage-b"`. Resume only missing `(policy, suite, task_id, reset_seed)` keys; do not tune checkpoints on Stage B outcomes.
