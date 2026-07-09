# All Killed Routes Summary

Current killed or reframed main-route candidates:
- Target-Prior TCA-Map
- CSS-Shield
- ExecSpec-Repair
- AMP-GD
- ResetSpec-Retarget
- Phase-Locked Retiming
- TL-ChunkRepair
- ContactTube-Aug
- PRISM-VLA
- ContactSet-VLA
- ActionMap Mini-Anchor
- SafeTrace-VLA
- PatchGuard-VLA

## Target-Prior TCA-Map

- original hypothesis: target-prior conditioned action decoding can improve counterfactual robustness and wrong-target behavior under low compute.
- strongest positive evidence: fixed-prior TCA improved offline proxy metrics; prior-source audit passed; fixed-prior TCA beat ActionMap in some 7D diagnostics.
- decisive negative evidence: the online 7D action-quality gate failed because the best TCA head did not beat the mean-action baseline.
- exact kill criterion triggered: no credible rollout-level support because the action decoder failed the pre-rollout quality gate.
- strongest trivial baseline that killed it: mean-action baseline.
- why not RA-L-stable: the route relied on offline/diagnostic evidence and the action source was not strong enough for closed-loop claims.
- reusable artifacts: target-prior split/audit code, offline ActionMap/TCA comparisons, 7D bridge, expert replay, online 7D diagnostic heads.

## CSS-Shield

- original hypothesis: a counterfactual semantic/safety shield can reduce wrong-target and unsafe VLA actions at inference time.
- strongest positive evidence: controlled and randomized proposal diagnostics showed semantic wrong-target reductions beyond clipping-only and safety-only.
- decisive negative evidence: native-action Phase 2 diagnostic showed no wrong-target gain beyond safety-only and full intervention rate `1.0`.
- exact kill criterion triggered: full shield failed the native-action semantic novelty gate.
- strongest trivial baseline that killed it: safety-only.
- why not RA-L-stable: native-action evidence supports safety damping against clipping-only, not a semantic contribution beyond safety-only.
- reusable artifacts: WSL simulator path, native SmolVLA CPU inference, shield variants, object/safety metrics, bounded autopilot pattern.

## ExecSpec-Repair

- original hypothesis: mismatch-aware executable-spec repair can recover robot policy execution failures beyond simple action-space baselines.
- strongest positive evidence: STATE 3 full repair recovered `17 / 19` degraded exact-init replay cases.
- decisive negative evidence: STATE 3.5 found diagonal affine calibration also recovered `17 / 19`, with full-minus-best-simple gain `0.0`.
- exact kill criterion triggered: best single simple baseline matched full repair within the predeclared tolerance.
- strongest trivial baseline that killed it: diagonal affine calibration.
- why not RA-L-stable: the broad contribution collapsed to per-dimension affine calibration under current evidence.
- reusable artifacts: executable mismatch diagnostics, exact-init replay, calibration baselines, replay validation, baseline dominance audit.

## AMP-GD

- original hypothesis: active micro-probes can disambiguate intended targets before commitment and reduce wrong-target behavior.
- strongest positive evidence: toy point-world AMP-GD reached wrong-target rate `0.0` and success `1.0`.
- decisive negative evidence: deterministic informative-probe and entropy-greedy heuristics matched toy AMP-GD, and the LIBERO port did not beat safety-only or random-probe on the tiny diagnostic.
- exact kill criterion triggered: simple probe/safety baselines matched or beat the method outside the toy claim.
- strongest trivial baselines that killed it: informative-probe heuristic, safety-only, and random-probe.
- why not RA-L-stable: the method did not show active-probe value beyond simple heuristics in the real simulator port.
- reusable artifacts: object/EEF observability inventory, instruction-to-visible-object resolver, tiny LIBERO micro-probe harness.

## ResetSpec-Retarget

- original hypothesis: object-relative and EEF-relative replay retargeting can recover execution under reset/object-pose mismatch better than raw replay and action-only calibration baselines.
- strongest positive evidence: exact-init expert replay succeeded, default-reset raw replay failed, and object-relative retargeting improved EEF-object progress and shifted-trajectory drift.
- decisive negative evidence: fixed global-scale replay from default reset succeeded with reward/success `1.0 / true`, while object-relative retargeting stayed `0.0 / false`.
- exact kill criterion triggered: object-relative retargeting did not beat the strongest simple baseline.
- strongest trivial baseline that killed it: fixed global scale.
- why not RA-L-stable: the novelty is not separable from a trivial action-scale change on the tested task.
- reusable artifacts: reset-mismatch replay runner, object/EEF state capture, object-shifted trajectory drift metric, baseline-first retarget report.

## Phase-Locked Retiming

- original hypothesis: event-locked timing should recover replay/control success when action chunks are temporally out of phase.
- strongest positive evidence: exact-init expert replay succeeded, and all nine synthetic phase perturbations degraded replay.
- decisive negative evidence: event-locked retiming recovered over raw on `0 / 9` perturbations and beat the best simple baseline on `0 / 9`.
- exact kill criterion triggered: event-locked retiming improved neither replay/progress over raw perturbed replay nor best-simple-baseline performance.
- strongest trivial baselines that exposed the failure: gripper-only timing correction, fixed time shift, repeat-last/hold, linear time warp, and diagonal affine depending on perturbation.
- why not RA-L-stable: the method did not produce a positive replay/control recovery signal, and simple timing/action baselines already recovered or matched several perturbation families.
- reusable artifacts: phase perturbation generator, event-anchor extraction, exact-init phase replay runner, baseline table, and result report.

## TL-ChunkRepair

- original hypothesis: a finite-state temporal-logic/event monitor can identify causal violation boundaries inside action chunks and minimally repair them to reduce temporal manipulation violations while preserving utility.
- strongest positive evidence: real exact-init LIBERO/RoboSuite replay/control metrics were produced; `7 / 8` temporal perturbations degraded replay; TL-ChunkRepair reduced symbolic temporal violations on `8 / 8`; exact-init replay infrastructure, temporal perturbation runner, monitor, metrics, and baseline suite worked.
- decisive negative evidence: TL safe-success was `0 / 8`; TL reward/success was `0.0 / 0`; the best single simple baseline, `no_repair`, achieved reward/success `1.0 / 1`; TL failed both the best single simple baseline gate and the best per-failure-mode simple baseline gate.
- exact kill criterion triggered: symbolic/property repair did not translate to replay/control utility and did not beat the required simple baselines.
- strongest trivial baseline that killed it: `no_repair`, with additional weakening from clipping-only, safety-only one-step filter, repeat-last/hold, and fixed-delay timing baselines.
- why not RA-L-stable: the route improved monitor satisfaction but not robot execution utility; RA-L-stable continuation requires safe-success, reward, success, done/progress, or comparable replay/control gains beyond simple baselines.
- reusable artifacts: temporal perturbation runner, exact-init replay diagnostic, temporal property monitor, violation metrics, simple baseline suite, focused tests, and STATE 1 result reports.

## ContactTube-Aug

- original hypothesis: successful demonstrations contain contact tubes, including EEF-object relative trajectories, gripper timing, object motion onset, lift/place phases, and contact/proximity windows; preserving those tubes while retargeting object/reset/distractor conditions should create useful training demonstrations without new teleoperation.
- strongest positive evidence: contact-tube extraction succeeded using HDF5 EEF/gripper traces plus runtime object traces; bounded LIBERO/RoboSuite replay/control diagnostics ran for `1621` simulator steps across `6` variants; exact-init no-op replay succeeded; ContactTube-Aug beat random action jitter and random pose jitter on tube preservation.
- decisive negative evidence: ContactTube-Aug generated invalid/clipped actions with controller-valid action rate `0.849265` and clip-step rate `0.150735`; simple object-relative translation retargeting beat ContactTube-Aug on tube preservation (`0.009154` versus `0.015226`); HDF5 object pose was unavailable.
- exact kill criterion triggered: augmented actions were not controller-valid enough, and simple object-relative retargeting matched or beat ContactTube-Aug before training.
- strongest trivial baseline that killed it: simple object-relative translation retargeting.
- why not RA-L-stable: a data-augmentation method must first produce physically valid demonstrations and beat simple retargeting before BC/action-head or VLA training; ContactTube-Aug failed that gate.
- reusable artifacts: contact-tube extraction, runtime object trace collection, augmentation-validity diagnostics, random jitter baselines, simple object-relative retarget baseline, gated replay smoke, and focused tests.

## PRISM-VLA

- original hypothesis: VLA policies should produce consistent action distributions for task-preserving paraphrases while preserving action-distribution differences for true object or target changes.
- strongest positive evidence: official LIBERO-Para metadata was integrated with local LIBERO HDF5 action chunks; a deterministic held-out paraphrase group split was created with no group leakage; base held-out paraphrase degradation was measurable (`0.062428`); PRISM+canonicalization beat simple augmentation on primary held-out robustness (`+0.055205`).
- decisive negative evidence: canonicalization-only beat the best PRISM variant on held-out paraphrase proxy (`0.474066` versus `0.436356`) and PRIDE (`46.686731` versus `31.985592`); best PRISM primary held-out delta versus canonicalization was `-0.030420`; counterfactual sensitivity was not preserved.
- exact kill criterion triggered: canonicalization-only matched or beat every PRISM variant on primary held-out paraphrase/PRIDE metrics, and the best PRISM variant weakened counterfactual/object sensitivity.
- strongest trivial baseline that killed it: canonicalization-only.
- why not RA-L-stable: the route targets a real language robustness problem but does not beat a simple lexical normalization baseline on the primary held-out gate; auxiliary consistency gains are not enough when object/target sensitivity weakens.
- reusable artifacts: LIBERO-Para metadata integration, held-out paraphrase group split, gated paraphrase diagnostic runner, PRIDE/difficulty-weighted robustness metrics, consistency metrics, object/syntactic subset metrics, and counterfactual sensitivity checks.

## ContactSet-VLA

- archive status: complete.
- original hypothesis: a structured source/destination/support/safety/normal contact set injected into the action head can improve contact-rich and multi-stage manipulation beyond the single 3D point injection result from the anchor paper.
- strongest positive evidence: the local diagnostic extracted source object, destination/support, safety, and normal proxy points from `6` local LIBERO HDF5 demos without eval-label leakage; all required variants ran; tiny CPU action-head loss was computed.
- decisive negative evidence: full contact-set action L2 was `1.105028754`, worse than active single-point injection (`0.930495702`), destination-only (`0.86372`), and no-geometry (`0.851451`) on the held-out action metric.
- exact kill criterion triggered: single-point and simple point/no-geometry baselines matched or beat full contact-set injection before any full VLA training.
- strongest trivial baselines that killed it: active single 3D point, destination-only point, and no-geometry action-head baseline.
- why not RA-L-stable: the method-level extension did not improve the first bounded local action-head metric; scaling to VLA fine-tuning or replay would violate the baseline-first gate.
- reusable artifacts: HDF5/XML free-joint geometry extraction, qpos-offset audit, instruction-based source/destination selector, permutation-aware point-set encoder, gated diagnostic runner, and focused tests.

## ActionMap Mini-Anchor

- archive status: killed/reframe at STATE 1 anchor gate.
- original hypothesis: a local ActionMap-style voxel heatmap/candidate action head should beat mean-action, linear/L1, and cheap MLP baselines on held-out LIBERO HDF5 action quality before any extension or failure mining.
- strongest positive evidence: the diagnostic used `8` local HDF5 demos with `1008 / 432` deterministic train/eval records and showed the candidate grid had strong oracle headroom (`0.065653208` action L2).
- decisive negative evidence: ActionMap-style action L2 was `0.529931357`, worse than mean-action (`0.466767673`) and matched/beat by cheap MLP (`0.501926707`); candidate top1 was `0.018518519`; the learned heatmap also collapsed to one rotation bin (`unique trans/rot/grip = 5/1/2`).
- exact kill criterion triggered: mean-action and cheap MLP matched or beat the ActionMap-style head, and candidate collapse was detected before replay/control or any new extension.
- strongest trivial baselines that killed it: mean-action and cheap MLP action heads.
- why not RA-L-stable: the local anchor did not pass the reproduction-first simple-baseline gate; no standard LIBERO success, exact-init replay, VLA metric, GPU training, or full ActionMap reproduction occurred. Target-Grounded ActionMap cannot proceed from this result.
- reusable artifacts: ActionMap anchor task docs, gated tiny CPU diagnostic runner, HDF5 feature/action loader, translation/rotation/gripper candidate heads, top-k/NLL/collapse metrics, oracle nearest-candidate upper bound, and focused tests.
- revival requirement: official ActionMap reproduction with official code/assets, or a stronger non-collapsed heatmap implementation that first beats mean-action, linear/L1, and cheap MLP baselines.

## SafeTrace-VLA

- archive status: complete.
- original hypothesis: temporal safety monitors can generate preference pairs, such as safer trajectory/action chunk over unsafe trajectory/action chunk, and temporal safety preference optimization can reduce temporal safety violations while preserving task utility better than filters or generic DPO.
- strongest positive evidence: local standard LIBERO HDF5 proxy traces produced temporal safety metrics with risk exposure time `0.519444`, cumulative safety cost `939.0`, and `800` valid preference pairs including `10` nontrivial pairs; official safety anchors SafeManip, LIBERO-Safety, and ForesightSafety-VLA were identified.
- decisive negative evidence: safety-only/risk-only preference accuracy was `1.0`; generic DPO proxy accuracy/loss was `1.0 / 0.052119`; SafeTrace proxy accuracy/loss was `1.0 / 0.052120`; task-success labels were unavailable in sampled proxy traces; official safety benchmark assets were not locally reproduced.
- exact kill criterion triggered: safety-only/risk-only and generic preference/DPO proxy matched the intended SafeTrace preference objective before STATE 2.
- strongest trivial baselines that killed it: safety-only/risk-only monitor scoring and generic DPO/preference proxy.
- why not RA-L-stable: the method claim collapsed to generic monitor-derived preference labels, utility retention was not established, and no official safety benchmark reproduction backed the route.
- reusable artifacts: SafeTrace source-audit table, local HDF5 temporal safety smoke, temporal violation/risk exposure/cumulative-cost metrics, preference-pair headroom diagnostics, safety-only/generic DPO comparison, bounded runner, focused tests, and STATE 1 reports.

## PatchGuard-VLA

- archive status: complete.
- original hypothesis: kinematic-consistent action-path defense can suppress physical patch induced visual-proprioceptive hijacking in a VLA while preserving clean behavior.
- strongest positive evidence: local SmolVLA showed a measurable patch effect, with max attacked policy-action L1 `0.181765` and max attacked translation-action L2 `0.213965`; kinematic/proprioceptive signal was available; PEFT `0.19.1`, bitsandbytes `0.49.2`, CUDA on RTX 5080, SmolVLA LoRA injection, and a tiny rank-4 training smoke all worked.
- decisive negative evidence: PatchGuard metric `0.13356` did not earn a robust win over the predeclared baselines: generic adversarial LoRA `0.142803` and cutout/random-erasing `0.02973`.
- exact kill criterion triggered: PatchGuard LoRA did not beat both generic adversarial LoRA and cutout/random-erasing in STATE 1B.
- strongest trivial baselines that killed it: cutout/random-erasing and generic adversarial LoRA.
- why not RA-L-stable: the route was no longer blocked by the environment; after the adapter path worked, the method-specific defense failed the baseline-first gate. Continuing to STATE 2 would scale a baseline-dominated defense.
- reusable artifacts: patch diagnostic, dependency smoke, LoRA injection path, tiny training runner, RTX 5080 memory/runtime data, focused tests, and STATE 1/1B reports.

## Common Failure Pattern

The method must not merely beat no-method. It must beat the strongest trivial baseline available for the failure mode:
- mean-action,
- clipping-only,
- safety-only,
- nearest-target,
- random-probe,
- informative-probe heuristic,
- diagonal affine,
- global scale,
- oracle/replay leakage,
- exact-init expert replay upper bound.

Any future route that only beats no-method, raw replay, or a weak ablation should be killed or reframed before implementation scale-up.

New rule from Phase-Locked Retiming: a topic is invalid if each targeted failure mode can be solved by a separate obvious simple baseline. A method must beat the best single simple baseline and the best per-failure-mode simple baseline, not only the weakest baseline in the table.

New rule from TL-ChunkRepair: a topic is invalid for RA-L-stable continuation if it improves symbolic, proxy, or constraint-satisfaction metrics but degrades or fails real replay/control utility compared with a simple baseline.

New rule from ContactTube-Aug: a data-augmentation method is invalid for continuation if generated actions are not controller-valid, or if simple object-relative retargeting preserves trajectory/contact metrics better than the proposed augmentation.

New rule from PRISM-VLA: a language-robustness method is invalid for continuation if canonicalization-only beats it on held-out paraphrase robustness, or if it improves paraphrase consistency by weakening counterfactual/object sensitivity.

New rule from ContactSet-VLA: a richer action-head geometry method is invalid for continuation if active single-point, source-only, destination-only, source+destination, or no-geometry action-head baselines match or beat it on the first held-out action metric.

New rule from ActionMap Mini-Anchor: an action-decoder anchor or extension is invalid if mean-action, linear/L1, or cheap MLP action heads match or beat it on held-out 7D action quality, or if heatmap/candidate predictions collapse before replay/control evidence. Oracle candidate headroom is invalid as method evidence unless the learned head exploits it.

New rule from SafeTrace-VLA: a temporal safety preference method is invalid if safety-only/risk-only monitor scoring or generic DPO/preference labels match the proposed temporal preference objective, or if utility retention cannot be measured on an official safety benchmark/source.

New rule from PatchGuard-VLA: a VLA robustness or defense route is invalid if it does not beat both a cheap image-space defense such as cutout/random-erasing and a generic adversarial augmentation LoRA after the real LoRA path is available.

Additional reset after PatchGuard-VLA: local proxy idea generation should remain stopped. The real SmolVLA LoRA path is now available, so the next valid step is standard SmolVLA LoRA baseline reproduction on an official or standard task split, not another custom method.

Update from SmolVLA LoRA Baseline STATE 1: the baseline reproduction ran and standard LoRA learned the training loss, but it was mean-action dominated on held-out local demos. Standard LoRA eval action L2 was `0.940196`, mean-action eval action L2 was `0.486561`, and frozen/base SmolVLA eval action L2 was `1.6029`.

Follow-up diagnosis: final decision `ACTION_INTERFACE_BUG`. The local data has enough raw timesteps (`13298`) and a larger deterministic split is possible (`300 / 100`), but local LIBERO labels are 7D while the SmolVLA action head and normalizer are 6D SO100-style. One-sample and one-demo overfit failed in action space.

Interface fix update: final decision `READY_FOR_REAL_METHOD_AFTER_INTERFACE_FIX`. A separate LIBERO_7D adapter path now keeps labels 7D, uses train-split-only 7D normalization, learns the gripper dimension, passes one-sample/one-demo overfit, and beats mean-action/frozen-base action L2 in the bounded diagnostic. This removes the local action-interface blocker for baseline reproduction, not the need for official or standard fixed-interface baseline evidence before any new method.
