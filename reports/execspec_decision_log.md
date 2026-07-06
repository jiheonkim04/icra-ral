# ExecSpec-Repair Decision Log

## STATE 0 Initialization

Decision: start ExecSpec-Repair as a fresh rollout/replay-first direction and archive Target-Prior TCA-Map and CSS-Shield as negative evidence.

Reason: the killed routes exposed a repeated deployment failure mode: action dimensions, gripper conventions, metadata scales, and controller interfaces can collapse execution even when checkpoints or offline proxies look plausible.

Consequence: the first branch must produce a real mismatch or replay metric, not only planning documents.

## STATE 1 HDF5 Mismatch and Exact-Init Replay Result

Decision: continue ExecSpec-Repair to a calibrated repair replay state.

Reason: local LIBERO HDF5 action metrics reproduced substantial executable-spec mismatch, and bounded exact-init replay showed degradation from plausible mismatches. Correct 7D expert replay reached reward/success `1.0 / true` at observed first reward/done index `260`. Gripper sign flip replay and translation-scale mismatch replay both degraded to `0.0 / false`.

Key HDF5 metric: `gripper_sign_flip` had action L2 mean `2.0` and gripper mismatch rate `1.0`. Translation-scale mismatch had action L2 mean `0.363970119`.

Repair metric: supervised diagonal calibration, labeled as HDF5 calibration/evaluation rather than rollout policy action generation, beat identity, clipping-only, and naive global affine baselines on global scale, per-dimension scale, gripper sign flip, gripper threshold, translation scale, rotation scale, and 6D/7D zero-gripper bridge variants.

Consequence: STATE 2 should replay a minimal calibrated repair for the strongest degraded mismatch under the same exact-init boundary. Do not claim paper-grade results or deployable policy repair until calibrated replay evidence exists without using future expert actions as rollout actions.

Training happened: false. LoRA training happened: false. Loss was computed: false. Replay/rollout happened: true, bounded exact-init diagnostic only. GPU/download/OpenVLA-OFT happened: false. Paper-grade claim: false.

## STATE 3.5 Baseline Dominance and Reframe Audit

Decision: kill the broad ExecSpec-Repair route as a main RA-L topic.

Reason: the report-only audit over the existing STATE 3 replay results showed that the best single simple baseline, `diagonal_affine_calibration`, exactly matched full ExecSpec-Repair. Both recovered `17 / 19` degraded replay cases, with success recovery `0.894736842` and action-drift recovery `1.0`. The full-minus-best-single-simple-baseline gain was `0.0`, which triggers the predeclared kill threshold.

Matched-case analysis: the four STATE 3 simple-baseline matched recovery cases were the living-room `global_action_scale_mismatch` and `range_clipping_mismatch` cases, where `global_affine_calibration` reached the same `1.0 / true` reward/success as full repair. Global affine did not explain the full result, recovering only `4 / 19` degraded cases; gripper-only recovered `6 / 19`.

Routing audit: a mismatch-aware selector recovered the same `17 / 19` degraded cases as full repair and diagonal affine, but did not beat diagonal affine. The selector/routing gain over the best single simple baseline was `0.0`, so mismatch-aware repair selection is not meaningful enough to rescue the route under the predeclared criteria.

Consequence: archive ExecSpec-Repair as useful negative/diagnostic evidence. Do not proceed to STATE 4, do not package it as paper-ready, and do not claim novelty beyond a diagonal affine repair baseline. The next work should either select a new rollout-first route or explicitly build a harder executable-spec benchmark before reviving this topic.

Training happened: false. LoRA training happened: false. Loss was computed: false. New replay/rollout happened: false. GPU/download/OpenVLA-OFT happened: false. Paper-grade claim: false.

## STATE 2 Calibrated Repair Replay Result

Decision: continue ExecSpec-Repair to STATE 3 replay/rollout validation.

Reason: calibrated repair was fit only on calibration demonstrations and evaluated on a held-out exact-init replay demo without eval-action leakage. The split used `5` calibration demos (`1403` action samples) and `1` held-out eval demo (`272` action samples), with leakage detection reported as `false`.

Held-out action result: seven mismatch types were tested: `gripper_sign_flip`, `translation_scale_mismatch`, `rotation_scale_mismatch`, `global_action_scale_mismatch`, `per_dimension_scale_mismatch`, `gripper_threshold_0_1_mismatch`, and `range_clipping_mismatch`. Aggregate held-out mean action L2 was identity `0.565447642`, clipping-only `0.565447642`, global affine `0.308794194`, and full ExecSpec-Repair `0.0`. Full repair beat identity, clipping-only, and global affine on the aggregate held-out action-drift criterion. Per-mismatch beat counts for full repair were `7 / 7` versus identity, `7 / 7` versus clipping-only, and `5 / 7` versus global affine.

Exact-init replay result: bounded replay was run for `gripper_sign_flip` and `translation_scale_mismatch`. Correct expert replay reached reward/success `1.0 / true`; wrong executable spec replay degraded to `0.0 / false`; full ExecSpec-Repair recovered reward/success to `1.0 / true` for both tested mismatches. The best repair method by mean recovery was `diagonal_affine_calibration`, and full repair mean recovery fraction was `1.0`.

Consequence: STATE 3 may broaden replay/rollout validation cautiously, while preserving exact-init controls, split discipline, no eval leakage, and no paper-grade claims.

Training happened: false. LoRA training happened: false. Loss was computed: false. Replay/rollout happened: true, bounded exact-init calibrated repair replay only. GPU/download/OpenVLA-OFT happened: false. Paper-grade claim: false.

## STATE 3 Multi-Demo Replay Validation Result

Decision: kill or reframe the broad ExecSpec-Repair RA-L route instead of advancing to STATE 4 as a general method claim.

Reason: STATE 3 expanded to `3` held-out eval demos (`805` eval action samples) with `5` calibration demos (`1403` calibration action samples) and no eval-action leakage. Full repair remained strong on held-out action drift: mean action L2 was identity `0.554723914`, clipping-only `0.554723914`, global affine `0.318936592`, and full ExecSpec-Repair `0.0`; full repair beat identity, clipping-only, and global affine on aggregate action drift.

Replay result: exact-init replay ran `21` demo/mismatch cases across all seven mismatch types. Wrong executable specs degraded `19` cases. Full repair recovered reward/success/done behavior in `17 / 19` degraded cases, for success and reward recovery rates of `0.894736842`. Recovery appeared across all `3` held-out eval demos and all `7` mismatch types where degradation was observed.

Blocker: simple baselines matched full repair in `4` degraded replay cases, especially global/range-scale cases on held-out living-room demos where global affine also reached reward/success. Two rotation-scale cases did not degrade under wrong spec. This violates the predeclared STATE 3 continuation criterion that clipping-only or global affine must not match full repair.

Default reset: a tiny non-primary default-reset sanity check failed for both expert replay and full repair (`0.0 / false`), so the claim remains limited to exact-init executable-spec repair under matched replay conditions.

Consequence: do not advance the broad ExecSpec-Repair claim to STATE 4 as-is. Honest continuations are to reframe around mismatch classes where simple affine repair is insufficient, build a harder executable-spec benchmark with predeclared nontrivial baselines, or select a new rollout-first route.

Training happened: false. LoRA training happened: false. Loss was computed: false. Replay/rollout happened: true, bounded exact-init diagnostic only. GPU/download/OpenVLA-OFT happened: false. Paper-grade claim: false.
