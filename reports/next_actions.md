# Next Actions

Date: 2026-07-10 KST

Current decision: `PROTOCOL_DRIFT_FOUND`

## Immediate Rule

Do not run official rollout.

The regenerated persisted LoRA checkpoints are complete and deterministic under fixed-seed repeated disk evaluation, but they are not accepted as the canonical rollout baseline set because the historical in-memory evaluation protocol differs from the regenerated persisted-reload protocol and because the saved regenerated artifact metrics differ from fixed-seed disk re-evaluation metrics.

## Evidence To Preserve

- historical result: `reports/official_smolvla_lora_seed_repro_result.json`
- regenerated result: `reports/official_smolvla_lora_checkpoint_regen_result.json`
- checkpoint manifest: `reports/official_smolvla_lora_checkpoint_manifest.json`
- drift audit: `reports/official_smolvla_lora_drift_audit.md`
- config diff: `reports/official_smolvla_old_vs_regen_config_diff.md`
- artifact alignment: `reports/official_smolvla_artifact_alignment_audit.md`
- deterministic eval: `reports/official_smolvla_eval_determinism_check.md`
- training identity status: `reports/official_smolvla_training_determinism_status.md`
- canonical proposal: `reports/official_smolvla_canonical_checkpoint_proposal.md`
- drift decision: `reports/official_smolvla_lora_drift_decision.md`

## Required Before Canonicalization

1. Decide how to handle the PEFT protocol difference:
   - old path: in-memory policy evaluation, no adapter save/reload, no assigned `wrap_with_peft` return;
   - regenerated path: assigned PEFT wrapper, adapter save, `PeftModel.from_pretrained` reload, disk evaluation.
2. Decide and pin the evaluation RNG-state policy, since fixed-seed disk re-evaluation is repeatable but does not exactly reproduce the saved regenerated artifact metrics.
3. Do not claim the regenerated persisted checkpoints are identical to the historical ephemeral runs.
4. Preserve old metrics as historical.
5. If re-baselining is approved, adopt the persisted checkpoint metrics explicitly as a new canonical baseline and update the reproducibility lock with checkpoint hashes and evaluation seed policy.
6. If the protocol difference is treated as a bug, fix the protocol first and run a new explicitly authorized bounded check.

## Still Forbidden

- no closed-loop rollout
- no simulator dependency installation
- no OpenVLA-OFT
- no FCAR revival
- no method design
- no full benchmark
- no asset downloads
- no static-alpha tuning on test
- no favorable-seed-only reporting
- no silent historical metric replacement

## Exact Next Step

Create a no-rollout protocol-adjudication branch that either fixes the PEFT in-memory vs persisted-reload mismatch and pins evaluation RNG state, or explicitly records a re-baselining decision before any canonical rollout baseline is used.

## 2026-07-10 Canonical Next Action

Current decision: `NEEDS_WSL_OR_LINUX_OFFICIAL_ROLLOUT`

Move the same canonical artifacts/checkpoints into the verified WSL/Linux LeRobot LIBERO environment, install only official `lerobot[libero]` dependencies, then run the official smoke before any bounded pilot. Do not retrain, select a LoRA seed from rollout outcomes, revive FCAR, or use the old custom LIBERO_7D route.

## 2026-07-10 WSL Official Rollout Next Action

Current decision: `OFFICIAL_ROLLOUT_BASELINE_READY`

Exact next step: run a larger predeclared official baseline rollout/failure-mining pass with frozen base and all three LoRA seeds, using official videos for failed episodes. Keep static mixes skipped at alpha `0.0`, keep all seeds reported, and do not select a winning LoRA seed or design a new method from the 48-episode pilot alone.

## 2026-07-11 Closed-Loop Scaleup Next Action

Current decision: `OFFLINE_ONLINE_MISMATCH_CONFIRMED`

Exact next step: run a bounded visual review pass on the already identified repeated failures, with official video capture enabled and no policy changes.

Start with:

1. `libero_10/task_4/seed_20260713`
2. `libero_10/task_4/seed_20260715`
3. `libero_spatial/task_4/seed_20260712`
4. `libero_spatial/task_4/seed_20260713`
5. `libero_spatial/task_4/seed_20260714`

Rules for the next pass:

- do not retrain any policy
- do not select seed `11` as best after outcomes
- do not run static-mix duplicates
- do not revive FCAR
- do not design routing, retention, prior, correction, or chunking methods yet
- do not use old custom `LIBERO_7D` or exact-init replay routes
- keep frozen_base and all LoRA seeds paired on identical task/reset cases
- stop at video/phase annotation unless a repeated mechanism is visually supported

The novelty/method-design gate can only reopen if the bounded review converts the current `ambiguous_or_unclassified` failures into a repeated, success-critical, mechanism-linked phase failure that survives frozen-base, LoRA-seed, task, and reset explanations.
