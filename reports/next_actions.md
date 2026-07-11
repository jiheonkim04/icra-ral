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

## 2026-07-11 Closed-Loop Visual Gate Next Action

Current decision: `NO_SAFE_RA_L_METHOD_YET`

Do not implement a method from the current evidence.

The bounded video review found real visible failures, but not a safe RA-L method route:

- `libero_spatial/task_4` shows a drawer/bowl stable-grasp extraction failure on only two independent rerun-failure reset seeds.
- `libero_10/task_4` shows a different multi-object long-horizon failure.
- `8/24` same-identity reruns changed success status, so original failure identity is not stable enough for a narrow causal method claim.
- recent work kills generic confidence, verification, correction, adaptive chunking, progress/recovery, failure-negative learning, and adapter-routing routes.

Allowed next actions:

1. Archive this method gate as a no-implementation result.
2. If reopening later, collect new bounded visual evidence only for a predeclared mechanism and stop once either three independent reset seeds or two tasks are verified.
3. Before any implementation, predeclare a second-backbone plan, a second-benchmark plan, and simple baseline kill tests.

Still forbidden:

- no LoRA training as a method contribution
- no best-seed selection
- no FCAR revival
- no generic correction/replanning/chunking/progress method
- no full sweep just to rescue the gate
- no paper claim from SmolVLA-only evidence

## 2026-07-11 Cross-Model Gate Next Action

Current decision: `SECOND_BACKBONE_OR_BENCHMARK_BLOCKED`

Do not implement a method.

The selected second backbone is `OpenVLA-OFT` with checkpoint `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`. The selected second benchmark is `LIBERO-PRO`.

Immediate next action:

1. Ask for explicit approval before downloading the `14.845` GiB OpenVLA-OFT checkpoint.
2. Choose the hardware path:
   - preferred: lab GPU path with 24GB+ VRAM per inference process;
   - local RTX 5080 16GB only with a predeclared offload/quantization risk note, because official full-precision inference is not proven.
3. After approval, run only the frozen protocol in `reports/cross_model_failure_manifest.json`.

Still forbidden:

- no SmolVLA retraining
- no OpenVLA-OFT fine-tuning
- no FCAR revival
- no method implementation
- no full benchmark
- no generic retry/progress/verification/replanning/chunking method
- no claim that either mechanism generalizes before OpenVLA-OFT and LIBERO-PRO evidence exists

## 2026-07-11 Quantized OpenVLA-OFT Gate Next Action

Current decision: `FAILURE_NOT_REPRODUCED_IN_SECOND_ARCHITECTURE`

Do not implement a method and do not proceed to LIBERO-PRO from the current evidence.

Why:

- quantized OpenVLA-OFT INT4 succeeded on all exact hard-slice and matched-control episodes (`20/20`)
- SmolVLA frozen-base still failed on the exact hard slices (`libero_spatial/task_4 = 1/5`, `libero_10/task_4 = 1/5`)
- visual comparison therefore does not support a shared cross-backbone mechanism
- INT4 is quantized, so this is not a full-precision OpenVLA-OFT claim

Allowed next actions:

1. Archive the result as `FAILURE_NOT_REPRODUCED_IN_SECOND_ARCHITECTURE`.
2. If continuing later, predeclare a new second-backbone or full-precision hardware run before any method design.
3. Keep LIBERO-PRO blocked unless a future decision is one of the confirmed cross-backbone failure decisions.

Still forbidden: no FCAR revival, no LoRA training, no OpenVLA-OFT fine-tuning, no generic correction/chunking/progress method, no LIBERO-PRO run from this decision.
