# Next Actions

## 2026-07-15 Epoch 4 Cycle 15 Current Action

Active governance: `reports/current_research_governance.md`

Current decision: `LIFT_PROPOSAL_FROZEN_REVIEWER_ATTACK_PENDING`

Immediate next action: conduct an independent Reviewer B attack on the frozen LIFT proposal, then allow exactly one Researcher A rebuttal before mathematical audit or preregistration.

COVI Stage 0 is complete and preserved. It stopped as `COVI_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE_NO_SCIENTIFIC_KILL`; the objective-gradient ratio exceeded the frozen maximum, action validity failed, and no headroom existed. The one-check set remained sealed and no validation search or rollout ran.

Do not rescue G3P, CALA, RAR, or COVI by changing their frozen labels, objectives, thresholds, source gates, validation configs, or baselines. LIFT must retain its narrow pathwise-flow claim, transparent training-free CAG prior, last-step ablation, three-scale validation cap, and official-LIBERO-CF non-equivalence disclosure.

## 2026-07-13 Governance V2 Current Action

Active governance: `reports/current_research_governance.md`

Current decision: `EPOCH_4_CYCLE_1_RCV_KILLED_CONTINUE_CYCLE_2`

Immediate next action: start Epoch 4 Cycle 2 autonomous research. Generate exactly three candidates under the post-PSE problem-first and external-prior-early quality gate, select exactly one, freeze/hash the Researcher proposal, run Reviewer B novelty and mathematical mechanism attack, then implement unless exact duplication, trivial equivalence, mathematical invalidity, or hard infeasibility is proven.

Do not stop after governance migration, method failure, three historical method failures, or prototype GO unless an allowed final state in `reports/current_research_governance.md` is reached. Epoch 4 Cycle 2 must not be a cosmetic variant of CBFD-VLA, SCVC-VLA, PSE-VLA, or RCV-VLA. In particular, do not rescue RCV by threshold retuning, a renamed verifier, or another receding-chunk replanning ablation; change at least two core dimensions relative to RCV's frozen-policy disagreement verifier, inference-time replanning intervention, and efficiency-versus-stateless claim.

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

## 2026-07-11 Paper-First VLA Method Design Next Action

Current decision: `READY_TO_IMPLEMENT_PRIMARY_VLA_METHOD`

Immediate next action: implement the first bounded `ECHO-VLA` prototype only.

Allowed first implementation scope:

1. Use official SmolVLA-LIBERO as the only first backbone.
2. Use the four predeclared predicate-diversity tasks:
   - `libero_spatial/task_0`
   - `libero_object/task_4`
   - `libero_goal/task_0`
   - `libero_10/task_0`
3. Build a small effect-label dataset from official demonstrations and training-time BDDL/simulator predicate labels.
4. Train only lightweight visual predicate/effect heads and the ECHO counterfactual ranking objective.
5. Compare against frozen SmolVLA, heuristic effect/progress, progress/value head, Pre-VLA-style validity/advantage head, and no-counterfactual ECHO.

Kill before scaling if ECHO full fails to beat the strongest simple baseline by at least `5` absolute task-balanced success points or fails to beat its no-counterfactual ablation.

Still forbidden before the first gate passes:

- no OpenVLA-OFT INT4 validation,
- no full benchmark,
- no LIBERO-PRO,
- no broad failure mining,
- no generic confidence/verification/progress/replanning/chunking method,
- no LoRA, SmolVLA, or quantization as novelty.

## 2026-07-11 ECHO-VLA First Prototype Next Action

Current decision: `NO_ECHO_CANDIDATE_HEADROOM`

Immediate next action: stop the current ECHO implementation path.

Why:

- the focused novelty gate passed, but only for the strict same-state intervention effect-mediator claim;
- the bounded same-state candidate-headroom gate generated `4` intervention groups and `16` candidate records;
- all same-state identity proofs passed;
- oracle realized-effect selection achieved `0.0` percentage-point improvement over the default candidate;
- `0.0` of default-failure states contained a successful or materially better candidate;
- therefore training lightweight ECHO heads would only train a selector over a candidate set with no recoverable headroom.

Allowed next actions:

1. Archive this as a no-headroom kill for the current ECHO candidate generator.
2. If reopening ECHO, design a new candidate generator or longer-horizon state selection protocol first, then freeze a new headroom gate before training.
3. Preserve the no-privileged-inference and same-state-intervention tests.

Still forbidden:

- no ECHO head training on the current no-headroom data;
- no OpenVLA-OFT validation;
- no full benchmark;
- no claim that a local Pre-VLA-style proxy was officially reproduced;
- no causality claim from ordinary demonstration transitions.

## After ECHO Final Candidate Headroom Gate - 2026-07-11

- decision: `NO_ECHO_HEADROOM_CONFIRMED`
- next: `Archive ECHO and return to the paper-first candidate portfolio.`

## After Implementation V2 Empirical Postmortem - 2026-07-12

- decision: `PROTOTYPE_EVIDENCE_INSUFFICIENT_FOR_TERMINAL_CLAIM`
- next: `Do not start another autonomous campaign from the implementation-v2 terminal claim.`

Allowed only if explicitly reopened later:

1. A bounded CensorCredit repair that first proves censored and uncensored labels differ on held-out generated intervention records, then repeats an adequately powered evaluation.
2. A genuinely distinct policy-distribution training method using intervention-generated sequence-level supervision, after targeted novelty review against SDP, TORL-VLA, ConRFT, VLA-Corrector, and OpenVLA-OFT.

Still forbidden:

- no PhaseBarrier threshold tuning as a rescue;
- no CensorCredit hold-strength tuning as a rescue;
- no rollout rerun inside this postmortem;
- no claim that `TWO_IMPLEMENTED_METHODS_KILLED` is a review-resistant scientific terminal result;
- no final method promotion from the current evidence alone.

## After PhaseBarrier Bounded Adjudication - 2026-07-12

- decision: `PHASEBARRIER_COMPONENT_NOT_USEFUL`
- next: `Archive PhaseBarrier; do not rescue or tune it.`

Evidence:

- valid bounded result reused original saved PhaseBarrier weights;
- full PhaseBarrier completed `20/20` held-out episodes and changed actions in every episode;
- full PhaseBarrier success was `0/20`;
- no-phase ablation success was `9/20`;
- frozen SmolVLA success was `8/20`.

Allowed later action:

1. Inspect CensorCredit's documented implementation failure only if explicitly reopened.
2. A CensorCredit repair is allowed only if it first demonstrates that censored and uncensored labels differ and that the intended component changes actions.

Still forbidden:

- no PhaseBarrier threshold tuning;
- no new PhaseBarrier repeat;
- no PhaseBarrier redesign under another name;
- no CensorCredit repair inside the PhaseBarrier adjudication branch.

## After CensorCredit One-Repair Gate and Final Method - 2026-07-12

- decision: `NO_VALID_CENSORCREDIT_REPAIR_FINAL_METHOD_KILLED`
- next: `Stop this autonomous method chain unless new intervention/correction data or genuinely new primary-source gap appears.`

What happened:

- CensorCredit was diagnosed exactly as `LABEL_OR_DATA_FAILURE`.
- No repair was allowed because the objective only allowed concrete implementation or optimization bugs.
- The final distinct candidate, `ISAC-VLA`, was killed before implementation due SDP/TORL-VLA/ConRFT overlap and unavailable paired intervention/correction chunk data.

Allowed reopen conditions:

1. Add real paired negative-policy and corrective-action chunk data from human, robot, or validated intervention simulator.
2. Identify a new primary-source gap not equivalent to action-chunk correction learning, intervention-censored VLA refinement, contact barriers, candidate selection, or post-hoc temporal wrappers.
3. Explicitly ask for a non-paper engineering prototype that is allowed to be incremental rather than RA-L novel.

Still forbidden:

- no CensorCredit relabeling as a repair;
- no CensorCredit hold-strength or threshold tuning;
- no final-method implementation using synthetic local labels as a substitute for intervention chunks;
- no PhaseBarrier rescue or rename.

## After Autonomous Dual-Review RA-L Campaign - 2026-07-11

- decision: `NO_METHOD_AFTER_3_VALID_CYCLES`
- next: `Do not start another generic VLA method cycle from the current evidence.`

Allowed reopen conditions:

- reproduce an official action-representation baseline and find a residual not solved by mean-action, MLP, or the official baseline;
- find a new matched exact-state cross-backbone failure affecting both SmolVLA and Quantized OpenVLA-OFT INT4;
- add new physical robot intervention, tactile/force, or larger-GPU resources;
- identify a new primary-source gap not already occupied by action-conditioning, correction, verification, progress, chunking, contact-adaptation, or prior-preservation papers.

Still forbidden:

- ECHO rescue without a new predeclared headroom gate;
- generic confidence, verification, correction, adaptive chunking, progress, failure-negative, adapter-routing, or LoRA-as-novelty claims;
- main-branch pollution from abandoned method implementations.

## After Autonomous RA-L Research Implementation V2 - 2026-07-11

- decision: `TWO_IMPLEMENTED_METHODS_KILLED`
- next: `Stop autonomous no-method campaign unless the user explicitly reopens a stronger repeat or a new mechanism.`

Reason:

- `PhaseBarrier-VLA` was implemented, trained, evaluated closed-loop, and killed.
- `CensorCredit-VLA` was implemented, trained, evaluated closed-loop, and killed because its key uncensored ablation matched the full method.

Allowed reopen:

- one explicitly requested repeat of CensorCredit on more held-out resets, because it showed a weak positive signal over frozen/simple baselines but failed the ablation gate;
- a genuinely new mechanism that differs from both physical feasibility projection and temporal credit/action-history blending.

## After ECHO Final Candidate Headroom Gate - 2026-07-11

- decision: `NO_ECHO_HEADROOM_CONFIRMED`
- next: `Archive ECHO and return to the paper-first candidate portfolio.`
