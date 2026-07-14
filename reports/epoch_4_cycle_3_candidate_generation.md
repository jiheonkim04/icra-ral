# Epoch 4 Cycle 3 Candidate Generation

Date: 2026-07-14 KST

Decision: `SELECT_FANG_VLA`

Governance applied: post-CAVM performance-oriented research design. Exactly three candidates were generated and scored. CAVM results remain fixed and are used only as prior evidence, not as a source for retuning the CAVM method.

## Candidate 1: FANG-VLA

Name: `FANG-VLA`, Identity-Preserving Failure-Aware Negative Guidance for Frozen VLAs.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: AFIL, Failing Forward: Adaptive Failure-Informed Learning for Vision-Language-Action Models, https://arxiv.org/abs/2605.08434.

Positive prior result: AFIL reports that dual success/failure action generators with failure-informed guidance improve task success and robustness for diffusion/flow VLA policies across in-domain and OOD manipulation tasks.

Official code/checkpoint/reproducible mechanism: no official repository or checkpoint link was verified from the AFIL arXiv record during this pass. The reproducible local mechanism is a transparent AFIL-style proxy: dual success/failure action predictors trained from frozen SmolVLA rollout traces, with the base SmolVLA action kept as the default action.

Assumption extended: AFIL changes the action generator itself. The local extension tests whether the same failure-guidance principle can be made safe for a frozen off-the-shelf SmolVLA through a zero-initialized, validation-calibrated residual gate.

Minimal technical difference proposed by Ours:

- learn success and failure action fields around the frozen 7D action, then derive a bounded residual at inference rather than replacing the action;
- initialize the residual path at base-policy passthrough;
- add a reliability gate calibrated on validation traces;
- include a clean-retention/action-delta penalty;
- compare against an AFIL-style local proxy that lacks the identity-preserving gate and against nearest-success replay.

Why it could improve the same claim axis: CAVM showed that outcome contrast can act but non-parametric memory was too weak. AFIL shows that learned failure guidance can be positive. FANG tests whether parametric negative guidance can preserve base competence while producing a stronger and smoother action change than local memory replay.

### Quality Screen

Provisional novelty:

- Distinct from AFIL because the core technical object is a frozen-policy residual guidance field with base-passthrough initialization and validation-calibrated activation, not end-to-end dual action generator training.
- Distinct from CAVM because it uses a trained parametric residual objective, validation search, and identity-preserving integration rather than nearest-neighbor action memory.
- Not a renamed loss only; the policy mechanism is a bounded, gated residual action-generation path.

Prior-anchor strength:

- Strong positive anchor from AFIL.
- Local faithful proxy is feasible using existing success/failure frozen SmolVLA traces.
- Matched comparison can use the same backbone, tasks, reset identities, inference budget, and five-policy manifest.

Mechanism plausibility:

- Problem condition -> frozen SmolVLA failures contain repeated local action patterns that terminal failure labels alone expose coarsely.
- Intermediate failure mechanism -> non-parametric memory is sparse and noisy; success/failure neighborhoods can conflict in common approach phases.
- Policy representation/action behavior -> dual action-field heads learn smooth success and failure action fields conditioned on state, base action, previous action, task, and chunk phase.
- Closed-loop failure -> bad local actions compound into task failure.
- Proposed method -> residual gate activates only where success/failure predictions separate and validation reliability is high.
- Intended action behavior -> small bounded shift toward success residual and away from failure residual.
- Expected improvement -> better held-out closed-loop success than AFIL proxy, no-failure ablation, nearest-success replay, and Base, with clean retention.

Data and supervision viability:

- Existing CAVM acquisition data provide `10801` non-confirmatory trace rows with terminal success labels, two hard task keys, 8D state, 7D actions, previous action, chunk fraction, and task identity.
- Quick development count: `1761` success-labeled rows and `9040` failure-labeled rows across `16` identities and two tasks.
- Both tasks have successes and failures: `libero_spatial/task_4` has `791` success rows and `2800` failure rows; `libero_10/task_4` has `970` success rows and `6240` failure rows.
- Confirmatory CAVM Stage 2B identities `20260922..20260950` are not used for training or validation.
- Privileged terminal success labels are used only for training; inference uses no success label.

Identity-preserving integration:

- Residual branch initialized to output zero.
- Gate initialized to base-policy passthrough.
- Residual norm clipped per 7D action.
- Clean retention and action-delta diagnostics required before rollout.
- Any configuration that globally changes actions receives an implementation/design failure, not a closed-loop result.

Decisive experiment feasibility:

- Stage 0 development audit checks label health, separability, and disruption risk.
- Bounded validation search uses at most six configurations on discovery/validation identities.
- First serious rollout uses exactly five policies: Base, AFIL local proxy, FANG full, no-failure/no-gate ablation, and nearest-success replay.
- Stage A uses about 10 paired episodes per policy; Stage B uses at least 40 paired episodes per key policy if non-catastrophic.

Score:

- provisional novelty: `19 / 25`
- importance of problem: `13 / 15`
- strength of positive prior anchor: `19 / 20`
- technical mechanism quality: `17 / 20`
- data/supervision feasibility: `9 / 10`
- decisive experiment feasibility: `9 / 10`
- total: `86 / 100`

## Candidate 2: DREAM-Lite-VLA

Name: `DREAM-Lite-VLA`, Lightweight Latent Execution-Mismatch Chunk Guidance.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: DREAM-Chunk, Reactive Action Chunking with Latent World Model, https://arxiv.org/abs/2606.18589.

Positive prior result: DREAM-Chunk reports improved robustness of action-chunking policies under stochastic dynamics, action noise, and partial observability by sampling candidate chunks and selecting by predicted/observed latent-state agreement.

Official code/checkpoint/reproducible mechanism: no official repository or checkpoint link was verified from the arXiv record during this pass. The local reproducible proxy would train a small latent next-state predictor on frozen traces and use a small candidate budget.

Assumption extended: DREAM-Chunk uses a latent world model at test time. The local extension would test a much smaller execution-mismatch model for frozen SmolVLA without full policy fine-tuning.

Minimal technical difference proposed by Ours:

- train a trace-local latent effect model from state/action/phase features;
- use only a small candidate set from available SmolVLA stochastic action chunks;
- choose or gently adjust the chunk when predicted state drift matches observed execution mismatch.

Why it could improve the same claim axis: it targets brittleness during committed chunks, a known VLA issue with a positive prior.

### Quality Screen

Provisional novelty:

- Some novelty as a local lightweight version, but candidate selection and latent future matching are directly close to DREAM-Chunk and prior verifier/ranker routes.

Prior-anchor strength:

- Strong prior, but local reproduction depends on useful candidate diversity and latent prediction quality.

Mechanism plausibility:

- Problem condition -> committed chunks are brittle under stochastic execution.
- Intermediate failure mechanism -> the current chunk becomes inconsistent with observed execution.
- Proposed method -> latent effect predictor detects mismatch and selects or adjusts a better candidate chunk.
- Expected outcome -> more reactive execution under perturbation.

Data and supervision viability:

- Existing traces include state/action sequences, but not a dedicated stochastic dynamics or action-noise condition.
- Candidate diversity may be weak locally; ECHO previously found no useful oracle headroom for candidate ranking on its tested axis.

Identity-preserving integration:

- Can default to base chunk when latent confidence is low.
- Still risks becoming a candidate ranker or replanner under another name.

Decisive experiment feasibility:

- Feasible only after a new headroom audit proves useful candidate diversity under a preregistered stochastic perturbation condition.
- More moving parts than FANG: latent model, candidate generation, and mismatch selection must all work.

Score:

- provisional novelty: `14 / 25`
- importance of problem: `12 / 15`
- strength of positive prior anchor: `18 / 20`
- technical mechanism quality: `14 / 20`
- data/supervision feasibility: `5 / 10`
- decisive experiment feasibility: `6 / 10`
- total: `69 / 100`

## Candidate 3: TACR-VLA

Name: `TACR-VLA`, Time-Aware Chunk Residual Correction for Frozen SmolVLA.

Contribution type: `PRIOR_EXTENSION`.

Closest external prior: A2C2, Leave No Observation Behind: Real-time Correction for VLA Action Chunks, https://arxiv.org/abs/2509.23224.

Positive prior result: A2C2 reports consistent improvements under inference delay and long horizons, including `+23` percentage points on Kinetix and `+7` percentage points on LIBERO Spatial compared with RTC.

Official code/checkpoint/reproducible mechanism: no official repository or checkpoint link was verified from the arXiv record during this pass. The local reproducible mechanism is a small correction head conditioned on state, base action, previous action, and chunk phase.

Assumption extended: A2C2 shows that per-step correction can restore reactivity. The local extension would test whether correction can be trained only from existing frozen traces.

Minimal technical difference proposed by Ours:

- remove base-policy internal features not accessible in the current runner;
- train a small correction head from local hard-task traces;
- initialize the correction to zero and enforce clean retention.

Why it could improve the same claim axis: if SmolVLA errors are chunk-reactivity errors, a small correction may improve execution without full fine-tuning.

### Quality Screen

Provisional novelty:

- Low to moderate. This is very close to A2C2 and to prior action residual correction attempts in this campaign.

Prior-anchor strength:

- Strong positive external prior, but the local problem condition is weaker because no new delay or dynamic condition is currently preregistered.

Mechanism plausibility:

- Problem condition -> stale action chunks miss latest observations.
- Proposed method -> time-aware residual corrects each step.
- Expected outcome -> improved reactivity and long-horizon success.

Data and supervision viability:

- Local traces contain states/actions/chunk fraction but not oracle corrective targets.
- Training from successful traces alone risks behavior cloning the same base policy and repeating a killed residual route.

Identity-preserving integration:

- Straightforward zero residual and action clipping.
- High disruption-risk penalty if the head learns global corrections.

Decisive experiment feasibility:

- Implementation is easy, but the experiment would likely test known A2C2-like correction rather than a defensible new method.

Score:

- provisional novelty: `9 / 25`
- importance of problem: `10 / 15`
- strength of positive prior anchor: `17 / 20`
- technical mechanism quality: `11 / 20`
- data/supervision feasibility: `7 / 10`
- decisive experiment feasibility: `8 / 10`
- total: `62 / 100`

## Selection

Selected method: `FANG-VLA`.

Selection reason:

FANG has the strongest positive external prior, the cleanest local supervision, and the most decisive first comparison. It also directly absorbs the CAVM lesson without rescuing CAVM: outcome contrast may matter, but the non-parametric memory formulation was too weak. The selected method must now be treated as a new AFIL-anchored method cycle with its own development-only audit, bounded validation search, mathematical objective audit, proposal hash, reviewer attack, preregistration, and confirmatory freeze.

Immediate next steps:

1. Freeze Researcher A's FANG proposal and hash it.
2. Run Reviewer B's independent attack against AFIL, A2C2, DREAM-Chunk, Pre-VLA, VeriSpace, and the simplest residual/nearest-memory baselines.
3. Perform a development-only headroom/data audit before any expensive training or rollout.
4. If the audit passes, implement the smallest FANG training and validation smoke.
