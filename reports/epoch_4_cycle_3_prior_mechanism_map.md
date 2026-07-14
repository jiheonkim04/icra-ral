# Epoch 4 Cycle 3 Prior Mechanism Map

Date: 2026-07-14 KST

Purpose: select the first post-CAVM method under the performance-oriented governance. CAVM remains fixed and archived as `STAGE_2B_EXPANDED_NON_GO_NO_THIRD_EXPANSION`; this map must not reinterpret, rescue, retune, or extend the CAVM protocol.

## Local Constraints From Prior Results

The next method must not be:

- another CAVM threshold, memory reconstruction, or expansion variant;
- another RCV-style verifier, stateless replanner, or chunk reset rule;
- another Epoch 3 observation-side teacher, canonicalization, or photometric ensemble;
- a generic confidence, progress, value, candidate-ranker, or action-L2 method.

CAVM did show a small positive signal from success/failure outcome contrast, but the non-parametric nearest-memory formulation produced only `+1 / 58` over the strongest baseline after the only allowed expansion. Therefore a future method may use failure trajectories only if it changes the mechanism in a material way and compares early against the closest external prior and a simple baseline.

## Close Sources

### AFIL

Full title: Failing Forward: Adaptive Failure-Informed Learning for Vision-Language-Action Models.

URL: https://arxiv.org/abs/2605.08434

Status checked: arXiv page verified on 2026-07-14. The page lists the paper as submitted 2026-05-08 and revised 2026-05-12. No official code or checkpoint link was verified from the arXiv record during this pass.

AUTHOR_STATED:

- Success-only behavioral cloning leaves VLA policies brittle under compounding errors.
- AFIL uses failure trajectories as adaptive negative guidance for diffusion- and flow-based VLA policies.
- It trains dual action generators for successful and failed behaviors while sharing a vision-language backbone.
- During sampling, failure guidance steers generation away from failure-prone regions and toward success modes.
- Reported experiments show improved task success and robustness over VLA baselines across in-domain and OOD manipulation.

INDEPENDENTLY_INFERRED:

- AFIL is the strongest positive prior for a post-CAVM method because it already demonstrates that failure trajectories can improve flow/diffusion VLA action generation.
- A generic "use failures as negatives" claim is not novel after AFIL.
- The locally feasible extension is not to reproduce AFIL end to end, but to build a transparent AFIL-style proxy around frozen SmolVLA traces and test a minimal technical difference under matched tasks, inference budget, and baselines.
- The key local risk is disruption: a failure generator can globally push actions away from common approach motions that appear in failed episodes. Identity-preserving integration must be mandatory.

CROSS_PAPER_SYNTHESIZED:

- CAVM used failure traces non-parametrically and barely cleared baselines numerically; AFIL uses a learned dual-generator mechanism and reports positive results.
- A defensible next step is a parametric, bounded, identity-preserving failure-aware guidance head: keep the base action as default, learn success and failure action fields, and convert them into a bounded residual only at inference when validation-only calibration says the field contrast is reliable.
- This is a `PRIOR_EXTENSION`, not an invention of failure-negative VLA learning.

Mechanism fields:

- observation/input: VLA observation, language, proprioception, base action, generated success/failure rollouts;
- learned representation: success and failure action generators sharing a policy backbone;
- supervision: online failure rollouts and successful behavior;
- objective: dual generator training plus failure-informed sampling guidance;
- policy component changed: action generation distribution;
- action-generation mechanism: failure generator repels sampling away from failure-prone regions;
- inference-time intervention: guidance during diffusion or flow sampling;
- assumed feedback: rollout success/failure labels;
- benchmark condition: in-domain and OOD manipulation tasks;
- primary metric: closed-loop success and robustness;
- demonstrated causal link: dual generator/guidance reports positive closed-loop gains;
- untested causal link locally: whether a lightweight frozen-SmolVLA residual proxy can retain base competence while using failure contrast better than nearest-success memory.

### A2C2

Full title: Leave No Observation Behind: Real-time Correction for VLA Action Chunks.

URL: https://arxiv.org/abs/2509.23224

Status checked: arXiv page verified on 2026-07-14. No official code or checkpoint link was verified from the arXiv record during this pass.

AUTHOR_STATED:

- VLA action chunking harms reactivity under inference delay and long horizons.
- A2C2 adds a lightweight per-step correction head using the latest observation, base action, chunk index, and base-policy features.
- It preserves base competence while restoring closed-loop responsiveness.
- Reported gains include `+23` percentage points on Kinetix and `+7` percentage points on LIBERO Spatial compared with RTC.

INDEPENDENTLY_INFERRED:

- A2C2 is a strong positive prior for identity-preserving residual correction.
- It is also dangerous for novelty here: a local version could collapse to a direct action residual head, a family repeatedly killed or penalized in this campaign unless it targets a new mechanism with strong prior anchoring.
- The local setup does not currently have a preregistered delay/stochastic chunk condition that clearly leaves headroom after RCV and CAVM.

CROSS_PAPER_SYNTHESIZED:

- A2C2 supplies a design principle rather than the next core method: small correction heads should be base-action-conditioned and identity-preserving.
- If used as a candidate, it must be selected for a delay/reactivity claim, not for generic action correction.

Mechanism fields:

- observation/input: latest observation, base action, chunk-index positional feature, base features;
- learned representation: real-time correction state;
- supervision: correction learning for chunked policies;
- objective: per-step correction loss;
- policy component changed: action chunk execution at each control step;
- action-generation mechanism: add a small correction to the base chunk action;
- inference-time intervention: lightweight head runs every step;
- assumed feedback: delay/reactivity failure signal;
- benchmark condition: dynamic Kinetix, LIBERO Spatial, long horizons;
- primary metric: closed-loop success under delay and horizon changes;
- demonstrated causal link: reported gains against RTC;
- untested causal link locally: whether this is more than a known residual-correction mechanism on current hard LIBERO tasks.

### DREAM-Chunk

Full title: DREAM-Chunk: Reactive Action Chunking with Latent World Model.

URL: https://arxiv.org/abs/2606.18589

Status checked: arXiv page verified on 2026-07-14. No official code or checkpoint link was verified from the arXiv record during this pass.

AUTHOR_STATED:

- Committed action chunks can be brittle under stochastic dynamics, hardware execution errors, and partial observability.
- DREAM-Chunk augments chunked policies with a lightweight latent world model and requires no policy fine-tuning.
- At test time it samples multiple candidate chunks, predicts latent futures, and selects the chunk whose predicted state best matches observed rollout.
- It improves robustness under action noise and across manipulation tasks, robot platforms, and VLA policies.

INDEPENDENTLY_INFERRED:

- DREAM-Chunk is a positive prior for world-model-assisted chunk reactivity.
- It overlaps with candidate selection and verifier/ranker routes if reduced to "sample chunks and choose one".
- ECHO already found no useful oracle headroom among local policy candidates in its target setting; any DREAM-like local method needs a different headroom audit before implementation.

CROSS_PAPER_SYNTHESIZED:

- DREAM-Chunk suggests using observed execution mismatch as feedback during chunk execution.
- The locally feasible version would need a latent effect predictor trained from traces and a small candidate budget, but its decisive experiment is less direct than AFIL because candidate generation and world-model validation are both unresolved locally.

Mechanism fields:

- observation/input: chunked policy observations, candidate action chunks, observed rollout state;
- learned representation: latent world model;
- supervision: latent future prediction;
- objective: predict future latent state under candidate chunks;
- policy component changed: chunk selection at test time;
- action-generation mechanism: sample multiple chunks and select by predicted/observed latent match;
- inference-time intervention: test-time scaling over candidates;
- assumed feedback: observed rollout matching;
- benchmark condition: stochastic dynamics, action noise, partial observability;
- primary metric: robustness and closed-loop success;
- demonstrated causal link: reported robustness gains under stochasticity;
- untested causal link locally: whether frozen SmolVLA can produce useful candidate diversity under the selected hard tasks without replaying the killed ECHO route.

### Pre-VLA And VeriSpace

URLs:

- Pre-VLA: https://arxiv.org/abs/2605.22446
- VeriSpace: https://arxiv.org/abs/2606.10568

Status checked: arXiv pages verified on 2026-07-14.

AUTHOR_STATED:

- Pre-VLA predicts safety confidence and critic-derived advantage for candidate action chunks, then filters or resamples under a limited computation budget. Its LIBERO result improves average closed-loop success from `30.79%` to `37.62%` over RynnVLA-002.
- VeriSpace is a 3D-aware verifier for test-time action selection, using scene encoding and spatially grounded action reasoning. It reports gains over underlying policies and prior verification methods.

INDEPENDENTLY_INFERRED:

- Both are strong positive priors for verification and candidate selection.
- Both make a new verifier/ranker contribution hard to defend locally after RCV and ECHO.
- The current campaign should use them as negative boundary conditions and possible baselines, not as the core selected method.

CROSS_PAPER_SYNTHESIZED:

- The 2026 literature has largely occupied generic action verification, safety confidence, advantage heads, spatial candidate verification, and resampling. A viable post-CAVM method should change action generation or training guidance rather than adding another verifier.

Mechanism fields:

- observation/input: observations plus candidate action chunks;
- learned representation: safety/advantage or 3D spatial verifier;
- supervision: verifier labels, critic advantage, spatial validity/progress;
- objective: classification/regression or spatial reasoning over candidates;
- policy component changed: candidate selection before execution;
- action-generation mechanism: filter, resample, or select among candidates;
- inference-time intervention: verifier/ranker call before physical execution;
- assumed feedback: safety, advantage, or geometric progress labels;
- benchmark condition: LIBERO and public/real manipulation tasks;
- primary metric: success and decision reliability;
- demonstrated causal link: reported verifier gains;
- untested causal link locally: whether such verification leaves any residual novelty after RCV and ECHO.

## Cycle 3 Opportunity

The strongest post-CAVM opportunity is AFIL-anchored failure-aware action guidance with strict identity preservation:

successful and failed frozen-policy traces may train parametric local success/failure action fields that are smoother than nearest-neighbor CAVM memory, but only if validation proves the derived residual is bounded, acts selectively, and beats an AFIL-style transparent proxy plus the nearest-success simple baseline.

This changes at least four dimensions relative to CAVM:

- representation: parametric success/failure action fields instead of non-parametric action means;
- objective: supervised dual-head residual learning with clean retention instead of retrieval calibration only;
- action-generation mechanism: identity-preserving residual guidance around the frozen action instead of memory target blending;
- development protocol: bounded validation search before confirmatory testing instead of a single fixed memory configuration.

It also changes at least three dimensions relative to RCV:

- supervision: rollout success/failure action traces instead of queued-vs-fresh disagreement;
- intervention: continuous residual guidance instead of replan/reset decisions;
- prior anchor: AFIL positive failure-guidance prior instead of verifier/replanner literature.
