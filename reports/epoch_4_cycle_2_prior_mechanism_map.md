# Epoch 4 Cycle 2 Prior Mechanism Map

Date: 2026-07-13 KST

Purpose: identify a non-RCV method family after `RCV-VLA` was validly killed by no-context and stateless baselines. The next method must not be another renamed verifier, threshold tuner, or receding-chunk replanning variant.

## Close Sources

### Retrieve-then-Steer / Online Success Memory

- URL: https://arxiv.org/html/2605.10094v1
- AUTHOR_STATED: repeated deployment is not independent zero-shot evaluation; successful test-time experience should be stored, retrieved, consistency-filtered, aggregated into an elite action prior, and injected into a generative VLA sampler without parameter updates.
- INDEPENDENTLY_INFERRED: the method is success-only at memory construction time. Failed trajectories are mostly discarded rather than used as local negative evidence. Its progress critic and flow-sampler injection are not directly available in this repository's frozen SmolVLA runner, so a faithful local proxy must be labeled as a proxy.
- CROSS_PAPER_SYNTHESIZED: success memory is promising, but after RCV's no-context ablation kill, a useful local method needs more than "retrieve similar successful action"; it needs to prove that success-conditioned memory adds information not captured by nearest-success replay, stateless replanning, or task/reset bias.

Mechanism fields:

- observation/input: RGB, proprioception, instruction, accumulated successful deployment traces
- learned representation: retrieval key and elite action prior
- supervision: successful executions and progress-calibrated prefixes
- objective: non-parametric retrieval/aggregation, no policy gradient
- policy component changed: action sampler initialization/guidance
- action-generation mechanism: success-memory prior steers generated action chunk
- inference-time intervention: retrieve/filter/aggregate prior before action generation
- assumed feedback: success or progress signal after/within episode
- benchmark condition: persistent local deployment, LIBERO-10 and SimplerEnv
- primary metric: closed-loop success and stability
- demonstrated causal link: component ablations support memory/guidance utility
- untested causal link locally: whether failure traces contain action-contrast signal unavailable to success-only memory

### HELM

- URL: https://arxiv.org/abs/2604.18791
- AUTHOR_STATED: long-horizon VLA failures arise from memory, verification, and recovery gaps; HELM combines episodic memory, a learned state verifier, and a harness controller with rollback/replanning.
- INDEPENDENTLY_INFERRED: HELM's core state verifier plus rollback overlaps with verification/recovery methods and with the now-killed RCV family if reduced to a local verifier. Its useful distinction is memory-conditioned verification, not generic failure prediction.
- CROSS_PAPER_SYNTHESIZED: memory and recovery should be separated from a standalone verifier. A local method should avoid making "verification" the contribution and instead use memory as an action-generation prior with explicit baselines.

Mechanism fields:

- observation/input: observation, proposed action, subgoal, episodic memory
- learned representation: memory-conditioned failure verifier
- supervision: long-horizon success/failure and perturbation recovery labels
- objective: predict action failure before execution
- policy component changed: harness controller around frozen VLA
- action-generation mechanism: rollback/replan when verifier predicts failure
- inference-time intervention: memory retrieval, verifier, rollback
- assumed feedback: success/failure and perturbation recovery signal
- benchmark condition: LIBERO-LONG, CALVIN, LIBERO-Recovery
- primary metric: long-horizon task success and recovery success
- demonstrated causal link: ablations isolate memory-conditioned verifier and harness
- untested causal link locally: whether action memory can improve without rollback, subgoal labels, or privileged recovery state

### LaMem-VLA / Dual Latent Memory

- URL: https://arxiv.org/abs/2607.07608
- AUTHOR_STATED: mainstream VLAs assume current observations are sufficient; LaMem reconstructs history into short- and long-term latent memory tokens and weaves them directly into the VLA embedding sequence.
- INDEPENDENTLY_INFERRED: this is architecture/training-heavy and not locally feasible as a first prototype for frozen SmolVLA without modifying internal latent sequences or retraining. The useful gap is that memory should influence action formation natively, not merely appear as an external report.
- CROSS_PAPER_SYNTHESIZED: a local method can test the weaker but feasible version: memory influences the actual 7D action vector through a mathematically explicit action-prior field, while direct latent-token integration remains a final-paper or different-backbone issue.

Mechanism fields:

- observation/input: current observation/instruction plus short- and long-term history
- learned representation: latent memory tokens in the model's native embedding space
- supervision: long-horizon manipulation data
- objective: end-to-end memory-conditioned action learning
- policy component changed: embedding sequence and action formation
- action-generation mechanism: memory tokens interleaved with reasoning/action tokens
- inference-time intervention: memory retrieval and weaving
- assumed feedback: historical experience relevance
- benchmark condition: SimplerEnv and LIBERO
- primary metric: closed-loop success on long-horizon tasks
- demonstrated causal link: paper reports memory-aware gains
- untested causal link locally: whether a frozen-policy external memory prior can approximate any of that gain

### Harness VLA

- URL: https://arxiv.org/abs/2607.08448
- AUTHOR_STATED: frozen VLAs are strong contact-rich primitives but brittle under retargeting, layout shifts, and unstable contacts; an agentic harness composes frozen VLA primitives with fixed analytic primitives and learned operating ranges.
- INDEPENDENTLY_INFERRED: the method depends on a richer primitive library, semantic re-grounding, and planner-level staging that are not available in the current LIBERO-only SmolVLA setup. A direct local reproduction is infeasible.
- CROSS_PAPER_SYNTHESIZED: the useful transfer is not "add an agent"; it is treating frozen VLA behavior as a local primitive whose reliable operating range can be learned from traces. That can be approximated by outcome-contrastive local action memory.

Mechanism fields:

- observation/input: language, scene, execution traces, fixed primitive library
- learned representation: primitive operating ranges and failure models
- supervision: success rules and execution traces
- objective: compose/retry primitives under learned reliability bounds
- policy component changed: planner/harness around frozen VLA
- action-generation mechanism: analytic primitives plus frozen VLA local contact phases
- inference-time intervention: semantic re-grounding, staging, retry
- assumed feedback: execution success/failure
- benchmark condition: LIBERO-Pro, RoboCasa365, RoboTwin C2R
- primary metric: perturbed deployment success
- demonstrated causal link: large improvements under deployment perturbations
- untested causal link locally: whether local trace reliability can improve action chunks without external primitives

### Affordance Field Intervention

- URL: https://arxiv.org/abs/2512.07472
- AUTHOR_STATED: VLAs fall into memory traps under distribution shifts by replaying memorized trajectories; 3D spatial affordance fields can intervene by proposing high-affordance waypoints and scoring trajectories.
- INDEPENDENTLY_INFERRED: the core 3D SAF requirement is not locally available without depth/3D affordance assets or a new perception model. A purely 2D heuristic would likely collapse to a simple visual retargeting baseline.
- CROSS_PAPER_SYNTHESIZED: memory-trap language is relevant, but the feasible local test should not pretend to be AFI. If selected, it must compare against simple image/position baselines and disclose missing 3D affordance supervision.

Mechanism fields:

- observation/input: RGB plus 3D spatial affordance field
- learned representation: actionable spatial affordance map
- supervision: affordance/waypoint cues
- objective: choose high-affordance trajectories under shift
- policy component changed: waypoint/action trajectory selection
- action-generation mechanism: VLA actions anchored by SAF waypoints
- inference-time intervention: trap detection, waypoint proposal, affordance scoring
- assumed feedback: proprioceptive trap detection and affordance geometry
- benchmark condition: OOD real robot tasks and LIBERO-Pro
- primary metric: robustness under distribution shift
- demonstrated causal link: reported backbone improvements
- untested causal link locally: non-privileged 3D affordance availability

### Critical-Moment Uncertainty and Adaptive Chunking

- URLs: https://arxiv.org/html/2603.18342v1 and https://arxiv.org/html/2604.04161v2
- AUTHOR_STATED: critical segments and action-chunk duration matter; global uncertainty or fixed chunking can hide transient failure signals.
- INDEPENDENTLY_INFERRED: after RCV, another chunk/replan/uncertainty detector is high-risk unless it changes the intervention family and beats stateless/no-context baselines.
- CROSS_PAPER_SYNTHESIZED: these papers remain useful as baselines or diagnostics, but not as the next core contribution.

## Cycle 2 Opportunity

The strongest locally feasible gap is not memory alone. Existing memory methods emphasize successful experience, latent history, or harnessed rollback. The missing locally testable interaction is:

success and failure traces from the same frozen policy may define a local action-advantage field, where useful intervention depends on how nearby successful actions differ from nearby failed actions, not on success-only retrieval.

This changes at least four dimensions relative to RCV:

- representation: outcome-contrastive trace memory instead of queued-vs-fresh disagreement features;
- supervision: episode success/failure outcome labels instead of self-disagreement labels;
- objective: local action-advantage estimation instead of replan classification;
- action generation: continuous action-prior adjustment instead of binary chunk reset/replan.
