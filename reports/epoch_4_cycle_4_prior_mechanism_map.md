# Epoch 4 Cycle 4 Prior Mechanism Map

Date: 2026-07-14 KST

Purpose: select the first method after the valid FANG-VLA Stage B kill. FANG is archived as `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT` and must not be rescued, threshold-tuned, or reframed after seeing Stage B.

## Local Constraints From Prior Results

The next method must not be:

- another success/failure action-field residual like FANG;
- another contrastive nearest-memory method like CAVM;
- another no-context replanner, verifier, or stateless chunk-reset rule like RCV;
- another photometric perturbation ensemble like PSE unless the mechanism is materially different and the closest prior is tested early;
- another generic action residual, value head, progress head, or candidate ranker without a new mechanism and positive external prior.

FANG showed that a learned success/failure residual can act safely enough to roll out, but Stage B full FANG reached only `11 / 40`, below Base at `16 / 40`, AFIL proxy at `15 / 40`, nearest-success replay at `14 / 40`, and tied with its no-failure ablation at `11 / 40`. Therefore failure-aware action guidance is not the next route.

## Close Sources

### EvoScene-VLA

Full title: EvoScene-VLA: Evolving Scene Beliefs Inside the Action Decoder for Chunked Robot Control.

URL: https://arxiv.org/abs/2605.21862

AUTHOR_STATED:

- Chunked VLA control uses low-frequency perception to drive high-frequency execution.
- Current-frame spatial representations and temporal frame aggregation do not maintain an action-updated scene prior across chunks.
- EvoScene carries a recurrent geometry-aware scene state inside the action decoder and updates it with both actions and new observations.
- Reported gains include improved fixed and randomized RoboTwin success and real-robot improvement over baselines.

INDEPENDENTLY_INFERRED:

- The strongest useful mechanism is not "more memory" but action-updated belief: the method predicts how the scene should evolve under the executed chunk, then reconciles that prior with the next observation.
- A direct local reproduction is infeasible because it depends on scene tokens, depth/3D teachers, and architecture-level action decoder modification.
- A locally feasible proxy can test the same causal idea in the low-dimensional state/action interface already used by official SmolVLA: maintain an action-evolved proprioceptive latent state, measure observed-vs-predicted mismatch, and apply bounded model-based corrections only when the mismatch is controllable and validation-calibrated.

CROSS_PAPER_SYNTHESIZED:

- RCV showed that simply refreshing or discarding chunks can be beaten by no-context/stateless baselines.
- DREAM-Chunk shows that latent future prediction can improve chunk robustness under stochastic execution.
- EvoScene suggests the missing piece is persistent action-updated state, not just current-frame replanning.
- A defensible next local method should therefore be a compact action-evolved state model with an identity-preserving correction rule, compared early against DREAM/AAC-style chunk baselines and a simple inverse-dynamics controller.

Mechanism fields:

- observation/input: current 8D robot state, base 7D action, previous action, task key, chunk phase, and an internal predicted state;
- learned representation: compact action-evolved proprioceptive latent state and controllable mismatch direction;
- supervision: next-state prediction from non-confirmatory frozen SmolVLA traces;
- objective: one-step and short-horizon state prediction with calibration of controllable mismatch;
- policy component changed: execution of the current chunk action, not the VLA backbone;
- action-generation mechanism: bounded inverse-dynamics correction toward the action-evolved expected state;
- inference-time intervention: default base action unless a validation-calibrated mismatch gate activates;
- assumed feedback: deployment-observable proprioceptive state only;
- benchmark condition: controlled chunk/execution mismatch and retained clean hard-task behavior;
- primary metric: closed-loop success and clean retention;
- demonstrated causal link externally: action-updated state priors improve chunked control in EvoScene and latent world models improve robust chunk execution in DREAM-Chunk;
- untested causal link locally: whether proprioceptive action-evolved state is enough without privileged object state or scene-token teachers.

### DREAM-Chunk

Full title: DREAM-Chunk: Reactive Action Chunking with Latent World Model.

URL: https://arxiv.org/abs/2606.18589

AUTHOR_STATED:

- Committed action chunks are brittle under stochastic dynamics, hardware execution errors, and partial observability.
- DREAM-Chunk uses a lightweight latent world model at test time, samples candidate chunks, predicts latent futures, and selects by latent agreement with observed rollout.
- It reports robustness gains under action noise and across manipulation tasks, robot platforms, and VLA policies.

INDEPENDENTLY_INFERRED:

- DREAM-Chunk is the closest comparison for latent-world-model chunk robustness.
- A local method that only samples and ranks chunks would collide with ECHO, RCV, Pre-VLA, VeriSpace, and other verifier routes.
- The useful transplant is latent mismatch as a deployment signal, not generic best-of-N action selection.

CROSS_PAPER_SYNTHESIZED:

- DREAM supplies the validation baseline: if a lightweight local latent model cannot outperform a transparent DREAM-style proxy, it is not a paper candidate.
- The local experiment must separate three explanations: world-state mismatch, simple adaptive chunking, and static inverse dynamics.

Mechanism fields:

- observation/input: observations, candidate chunks, and observed rollout state;
- learned representation: latent world dynamics;
- supervision: latent future prediction;
- objective: predict future latent state under candidate chunks;
- policy component changed: chunk choice or chunk execution;
- action-generation mechanism: test-time latent model over possible futures;
- inference-time intervention: select or adjust chunk;
- assumed feedback: observed rollout matching;
- benchmark condition: stochastic dynamics/action noise/partial observability;
- primary metric: robustness and success;
- demonstrated causal link externally: reported robustness gains from latent future matching;
- untested causal link locally: whether frozen SmolVLA traces provide enough state-transition coverage.

### PDF

Full title: Test-Time Perturbation Learning with Delayed Feedback for Vision-Language-Action Models.

URL: https://arxiv.org/abs/2604.18107

AUTHOR_STATED:

- VLAs can overfit to trajectory-like visual correlations and become brittle under small environmental shifts.
- PDF uses uncertainty-based observation augmentation, action voting, and delayed-feedback perturbation learning while freezing the base VLA.
- It reports LIBERO gains and code availability.

INDEPENDENTLY_INFERRED:

- PDF is the strongest positive prior for test-time perturbation, but it is close to the local PSE failure if reduced to photometric voting.
- SmolVLA continuous 7D actions are not discrete logits, so any local PDF proxy must avoid invalid action-token entropy or KL assumptions.
- PDF is useful as a comparison and design warning, but less attractive as the next selected method because PSE already tested observation perturbation/ensemble behavior and lost to a simple bright-single baseline.

CROSS_PAPER_SYNTHESIZED:

- Use PDF as a boundary condition for any future perturbation method.
- Do not select another perturbation ensemble unless it has delayed-feedback learning or another mechanism beyond view averaging.

### AffordVLA And ProgressVLA

URLs:

- AffordVLA: https://arxiv.org/abs/2605.17517
- ProgressVLA: https://arxiv.org/abs/2603.27670

AUTHOR_STATED:

- AffordVLA aligns VLA intermediate visual representations with manipulation-centric affordance representations from a zero-shot affordance teacher.
- ProgressVLA learns robust progress estimation and uses an inverse-dynamics world model for progress-guided action refinement.
- Both report improved simulation and real-world manipulation performance.

INDEPENDENTLY_INFERRED:

- These are strong positive priors for representation-level action guidance.
- Local feasibility is weaker: affordance teachers, dense affordance labels, future visual latents, and action-token gradients are not already available in the verified SmolVLA stack.
- A local version may still become viable after a data audit, but it should not be selected before proving label generation and inference-time observability.

CROSS_PAPER_SYNTHESIZED:

- Affordance and progress are good second-wave routes if action-updated state fails.
- They require stronger data-health evidence than EvoState because their labels are less directly present in the existing traces.

## Cycle 4 Opportunity

The strongest post-FANG opportunity is `EvoState-VLA`: an EvoScene/DREAM-anchored method that uses existing non-confirmatory trace rows to learn a compact action-evolved state model, then applies a bounded identity-preserving correction when observed state deviates from the predicted state in a controllable direction.

This changes at least four dimensions relative to FANG and CAVM:

- representation: action-evolved proprioceptive latent state instead of success/failure action fields or nearest outcome memory;
- supervision: next-state dynamics from trace transitions instead of terminal success/failure contrast;
- action-generation mechanism: model-based mismatch correction instead of residual guidance away from failures;
- claim axis: chunk/execution mismatch robustness plus clean retention instead of failure-negative action shaping.

It also changes at least three dimensions relative to RCV:

- persistent state prior instead of stateless/no-context replanning;
- model-based controllability gate instead of verifier threshold;
- explicit comparison against DREAM/AAC-style latent/chunk priors and static inverse dynamics.
