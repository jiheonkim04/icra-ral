# Epoch 4 Cycle 6 Prior Mechanism Map

Date: 2026-07-14 KST

Purpose: select the first method after the closed RAC-VLA Stage B kill. RAC is archived as `STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT` and must not be rescued by retuning `rac_h4_a0.05`, changing the hidden shift, adding another consequence-context ablation, or reinterpreting the fixed Stage B result.

## Local Constraints From Prior Results

The next method must not be:

- another Reflective-style consequence calibration wrapper like RAC;
- another action-evolved state controller like EvoState;
- another success/failure residual field like FANG;
- another nearest-memory or contrastive action method like CAVM;
- another no-context replanner or stateless chunk reset like RCV;
- another photometric perturbation ensemble like PSE;
- another generic progress head, verifier, ranker, or action residual without a positive prior and a first-class prior comparison.

Repeated local evidence also warns that generic rank-4 LoRA is not a paper method by itself. In the official closed-loop baseline, frozen SmolVLA reached `74 / 100`, rank-4 LoRA seed 11 reached `74 / 100`, seed 22 reached `68 / 100`, and seed 33 reached `66 / 100`. Any future adapter training route must prove that the new supervision or sampling mechanism matters beyond ordinary LoRA, and it must include a uniform-sampling LoRA simple killer.

## Close Sources

### FrameSkip

Full title: FrameSkip: Learning from Fewer but More Informative Frames in VLA Training.

URL: https://arxiv.org/abs/2605.13757

AUTHOR_STATED:

- Dense robot demonstrations are usually sampled as if every frame provides equal supervision.
- This creates temporal supervision imbalance: long low-change segments dominate while alignment, contact, grasping, and release frames are sparse.
- FrameSkip scores frames using action variation, visual-action coherence, task-progress priors, and gripper-transition preservation.
- It leaves the VLA architecture, action head, objective, and inference procedure unchanged.
- It reports a macro-average success rate of `76.15%` across RoboCasa-GR1, SimplerEnv, and LIBERO versus `66.50%` for full-frame training while retaining `20%` of unique frames in the main setting.

INDEPENDENTLY_INFERRED:

- The main positive prior is a data-layer effect: the same model can learn better when the training stream emphasizes manipulation-critical transitions instead of plateau frames.
- A local official reproduction is not currently installed, but a faithful transparent proxy is feasible because LIBERO demonstrations and prior SmolVLA trace records include actions, gripper values, task keys, phases, and robot states.
- FrameSkip is not identity-preserving by itself. It changes training sample distribution but does not explicitly constrain an adapted policy to remain close to a strong pretrained policy on plateau or clean-retention states.
- Because local standard LoRA did not improve closed-loop success, the next method must test whether critical-frame sampling plus retention-specific supervision changes the outcome, not whether LoRA exists.

CROSS_PAPER_SYNTHESIZED:

- FrameSkip gives the strongest immediate positive-prior anchor because it directly addresses a plausible reason ordinary imitation and LoRA waste capacity.
- StructVLA suggests that the sparse frames should be physically meaningful milestones rather than only high action-norm frames.
- The local method should therefore combine high-information transition sampling with explicit pretrained-policy retention on low-information states, then compare against both a FrameSkip proxy and uniform retained-ratio LoRA.

Mechanism fields:

- observation/input: LIBERO RGB observations when available, 8D proprioceptive state, 7D expert action, task key, timestep or chunk phase, and gripper state;
- learned representation: no new inference-time representation required for the selected route; the learned object is an adapted policy trained on transition-focused and retention-focused sample weights;
- supervision: expert actions on high-information milestone frames plus frozen-base retention targets on plateau or clean-retention frames;
- objective: weighted supervised action imitation with a mathematically simple Huber/L2 action loss and a base-retention loss, not KL over deterministic actions;
- policy component changed: lightweight SmolVLA adapter or LoRA weights; inference procedure remains the ordinary policy call;
- action-generation mechanism: adapted policy emits one action chunk without extra test-time ranking or wrappers;
- inference-time intervention: none beyond loading the selected adapter;
- assumed feedback: no deployment feedback at inference;
- benchmark condition: standard and transition-heavy official LIBERO conditions with clean retention;
- primary metric: paired closed-loop success and task-balanced success;
- demonstrated causal link externally: data-layer frame selection improves VLA success-retention trade-off across multiple benchmarks;
- untested causal link locally: whether structured milestone selection plus base-retention supervision can improve SmolVLA closed-loop success beyond a FrameSkip-style proxy and uniform-sampling LoRA.

### StructVLA

Full title: Beyond Dense Futures: World Models as Structured Planners for Robotic Manipulation.

URL: https://arxiv.org/abs/2603.12553

AUTHOR_STATED:

- Dense future prediction is visually redundant and can accumulate long-horizon plan drift.
- StructVLA predicts sparse physically meaningful structured frames derived from intrinsic kinematic cues such as gripper transitions and kinematic turning points.
- These sparse frames provide spatiotemporal milestones aligned with task progress.
- It reports `75.0%` success on SimplerEnv-WidowX and `94.8%` on LIBERO, plus real-world deployment evidence.

INDEPENDENTLY_INFERRED:

- The useful local mechanism is not a large generative world model; it is the idea that gripper transitions and kinematic turning points identify physically meaningful supervision points.
- A direct StructVLA reproduction would require a world-model architecture, structured-frame tokenization, and training stages that are not locally cheap.
- A local proxy can use the same milestone definition at the data layer: oversample or balance frames around gripper sign changes, velocity extrema, acceleration extrema, and phase boundaries.

CROSS_PAPER_SYNTHESIZED:

- FrameSkip identifies the training-stream imbalance; StructVLA identifies which sparse frames are physically meaningful.
- Combining them gives a testable local method: milestone-transition-focused adapter training, with no new inference-time module and no privileged inference input.
- Reviewer B must force a comparison to a FrameSkip proxy because gripper-transition preservation and action-variation scoring already appear in the closest prior.

Mechanism fields:

- observation/input: demonstration observations and robot state/action traces;
- learned representation: optionally a scalar milestone score for data selection, not an inference-time latent;
- supervision: expert action labels and base-retention labels;
- objective: milestone-weighted imitation plus retention;
- policy component changed: lightweight adapter only;
- action-generation mechanism: ordinary adapted policy action;
- inference-time intervention: none;
- assumed feedback: none at inference;
- benchmark condition: clean and transition-heavy LIBERO task/reset manifests;
- primary metric: closed-loop success and clean retention;
- demonstrated causal link externally: structured physical milestones improve planning-to-control alignment;
- untested causal link locally: whether milestone selection alone, without a world model, improves adapter training.

### TT-VLA

Full title: On-the-Fly VLA Adaptation via Test-Time Reinforcement Learning.

URL: https://arxiv.org/abs/2601.06748

AUTHOR_STATED:

- Existing VLAs rely mostly on supervised fine-tuning or training-time RL and struggle in evolving deployments.
- TT-VLA performs on-the-fly policy adaptation at inference using dense step-by-step task-progress rewards.
- It reports better adaptability, stability, and task success in simulated and real-world settings while preserving trained priors.

INDEPENDENTLY_INFERRED:

- The positive prior is strong, but local reproduction is risky because dense progress reward may require privileged simulator state or task-specific reward engineering.
- Test-time updating on confirmatory identities is not the same as validation retuning, but it must be frozen as part of the method and cannot inspect held-out outcomes to change its update rule.
- A local method would need a non-privileged progress signal observable from RGB/proprioception/instruction, or else it becomes a simulator-reward method with weak real-world relevance.

CROSS_PAPER_SYNTHESIZED:

- TT-VLA is a boundary candidate, not the strongest next selection, because the campaign has repeatedly killed generic progress and recovery heads and lacks a verified dense reward predictor.
- It may become viable only after a Stage 0 progress-label health audit proves noncollapsed labels and above-trivial predictability from deployment inputs.

### AffordanceVLA And DAM-VLA

URLs:

- AffordanceVLA: https://arxiv.org/abs/2606.06155
- DAM-VLA: https://arxiv.org/abs/2603.00926

AUTHOR_STATED:

- AffordanceVLA introduces Which2Act, Where2Act, and How2Act affordance forecasting to bridge VLM semantics and embodied action generation, reporting strong simulated and real-world performance.
- DAM-VLA introduces dynamic action routing, specialized arm/gripper action models, and dual-scale action weighting for complex manipulation, reporting superior simulation and real-world success.

INDEPENDENTLY_INFERRED:

- Both are strong positive priors for structured intermediate representations and action decomposition.
- Direct local reproduction is weak because dense affordance labels, 3D affordance teachers, specialized action models, and the full architecture are unavailable in the current verified SmolVLA stack.
- A reduced local method would risk becoming another hand-engineered gripper/contact adapter unless data and label health are proven first.

CROSS_PAPER_SYNTHESIZED:

- Affordance and arm/gripper decomposition should remain reviewer boundaries for any method that claims manipulation-aware supervision.
- For Cycle 6, they are not as attractive as FrameSkip/StructVLA because the selected route can use existing demonstration labels without introducing unavailable dense affordance supervision.

## Cycle 6 Opportunity

The strongest post-RAC opportunity is `MTF-VLA`: Milestone-Transition Focused VLA Adaptation.

It is a FrameSkip-anchored and StructVLA-informed data-supervision method. It trains a lightweight SmolVLA adapter from a development-only selected training stream that emphasizes physically meaningful transition frames while explicitly retaining the pretrained policy on low-information or clean-retention frames. Inference remains unchanged.

This changes at least four dimensions relative to RAC:

- representation: data-layer milestone and retention scores rather than action-consequence calibration context;
- supervision: expert-action emphasis plus base-retention targets rather than synthetic action-channel shift labels;
- policy generation: ordinary adapted policy action rather than a gated residual wrapper;
- claim axis: improving closed-loop success through transition-balanced supervision and clean retention rather than deployment action-channel calibration.

The critical Reviewer B baselines are:

- unmodified frozen SmolVLA;
- a faithful transparent FrameSkip proxy using action variation, state or visual-action coherence when available, task-progress priors, and gripper-transition preservation;
- `MTF-VLA` full;
- a no-retention ablation using the same milestone sampling but no base-retention term;
- one simple uniform retained-ratio LoRA baseline.
