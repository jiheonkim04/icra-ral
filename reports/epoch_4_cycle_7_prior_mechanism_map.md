# Epoch 4 Cycle 7 Prior Mechanism Map

Date: 2026-07-14 KST

Purpose: select the next method after the closed MTF-VLA Stage B kill. MTF-VLA is archived as `MTF_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD` and must not be rescued by retuning `mtf_r20_ret100`, changing retention, changing task/reset identities, changing the policy list, or reinterpreting the fixed Stage B result.

## Local Constraints From Prior Results

The next method must not be:

- another milestone-frame selection or retained-frame sampling method like MTF;
- another ordinary LoRA or uniform data-reweighting variant without a stronger mechanism;
- another no-retention variant of MTF, since the no-retention ablation explained the full method;
- another Reflective-style consequence calibration wrapper like RAC;
- another success/failure residual field like FANG;
- another action-evolved state controller like EvoState;
- another nearest-memory contrastive action method like CAVM;
- another no-context replanner or stateless chunk reset like RCV;
- another photometric perturbation ensemble like PSE.

MTF showed that local adapter training can produce disk-reloadable, bounded policies, but the claimed retention/milestone mechanism did not improve closed-loop behavior. The no-retention ablation reached `32 / 40`, while MTF full reached `26 / 40`. A new method may still use an identity-preserving adapter, but the novelty must be in a different representation, objective, supervision, or action-generation mechanism.

## Close Sources

### DAM-VLA

Full title: DAM-VLA, Dynamic Action Model for Vision-Language-Action Policies.

URL: https://arxiv.org/abs/2603.00926

AUTHOR_STATED:

- Manipulation actions are heterogeneous: arm motion, wrist rotation, and gripper decisions do not require identical temporal or representational treatment.
- DAM-VLA introduces dynamic action routing, specialized arm and gripper action models, and dual-scale action weighting.
- It reports superior simulated and real-world manipulation success compared with less structured action decoders.

INDEPENDENTLY_INFERRED:

- The central positive prior is action-factorization, not another data-selection scheme.
- A direct local reproduction is unavailable in the current SmolVLA stack because the official architecture/checkpoint is not installed.
- A faithful transparent proxy is feasible: split the 7D action into translation, rotation, and gripper groups; learn group-specific residuals and route gates from deployment-observable inputs; compare against a static component-weighted adapter and a simple gripper-transition heuristic.
- The local method must show that dynamic routing matters beyond a shared residual adapter and beyond a static arm/gripper weighting proxy.

CROSS_PAPER_SYNTHESIZED:

- MTF and StructVLA both touched gripper transitions as frame-selection signals, but did not route action generation by component.
- DAM-VLA suggests a stronger local claim axis: separate action components should be generated or corrected differently because their closed-loop failure modes differ.
- Identity-preserving integration is natural because the router can default to base passthrough and only allow bounded group-specific residuals.

Mechanism fields:

- observation/input: official SmolVLA RGB observations, language tokens, 8D proprioceptive state, base 7D action chunk, task key, and chunk phase;
- learned representation: deployment-observable arm/rotation/gripper route logits and group-specific residual action features;
- supervision: expert-minus-base residual action targets split into translation, rotation, and gripper groups, plus route labels from group-wise action change and gripper-transition events on discovery/validation identities only;
- objective: group-normalized Huber residual loss with route-gated group masks, route-label cross entropy or focal loss, and a clean action-delta regularizer; no KL over deterministic actions;
- policy component changed: lightweight route head and group residual adapter around frozen SmolVLA actions;
- action-generation mechanism: base action plus clipped group residuals, initialized to zero and gated by route confidence;
- inference-time intervention: one ordinary base policy call plus a small router/residual module; no privileged state, reward, or future observation;
- assumed feedback: none at inference;
- benchmark condition: standard official LIBERO paired manifest, with action-group diagnostics and clean retention;
- primary metric: paired closed-loop success and task-balanced success;
- demonstrated causal link externally: dynamic action routing and specialized arm/gripper models improve manipulation success;
- untested causal link locally: whether a small identity-preserving arm/gripper router can improve SmolVLA beyond a DAM-style static proxy, a shared residual ablation, and a gripper heuristic.

### ReactVLA

Full title: ReactVLA: Fast and Lightweight Reactive Robot Manipulation via Improved Mean Flow Action Generation.

URL: https://arxiv.org/abs/2606.14255

AUTHOR_STATED:

- Slow action generation limits reactive manipulation.
- Improved Mean Flow action generation and attention residual routing improve precision-task performance and inference latency.
- Reported results include stronger LIBERO/RoboIMI performance and real-world latency below `38.6 ms`.

INDEPENDENTLY_INFERRED:

- ReactVLA is a strong positive prior for changing action generation rather than adding post-hoc rankers or wrappers.
- A direct local reproduction would require architecture-level action decoder training that is heavier than the current single-GPU cycle.
- A local lightweight method can borrow only the action-generator discipline: avoid extra VLA calls, keep inference cheap, and report latency.

CROSS_PAPER_SYNTHESIZED:

- If Cycle 7 uses a route head, it must be lightweight enough that the first comparison includes latency and cannot hide a success gain behind a large inference budget.

### AffordanceVLA And GEAR-VLA

URLs:

- AffordanceVLA: https://arxiv.org/abs/2606.06155
- GEAR-VLA: https://arxiv.org/abs/2606.08530

AUTHOR_STATED:

- AffordanceVLA introduces Which2Act, Where2Act, and How2Act affordance forecasting to connect semantics and embodied action generation, reporting strong simulated and real-world performance.
- GEAR-VLA reports geometry-aware action representations, semantic-aligned 3D integration, and embodiment canonicalization across LIBERO, LIBERO-Plus, RoboTwin, and real robots.

INDEPENDENTLY_INFERRED:

- These are strong positive priors for manipulation-centric intermediate representations.
- Direct local reproduction is weak because dense affordance labels, 3D teachers, and official compatible checkpoints are not locally available.
- A local candidate could use weak contact/affordance proxies from state and gripper traces, but it must first prove label health and predictability from deployment inputs.

CROSS_PAPER_SYNTHESIZED:

- Affordance and geometry priors are reviewer boundaries for any action-routing method: if DAGR claims action-group routing, it must not overclaim true 3D affordance learning.

### PDF And TT-VLA

URLs:

- PDF: https://arxiv.org/abs/2604.18107
- TT-VLA: https://arxiv.org/abs/2601.06748

AUTHOR_STATED:

- PDF reports test-time perturbation learning with delayed feedback, uncertainty-based augmentation, action voting, and a lightweight perturbation module.
- TT-VLA reports on-the-fly test-time reinforcement learning with dense task-progress rewards.

INDEPENDENTLY_INFERRED:

- Both are positive priors for adaptation after deployment begins.
- Local versions are high-risk because PSE already showed that simple observation perturbation can explain related behavior, and progress/delayed-feedback labels are not yet verified as non-privileged and noncollapsed.

CROSS_PAPER_SYNTHESIZED:

- These priors should remain candidate boundaries, not the selected Cycle 7 path, unless a development audit first proves delayed-feedback or progress signals are observable from deployment-time inputs.

## Cycle 7 Opportunity

The strongest post-MTF opportunity is `DAGR-VLA`: Dynamic Arm-Gripper Routing for frozen SmolVLA adaptation.

It is a DAM-VLA-anchored prior extension. Instead of selecting training frames or adding a global residual, DAGR factorizes the 7D action into translation, rotation, and gripper groups and learns a lightweight route-gated residual module initialized to base passthrough. It tests whether action-component specialization can improve closed-loop success beyond a static component-weighted DAM proxy, a shared-residual ablation, and a simple gripper-transition heuristic.

This changes at least four dimensions relative to MTF:

- representation: action-group route logits rather than milestone-frame scores;
- supervision: expert-minus-base group residuals and route labels rather than high-frame imitation plus retained-frame targets;
- objective: group-normalized residual routing loss rather than milestone/retention data objective;
- policy generation: route-gated clipped group residuals around a frozen action rather than an unchanged adapted policy call;
- claim axis: action-component specialization and gripper/arm timing rather than training-stream informativeness.

The critical Reviewer B baselines are:

- unmodified frozen SmolVLA;
- a faithful transparent DAM-style static arm/gripper weighted proxy;
- `DAGR-VLA` full;
- a no-dynamic-route shared residual ablation;
- one simple gripper-transition heuristic killer baseline.
