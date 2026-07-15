# Epoch 4 Cycle 12 Prior Mechanism Map

Date: 2026-07-15 KST

Purpose: select the next method after `G3P-VLA` stopped before rollout as
`DATA_OR_SUPERVISION_FAILURE`. G3P is not a closed-loop scientific kill, but
the fixed Stage 0 stop is valid for that protocol. It must not be rescued by
changing the material-point threshold, label construction, source gate,
validation search, Stage 0 criteria, or policy list.

## Local Constraints From Prior Results

The next method must not be:

- another 3D point, 2D point, nearest-object, or future-waypoint label rescue
  of G3P;
- another adaptive chunking, queue-commitment, entropy/dispersion, or
  fixed-replan variant of EAC or RCV;
- another prior-query or spectral-capacity PESA rescue;
- another median-anchor, static L1 mixture, or disagreement-gated MARC rescue;
- another dynamic arm/gripper route residual like DAGR;
- another milestone-frame sampling or retained-frame MTF rescue;
- another reflective consequence-calibration wrapper like RAC;
- another failure-aware residual field like FANG;
- another action-evolved state controller like EvoState;
- another nearest-memory contrastive action method like CAVM;
- another photometric-only perturbation ensemble like PSE or GCAP.

The next method should change the mechanism axis away from output-action
correction, queue scheduling, point labels, memory replay, and visual
preprocessing. The strongest current opportunity is to make the intermediate
representation that reaches the action expert explicitly action-structured,
while preserving frozen SmolVLA behavior by default.

## Close Sources

### CAC-VLA

Full title: Context-Gated Action Conditioning for Vision-Language-Action
Models.

URL: https://arxiv.org/abs/2607.04816

AUTHOR_STATED:

- CAC-VLA identifies a gap between VLM representations and continuous motor
  control: visual-language features are not explicitly optimized for action
  conditioning.
- It predicts coarse-to-fine latent actions from visual-language context,
  using future action segments encoded by an ordered action tokenizer as
  supervision.
- It injects latent-action conditioning into the action expert through
  cross-attention and a context gate, rather than treating the latent action as
  a fixed command.
- It reports `98.3%` average success on LIBERO and `89.5%` on LIBERO-Plus.

INDEPENDENTLY_INFERRED:

- The key positive mechanism is not generic LoRA, action residual correction,
  or prompt rewriting. The causal claim is that a VLM-native latent-action
  interface gives the action expert a structured summary of future action
  intent, and the context gate prevents unreliable latent guidance from
  dominating expert control.
- The prior uses future action segments only as training supervision. Inference
  uses predicted latent actions from deployment inputs, so a local version can
  be made leakage-safe if the label and split gates are enforced.
- The closest prior does not automatically solve the local identity-preserving
  risk: directly injecting latent-action updates into a strong frozen SmolVLA
  flow expert could globally disturb action chunks unless the gate is
  initialized to Base passthrough and action deltas are bounded.
- Official code or checkpoints were not located during the Cycle 12 scan, so
  the first local comparison must use a faithful transparent proxy unless exact
  official equivalence is later established.

CROSS_PAPER_SYNTHESIZED:

- CAC-VLA, VLS, and World Pilot all move useful structure closer to the action
  generator. CAC-VLA is the most local-budget-compatible because its
  supervision is already present in robot trajectories as future action
  segments.
- CAC-VLA differs from G3P: it does not need object pose, point labels, depth,
  or waypoint materiality thresholds. Its Stage 0 can audit action-latent
  variance and predictability directly from demonstration actions and
  deployment-observable inputs.
- CAC-VLA also differs from MARC/DAGR/MTF/PESA: it is not an output action
  residual, component route, frame selector, or prior-query adapter. The
  representation is a latent summary of future action structure that conditions
  hidden action states.

Mechanism fields:

- observation/input: current RGB, proprioception, language instruction, Base
  SmolVLA features or action-expert hidden states available through the local
  implementation path; no reset identity, success label, future observation, or
  simulator object state at inference;
- learned representation: predicted latent action vector or token set
  summarizing a bounded future action segment;
- supervision: discovery/validation-only future action segments from
  demonstration trajectories, encoded by a frozen local action tokenizer or
  deterministic DCT/PCA/OAT-lite encoder;
- objective: latent-action prediction loss, action imitation loss for the
  adapter path, clean-retention loss, bounded action-delta penalty, and
  validation score combining latent predictability, clean retention, mechanism
  activation, action validity, and simple-baseline margin;
- policy component changed: action-expert conditioning adapter or hidden-state
  conditioning path only, not the VLM backbone or final action values as an
  unconditional replacement;
- action-generation mechanism: Base SmolVLA remains the default; latent-action
  conditioning can add a bounded, gated hidden-state update;
- inference-time intervention: predicted latent action from current
  deployment-observable inputs, passed through a zero-initialized context gate;
- assumed feedback: current observation and proprioception only;
- benchmark condition: official paired LIBERO manifest after Stage 0, bounded
  validation search, mechanism smoke, and checkpoint reload pass;
- primary metric: task-balanced official closed-loop success, paired deltas,
  latent predictability, action validity, clean retention, mechanism activation,
  latency, and VRAM;
- demonstrated causal link externally: CAC-VLA reports strong LIBERO and
  LIBERO-Plus success with latent-action conditioning and context-gate
  ablations;
- untested causal link locally: whether a small, identity-preserving
  SmolVLA-compatible latent-action adapter can improve closed-loop behavior
  beyond Base, a CAC-style proxy, a no-gate ablation, and a task-mean latent
  action baseline.

### STRONG-VLA

Full title: STRONG-VLA: Decoupled Robustness Learning for Vision-Language-Action
Models under Multimodal Perturbations.

URL: https://arxiv.org/abs/2604.10055

AUTHOR_STATED:

- VLA policies are fragile under visual and language perturbations.
- Joint robust training can create a conflict between robustness and clean task
  fidelity.
- STRONG-VLA separates robustness acquisition from clean task-aligned
  refinement.
- It reports gains up to `12.60%` on OpenVLA, `14.48%` on OpenVLA-OFT, and
  `16.49%` on `pi0` under seen perturbations, with additional gains on unseen
  perturbations.

INDEPENDENTLY_INFERRED:

- The useful mechanism is the training schedule and data partition, not a new
  action representation.
- Local implementation is feasible because image and language perturbations can
  be generated from existing demonstrations, but prior local perception-repair
  failures mean Stage 0 must prove actual perturbed-condition headroom and clean
  retention before rollout.
- A local variant should not become a PSE/GCAP rescue; it must include
  multimodal perturbation partitions, decoupled robust-then-clean training, and
  a direct comparison against a joint-augmentation proxy.

CROSS_PAPER_SYNTHESIZED:

- STRONG-VLA is strong on robustness but weaker than CAC-VLA for the immediate
  local campaign because it optimizes input robustness rather than the
  representation that reaches the action expert.
- It remains a serious backup route if latent-action supervision fails, because
  it has a strong positive prior and clear clean-retention governance.

### VLS

Full title: VLS: Steering Pretrained Robot Policies via Vision-Language Models.

URL: https://arxiv.org/abs/2602.03973

AUTHOR_STATED:

- VLS is a training-free framework for steering frozen diffusion or
  flow-matching robot policies at inference time.
- It uses VLM-synthesized trajectory-differentiable reward functions to guide
  denoising toward trajectories satisfying spatial and task requirements.
- It reports a `31%` improvement on CALVIN and a `13%` gain on LIBERO-PRO, with
  real-world Franka deployment.

INDEPENDENTLY_INFERRED:

- VLS is attractive for identity preservation because policy weights remain
  frozen.
- Its local feasibility is weaker: a faithful version needs a deployable
  reward source, differentiable access to the SmolVLA flow sampling process, and
  an inference-time compute budget that does not silently use privileged
  simulator state.
- A local proxy would have to be very explicit about omitted VLM reward
  synthesis and any non-differentiable approximations.

CROSS_PAPER_SYNTHESIZED:

- VLS and CAC-VLA both target the gap between static imitation priors and
  test-time action needs. VLS steers sampling with rewards; CAC-VLA trains an
  action-structured latent interface.
- Given repeated local failures from action-value interventions, VLS is
  promising but should follow a source-fidelity and compute-feasibility gate
  before selection.

### VLA Grounder

Full title: VLA Grounder: Language-Conditioning Space Optimization for
Black-Box VLA Models.

URL: https://arxiv.org/abs/2607.04517

AUTHOR_STATED:

- VLA behavior can be sensitive to how an instruction is phrased.
- The method learns a language-conditioning policy that rewrites a human
  instruction into a VLA-grounded command while keeping the downstream VLA
  frozen.
- It optimizes the command policy with sparse rollout rewards and reports
  improved success on frozen `pi0` and OpenVLA backbones in object-grounding and
  multi-object manipulation settings.

INDEPENDENTLY_INFERRED:

- The method changes the conditioning input rather than action weights, so it
  is non-invasive.
- Local use would require online validation rollouts for command optimization
  and a reliable non-privileged command generator. That is higher leakage and
  compute risk than CAC-VLA's offline action-latent Stage 0.

CROSS_PAPER_SYNTHESIZED:

- VLA Grounder is useful as a future language-conditioning route, especially if
  a method cycle targets instruction sensitivity. It is not the strongest Cycle
  12 selection because the current local evidence does not yet isolate
  instruction phrasing as the main residual failure mechanism.

## Cycle 12 Opportunity

The strongest post-G3P opportunity is `CALA-VLA`: Context-Gated Action-Latent
Adapter for frozen SmolVLA.

It is anchored primarily to CAC-VLA. The local extension is to make the CAC
mechanism compatible with a frozen SmolVLA flow policy and the current
honest-performance governance:

- future action segments may be used only as discovery/validation training
  labels, never as inference input;
- latent-action labels must be noncollapsed, task/phase-covered, and
  predictable above trivial action-only and task-mean baselines from
  deployment-observable inputs;
- the context gate is initialized to exact Base passthrough;
- rollout is forbidden if the latent-action module is nonacting, globally
  destructive, not predictably different from a task-mean latent prior, or
  indistinguishable from the no-gate ablation;
- the closest-prior comparison remains a transparent CAC-style proxy until
  official code/checkpoint equivalence is established.

This changes the axis relative to G3P, EAC, PESA, MARC, DAGR, MTF, RAC, FANG,
EvoState, CAVM, and RCV:

- representation: latent future-action structure rather than point labels,
  uncertainty, spectral queries, median anchors, route labels, milestone
  frames, consequence histories, failure memories, evolved states, or memory
  retrieval;
- supervision: future action segment encoding from demonstrations, not
  simulator object pose, success labels, or confirmatory outcomes;
- objective: latent-action prediction and bounded gated hidden-state
  conditioning, not final 7D action correction as the novelty;
- policy generation: Base remains default, with a context-dependent hidden
  update only when the learned gate activates;
- claim axis: whether action-structured latent conditioning improves
  closed-loop success beyond Base, a CAC-style proxy, a no-gate ablation, and a
  task-mean latent-action baseline.

The critical Reviewer B baselines are:

- unmodified frozen SmolVLA;
- `cac_vla_latent_action_proxy`, a faithful transparent proxy for the closest
  prior's latent-action conditioning when official equivalence is not
  established;
- `cala_full`;
- `cala_no_context_gate_ablation`;
- `task_mean_latent_action_baseline`, the strongest simple reviewer-killer
  testing whether task-level action prototypes explain any gain.
