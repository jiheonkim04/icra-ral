# MTF-VLA Researcher A Proposal

Date: 2026-07-14 KST

Method: `MTF-VLA`, Milestone-Transition Focused VLA Adaptation.

Contribution type: `CROSS_PAPER_SYNTHESIS`.

Closest external prior: FrameSkip, https://arxiv.org/abs/2605.13757.

Secondary positive prior: StructVLA, https://arxiv.org/abs/2603.12553.

## One-Sentence Claim

Training a lightweight VLA adapter on physically meaningful manipulation-transition frames while explicitly retaining the frozen pretrained policy on low-information frames can improve closed-loop success over frozen SmolVLA, a FrameSkip-style proxy, a no-retention ablation, and a uniform-sampling LoRA simple baseline.

## Positive Prior Anchor

FrameSkip reports that data-layer selection of informative frames improves VLA success-retention trade-off across RoboCasa-GR1, SimplerEnv, and LIBERO while leaving architecture and inference unchanged. Its core positive mechanism is that dense teleoperation trajectories overrepresent low-change frames and underrepresent alignment, contact, grasping, and release.

StructVLA reports that sparse physically meaningful structured frames derived from gripper transitions and kinematic turning points can better align planning and low-level control. MTF-VLA uses those physical cues only as a local data-selection and retention mechanism, not as a world-model claim.

MTF-VLA does not claim to reproduce FrameSkip or StructVLA officially. The first experiment includes a transparent FrameSkip proxy as the closest-prior comparison.

## Local Problem

The official SmolVLA closed-loop baseline leaves meaningful headroom: frozen SmolVLA reached `74 / 100` on the official paired LIBERO scale-up, while rank-4 LoRA seeds did not reliably improve it (`74 / 100`, `68 / 100`, and `66 / 100`).

This suggests that simply adding adapter capacity is not enough. A plausible failure mechanism is that ordinary training or fine-tuning spends much of its gradient budget on abundant low-change frames and can disrupt a strong pretrained policy without specifically improving rare manipulation transitions.

## Proposed Method

MTF-VLA builds a development-only scored training stream.

For each demonstration frame or action-chunk record, compute:

- action variation score from local changes in 7D expert action;
- gripper-transition score from gripper sign or open/close changes;
- kinematic-turning score from local changes in robot state velocity or action curvature;
- phase-coverage bin from normalized trajectory progress.

The high-milestone set contains per-task and per-phase top-scoring frames. The retention set contains low-score frames and clean-retention frames where the adapted policy should remain close to the frozen base.

The adapter is trained with:

- weighted expert imitation on high-milestone frames;
- frozen-base retention on low-information frames;
- no KL between deterministic 7D actions;
- no privileged inference input;
- no test-time ranking, replanning, perturbation ensemble, or consequence wrapper.

Inference is ordinary: load the selected checkpoint and call the policy once in the same way as the base policy.

## Falsifiable Mechanism Chain

Observed condition:

- Dense demonstration training contains long low-change plateaus and sparse manipulation-critical transitions.

Intermediate failure mechanism:

- Generic adapters receive many gradients from plateau behavior and too few from contact, grasp, release, alignment, or turning frames.

Policy behavior:

- Standard LoRA changes actions without reliably improving transition states and may disturb clean pretrained behavior.

Closed-loop outcome:

- Success-critical transitions remain unreliable, and overall closed-loop success does not improve.

Proposed method:

- Select physically meaningful transition frames and pair them with base-retention frames.

Intended internal change:

- Adapter gradients concentrate on transition-relevant states while retention gradients preserve base behavior elsewhere.

Intended action behavior:

- Bounded action differences appear primarily near transition-like states; low-score states remain base-like.

Expected closed-loop improvement:

- Higher task-balanced success on held-out paired LIBERO rollouts with clean retention.

## Data And Supervision

Required data:

- official LeRobot LIBERO demonstrations;
- 7D action chunks or first actions;
- proprioceptive state;
- gripper component;
- task key;
- trajectory or chunk phase;
- frozen-base action targets for retention frames.

No required inference input is privileged. Development may use expert demonstrations and frozen-base outputs. Confirmatory rollouts use only the frozen selected checkpoint and ordinary observations.

## Development Partitions

`DISCOVERY`:

- historical official SmolVLA and LoRA closed-loop evidence;
- existing non-confirmatory reports and trace records;
- literature mechanism map;
- data/label construction logic.

`VALIDATION`:

- held-out development identities and demonstration splits used for selecting one retained-frame ratio and one retention coefficient;
- optional small validation rollouts or the closest feasible closed-loop proxy;
- all tried configurations saved.

`CONFIRMATORY_TEST`:

- task/reset manifest frozen only after proposal, audit, validation search, checkpoint selection, baselines, ablation, metrics, and thresholds are frozen.
- outcomes may not retune MTF-VLA.

Stage 0 must persist split manifests and prove zero identity overlap.

## Bounded Search

At most six configurations:

1. retained high-frame ratio `0.20`, retention coefficient `0.25`
2. retained high-frame ratio `0.20`, retention coefficient `0.50`
3. retained high-frame ratio `0.20`, retention coefficient `1.00`
4. retained high-frame ratio `0.30`, retention coefficient `0.25`
5. retained high-frame ratio `0.30`, retention coefficient `0.50`
6. retained high-frame ratio `0.30`, retention coefficient `1.00`

Use at most two lightweight training seeds per selected configuration if needed. Do not add configurations after seeing confirmatory results.

Validation selection score:

- validation closed-loop success or closest feasible proxy: `35%`
- clean retention: `25%`
- milestone activation and high/low score health: `20%`
- action validity and bounded deltas: `10%`
- compute overhead: `10%`

Offline action L2 alone cannot select the final configuration.

## First Serious Comparison

Exactly five policies:

1. `base_smolvla`
2. `frameskip_proxy_lora`
3. `mtf_full`
4. `mtf_no_retention_ablation`
5. `uniform_retained_ratio_lora`

`frameskip_proxy_lora` is the closest external-prior proxy. It uses the same retained ratio and a faithful transparent approximation of FrameSkip scoring, but without MTF's base-retention objective.

`uniform_retained_ratio_lora` is the one strongest simple reviewer-killer baseline. If it matches or beats MTF, then the selected sampling and retention mechanism is not supported.

## Required Success Conditions

MTF-VLA becomes a paper candidate only if:

- full MTF beats frozen Base;
- full MTF beats the FrameSkip proxy;
- full MTF beats the no-retention ablation;
- full MTF beats uniform retained-ratio LoRA;
- clean behavior is retained;
- the milestone-retention mechanism is active and bounded;
- the result is not explained by best-seed selection.

If Stage B shows that the FrameSkip proxy, no-retention ablation, or uniform LoRA matches or beats full MTF, the method is killed or classified as explained by that baseline.

## Second Backbone And Second Condition

If SmolVLA reaches prototype GO, immediately plan the Quantized OpenVLA-OFT INT4 integration and same-backbone comparison:

- Quantized OpenVLA-OFT INT4
- Quantized OpenVLA-OFT INT4 plus MTF-style selected training or the closest feasible adapter/selection equivalent

Do not train OpenVLA-OFT before SmolVLA GO and a risk-assessed local feasibility plan.

The second condition should directly test the transition-supervision claim, such as a held-out transition-heavy LIBERO slice or a clean-retention condition with predeclared task/reset identities.

## Hard Stops Before Rollout

Stop before expensive training or rollout if:

- frame scores collapse to all high or all low;
- high and low sets are not distinct;
- task or phase coverage collapses;
- split overlap is nonzero;
- frozen-base retention targets cannot be generated and persisted;
- the full and no-retention ablation receive effectively identical targets;
- the adapter is not identity-preserving at initialization;
- preliminary action deltas are globally destructive;
- the closest-prior FrameSkip proxy cannot be constructed transparently.
