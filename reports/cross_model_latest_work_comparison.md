# Cross-Model Latest Work Comparison

Date: 2026-07-11 KST

This comparison is scoped as a readiness audit. Because no cross-backbone rollout ran, no method route is allowed to claim novelty.

## Stable-Grasp Route

If the drawer/bowl `stable_grasp` failure later generalizes, the nearest recent work is already crowded:

| Work | Source | Relevance |
| --- | --- | --- |
| GraspCorrect | `https://arxiv.org/abs/2503.15035` | Directly targets unstable robotic grasping through VLM-guided feedback, grasp-guided prompting, and object-aware candidate sampling. |
| CRAFT | `https://arxiv.org/abs/2602.12532` | Force-aware curriculum fine-tuning for contact-rich manipulation across VLA architectures. |
| HapticVLA | `https://arxiv.org/abs/2603.15257` | Tactile-aware VLA training and distillation for contact-rich manipulation without inference-time tactile sensing. |
| UniTacVLA | `https://arxiv.org/abs/2606.31723` | Tactile understanding/prediction and high-frequency tactile-action correction for contact-rich manipulation. |
| VLA-Corrector | `https://arxiv.org/abs/2607.01804` | Generic event-triggered correction/replanning route already blocks monitor-and-correct claims. |

Rejected stable-grasp method framings:

- predict contact;
- distill tactile or force information;
- use force-aware fine-tuning;
- use VLM grasp feedback;
- correct or resample grasp candidates.

## Long-Horizon Route

If the LIBERO-10 `long_horizon_compounding` failure later generalizes, the nearest recent work is also crowded:

| Work | Source | Relevance |
| --- | --- | --- |
| VLA-Reasoner | `https://arxiv.org/abs/2509.22643` | Test-time world-model rollout and MCTS for long-horizon reasoning. |
| AFIL | `https://arxiv.org/abs/2605.08434` | Uses online failure rollouts as adaptive negative guidance for VLA policies. |
| FAR | `https://arxiv.org/abs/2607.01111` | Failure-aware retry at test time plus continual policy improvement from recovery trajectories. |
| SPR | `https://arxiv.org/abs/2603.09292` | Progress monitoring, spatial subgoals, and rewind-style recovery. |
| ProgressVLA | `https://arxiv.org/abs/2603.27670` | Progress estimator and differentiable progress guidance. |
| ProgVLA | `https://arxiv.org/abs/2605.28231` | Progress heads and advantage/success-weighted flow imitation. |
| VLA-Corrector | `https://arxiv.org/abs/2607.01804` | Corrective replanning after latent visual dynamics divergence. |

Rejected long-horizon method framings:

- replan more often;
- estimate progress;
- retry after failure;
- sample more action candidates;
- use a world model to rank trajectories;
- train on failure rollouts as negative data.

## Novelty Status

No review-resistant method survives from the current evidence. A later method would need a very specific physical mechanism that reproduces across OpenVLA-OFT and SmolVLA, survives LIBERO-PRO perturbations, and avoids the crowded grasp/contact and long-horizon correction families above.
