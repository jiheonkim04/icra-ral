# Final Method Mechanism Synthesis

Date: 2026-07-12 KST

Synthesis result: no final method direction is promoted from the current evidence.

## Empirical Mechanism Found

The two prototypes failed at the same abstraction layer: each used a small auxiliary head to rewrite the frozen policy's already-generated actions, but neither created a learned action distribution trained on sequence-level success or differentiated intervention data.

PhaseBarrier shows that action projection can materially perturb execution, but its labels were too weakly connected to task success. CensorCredit shows an even sharper failure: the supposedly censored and uncensored credit targets were identical, so the full method became the ablation.

The unresolved mechanism is therefore not "add a better margin head" or "tune the hold strength." It is:

`How can a locally feasible VLA prototype use intervention-generated, sequence-level supervision to change the policy's emitted action distribution rather than post-process frozen actions?`

## Focused Literature Check

This check was restricted to the empirical mechanism above.

1. OpenVLA-OFT, "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success" (`https://arxiv.org/html/2502.19645v1`)

Relevance: shows that VLA performance can be substantially changed by fine-tuning action decoding, action representation, action chunking, and the supervised action objective rather than wrapping frozen actions. It is direct evidence that policy-distribution training is the right abstraction level, but it also occupies much of the generic "fine-tune VLA action generation" space.

2. VLA-Corrector, "Lightweight Detect-and-Correct Inference for Adaptive Action Horizon" (`https://arxiv.org/html/2607.01804`)

Relevance: addresses action-chunk brittleness with a lightweight external monitor, truncation, and corrective replanning without modifying backbone weights. This overlaps with post-hoc correction/chunk repair and makes a cosmetic stronger wrapper hard to defend as novel.

3. Set-Supervised Diffusion Policy, "Learning Action-Chunking Diffusion through Corrections" (`https://arxiv.org/html/2606.01865v1`)

Relevance: directly uses paired positive and negative action chunks from corrections to train an action-chunking policy. This is close to the missing ingredient identified here: contrastive intervention-generated supervision that changes the action generator.

4. TORL-VLA, "Tactile Guided Online Reinforcement Learning for Contact-Rich Manipulation" (`https://arxiv.org/html/2606.09337v3`)

Relevance: directly contains an intervention-censored critic to prevent post-intervention success from being credited to preceding policy actions. It covers the conceptual core that CensorCredit attempted to proxy, but with real interventions, tactile/wrench observations, and online RL.

5. ConRFT, "A Reinforced Fine-tuning Method for VLA Models via Consistency Policy" (`https://arxiv.org/html/2502.05450v2`)

Relevance: uses offline and online reinforced fine-tuning with human intervention for sample-efficient VLA adaptation. It further reduces novelty room for a generic "sequence reward fine-tune VLA" method.

## Candidate Final Direction Audit

The only empirically grounded direction would be a policy-distribution training method using intervention-generated positive/negative action chunks and sequence-level credit. It would differ from PhaseBarrier and CensorCredit in at least:

- action generation: train the action distribution instead of post-processing frozen actions;
- supervision: use paired intervention/counterfactual chunks instead of passive short-horizon score thresholds;
- objective: optimize sequence/chunk likelihood or preference over action sets instead of a per-step margin;
- representation: preserve chunk-level temporal structure instead of one-step scalar trust.

However, this direction is not promoted now.

Reasons:

- The current artifacts do not contain enough differentiated intervention examples to train it.
- The closest primary sources already cover large parts of this space: SDP covers set-supervised correction chunks, TORL-VLA covers intervention-censored credit in VLA/robotics, ConRFT covers reinforced VLA fine-tuning, and OpenVLA-OFT covers action-distribution fine-tuning.
- A locally feasible version on RTX 5080 would need a new preregistered prototype and fresh evidence, which this postmortem explicitly must not run.
- Promoting it now would turn an empirical postmortem into another speculative portfolio entry.

## Synthesis Decision

No final method is justified from this evidence.

Permissible future research direction, if explicitly reopened later: a preregistered policy-distribution training prototype with intervention-generated positive/negative action chunks and a direct simple baseline such as temporal EMA plus ordinary behavior cloning. That future route must survive a targeted novelty check against SDP, TORL-VLA, ConRFT, VLA-Corrector, and OpenVLA-OFT before implementation.
