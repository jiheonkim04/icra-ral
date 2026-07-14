# MTF-VLA Reviewer B Attack

Date: 2026-07-14 KST

Proposal under review: `reports/mtf_vla/researcher_proposal.md`.

## Primary Attack: FrameSkip Overlap

FrameSkip already claims the central idea that dense VLA demonstrations should be remapped toward informative frames using action variation, visual-action coherence, task-progress priors, and gripper-transition preservation. MTF-VLA cannot claim novelty for "important frame selection" or "transition frames matter."

Allowed framing:

- MTF-VLA may proceed only as a `CROSS_PAPER_SYNTHESIS` or strong prior extension.
- The technical novelty must be the combination of structured physical milestone selection with explicit frozen-base retention for adapter training.
- The FrameSkip proxy must enter the first serious comparison.

Kill condition:

- If the FrameSkip proxy matches or beats MTF, the local extension is not a paper candidate.

## Secondary Attack: This Could Be Ordinary LoRA

The historical official closed-loop evidence shows that rank-4 LoRA did not reliably improve frozen SmolVLA. MTF must not be reported as "LoRA works."

Mandatory simple killer:

- `uniform_retained_ratio_lora` with the same adapter family, retained-frame ratio, training budget, and checkpoint-selection rule.

Kill condition:

- If uniform retained-ratio LoRA matches or beats MTF, the milestone-retention mechanism is not supported.

## Mathematical Validity Attack

Do not use KL, entropy, mutual information, or contrastive language unless the arguments are valid distributions and the gradient path is justified. MTF has deterministic 7D action targets and frozen-base action targets. Huber/L2 action discrepancies are sufficient and more honest.

Required:

- define variables and tensor shapes;
- define the exact score function;
- define loss scales and units;
- measure loss magnitudes and gradient norms before full training;
- verify that the retention loss does not overwhelm transition imitation.

## Data Leakage Attack

Frame scores may be computed from demonstrations, but confirmatory rollout identities and outcomes must not enter score construction, validation selection, checkpoint selection, or baseline selection.

Required:

- persist discovery, validation, and confirmatory splits;
- prove zero overlap in task/reset identities for rollouts and demo/frame identities for training where applicable;
- preserve all validation configurations and failed runs.

## Proxy-Faithfulness Attack

If official FrameSkip code is unavailable, the proxy must still preserve the essential mechanism:

- action variation;
- gripper-transition preservation;
- task-progress or phase coverage;
- visual-action coherence when image access is feasible, or a documented state-action proxy when it is not.

The proxy must not be intentionally weak. If MTF omits a FrameSkip component for local feasibility, the omission must be listed.

## Identity-Preservation Attack

MTF changes policy weights. It can harm clean behavior more easily than an inference-time passthrough wrapper.

Required:

- adapter initialization close to Base;
- action-delta measurement before rollout;
- clean validation retention gate;
- no best-seed selection on confirmatory test;
- no checkpoint chosen by confirmatory outcomes.

## Stage 0 Attack

Do not proceed to training or rollout if:

- score variance is too low to produce a real high/low contrast;
- high-score frames are just one task or one phase;
- gripper-transition positives are all zero or almost all examples;
- the no-retention ablation receives the same effective targets as full MTF;
- uniform sampling and MTF sampling overlap so much that the comparison is meaningless;
- frozen-base retention targets cannot be generated from the same policy identity used in Base rollout.

## Reviewer Decision

Decision: `ALLOW_WITH_STRONG_PRIOR_COMPARISON`.

Reason: MTF is very close to FrameSkip, but it is not an exact duplicate if and only if the retention objective and structured milestone pairing are central and the first comparison includes FrameSkip proxy, no-retention ablation, and uniform LoRA. Unknown empirical performance is not a rejection reason under active governance.
