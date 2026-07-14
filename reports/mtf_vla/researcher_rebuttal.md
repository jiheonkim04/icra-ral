# MTF-VLA Researcher A Rebuttal

Date: 2026-07-14 KST

Reviewer B's overlap attack is accepted.

MTF-VLA will not claim that informative frame selection itself is novel. FrameSkip is the closest external prior and must be compared in the first serious experiment. The local contribution is narrower:

- use StructVLA-style physical milestone cues as the transition score backbone;
- pair high-milestone expert imitation with low-milestone frozen-base retention;
- test whether this identity-preserving data objective improves a local SmolVLA adapter where ordinary LoRA did not.

If the FrameSkip proxy matches or beats MTF, the extension fails.

If uniform retained-ratio LoRA matches or beats MTF, the mechanism is explained by ordinary adapter training.

If the no-retention ablation matches or beats MTF, the base-retention component is not useful.

The method will use Huber/L2 discrepancies over deterministic 7D actions and action chunks. No KL term is allowed.

Stage 0 will stop before training or rollout if score health, split separation, base-retention target persistence, identity-preserving initialization, or proxy-faithfulness checks fail.

This preserves the strongest honest version of the method without weakening the confirmatory test.
