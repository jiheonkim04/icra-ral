# Final Research Direction Recommendation

Date: 2026-07-08

## Final Decision

`KILL_ACTIONMAP_ANCHOR`

The local ActionMap mini-anchor route is killed. Target-Grounded ActionMap must not proceed from the current local anchor result.

## Recommendation

Do not start a new method under the current local-proxy constraints.

The only viable next steps are:

A. Official ActionMap reproduction with official code/assets.

B. Official LIBERO-Safety/SafeManip benchmark reproduction.

C. Stop VLA method search under current constraints.

## Why The Previous Direction Is Closed Locally

The previous reset preserved Target-Grounded ActionMap only behind an ActionMap anchor gate. That gate has now run and failed.

Key evidence:

- mean-action action L2: `0.466767673`;
- simple MLP action L2: `0.501926707`;
- ActionMap-style action L2: `0.529931357`;
- linear/L1 action L2: `0.812610317`;
- oracle candidate upper bound action L2: `0.065653208`;
- candidate top1: `0.018518519`;
- candidate collapse: yes, unique translation/rotation/gripper bins `5 / 1 / 2`.

The oracle candidate upper bound shows candidate-space headroom, but the learned local heatmap head collapsed and did not exploit it. The learned head lost to mean action and was matched or beaten by cheap MLP.

## Interpretation Boundary

This kills the local minimal ActionMap approximation. It does not kill the official ActionMap paper.

The result also does not justify a target-grounded extension. Target grounding would be a new method on top of a failed local decoder, so the experiment would not answer whether target grounding helps a credible ActionMap substrate.

## Revival Requirement

This family can be revived only by:

- official ActionMap reproduction with official code/assets; or
- a stronger non-collapsed heatmap implementation that first beats mean-action, linear/L1, and cheap MLP baselines before any target grounding.

No further local proxy approximation should be attempted as the next research route.
