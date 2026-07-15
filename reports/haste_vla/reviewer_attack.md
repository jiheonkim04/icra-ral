# HASTE-VLA Reviewer B Attack

Date: 2026-07-15 KST

Proposal hash:
`5415BC1533A24EC55CC511DDEB014BB11D9C19F603C59D1F1D3E151E15B930A6`.

Decision: `HASTE_REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

## Strongest Novelty Attack

StaKe already derives manipulation stages and next gripper-transition targets
from demonstration gripper states. EventVLA and KEMO already use events to
select visual evidence and keyframes. FrameSkip already preserves and weights
gripper-transition frames. HASTE cannot claim event-aware VLA training,
gripper-event labels, stage structure, keyframes, or transition importance as
new.

The only provisional novelty is the joint replacement of StaKe's binary stage
and absolute keyframe targets by a censored time-to-event hazard and a relative
cumulative action displacement. If either target is merely a reparameterized
version of StaKe with no representation or action consequence, novelty fails.

## Target-Semantics Attack

The action gripper command may remain at `-1` or `+1` for many steps. A command
transition is not necessarily physical contact, successful grasp, or release.
HASTE must call these command events, not contact events.

Summed six-dimensional action increments are not automatically an SE(3)
displacement. Rotation coordinates may be local increments, postprocessed, or
clipped. The proposal must not claim exact kinematic composition. The target is
a relative cumulative action-coordinate displacement only.

Rows near the end of a demonstration are right-censored for data-boundary
reasons. Censoring is valid only if the likelihood masks unavailable intervals
and does not label them as genuine no-event behavior.

## Data And Headroom Attack

HEST established transition-containing windows, not balanced time-to-event
offsets or cross-task predictability. Dense action records can still produce:

- mostly censored rows;
- event offsets concentrated near one boundary;
- one task dominating transitions;
- low displacement variance;
- future targets unpredictable from current deployment inputs.

The fixed four tasks were reused from HEST development. That is legal discovery
reuse, but the campaign may not use HEST's support failure as positive HASTE
evidence.

Base near-event action error must be computed against correctly aligned
postprocessed action chunks. Offline action error is only a headroom diagnostic,
not paper evidence. If Base is not worse near events, the proposed problem has
no local headroom.

## Objective Attack

Long survival sequences can make `L_haz` scale with `H_e`; averaging over valid
intervals is necessary but may underweight rare events. Huber displacement is
masked on censored rows and may receive fewer gradients than flow or retention.
Magnitude and LoRA-gradient audits are mandatory before coefficient search.

Gradient conflict is plausible: retention requests Base identity while event
supervision changes the representation. A positive auxiliary-head fit with
zero policy consequence is insufficient.

The constant-hazard and discovery-mean displacement probes must be fitted only
on discovery rows and evaluated on validation rows. A leakage-prone per-task
constant is not a valid trivial baseline unless preregistered.

## Identity And Comparison Attack

Zero LoRA B matrices should preserve Base exactly, but trainable query tokens or
normalization changes can break identity even before optimization. Stage 0 must
hash Base parameters and compare actual flow vectors and decoded actions after
disk reload.

The StaKe proxy, HASTE, no-hazard ablation, and standard LoRA need matched:

- action rows;
- retention rows;
- optimizer steps and schedule;
- rank and adapter targets;
- seeds;
- checkpoint loading;
- inference budget.

Otherwise generic adaptation or compute explains any result.

## Decisive Failure Rules

Stop before training for collapsed labels, no Base event-near deficit,
unpredictable hazard/displacement targets, nonidentity initialization, split
leakage, or nonfinite data.

Stop before rollout if the micro-fit is nonacting, destroys action validity,
fails disk reload, only improves auxiliary heads, or degrades event-far
retention.

Do not rescue HASTE by changing event threshold, source tasks, horizon set,
relative target, coefficient grid, or head architecture after a valid gate
failure.

## Required Rebuttal

Researcher A must narrow claims to command-event timing and relative action
coordinates, specify boundary censoring, freeze trivial baselines and objective
scales, and accept that Stage 0 may end as data, no-headroom, design, or
implementation failure without scientific interpretation.
