# COVI-VLA Reviewer B Attack

Date: 2026-07-15 KST

Reviewed frozen proposal: `reports/covi_vla/researcher_proposal.md`

Proposal hash: `338430D2C6CF1D82410C036D79102ED3F38B2367BB35B9AE2811161698A3E621`

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

## Independent Source Check

Primary closest prior: LIBERO-Occ / Viewpoint Imagination,
https://arxiv.org/abs/2606.10862.

The closest prior is strong. LIBERO-Occ explicitly studies scene-induced
occlusion rather than generic image corruption, reports large VLA drops under
occlusion, and proposes VIM: generated complementary-view evidence followed by
action prediction conditioned on observed and imagined views. The authors also
report code at https://github.com/litsh/Libero-Occ, and the repository exposes
benchmark assets plus VIM training/evaluation scripts.

Key source-fidelity issue: VIM is not a small frozen-policy feature adapter. It
uses a UniVLA-derived, Emu3-MoE-style autoregressive image-token/action-token
pipeline and two-stage training for viewpoint generation plus action prediction.
The proposal must therefore keep calling the local comparison a transparent
proxy until official equivalence is established.

Secondary prior: CamVLA, https://arxiv.org/abs/2607.05396.

CamVLA attacks view robustness by changing the action representation: it
predicts camera-centric end-effector actions and a hand-eye transform, then
maps actions back to the robot base frame. COVI must not claim calibration-free
viewpoint robustness or camera-centric action generation.

Secondary prior: STRONG-VLA, https://arxiv.org/abs/2604.10055.

STRONG-VLA reports that decoupled robustness learning improves multimodal
perturbation robustness across OpenVLA, OpenVLA-OFT, and pi0. It is a direct
pressure test against the possibility that COVI is only ordinary perturbation
training or random cutout plus clean re-alignment.

Visual robustness/security pressure: Partially Observable Adversarial Patch
Attacks on VLAs, https://arxiv.org/abs/2606.03556.

This is not a defense method, but it shows that partial visual observability
and patch/occlusion-like visual disruption can drive long-horizon control
failure. COVI must distinguish physical scene-induced missing evidence from
adversarial patch or photometric robustness defenses.

Local historical pressure: PatchGuard-v1 was killed because random erasing /
cutout was stronger. COVI must keep the random-cutout clean-retention simple
killer live.

## Novelty Attack

1. VIM already contains the core idea of generating complementary visual
   evidence for occluded manipulation. COVI is not novel as "viewpoint
   imagination for occlusion." Its only defensible novelty is a locally feasible
   frozen-SmolVLA, identity-preserving complementary-feature adapter with a
   source-gated Stage 0 audit.

2. Standard two-camera fusion is a serious trivial-equivalence risk. Official
   SmolVLA already uses two image streams. If COVI simply learns to pass through
   or reweight an already available clean second view, it is not viewpoint
   imagination. Stage 0 must separate:
   - direct use of the available second stream;
   - target clean-view oracle as a diagnostic upper bound;
   - predicted complementary representation from legal occluded inputs.

3. Random cutout or STRONG-VLA-style robustness training may explain any gain.
   The first serious experiment may retain exactly one mandatory simple killer,
   but Stage 0 and the mathematical audit must explicitly prove why the
   random-cutout clean-retention baseline is the strongest simple explanation.

4. PatchGuard overlap is not fatal only if COVI's condition is physically
   grounded scene-induced occlusion with complementary-view target evidence.
   Synthetic rectangles alone are insufficient; they would collapse the claim
   back to a generic image-corruption defense.

5. The proposal's action formula risks overstating access to SmolVLA internal
   visual tokens. If local SmolVLA does not expose a stable visual-token
   intervention point, the implementation must use the smallest faithful
   available adapter hook and document the exact policy component affected.

## Leakage And Source-Gate Attack

Reviewer B will reject COVI if any of the following happens:

- clean unoccluded evaluation images are used at inference;
- ground-truth complementary camera frames from the confirmatory identity are
  used as policy inputs rather than diagnostic oracle labels;
- segmentation masks, object poses, visibility labels, reset identities, or
  task outcomes enter inference;
- confirmatory identities influence occlusion severity, task selection,
  checkpoint selection, or threshold tuning;
- the VIM proxy is silently treated as official VIM when only a local proxy was
  run;
- synthetic cutout labels replace the scene-induced occlusion claim without
  an explicit downgrade of the claim.

## Mathematical And Mechanism Attack

The method currently avoids invalid deterministic-action KL, which is good.
However, the mathematical audit must prove the adapter is functional rather
than decorative.

Required audit points:

- exact visual feature tensor source, shape, and intervention point;
- whether `delta_e_t` changes frozen visual tokens, post-encoder features,
  auxiliary conditioning, or pre-action hidden state;
- loss magnitudes and gradient norms for view-feature prediction, clean
  retention, and any action-preservation term;
- whether gradients reach the adapter and gate but not the frozen Base;
- how feature-predictability improvement is linked to an action-distribution
  consequence;
- direct report of Base action, COVI action, feature residual norm, gate value,
  dimensions or tokens changed, and occlusion context.

Feature reconstruction alone is insufficient. The smoke must show bounded,
localized action consequences under occlusion while preserving clean actions.

## Mandatory Rebuttal Conditions

Researcher A must accept all of the following before mathematical audit or
implementation:

1. Narrowed novelty:
   `COVI-VLA` is a frozen-SmolVLA identity-preserving complementary-feature
   adapter for scene-induced occlusion. It is not official VIM, not full
   viewpoint imagination novelty, and not calibration-free view-robust action
   generation.

2. Source fidelity:
   `vim_view_imagination_proxy` remains a transparent faithful proxy until the
   official LIBERO-Occ/VIM code and checkpoint equivalence are locally
   established.

3. Direct-fusion challenge:
   Stage 0 must include a diagnostic that separates predicted complementary
   evidence from direct two-camera pass-through or clean-view oracle use.

4. Random-cutout simple killer:
   `random_cutout_clean_retention_baseline` remains the only mandatory simple
   killer in the first serious comparison and must not be dropped, renamed, or
   weakened.

5. Physical occlusion claim:
   COVI must use or faithfully proxy scene-induced occlusion. If only synthetic
   image masks are used, the claim must be downgraded and may no longer be
   paper-worthy unless the connection to physical occlusion is validated.

6. Identity-preserving integration:
   The adapter gate must initialize to Base passthrough; clean validation
   action delta, action validity, translation/rotation/gripper deltas, and
   activation localization must be reported before rollout.

7. No privileged inference:
   clean images, future frames, segmentation, object pose, reset identity,
   success labels, and confirmatory outcomes remain forbidden inference inputs.

## Reviewer Decision

Conditional pass to Researcher A rebuttal.

COVI is not killed before implementation because the closest prior is positive
and directly relevant, the local extension changes more than two core
dimensions relative to RAR, and Stage 0 can cheaply falsify the main risks.
The method proceeds only under the narrowed novelty and source-gate constraints
above. If Researcher A refuses them, the proposal must be killed before
mathematical audit.
