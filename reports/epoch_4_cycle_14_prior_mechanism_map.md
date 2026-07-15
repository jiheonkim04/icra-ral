# Epoch 4 Cycle 14 Prior Mechanism Map

Date: 2026-07-15 KST

Purpose: select the next method after `RAR-VLA` stopped before rollout as
`DESIGN_FAILURE`. RAR is not a closed-loop scientific kill, but its fixed Stage
0 stop is valid for that protocol. It must not be rescued by changing history
features, residual labels, thresholds, source gates, validation configs, or
baseline interpretation.

## Local Constraints From Prior Results

The next method must not be:

- another causal action-history, autoregressive residual, EMA smoothing, or
  chunk-boundary repair of RAR;
- another future-action latent, point-label, waypoint, material-point, or
  context-gated label rescue of CALA or G3P;
- another adaptive chunking, retained-frame, receding-horizon, fixed-replan, or
  action-queue scheduler variant of EAC, MTF, RCV, or PSE;
- another median-anchor, static mixture, L1 proxy, reflective consequence, or
  generic output-action correction route;
- another photometric-only random-erasing defense in the killed PatchGuard
  family;
- a confidence head, progress head, generic reflection wrapper, or termination
  heuristic as the main contribution.

RAR's failed audit remains informative: the legal causal residual was not
predictable above the zero-residual trivial baseline. Cycle 14 should therefore
change the problem axis rather than trying a richer residual model. The strongest
unrepeated local opportunity is partial observability from scene-induced
occlusion and view dependence.

## Close Sources

### LIBERO-Occ / Viewpoint Imagination

Full title: LIBERO-Occ: Evaluating and Improving Vision-Language-Action Models
under Scene-Induced Occlusion via Viewpoint Imagination.

URL: https://arxiv.org/abs/2606.10862

Code URL reported by authors: https://github.com/litsh/Libero-Occ

AUTHOR_STATED:

- The paper identifies scene-induced occlusion as a VLA failure mode that is
  distinct from artificial image perturbations.
- It introduces LIBERO-Occ, an occlusion-oriented LIBERO extension with
  controlled occlusion types and severities.
- It proposes Viewpoint Imagination (VIM), which generates a complementary view
  from an occluded primary observation and conditions action prediction on both
  observed and imagined evidence.
- It reports that VIM improves LIBERO-Occ robustness without requiring
  additional cameras at deployment.

INDEPENDENTLY_INFERRED:

- The positive prior is not random masking or augmentation alone. The mechanism
  is missing-evidence recovery through a complementary-view intermediate that is
  action-grounded.
- The paper reports a large occlusion headroom axis: strong VLA baselines drop
  substantially on LIBERO-Occ, while VIM reduces the drop and improves
  occlusion success.
- The official code link makes this an unusually strong local anchor, but local
  use must still be transparent until the exact benchmark and method are run in
  the repository environment.

CROSS_PAPER_SYNTHESIZED:

- COVI-VLA should not reproduce full VIM with a large generative world model.
  The locally feasible extension is an identity-preserving visual adapter for
  frozen SmolVLA that learns an occlusion-conditioned complementary-view
  representation and uses it only under a bounded gate.
- The mandatory simple killer is a clean-retention random-cutout or masked-view
  augmentation baseline, because PatchGuard already showed that many visual
  robustness ideas collapse to ordinary augmentation.

### CamVLA

Full title: From Fixed to Free Cameras: Calibration-Free View-Robust
Vision-Language-Action Model.

URL: https://arxiv.org/abs/2607.05396

Project URL reported by authors: https://alibaba-damo-academy.github.io/CamVLA/

AUTHOR_STATED:

- CamVLA targets camera remounting and unseen viewpoints without explicit camera
  extrinsics.
- It decouples camera-centric action prediction from hand-eye transformation
  prediction, then composes them into robot-base-frame actions.
- It reports improved success under diverse unseen viewpoints in simulation and
  real-world data.

INDEPENDENTLY_INFERRED:

- CamVLA reinforces that view geometry is an under-modeled deployment variable
  for VLAs.
- It is less locally feasible than LIBERO-Occ for the next method because exact
  camera-pose supervision and hand-eye labels are not already established in the
  repository.
- It should serve as a Reviewer B novelty/source-fidelity challenge: a COVI
  method may claim occlusion-completion robustness, not full calibration-free
  camera-centric action generation.

### SUREFlow

Full title: SUREFlow: State-space Uncertainty-aware REsidual Flow Matching for
Robust Robot Manipulation.

URL: https://arxiv.org/abs/2607.10504

Code URL reported by authors: https://github.com/tanvirnwu/SUREFlow

AUTHOR_STATED:

- SUREFlow targets instability under noise, partial observability, and
  stochastic initial conditions.
- It jointly predicts action velocities and input-dependent residual uncertainty
  to selectively refine unreliable action dimensions.
- It reports high LIBERO success and strong LIBERO-PRO performance with a small
  Mamba backbone.

INDEPENDENTLY_INFERRED:

- The positive prior is uncertainty-aware action generation, not generic
  confidence estimation.
- A local SmolVLA method could use uncertainty to decide when an imagined-view
  or robust adapter should act, but making uncertainty the main contribution
  risks returning to output-residual methods after RAR, MARC, and DAGR failures.
- SUREFlow is a strong secondary prior for candidate scoring and future
  mechanism audits.

### DFM-VLA

Full title: DFM-VLA: Iterative Action Refinement for Robot Manipulation via
Discrete Flow Matching.

URL: https://arxiv.org/abs/2603.26320

Project URL reported by authors: https://chris1220313648.github.io/DFM-VLA/

AUTHOR_STATED:

- DFM-VLA argues that discrete action-token decoders lock early token errors and
  therefore cannot revise full action sequences after initial generation.
- It learns a token-level probability velocity field that iteratively updates
  the full action sequence, followed by deterministic validation.
- It reports strong results on CALVIN, LIBERO, and real-world manipulation with
  high inference efficiency.

INDEPENDENTLY_INFERRED:

- The positive prior is iterative action refinement, not simply a smoother or
  residual regressor.
- Local feasibility is weaker because SmolVLA uses continuous 7D action chunks,
  not the same discrete token refinement interface.
- A direct Cycle 14 selection around DFM risks another output-action correction
  route unless a new, nontrivial tokenization or refinement target is justified.

### STRONG-VLA

Full title: STRONG-VLA: Decoupled Robustness Learning for
Vision-Language-Action Models under Multimodal Perturbations.

URL: https://arxiv.org/abs/2604.10055

AUTHOR_STATED:

- STRONG-VLA separates robustness acquisition under multimodal perturbations
  from clean task re-alignment.
- It reports cross-architecture robustness gains on LIBERO, including OpenVLA,
  OpenVLA-OFT, and pi0.

INDEPENDENTLY_INFERRED:

- STRONG-VLA is a useful robustness prior, but by itself it is a training recipe
  and is too close to ordinary augmentation unless paired with a more specific
  mechanism.
- For Cycle 14 it is best used as a simple-baseline pressure test: COVI must
  beat a clean-retention cutout/perturbation baseline, not merely demonstrate
  that augmentation helps.

## Cycle 14 Opportunity

The strongest immediate opportunity is `COVI-VLA`: Complementary Occlusion View
Imagination for frozen SmolVLA.

It is anchored primarily to LIBERO-Occ and VIM. The local extension is not a
full official VIM reproduction. It is a frozen-backbone, identity-preserving
visual-representation adapter that learns from paired LIBERO camera streams and
controlled development-only occlusion to infer a complementary-view feature
when task-relevant evidence is hidden.

The proposed deployment-time inputs are legal:

- current official RGB observations;
- current proprioception;
- task or language instruction;
- frozen Base action chunk;
- an internally predicted complementary-view representation.

Forbidden inference inputs remain forbidden:

- future frames;
- success labels;
- reset identities;
- simulator object state;
- segmentation masks;
- task outcome labels;
- confirmatory-test outcomes.

The critical Reviewer B baselines are:

- unmodified frozen SmolVLA under the same occlusion condition;
- `vim_view_imagination_proxy`, a faithful transparent local proxy for the
  closest prior until official equivalence is established;
- `covi_full`;
- `covi_no_imagined_view_ablation`;
- `random_cutout_clean_retention_baseline`, the simple reviewer-killer that
  tests whether ordinary visual augmentation explains any gain.

The Stage 0 gate must prove:

- Base has meaningful development/validation headroom under a predeclared
  occlusion condition;
- target-relevant visibility or view-completion labels are noncollapsed;
- complementary-view features are predictable from deployment inputs above
  trivial image-statistic and cutout baselines;
- the adapter starts as exact Base passthrough or a bounded visual-feature gate;
- no segmentation, object pose, future frame, success, reset, or confirmatory
  identity leaks into inference.
