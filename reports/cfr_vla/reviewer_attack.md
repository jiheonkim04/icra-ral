# CFR-VLA Reviewer B Attack

Date: 2026-07-16 KST

Decision: `REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

Reviewed proposal: `reports/cfr_vla/researcher_proposal.md`

Reviewed proposal hash:
`9E2FC510B2D97C869F18BE6C5B339CE034DD98223802078358320AA8BEF3D0AE`

Reviewer B does not edit or soften Researcher A's proposal. This attack is an
independent adversarial review before mathematical audit, preregistration, or
implementation.

## Closest Current Papers

1. DFM-VLA, https://arxiv.org/abs/2603.26320 and
   https://chris1220313648.github.io/DFM-VLA/

   DFM-VLA is the direct closest prior. It already claims the central idea that
   full action sequences should be revisable through iterative refinement rather
   than committed after one decode. It reports discrete flow matching,
   token-level probability velocity, a two-stage refinement-plus-validation
   decoder, `95.7%` LIBERO average success, `4.44` CALVIN average success
   length, and real-world results. CFR's novelty survives only if the final
   method is specifically continuous, Base-start, identity-preserving
   full-chunk refinement around SmolVLA actions, not a relabeled DFM proxy.

2. Adaptive Action Chunking, https://arxiv.org/abs/2604.04161 and
   https://lance-lot.github.io/adaptive-chunking.github.io/

   AAC attacks a neighboring action-chunk failure mode: fixed action execution
   horizon trades off consistency against reactivity, and entropy selects chunk
   size at inference. CFR is not chunk-size selection, but any empirical gain
   that comes only from smoother or shorter effective execution can be explained
   by AAC-style scheduling. The first comparison must keep the method claim on
   iterative refinement of the predicted chunk, not adaptive horizon control.

3. CoLA-Flow Policy, https://arxiv.org/html/2601.23087v5

   CoLA-Flow is not the same VLA claim, but it is a very recent continuous
   trajectory-level flow policy. It weakens any broad novelty statement about
   continuous latent or trajectory flow refinement. CFR must claim only the
   narrow VLA setting: post-decode continuous full-chunk refinement around a
   frozen SmolVLA Base chunk with identity-preserving integration and a matched
   DFM-style prior proxy.

Additional nearby priors checked:

- RotVLA, https://arxiv.org/abs/2605.13403, for continuous structured latent
  actions on `SO(n)`;
- GEAR-VLA, https://arxiv.org/abs/2606.08530, for geometry-aware action
  representation and coarse-to-fine action learning;
- FASTER, https://arxiv.org/html/2603.19199v1, for flow-VLA real-time action
  sampling and horizon-aware schedules.

## Novelty Attack

CFR is dangerously close to DFM-VLA. The phrase "iterative full-chunk
refinement" is not novel by itself. Researcher A may claim novelty only in the
minimal technical difference:

- DFM-VLA refines discrete action tokens through token-level probability
  velocity.
- CFR refines continuous `[50,7]` SmolVLA action chunks through a bounded
  residual velocity field initialized as exact Base passthrough.

Reviewer B rejects any broad claim that CFR invented iterative action
refinement, full-sequence correction, or flow refinement for robot actions.
Those are prior-owned.

Required rebuttal condition:

`CFR_NOVELTY_NARROWED_TO_CONTINUOUS_BASE_START_IDENTITY_REFINEMENT`

## Prior-Comparison Attack

The proposal says official DFM-VLA code is not locally verified. A weak local
proxy could make CFR look better than the closest prior unfairly. The DFM proxy
must enter the first serious comparison early and transparently.

Required rebuttal conditions:

1. `dfm_vla_continuous_refinement_proxy` remains policy 2 unless official
   DFM-VLA is installed and verified before the first serious comparison.
2. The proxy must implement iterative full-sequence refinement, not a straw
   one-shot residual.
3. If official DFM-VLA code/assets become locally available before
   confirmatory testing, official DFM-VLA replaces or augments the proxy without
   retuning CFR on confirmatory outcomes.
4. CFR must beat the DFM proxy on the matched claim axis before paper
   viability.

## Simplest Equivalent Method Attack

The strongest simple alternative is not "no method"; it is a one-shot terminal
residual or matched standard LoRA trained on the same demonstrations. A result
where CFR only equals `cfr_no_iterative_refinement` is not a CFR result.

Required rebuttal conditions:

1. `cfr_no_iterative_refinement` remains the key ablation.
2. `standard_lora` remains the single simple reviewer-killer baseline because
   CFR uses trainable adapter infrastructure.
3. Offline terminal action Huber cannot by itself select or validate CFR.
4. The mechanism evidence must show a stepwise refinement consequence: later
   `C^k` states move coherently toward the target and not merely through one
   large terminal residual hidden inside the final step.

## Action-Validity Attack

Recent RAP and AMP Stage 0 stops both hit postprocessed action-validity gates
where `base_action_in_bounds` was false. CFR must not inherit a wrong or
overly literal `[-1,1]` gate as if it were a method failure. Conversely, CFR
must not loosen validity after seeing results.

Required rebuttal conditions:

1. Before any CFR Stage 0 run, define action validity from official LIBERO /
   SmolVLA postprocessor semantics and environment action limits, not from an
   ad hoc bound copied from RAP or AMP.
2. Persist the exact action-validity definition in the mathematical audit and
   prototype protocol.
3. Check Base, DFM proxy, no-iterative ablation, standard LoRA, and CFR with
   the same definition.
4. If Base fails the validity definition before CFR acts, classify the result
   as `CFR_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`, not a scientific
   kill, and do not rescue by clipping or changing the definition post hoc.

## Data And Supervision Attack

Base-to-demo residuals may be small, noisy, or explained by task/phase means.
Because the supervision is generated from demonstrations rather than outcomes,
Stage 0 can easily become another offline-L2 trap.

Required rebuttal conditions:

1. Residual targets must be noncollapsed by dimension, task, phase, and
   timestep.
2. A deployment-input refinement probe must beat task/phase residual baselines
   before training.
3. DFM proxy residual headroom must be positive before bounded validation.
4. The pre-rollout decision must use the false-negative safeguard: weak offline
   evidence cannot become a closed-loop scientific kill.

## Mathematical Attack

The proposed velocity target `D_t^k = stopgrad(A_t - C_t^k)/(K-k)` is plausible
but easy to mis-scale. `lambda_v`, `lambda_T`, `lambda_s`, and `lambda_clean`
can overwhelm each other. The unrolled refinement graph can also hide gradient
paths through `C_t^k`.

Required rebuttal conditions:

1. The mathematical audit must specify tensor shapes for every variable.
2. It must report small-batch term magnitudes and gradient norms before
   training.
3. It must explicitly state which paths use `stopgrad` and which paths carry
   gradient through the refinement unroll.
4. No KL may be computed between deterministic actions or SmolVLA flow vectors.
5. The audit must justify why Huber/vector-field consistency is used instead of
   JS, Wasserstein, MMD, Mahalanobis, or trajectory discrepancy.

## Inference-Legal Input Attack

CFR may use expert future action chunks for training targets, but inference
must use only legal current observations and Base chunks. No future observation,
success, done flag, reward, object pose, or reset identity can enter the
refinement module.

Required rebuttal condition:

`NO_PRIVILEGED_INFERENCE_INPUTS_CONFIRMED`

## Conditional Pass

Reviewer B does not kill CFR before implementation. The positive external prior
is strong, the proposed difference is technically meaningful if narrowed, and
the local experiment can be decisive. The proposal passes only conditionally:
Researcher A must accept the conditions above before mathematical audit or
preregistration.

If Researcher A rejects any condition, CFR must stop before implementation as
`CFR_PREIMPLEMENTATION_REVIEW_FAILURE`.
