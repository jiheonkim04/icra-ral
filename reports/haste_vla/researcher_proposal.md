# HASTE-VLA Researcher A Proposal

Date: 2026-07-15 KST

Decision: `HASTE_RESEARCHER_PROPOSAL_READY_FOR_REVIEW`

## Identity

Method: `HASTE-VLA`, Hazard-Anchored Stage-Transition Encoding for
Vision-Language-Action fine-tuning.

Contribution type: `PRIOR_EXTENSION`.

Closest positive prior: StaKe, Improving Vision-Language-Action Model
Fine-Tuning with Structured Stage and Keyframe Supervision,
https://arxiv.org/abs/2606.26801.

## Positive Prior And Difference

StaKe derives a binary stage target and the absolute joint action at the next
gripper transition, attaches two auxiliary query heads during VLA fine-tuning,
and leaves the inference loop unchanged. It reports relative success gains of
`14%` in bimanual simulation and `56%` on single-arm Franka tasks.

HASTE preserves the training-only auxiliary-supervision strategy but changes
the represented event information:

1. binary stage becomes a censored discrete hazard over the next transition;
2. absolute keyframe action becomes a current-centered cumulative arm
   displacement to the transition.

This separates when the event should occur from where the arm should reach.

## Problem And Mechanism

Condition:

- a manipulation trajectory approaches a gripper close/open transition;
- current images can be visually similar across different remaining distances
  to the event;
- ordinary action loss weights transition-near and transition-far frames
  equally.

Failure mechanism:

- the policy representation under-encodes remaining event time;
- absolute keyframe regression carries large task/pose variance;
- action errors around contact boundaries cause premature or late closure,
  missed grasps, early release, or post-grasp drift.

Intended HASTE mechanism:

- survival supervision makes remaining event time observable in the adapted
  representation;
- relative cumulative displacement removes current-pose offset and anchors the
  required arm motion in the policy's action coordinates;
- clean retention constrains non-event behavior;
- auxiliary heads are absent from the inference computation.

Expected effect:

- lower validation action error in event-near strata;
- improved closed-loop grasp/release success;
- no material degradation on event-far or clean validation cases.

## Frozen Labels

Let the postprocessed demonstration action sequence be
`a_t in R^7`, with arm coordinates `a_t^arm in R^6` and gripper command
`g_t = a_t[6]`.

Define a gripper transition at index `j` when
`abs(g_j - g_(j-1)) > 1e-8`.

For event horizon `H_e in {20, 50}`, define the first future transition offset

`tau_t = min{k in {1,...,H_e}: abs(g_(t+k)-g_(t+k-1)) > 1e-8}`.

If no such transition exists, the row is right-censored at `H_e`.

For uncensored rows, define relative event displacement

`d_t = sum_{j=0}^{tau_t} a_(t+j)^arm in R^6`.

Normalize each displacement coordinate using discovery-only mean and standard
deviation with floor `1e-6`. Validation and confirmatory statistics may not
change normalization.

## Model And Objectives

Backbone: official frozen-start SmolVLA LIBERO checkpoint.

Trainable policy parameters: fixed rank-4 LoRA on the preregistered attention
targets. Every LoRA B matrix initializes to zero.

Shared representation: one declared SmolVLA token or pooled latent
`z_t in R^960` after the frozen hook audit.

Hazard head:

- input `[B,960]`;
- hidden layer `[960,128]` with SiLU;
- output logits `[B,H_e]`;
- hazard probabilities `h_(t,k) = sigmoid(l_(t,k))`.

For an event at `tau`,

`L_haz = -sum_(k<tau) log(1-h_k) - log(h_tau)`.

For a censored row,

`L_haz = -sum_(k=1)^H_e log(1-h_k)`.

The hazard loss is averaged over valid intervals, not over padded positions.

Displacement head:

- input `[B,960]`;
- hidden layer `[960,128]` with SiLU;
- output `[B,6]`;
- loss is coordinate-mean Huber with delta `1.0` on normalized `d_t`;
- censored rows are masked from displacement loss.

Action objective:

- existing SmolVLA conditional flow-matching loss `L_flow`;
- unchanged action tensor, processor, horizon, and inference solver.

Clean retention:

`L_ret` is Huber distance between adapted and stop-gradient frozen-Base flow
vectors at the same noisy action state, time, images, language, and robot state
on preregistered retention rows.

Total:

`L = L_flow + lambda_h L_haz + 1.0 L_disp + 1.0 L_ret`.

No KL divergence is used. The deterministic 7D actions and flow vectors are not
treated as probability distributions.

## Gradient Paths

- `L_flow`: LoRA policy parameters;
- `L_haz`: hazard head and shared LoRA policy parameters;
- `L_disp`: displacement head and shared LoRA policy parameters;
- `L_ret`: LoRA policy parameters only; frozen Base target is detached;
- no auxiliary head is called to produce actions at inference.

Before training, Stage 0 must report each term magnitude, LoRA gradient norm,
head gradient norm, and pairwise LoRA-gradient cosine. Nonfinite, zero, or
greater-than-100x unexplained gradient ratios fail the audit.

## Closest Prior And Controls

Exactly five first-comparison policies:

1. frozen SmolVLA Base;
2. faithful transparent StaKe proxy: binary stage plus absolute next-event
   keyframe regression;
3. HASTE full;
4. HASTE without hazard, retaining relative displacement;
5. standard rank-4 LoRA with the same action rows, retention rows, optimizer,
   steps, seeds, and inference budget.

All trained policies use independent disk-reloadable checkpoints. The prior
proxy may not be called an official StaKe reproduction.

## Data Partitions

Fixed task families for the first development gate:

1. `libero_spatial/task_3`;
2. `libero_object/task_3`;
3. `libero_goal/task_5`;
4. `libero_10/task_5`.

Discovery demonstrations: IDs `0..7`.

Validation demonstrations: IDs `8..9`.

Confirmatory closed-loop reset identities are separate and sealed.

No reward, done, success, or confirmatory reset record may influence label
construction, objective coefficients, configuration selection, or task
selection within this cycle.

## Stage 0A Data And Headroom Gate

Stage 0A may load frozen SmolVLA and development demonstrations but performs no
adapter training and no simulator rollout.

It must establish:

- exact proposal and source hashes;
- no duplicate, missing, extra, or split-overlap keys;
- finite images, robot states, actions, latents, and labels;
- at least `128` uncensored and `128` censored discovery rows overall;
- at least `16` uncensored validation rows per task;
- at least `5` occupied transition-offset quartiles/bins overall;
- positive variance in all six normalized displacement coordinates;
- no task contributes more than `40%` of uncensored rows after equal-task
  sampling;
- Base arm-error mean in the event-near stratum (`tau <= 10`) is at least
  `10%` greater than event-far/censored error, or Base gripper sign error is at
  least `5` percentage points greater;
- a frozen-feature linear hazard probe beats the constant-hazard negative log
  likelihood by at least `2%` on validation;
- a frozen-feature linear displacement probe beats the discovery-mean target by
  at least `2%` normalized Huber loss on validation;
- zero-effect LoRA initialization reproduces Base flow vectors and actions
  within `1e-6`;
- exceptions are zero.

If labels collapse, classify `DATA_FAILURE`. If Base has no event-near deficit,
classify `NO_HEADROOM`. If targets are not observable from deployment features,
classify `DESIGN_FAILURE`. Hash, identity, source, finite, or execution defects
are `IMPLEMENTATION_FAILURE`. None is a scientific kill.

## Stage 0B Mechanism Smoke

Only after Stage 0A passes, run a fixed `20`-step micro-fit for HASTE, the StaKe
proxy, no-hazard ablation, and standard LoRA on discovery rows.

Required:

- finite nonzero expected gradients;
- objective and gradient-scale audit passes;
- all four policies differ after training and disk reload;
- HASTE validation hazard and displacement metrics beat their trivial
  baselines;
- HASTE event-near action proxy improves over Base and no-hazard without
  violating action bounds;
- event-far action delta p95 remains below the frozen validation threshold;
- no privileged inference input and no confirmatory access.

## Bounded Validation Search

Maximum six HASTE configurations:

- event horizon `H_e in {20,50}`;
- hazard coefficient `lambda_h in {0.25,0.5,1.0}`;
- Cartesian product only, exactly six maximum;
- at most two lightweight seeds per configuration;
- fixed displacement and retention coefficients `1.0`;
- no additional architecture, optimizer, rank, or threshold variants.

The preregistered validation score combines event-near action proxy, hazard
NLL, displacement Huber, event-far retention, action validity, and compute.
Freeze one configuration with a smaller-horizon then smaller-coefficient tie
break. Save all tried configurations and negative results.

## Confirmatory Boundary

Validation reset identities: `20262201..20262210`.

Stage A reset identities: `20262211..20262220`.

Stage B adds: `20262221..20262250`.

Stage A uses approximately ten paired episodes per policy. Stage B uses at
least forty paired episodes per key policy. One expansion to 80 is allowed only
when Stage B is genuinely unresolved under the frozen rule.

Confirmatory outcomes may not retune HASTE. A redesign starts a new method
cycle.

## Paper-Candidate Boundary

HASTE becomes a serious candidate only if it beats Base, the StaKe proxy, the
no-hazard ablation, and standard LoRA while retaining event-far behavior and
showing mechanism-consistent hazard/displacement evidence. Then immediately
verify Quantized OpenVLA-OFT INT4, one second condition, current baselines, and
compute/latency outside all recorded contention intervals.

## Automatic Continuation

After a valid Stage 0 or empirical failure, preserve the result, forbid rescue,
commit, push, and continue automatically to the next exact-three cycle. Routine
monitoring, validation, resume, commit, and push require no user approval.
