# PCAV-VLA Reviewer B Attack

Date: 2026-07-15 KST

Decision: `PCAV_REVIEWER_ATTACK_CONDITIONAL_PASS_REBUTTAL_REQUIRED`

The frozen proposal hash is
`E8B23C755C6D4E450FD193101CC0B15F88AAFE20E137A0F86830ED6D421E12AA`.
Reviewer B did not modify the proposal.

## Primary Sources Reviewed

- TACO paper: https://arxiv.org/abs/2512.02834
- TACO official code: https://github.com/breez3young/TACO
- ProgressVLA: https://arxiv.org/abs/2603.27670
- VLA-ATTC: https://arxiv.org/abs/2605.01194
- RoboMonkey: https://arxiv.org/abs/2506.17811
- RoboMonkey official code:
  https://github.com/robomonkey-vla/RoboMonkey

Reviewer B reconstructed the methods from equations, algorithms,
architectures, supervision, code availability, and reported ablations rather
than relying only on contribution text.

## Strongest Fair Reading

PCAV asks a legitimate question that TACO does not directly answer. TACO
selects the maximum pseudo-count candidate under the assumption that
demonstration density correlates with success. ProgressVLA shows that an
action-conditioned future-state model and task-progress estimator can steer
generation. PCAV tests whether a support filter followed by progress ordering
is better than either support-only selection or unconstrained progress
selection, while retaining exact Base fallback.

The support-then-progress lexicographic rule is cleaner than an arbitrary
weighted sum. The first comparison includes the closest prior, key ablation,
and one plausible simple baseline. The proposal also avoids privileged
inference signals and confirmatory tuning.

The method is not yet executable. The following issues are blocking.

## Blocking Findings

### 1. The Positive-Prior Generator Condition Is Missing

TACO applies its verifier to a policy already adapted to the downstream task.
PCAV proposes to sample from the untouched 40-task SmolVLA Base on three
identity-disjoint target tasks. A verifier cannot create an action mode absent
from every candidate.

Stage 0 must therefore include a development-only candidate-oracle audit. For
each fixed observation, report whether any alternative candidate is closer to
the demonstration action than Base in separately normalized translation,
rotation, and gripper units. Report the fraction of rows with a materially
better candidate and the attainable oracle reduction. A nonacting or uniformly
worse candidate set is `NO_USABLE_HEADROOM` or `DESIGN_FAILURE`, not evidence
against verification generally.

No stopped FAMR endpoint may be reused to manufacture this headroom.

### 2. `z_t + delta_z` Is Not Yet A Valid State Transition

The proposal writes
`P_phi(z_t + F_omega(z_t, s_t, a_i), ell)`. SmolVLA Transformer context is not
automatically an additive physical state space. Addition can be dimensionally
valid and scientifically meaningless.

The rebuttal must select a frozen state representation with an explicit target
and decoder. Acceptable choices include predicting a future pooled visual
feature directly, predicting a residual only after proving residual
reconstruction is better conditioned, or using a contrastive future-feature
objective. The audit must prohibit treating arbitrary token addition as a
world model.

### 3. The Proposal Omits ProgressVLA's Initial-State Anchor

ProgressVLA conditions progress on the initial observation, current
observation, and instruction. PCAV currently writes only current context and
instruction. Without an initial anchor, similar current images can correspond
to different task progress, especially before versus after a failed grasp.

The rebuttal must either add a deployment-available episode-initial context or
justify and test its omission as an ablation. Initial context is legal because
it is available at inference and does not reveal reset identity.

### 4. Normalized Time Can Be A Collection-Style Shortcut

ProgressVLA assumes curated expert trajectories advance approximately
monotonically. Local LIBERO demonstrations may contain variable approach,
holding, and terminal padding. A model can predict camera drift, proprioceptive
path length, or normalized episode duration rather than task advancement.

Before accepting progress labels, report episode-length variance, terminal
padding, repeated-frame/action frequency, progress-label distribution by task,
and ordering accuracy against task-only, proprioception-only, frame-difference,
and normalized path-length baselines. Temporal pairs must exclude terminal
padding and near-identical states.

### 5. The Consequence Horizon Is Undefined

SmolVLA produces a 50-step chunk, while the proposal leaves `delta` and the
relation between candidate horizon and future feature open. A one-frame target
can ignore most of the candidate; a 50-frame target can cross contacts and
subgoals that are not predictable from offline demonstrations.

Freeze one horizon before training. Demonstrate that the candidate action
prefix used by the model matches the future offset and postprocessed action
semantics. No horizon search may be hidden outside the six configurations.

### 6. TACO Proxy Fidelity Is Underspecified

TACO's official implementation uses high-fidelity feature search over noising
levels, a Coin Flipping Network, internal representation extraction, and
pseudo-count inversion. The proposal does not freeze:

- feature layer and pooling;
- noising levels and number of searches;
- Rademacher target dimension and seed;
- CFN width, depth, and optimizer;
- pseudo-count epsilon and monotone transform;
- exact support percentile population.

Without these details, `taco_support_proxy` could be a weak custom density head
rather than a fair prior proxy.

### 7. Head Capacity And Compute Are Not Yet Credible

ProgressVLA uses pretrained visual features, a spatiotemporal Transformer world
model, joint training, and substantial compute. VLA-ATTC trains its critic for
30,000 steps. RoboMonkey uses a 7B verifier and millions of comparisons. A
20-step local micro fit can establish gradients and serialization, not adequate
capacity for scientific rejection.

Stage 0B must be classified as an implementation/capacity screen only. A weak
head result cannot permanently kill the mechanism. Full training steps,
parameter counts, memory budget, and optimization diagnostics must be frozen
before a robust design conclusion.

### 8. Action Validity Limits Are Not Numerical

The proposal promises absolute and Base-relative limits but gives no values.
FAMR failed precisely because a learned action passed finite/max-absolute checks
while worsening range exceedance. PCAV must freeze numerical limits before
trained candidate selection and must apply them to every candidate and selected
action.

The correct units are postprocessed 7D action units with translation,
rotation, and gripper reported separately. No after-the-fact clipping is
allowed.

### 9. Base Inclusion Alone Does Not Prove Identity Preservation

A randomly initialized progress head can confidently replace Base everywhere.
The exact initial policy must set every learned branch to abstain before
training, persist over disk reload, and include deterministic tie behavior.

Report candidate index, support score, predicted progress, advantage, selected
index, intervention reason, and fallback reason on every Stage 0 row.

### 10. The Standard-LoRA Comparison Is Asymmetric

`standard_lora_new_task` receives target-task policy adaptation while PCAV is a
frozen-Base selector. This is acceptable as a reviewer-killer but not as a
matched explanation unless data and compute are clearly reported. It must not
be used as PCAV's candidate generator after seeing that Base candidates have no
headroom.

If a future cycle uses an adapted generator, that is a separately frozen method
cycle unless it was preregistered before any PCAV confirmatory result.

### 11. Clean Retention Source Is Missing

Target validation alone cannot establish clean retention. Use frozen rows from
the original 40-task stable artifact and, before confirmatory evaluation, a
small disjoint closed-loop clean manifest. PCAV must default to Base on clean
states unless validation evidence justifies a sparse intervention.

### 12. Intervention Sparsity Could Make PCAV Trivially Equivalent

An aggressive support threshold and positive margin can force zero
interventions. Conversely, a zero margin can intervene everywhere. The
validation score must include a nondegenerate activation term but may not
reward intervention for its own sake.

Exact equivalence is a fatal pre-rollout result only if candidate headroom,
labels, capacity, and optimization all pass. Otherwise it is
`IMPLEMENTATION_OR_DATA_FAILURE` or `UNDERPOWERED_OR_UNRESOLVED`.

### 13. Novelty Is Defensible Only As The Joint Rule

TACO owns coupled support verification. ProgressVLA owns action-conditioned
future progress. VLA-ATTC owns adaptive relative candidate selection. PCAV
cannot claim any component individually.

The novelty claim must remain the minimal joint mechanism: support as a hard
eligibility condition, progress as a within-support preference, and
Base-relative abstention. If the support-only or progress-only arm explains the
gain, the contribution fails.

### 14. Partial Results And Resource Evidence Need Durable Rules

Every long job must use PID, heartbeat, status, atomic partial JSON, final JSON,
log, and exit-code files. Resume only missing keys. Duplicate
`(policy, task, reset_identity)` rows are invalid. The recorded Windows
Efficiency Mode interval remains excluded from timing/resource evidence.

## Required Rebuttal

Researcher A must provide:

1. a candidate-oracle headroom test without FAMR reuse;
2. an explicit future representation and consequence target;
3. an initial-state progress anchor;
4. temporal-label and shortcut audits;
5. one frozen consequence horizon;
6. a faithful transparent TACO proxy specification;
7. capacity-versus-scientific-failure separation;
8. numerical action-validity gates;
9. exact pretraining Base identity behavior;
10. clean-retention data and intervention reporting;
11. a narrow novelty claim;
12. durable execution, manifest, duplicate, and resource-contamination rules.

## False-Negative Calibration

Before closed-loop rollout, only the following may permanently stop PCAV:

- impossible or privileged required inference input;
- exact candidate equivalence with no oracle headroom;
- proven absence of any usable candidate improvement;
- contradiction in action semantics that cannot be repaired without changing
  the method;
- a robust, adequately powered result showing the joint mechanism is
  nonacting or dominated.

Collapsed labels, failed extraction, insufficient micro-fit, missing capacity,
or an underpowered offline metric are not scientific kills.
