# Paperability Contract: Counterfactual Preservation of Task-Causal Low-Motion Actions

Status: archived unexecuted after official-code novelty collision
Frozen empirical values: none

This initial contract is preserved as an immutable selection record. Pinned
VLA-Arena code subsequently showed that its data regeneration already performs
outcome-based progressive no-op retention through simulator replay. The central
causal supervision and intervention therefore collide. Classification:
`TOO_OVERLAPPING_OR_TRIVIAL`. This contract authorizes no experiment, method,
or paper path.

## Claim and causal mechanism

One-sentence claim: in simulator-grounded VLA training data, preserving
low-motion action segments whose removal changes downstream task physics yields
better closed-loop policies than removing all such actions or preserving fixed
gripper-centered neighborhoods, without privileged inference inputs.

Claim-to-mechanism graph:

`magnitude-only filtering -> deletes physically necessary low-motion segments
-> removes supervision for constrained/settling states -> Base action errors or
closed-loop failures at those states`

`matched exact-init deletion/restoration -> training-only causal necessity
labels -> preserve/reweight necessary segments -> retain task-causal action
supervision -> improve held-out closed-loop success`

The second chain must not be asserted unless every empirical link passes.

## Closest-three-prior difference

| Prior | What it does | Required difference here |
|---|---|---|
| OpenVLA-OFT | Removes no-op actions broadly to prevent freezing. | Preserve only segments with matched downstream physical necessity, not all no-ops. |
| VLA-Arena | Preserves fixed action neighborhoods around transitions because wholesale deletion harms replay. | Replace fixed gripper/transition distance with intervention-derived necessity; fixed-N is the strongest control. |
| FrameSkip | Scores temporal importance using action variation, visual-action coherence, progress, and gripper transitions. | Use task-outcome-changing counterfactual replay within a trajectory, not a heuristic importance score or execution-duration predictor. |

## Expected contributions

1. A reproducible exact-init counterfactual protocol that separates physically
   necessary low-motion supervision from redundant near-zero actions.
2. A training-only preservation/weighting mechanism with no additional
   deployment sensor or inference module.
3. Closed-loop evidence against retain-all, remove-all, fixed-N, and matched-data
   controls on untouched tasks/resets, plus a failure and simulation-limit audit.

## Result shells

Primary table (all values `TBD` until frozen execution):

| Policy/data rule | Standard success | Held-out task success | Held-out reset success | Latency | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| Competent Base / remove-all | TBD | TBD | TBD | TBD | TBD |
| Retain-all control | TBD | TBD | TBD | TBD | TBD |
| Fixed-N gripper-neighborhood Prior | TBD | TBD | TBD | TBD | TBD |
| Selected causal preservation | TBD | TBD | TBD | TBD | TBD |

Key ablation table:

| Variant | Physical-necessity label | Gripper-only explanation controlled | Closed-loop result |
|---|---|---|---:|
| Full | yes | yes | TBD |
| No causal label / fixed-N | no | yes | TBD |
| Magnitude-only | no | no | TBD |
| Label-shuffled matched-retention control | shuffled | yes | TBD |

Main figure shell: one successful demonstration timeline showing action
magnitude, gripper transitions, the frozen low-motion candidate segments, and
matched delete/restore physical-state divergence; alongside the training-only
label path and the unchanged inference interface. Every visual element must
come from frozen artifacts.

## Reviewer defense and failure condition

Simulation-only defense: simulator state is necessary only to construct causal
training labels and matched interventions. The claimed contribution is limited
to simulator-grounded data curation and simulator closed-loop manipulation. The
evaluation must use diverse tasks, untouched partitions, paired uncertainty,
competent policies, strong heuristic controls, mechanism ablation, and explicit
resource/latency reporting. It will not claim hardware safety or sim-to-real.

Strongest likely reject reason: the proposed label is an expensive reformulation
of "keep actions near a gripper transition," and any gain comes from retaining
more data rather than causal information.

Evidence required to answer it: repeated necessary non-gripper segments;
matched retention-count and label-shuffle controls; fixed-N and gripper-event
baselines; a valid restore-only oracle; closed-loop improvement on untouched
tasks/resets; and a key ablation that removes causal labels while holding model,
data volume, training budget, and checkpoint selection fixed.

## Six-page allocation sketch

| Content | Target pages |
|---|---:|
| Motivation, exact claim, contributions | 0.6 |
| Closest work and distinction | 0.5 |
| Counterfactual labeling and training mechanism | 1.3 |
| Experimental protocol and integrity partitions | 0.8 |
| Main, ablation, generalization, and efficiency evidence | 1.5 |
| Failure analysis, limitations, conclusion | 0.5 |
| References | 0.8 |

The allocation is a design constraint, not permission to draft results or
promotional prose. Paper generation remains forbidden before
`PAPER_CANDIDATE_GO`.
