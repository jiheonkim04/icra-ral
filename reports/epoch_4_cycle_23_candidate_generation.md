# Epoch 4 Cycle 23 Candidate Generation

Date: 2026-07-15 KST

Exactly three candidates are evaluated under the active performance-oriented,
prior-first, minimum-sufficient-method governance. HASTE repair or rescue is
not a candidate.

## Candidate 1: KITE-VLA

Name: `KITE-VLA`, Kinematic Integration Targets for Execution.

Closest positive prior: GeoPredict.

### Scientific Method

Fit a frozen discovery-only realization operator that maps cumulative
demonstration commands to measured future end-effector displacement. During
training, reconstruct the clean action chunk from a SmolVLA flow state, pass
its cumulative arm commands through that operator at horizons `5` and `20`,
and penalize normalized Huber error to the demonstrated future state.

Unlike GeoPredict's hidden-state keypoint prediction, KITE sends the
kinematic gradient through the generated action chunk. At inference the
operator and loss are absent. Rank-4 LoRA is only the one-GPU parameterization.

Mechanism chain:

`per-step imitation with cumulative realization drift -> generated chunk does
not imply the demonstrated future state -> approach/placement endpoint error ->
closed-loop failure`

`multi-horizon action-realization loss -> predicted chunk integrates toward
measured future end-effector displacement -> lower accumulated geometric error
-> improved spatial and goal-conditioned success`.

### Quality Screen

- Provisional novelty: direct action-to-state realization differs from
  GeoPredict hidden-state prediction and StyleVLA driving kinematics.
- Prior anchor: GeoPredict reports positive LIBERO, RoboCasa, and real-world
  results.
- Data viability: local discovery affine maps improve validation MSE by
  `87.15%` at horizon `5` and `91.68%` at horizon `20`.
- Identity: zero-effect adapter; no inference module.
- Decisive experiment: Base, transparent GeoPredict proxy, KITE, endpoint-only
  KITE ablation, and data/compute-matched standard LoRA.

Score:

| Criterion | Weight | Score |
| --- | ---: | ---: |
| Provisional novelty | 25 | 24 |
| Problem importance | 15 | 14 |
| Positive prior anchor | 20 | 19 |
| Technical mechanism | 20 | 19 |
| Data/supervision feasibility | 10 | 10 |
| Decisive experiment feasibility | 10 | 10 |
| Total | 100 | 96 |

## Candidate 2: CIRR-VLA

Name: `CIRR-VLA`, Counterfactual Intent Representation Regularization.

Closest positive priors: ACoT-VLA and ERVLA.

### Scientific Method

Derive a coarse five-segment action intent from each demonstration chunk and
contrast the correct intent against task-matched, geometrically separated
counterfactual intents. The objective shapes policy representations during
training; no intent is decoded at inference.

The extension beyond ACoT/ERVLA is counterfactual intent discrimination rather
than positive-only trajectory or reasoning supervision.

### Quality Screen

- Positive priors and ACoT official code are strong.
- Labels are locally available without privileged inference input.
- Negative-pair health and phase leakage need careful auditing.
- Novelty is vulnerable to generic contrastive action-representation priors.
- The first gate can test noncollapsed negatives and frozen-feature retrieval.

Score:

| Criterion | Weight | Score |
| --- | ---: | ---: |
| Provisional novelty | 25 | 20 |
| Problem importance | 15 | 14 |
| Positive prior anchor | 20 | 20 |
| Technical mechanism | 20 | 17 |
| Data/supervision feasibility | 10 | 8 |
| Decisive experiment feasibility | 10 | 9 |
| Total | 100 | 88 |

## Candidate 3: HFC-VLA

Name: `HFC-VLA`, Hybrid Frequency Consistency for VLA action flows.

Closest positive priors: FreqPolicy and ManiFlow.

### Scientific Method

Apply flow-time frequency consistency to six continuous arm dimensions while
matching the gripper's temporal first-difference profile in the event domain.
The extension addresses the hybrid continuous/discrete topology that a single
all-dimension Fourier loss ignores.

### Quality Screen

- ManiFlow has official code and strong positive consistency results.
- Labels and objectives are fully local and cheap.
- Generic consistency and frequency losses are crowded prior art.
- The arm/gripper decomposition risks appearing adjacent to closed HEST/HASTE
  routes even though the objective and inference path differ.
- A decisive micro-fit is feasible, but novelty risk is higher than KITE.

Score:

| Criterion | Weight | Score |
| --- | ---: | ---: |
| Provisional novelty | 25 | 18 |
| Problem importance | 15 | 13 |
| Positive prior anchor | 20 | 20 |
| Technical mechanism | 20 | 17 |
| Data/supervision feasibility | 10 | 9 |
| Decisive experiment feasibility | 10 | 7 |
| Total | 100 | 84 |

## Selection

Select exactly one candidate: `KITE-VLA`, `96 / 100`.

KITE has the strongest combination of positive external prior, measurable
local headroom, noncollapsed supervision, identity-preserving integration, and
a direct reviewer comparison. CIRR remains plausible but has negative-pair
leakage risk. HFC is executable but occupies a denser prior-art neighborhood
and is too adjacent to recent hybrid-action routes.

## Baseline Rationale

| Comparison | Scientific question |
| --- | --- |
| Base vs KITE | Does action-realization supervision improve SmolVLA? |
| GeoPredict proxy vs KITE | Does coupling supervision through generated actions improve over latent kinematic prediction? |
| Endpoint-only KITE vs KITE | Is multi-horizon realization necessary beyond endpoint matching? |
| Standard LoRA vs KITE | Is any gain explained by ordinary adaptation with the same data and compute? |

The first serious comparison contains exactly five policies: Base,
transparent GeoPredict proxy, KITE, endpoint-only KITE, and standard LoRA.
