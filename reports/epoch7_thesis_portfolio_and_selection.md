# Epoch 7 Thesis Portfolio and Initial Selection

Decision date: 2026-07-20

This portfolio is a gate, not a commitment to write a paper. Ratings reflect novelty, scientific value, local artifact reality, resource fit, comparator strength, and ability to obtain official closed-loop evidence.

## Candidate A — paired semantic equivalence and intent selectivity

Paper archetype: method paper.

Question: can a VLA become invariant to meaning-preserving rewrites without becoming less discriminative between different feasible instructions in the same scene?

Proposed evidence boundary: use LIBERO-Para for within-intent variation and LIBERO-Goal/counterfactual instructions for between-intent sensitivity. The method, if authorized later, must use real demonstration supervision and explicitly optimize both sides. Plain paraphrase augmentation or action consistency is prohibited.

Closest work: RobustVLA, RoVLA, RSS, CAST, CAG, ProGAL-VLA, and step-wise language alignment.

Local feasibility: high for X-VLA inference and serial LIBERO-Para evaluation; moderate for parameter-efficient X-VLA training. A CAG-style mechanism-faithful Prior is locally implementable. The official benchmark and raw goal demonstrations are present.

Key reviewer risk: the idea may reduce to a combination of paraphrase augmentation and counterfactual supervision, or a simple canonicalizer may solve the closed set. The retained original X-VLA checkpoint is not yet proven bit/behavior identical to the LeRobot conversion used by the benchmark paper.

Rating: `PLAUSIBLE`, selected primary for Base/Prior/problem/headroom verification only.

Go requirement: local official closed-loop Base gap, a competent CAG-style Prior that leaves meaningful residual headroom, failure of a simple canonicalization/control explanation, and a novelty matrix showing at least two major dimensions beyond every direct prior.

## Candidate B — interference-resistant object-event memory

Paper archetype: method paper.

Question: can object-event memory retain task-relevant evidence under intervening irrelevant history while preserving a frozen VLA’s standard-task performance?

Closest work: LIBERO-Mem/Embodied-SlotSSM, OptimusVLA, RoboMME, and RoboMME-Interference, in addition to the closed Epoch 5/6 local memory routes.

Local feasibility: simulator and some OpenPI material are retained, but the now-released OptimusVLA recipe requires a separately obtained and converted pi0.5 LIBERO checkpoint that the official release explicitly does not supply. The repository’s default four-suite parallel execution also violates the host-memory contract until adapted.

Key reviewer risk: both the problem and retrieval/filtering solutions are directly occupied; a new local memory buffer would be incremental. Reconsideration would need a materially different benchmark, interference model, supervision, and memory mechanism from the archived local routes.

Rating: `BLOCKED` as an immediate fallback. It may be reconsidered only if a verified pi0.5 identity and a clearly non-duplicate mechanism become locally available.

## Candidate C — calibrated selective action-chunk execution

Paper archetype: method or systems paper.

Question: can a single VLA decide when to shorten, reject, or replan an action chunk using a calibrated risk signal?

Closest work: confidence calibration for VLAs, UQ for flow-based VLAs, SCALE, Adaptive Action Chunking, ReconVLA, ProGAL-VLA selective prediction, and the closed EAC/PESA family in the historical record.

Local feasibility: high for diagnostics and X-VLA inference; moderate for closed-loop evaluation.

Key reviewer risk: the contribution is already directly represented by action-wise calibration, ensemble flow discrepancy, single-pass uncertainty, conformal failure detection, and adaptive chunking. A new scalar or threshold would be a prohibited near-duplicate.

Rating: `WEAK`; not selected.

## Candidate D — progress-aware retry and recovery

Paper archetype: method paper.

Question: can a frozen VLA detect stalled progress and execute a bounded recovery without a separate large monitor?

Closest work: FLARE, FAR, See-Plan-Rewind, and multiple closed local residual/gate/recovery routes.

Local feasibility: closed-loop simulator execution is possible, but credible recovery training/evaluation requires failure and reset-skill evidence not retained as an official local artifact.

Key reviewer risk: direct contemporary methods already own retry, reset, progress monitoring, and rewind. A heuristic monitor would not clear the comparator bar.

Rating: `WEAK`; not selected.

## Candidate E — low-memory metamorphic VLA evaluation

Paper archetype: benchmark/evaluation or systems paper.

Question: can one-live-environment serial execution, paired seeds, and resumable manifests make robustness evaluation practical on a 24.87 GB host without changing benchmark semantics?

Closest work: Metamorphic Testing of VLA-Enabled Robots, LIBERO-Para, LIBERO-Plus, vla-evaluation-harness, and anytime-valid robot-policy comparison.

Local feasibility: high. The serial LIBERO-Para environment smoke already passes.

Key reviewer risk: engineering utility alone is insufficient for RA-L; broad interoperability, metamorphic relations, and statistically efficient evaluation are already published. The memory saving is an implementation property, not yet a robotics research contribution.

Rating: `WEAK`; retained as reproducibility infrastructure only.

## Initial selection

- Primary: Candidate A, gated at Base/Prior/problem/headroom. Status: `PRIMARY_PROBLEM_VERIFICATION_ONLY`.
- Fallback: none authorized for empirical execution. Candidate B is the highest scientific fallback but is artifact/resource blocked and directly crowded.
- Method naming: forbidden until Candidate A clears problem/headroom and method-discovery authorization.
- Paper package: forbidden until a paper-readiness GO.

## Rotation rule

Candidate A closes if any one of the following occurs:

1. local official X-VLA does not show a repeatable paraphrase problem under a frozen preregistered panel;
2. a simple canonicalization control removes the problem without a meaningful residual;
3. a competent CAG-style Prior removes the residual or establishes no legal headroom;
4. the proposed supervision cannot be constructed without synthetic/unverified action labels;
5. the novelty matrix collapses to RobustVLA/RoVLA plus CAST/CAG;
6. intended training cannot fit the resource contract.

On closure, perform a fresh portfolio cycle across method, benchmark, and systems archetypes; do not weaken the claim to keep the route alive.
