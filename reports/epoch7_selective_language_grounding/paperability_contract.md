# Epoch 7 selective-language-grounding paperability contract

Decision date: 2026-07-20

Status: `PROBLEM_VERIFICATION_ONLY`. No method, acronym, training run, or Ours rollout is authorized by this contract.

## Proposed paper archetype and claim

Preferred archetype: RA-L method paper.

Conditional one-sentence claim: a locally trainable VLA adaptation can improve closed-loop success under meaning-preserving instruction rewrites while retaining both canonical-task competence and sensitivity to genuinely different intents, outperforming the unmodified X-VLA Base and a mechanism-faithful Counterfactual Action Guidance Prior.

This is a hypothesis, not a result. The paper route exists only if every problem-verification gate below passes.

## Minimum reviewer-defensible novelty

Any later method must jointly address two failure modes: unwanted action change within an intent-equivalence class and insufficient action change between distinct feasible intents. It must use verified real robot-demonstration supervision, affect executed closed-loop actions, and differ by at least two major dimensions—objective, supervision, inference mechanism, or evaluation claim—from each of RobustVLA, RoVLA, Stable Language Guidance/RSS, STRONG-VLA, CAST, CAG, ProGAL-VLA, and selective hidden alignment.

Plain paraphrase augmentation, action consistency alone, text canonicalization alone, CAG-style guidance alone, LoRA alone, or a renamed archived residual/gate method cannot satisfy the novelty minimum. If legal state/action-aligned between-intent supervision cannot be constructed from retained real demonstrations without synthetic or unverified action labels, the method-paper route closes.

## Frozen Base, Prior, task, and condition identities

- Base: unmodified `2toINF/X-VLA-Libero`, revision `129e71460678b7236cee6fc9707f09d9fa0c3590`, loaded offline with official X-VLA source revision `6bc2513f5f1cbec715cc668b414392a6cae5c671`.
- Benchmark: LIBERO-Para revision `5a2198299a6d7a49bdb3cd519c7e92ed803adf5f` and its ten sorted LIBERO-Goal environments.
- Standard condition: the canonical `(:language ...)` instruction from the corresponding LIBERO-Goal BDDL.
- Claim condition: one outcome-independent hardest structural/keyword-similarity entry for each `(eval_id, paraphrase_family)` pair, where family is `act`, `obj`, or `comp`. The resulting 30 filenames are frozen in `problem_verification_protocol.json`.
- Pairing: `act` uses initial-state index 0 and seed 7; `obj` uses initial-state index 1 and seed 8; `comp` uses initial-state index 2 and seed 9. Canonical and paraphrased members of a pair receive the identical environment, initial state, seed, horizon, and model-query seed schedule.
- Environment semantics: exactly one `OffScreenRenderEnv` may be live; 360x360 agent and wrist views; `domain_id=3`; absolute EEF action mode; ten dummy settle steps; 300-step maximum; official `env.check_success()` endpoint.
- No confirmatory identity, Ours output, reward, success, or episode outcome may influence these identities.

The retained original checkpoint is not assumed bit-identical to the LeRobot conversion used by LIBERO-Para. A local canonical-competence failure triggers an identity audit against `lerobot/xvla-libero` before declaring `BASE_NOT_COMPETENT`.

## Comparators and legal information

Strong Prior: a mechanism-faithful local port of training-free Counterfactual Action Guidance (CAG-TF), using the same X-VLA as both branches and the published X-VLA guidance scale:

`a_CAG = a_empty + 1.5 * (a_cond - a_empty)`.

This is Eq. 4 of arXiv:2602.17659v2; at guidance scale 1, it recovers the conditional branch. A first implementation mistakenly used conditional-plus-guidance and was stopped after 14 invalid episodes. The primary-source-aligned repair changes no identity, seed, horizon, scale, or outcome rule, and the invalid episodes are excluded.

The conditional and empty-language chunks must be generated sequentially from the same observation and captured RNG state, with only one model resident. Mixing occurs in the model's native 10D absolute action representation before the existing LIBERO conversion. The port must be labeled local, not official, unless an author artifact is found and verified.

Simple Control: deterministic character-trigram cosine retrieval from the observed instruction to a fixed catalog of the ten canonical LIBERO-Goal instructions, followed by unmodified Base execution. It uses language only, no filename/eval ID, simulator state, reward, outcome, or hidden task label. The frozen implementation has 3,476/4,092 (84.95%) mapping accuracy over the complete benchmark metadata and 24/30 (80.0%) on the hard discovery panel before any rollout outcome was observed.

Oracle/headroom diagnostic: replace a paraphrase with its benchmark-provided canonical instruction while holding all other pair variables fixed. This is not a deployable method and may only establish that legal language information can recover success.

## Endpoints and decision thresholds

Main endpoint: official binary closed-loop task success, analyzed as matched pairs. Supporting endpoints are task coverage, paraphrase-family coverage, success-conditioned action divergence, latency, peak VRAM, host RAM, and swap use. PRIDE may be reported only for a benchmark-scale result and cannot replace success rate.

Base competence gate:

- canonical Base succeeds in at least 18/30 paired episodes (60%);
- successes span at least 6/10 tasks;
- action chunks are finite, converted correctly, and actually executed;
- no reset, camera, normalization, controller, or success-semantic defect is found.

Problem-gap gate:

- paraphrase success is at least 15 percentage points below paired canonical success;
- at least 5 discordant pairs are canonical-success/paraphrase-failure;
- the adverse direction appears in at least 4/10 tasks and at least 2/3 paraphrase families.

Headroom gate:

- at least 5 failed paraphrase pairs are recoverable by the paired canonical instruction; and
- failures are not explained by invalid instructions or infrastructure.

Prior-residual gate:

- CAG-TF is correctly action-connected and retains canonical success within 5 percentage points of Base; and
- after CAG-TF and the simple canonicalization Control, at least one reviewer-defensible residual remains. A route based only on closed-set instruction lookup is not paperable.

Minimum later method effect, if method discovery is authorized:

- at least +10 percentage points over Base on the frozen primary claim condition;
- at least +5 percentage points over CAG-TF, or a frozen meaningful Pareto advantage with greater success and lower compute;
- canonical retention no worse than 5 percentage points below Base;
- a positive paired effect across multiple tasks, not one selected task.

These are practical-effect gates; confirmatory uncertainty and multiplicity rules must be frozen before Ours evaluation.

## Closest-overlap falsification, ablation, and generalization

Closest-overlap falsification: before naming a method, compare equations, trainable modules, supervision, inference calls, and claimed evaluation axis against RobustVLA/RoVLA (paraphrase consistency), CAST (counterfactual labels/actions), CAG (dual-branch action guidance), RSS (language-neighborhood robustness), and ProGAL-VLA (grounding/ambiguity). If the proposed mechanism is a direct union or minor regularizer change, classify `TOO_OVERLAPPING_FOR_RAL_METHOD_CLAIM` and rotate.

Required later ablation: remove the between-intent selectivity term while keeping parameter count, update count, examples, and optimizer fixed.

Required simple control: the frozen character-trigram canonicalizer above. A stronger compact semantic-retrieval control must be added before paper confirmation if a locally licensed checkpoint becomes available.

Generalization priority: unseen paraphrase categories/identities within LIBERO-Para, then a second public paraphrase or counterfactual task suite, then a second locally runnable VLA only if the mechanism transfers faithfully. Generalization is evidence, not permission to tune on confirmatory outcomes.

## Resource and storage envelope

- At most one full X-VLA and one simulator environment resident.
- CAG branches execute sequentially; no duplicate full model.
- Zero WSL swap use at qualification and no sustained swap growth.
- Target peak VRAM below 13 GiB and host working set below 20 GiB.
- First Base/Prior discovery panel: at most 120 closed-loop episodes including repairs and reruns.
- Any later training pilot: at most 8 GiB new checkpoint/output storage and 12 active GPU-hours before a new gate.
- Maintain at least 20% free C: capacity and do not write frames/video unless explicitly required.

## Predicted main table

Rows: Base, CAG-TF Prior, text canonicalization Control, later Ours (only if authorized), selectivity-term ablation, and Oracle. Columns: canonical success, overall paraphrase success, `act`/`obj`/`comp` success, between-intent selectivity endpoint, paired gain versus Base, latency, peak VRAM, and second-suite/model generalization.

## Terminal decisions for this thesis

- Paper-GO: all paper-readiness requirements pass with a distinct method, positive official closed-loop Base and Prior gains, retention, statistics, ablation, control, and generalization evidence.
- Narrow: evidence supports only a smaller explicitly stated claim that still meets an RA-L archetype minimum.
- Kill exact mechanism: a valid method fails after the problem is verified; return to bounded method discovery without reusing its confirmatory outcomes.
- Rotate thesis: Base is not competent after identity repair; there is no repeatable gap or legal headroom; CAG/control saturates the problem; supervision is not legally constructible; or novelty collapses to direct prior work.

The next authorized action is Base/control/problem/headroom verification under `problem_verification_protocol.json`; it is not method design.
