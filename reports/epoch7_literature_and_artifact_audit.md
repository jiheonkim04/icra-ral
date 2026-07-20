# Epoch 7 Current Literature and Artifact Audit

Audit date: 2026-07-20 (Asia/Seoul)

## Rotation 2 update after the language-mechanism closure

The frozen action-energy falsifier closed the initial language mechanism. A second current-primary-source scan therefore re-opened the method, benchmark/evaluation, and systems archetypes rather than weakening the original claim.

The strongest remaining local boundary is a benchmark question, not a hidden-dynamics method claim. A recent position paper argues that VLA task success does not identify whether success comes from semantic mapping or physical action decisions and specifically calls for controlled interventions that hold semantics fixed while varying surface conditions or dynamics. It proposes separately observable semantic correctness and physical success, including tests involving mass, friction, or material. The paper is conceptual and releases no empirical benchmark artifact.

This direction is constrained by close empirical work:

- Eva-VLA varies target-object 3D pose, illumination, and adversarial-patch placement. These are visible/perceptual changes; the primary paper does not intervene on object mass, object/support friction, or articulated resistance.
- LIBERO-PRO varies objects, positions, instructions, task logic, and environments. Its official MIT code and CC-BY-4.0 data are public, but its perturbation axes do not isolate visually matched latent dynamics.
- J-PARC is the closest physical-execution precedent. It evaluates robot-side joint locks, range limits, and increased joint friction across OpenVLA-OFT and pi0.5, 50 episodes per task on all four LIBERO suites, then learns a history-conditioned residual calibrator. It reports official task success, not correct-target contact followed by completion conditional on grounding. Any Epoch 7 contribution must therefore stay on object/environment dynamics and causal failure attribution; a generic physical-fault robustness or residual-calibration claim is prohibited.
- LIBERO-CF already supplies contact-based target-grounding logic and evaluates X-VLA in its paper, while RoboSemanticBench explicitly separates semantic target choice from grasp success. A new benchmark cannot claim that decomposed grounding is itself novel.
- PhAIL and Beyond Binary Success make statistical reporting and distributional evaluation strong required controls rather than contributions by themselves.

The scan found no primary paper or official artifact, through 2026-07-20, that combines all of the following in manipulation VLA evaluation: identical instruction and initial rendering, controlled object/environment mass/friction/articulated-dynamics changes, correct-target-contact measurement, and task completion conditional on correct grounding. This is a bounded search inference and must not be written as an exhaustive universal absence claim.

### New language and adaptation overlap

The language route is even more crowded than at the initial audit. RoboSemanticBench tests knowledge-conditioned physical target selection. Breaking Lock-In/DeLock preserves visual grounding under low-data post-training. Anchor-Align, posted 2026-07-15 with linked code, uses frozen-VLM representation anchoring and motion-direction language-action alignment and reports gains across physical robots, LIBERO-PRO, LIBERO-Plus, and CALVIN. These works, together with the failed local mechanism test, close a simple language-alignment rescue.

Long-context and test-time adaptation are also occupied. RoboTTT scales robot-policy context to 8K timesteps using fast weights and reports on-the-fly improvement and perturbation robustness. J-PARC uses recent joint dynamics for physical-fault correction. VLS, TTT-VLA, and related work already steer or adapt policies at inference. A history-conditioned latent-physics adapter is therefore not selected as the fresh fallback.

### Outcome-free local simulator feasibility

Static source inspection and serial environment construction confirmed that the retained LIBERO runtime exposes writable MuJoCo arrays needed for a bounded preflight: `body_mass`, `geom_friction`, and `dof_damping`. The Goal scene includes free bodies for the bowl, cream cheese, wine bottle, and plate; drawer slide joints have named damping entries; the stove button has a named hinge damping entry. LIBERO-CF's public evaluator contains a root-body gripper-contact construction that can be adapted and independently tested. This establishes implementation feasibility only; it is not evidence that any intervention changes VLA outcomes or remains physically meaningful.

## Decision summary

The refresh found a real, large, executable language-robustness gap, but also found that the obvious method family is already crowded. LIBERO-Para reports an X-VLA drop from 97.8% on LIBERO-Goal to 62.1% on 4,092 meaning-preserving paraphrases. However, paraphrase augmentation, action consistency, residual language guidance, counterfactual guidance, step-wise language alignment, and explicit entity grounding all now have close primary precedents. A plain paraphrase-consistency adapter is therefore prohibited as an Epoch 7 thesis.

One narrower hypothesis remains eligible only for problem/headroom verification: learn invariance within meaning-preserving instruction classes while simultaneously preserving selectivity between different feasible intents. The distinction must survive comparisons to RobustVLA, RoVLA, RSS, CAST, CAG, and a simple canonicalization control. No method novelty is authorized yet.

## Executable Base and benchmark

### X-VLA

- Official source: [2toinf/X-VLA](https://github.com/2toinf/X-VLA), Apache-2.0.
- Local source: `C:\assets\repos\X-VLA` at `6bc2513f5f1cbec715cc668b414392a6cae5c671`, clean.
- Local official checkpoint: `2toINF/X-VLA-Libero`, revision `129e71460678b7236cee6fc9707f09d9fa0c3590`, Apache-2.0, approximately 3.52 GB of listed Hub files.
- Retained WSL snapshot: `/home/jiheon/assets/checkpoints/xvla_hf_cache/transformers/models--2toINF--X-VLA-Libero/snapshots/129e71460678b7236cee6fc9707f09d9fa0c3590`.
- Runtime smoke: CUDA import and official model path were retained from Epoch 6; LIBERO-Para serial environment creation was rechecked locally on 2026-07-20.
- The LIBERO-Para paper instead evaluates the LeRobot conversion `lerobot/xvla-libero`, revision `12e8783e996944f5c97e490d37d4c145484ed70a`, Apache-2.0, with 3,519,090,679 listed bytes. This is a released converted checkpoint, not yet a locally reproduced identity match to the retained original checkpoint. Any comparison must label this distinction.

### LIBERO-Para

- Paper: [LIBERO-Para](https://arxiv.org/abs/2603.28301), submitted 2026-03-30.
- Official code/data: [cau-hai-lab/LIBERO-Para](https://github.com/cau-hai-lab/LIBERO-Para), MIT.
- Verified remote/local revision: `5a2198299a6d7a49bdb3cd519c7e92ed803adf5f` on `master`.
- GitHub repository metadata at audit: 222,187 KiB reported; 457,087,965 local bytes after shallow clone; last push 2026-06-27; not archived.
- Contents: 4,092 paraphrases of 10 LIBERO-Goal instructions, organized by action, object, and compositional variation; published evaluator integrations for six VLA model families; PRIDE metadata and analysis.
- Published X-VLA evidence: 97.8% LIBERO-Goal, 62.1% LIBERO-Para, a 35.7 percentage-point drop; PRIDE 52.7. The paper reports five stochastic seeds (7--11) and about 11 X-VLA GPU-hours on an L40S with 6.5 GB peak model VRAM.
- Local simulator smoke: one official goal environment at 360x360, `eval0.pruned_init` state 0, both cameras present, and clean environment close all passed.
- Resource adaptation required: the released X-VLA evaluator pre-creates 10 MuJoCo environments. Epoch 7 must use a one-live-environment serial lifecycle while preserving BDDL, initial state, camera, action, success, and seed semantics.
- Dataset caveat: the paraphrase generation pipeline remains a repository TODO. Evaluation BDDLs and metadata are present; generation is not required for the intended audit.

## Closest language and grounding work

### Direct precedents that prohibit a plain consistency method

1. [On Robustness of VLA Models against Multi-Modal Perturbations / RobustVLA](https://arxiv.org/abs/2510.00037) trains semantic-preserving input consistency, including lexical and syntactic instruction variation, and worst-case action robustness. Official code is available at [gakakulicc/RobustVLA](https://github.com/gakakulicc/RobustVLA). This is the closest direct precedent to action-consistency under paraphrases.
2. [RoVLA](https://arxiv.org/abs/2605.19678) explicitly samples synonymous instructions during training and combines instructional, flow-evolution, and observational consistency. Its code was announced but not yet required for this audit.
3. [Stable Language Guidance / RSS](https://aclanthology.org/2026.acl-long.190/) uses LLM-generated dense syntactic neighborhoods and residual subtraction of a visual affordance prior. Official code is linked at [Doo-mon/RSS](https://github.com/Doo-mon/RSS).
4. [STRONG-VLA](https://arxiv.org/abs/2604.10055) uses decoupled robustness acquisition and clean task realignment over 28 multimodal perturbations.
5. [When Does Language Matter?](https://arxiv.org/abs/2606.11906) measures step-wise language sensitivity and applies selective representation alignment at language-critical steps. This closes a straightforward temporal-gating extension.

### Counterfactual selectivity and language reliance

1. [CAST](https://arxiv.org/abs/2508.13446) generates alternative feasible instructions and synthetic action branches so different instructions yield different actions under similar observations. Official MIT code/data/checkpoints are available at [catglossop/CAST](https://github.com/catglossop/CAST). Its data pipeline needs Gemini/GCP and an atomic policy; that exact official pipeline is not locally credential-free.
2. [When Vision Overrides Language / CAG](https://arxiv.org/abs/2602.17659) introduces LIBERO-CF and combines language-conditioned and language-unconditioned action branches at inference. Its [official MIT-licensed artifact](https://github.com/yuffish/LIBERO-CF) was verified locally at `8460457bfca6e0ef2e856bc104e2c60b023ef2a7`. The released source implements `uncond + scale*(cond-uncond)` and provides 65 counterfactual tasks across spatial, spatial-focused, object, long-horizon, and OOD suites. The artifact supports OpenPI pi0/pi0.5 and OpenVLA-OFT, not X-VLA.
3. [ProGAL-VLA](https://arxiv.org/abs/2604.09824) uses a symbolic planner, 3D entity graph, contrastive grounding, and ambiguity-aware selective prediction. It is a strong direct precedent against generic object-name canonicalization or entity-grounding claims.
4. [IGAR](https://arxiv.org/abs/2603.06001) is a train-free attention recalibration method for contradictory instructions, evaluated on a purpose-built instruction-conflict benchmark.
5. [Metamorphic Testing of VLA-Enabled Robots](https://arxiv.org/abs/2602.22579) already formalizes meaning-preserving input transformations as robot test relations across five VLAs, two robots, and four tasks. This limits a benchmark-only metamorphic-testing thesis.

### Interpretation of remaining boundary

The remaining distinction is not “add paraphrases,” “make actions consistent,” “amplify language,” “canonicalize object words,” or “contrast text and entities.” It would have to show that one-sided invariance can erode intent selectivity and that a paired, bidirectional objective improves both within-intent robustness and between-intent discrimination with real intended policy training and closed-loop outcomes. CAST is the closest conceptual neighbor; the local hypothesis is eligible only if it avoids synthetic action fabrication, uses real demonstration supervision, and produces a measurable advantage over both one-sided paraphrase training and a CAG-style prior.

## Other refreshed axes

### Memory and history

- [LIBERO-Mem](https://arxiv.org/abs/2511.11478) and its Embodied-SlotSSM already target object-level non-Markovian manipulation.
- [OptimusVLA](https://arxiv.org/abs/2602.20200) combines global retrieval memory, local action-history consistency, and adaptive flow steps. Official inference overlays and memory assets were released in May 2026 at [iLearn-Lab/CVPR26-OptimusVLA](https://github.com/iLearn-Lab/CVPR26-OptimusVLA).
- Artifact limitation: OptimusVLA does not release the required pi0.5 policy checkpoint; the user must separately obtain and convert `pi05_libero`. Its released recipe also runs four suites in parallel. The retained local OpenPI material is not an identity-verified converted pi0.5 LIBERO policy.
- [Benchmarking Robot Memory Under Interference](https://arxiv.org/abs/2606.22338) now directly evaluates distractor-history interference. This makes a simple memory-buffer or retrieval-filter thesis incremental.

### Uncertainty, abstention, and action chunks

- [Confidence Calibration in VLAs](https://arxiv.org/abs/2507.17383) uses prompt ensembles and action-wise calibration.
- [Uncertainty Quantification for Flow-Based VLAs](https://arxiv.org/abs/2606.18043) introduces ensemble flow discrepancy and uncertainty-aware action selection.
- [SCALE](https://arxiv.org/abs/2602.04208), [Adaptive Action Chunking](https://arxiv.org/abs/2604.04161), and [ReconVLA](https://arxiv.org/abs/2604.16677) cover single-pass uncertainty, adaptive execution horizons, and conformal action/state failure detection.
- ProGAL-VLA also includes ambiguity-aware selective prediction. Together with closed Epoch 5/6 uncertainty and chunking routes, the local residual is weak.

### Recovery and evaluation systems

- FLARE, FAR, and See-Plan-Rewind cover retry/reset, failure-aware retry, and progress-aware recovery.
- [vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness), local revision `a7eb023a962456bb0b6be40aa4336c31b7ac4ce6`, already supports broad model/benchmark interoperability, sharding, batching, and reproduced scores.
- [Beyond Binary Success](https://arxiv.org/abs/2603.13616) already supplies anytime-valid, sample-efficient robot-policy comparison. A generic sequential-evaluation thesis is therefore not novel.

## Artifact and licensing decision table

| Artifact | Status | License / access | Local decision |
|---|---|---|---|
| X-VLA source + original LIBERO checkpoint | verified and retained | Apache-2.0 | executable Base |
| LIBERO + raw goal demonstrations | verified and retained | Apache-2.0 | executable simulator/data |
| LIBERO-Para | cloned and smoke-tested | MIT | executable benchmark after serial lifecycle adaptation |
| `lerobot/xvla-libero` conversion | remote revision/size verified | Apache-2.0 | not downloaded; retained original preferred until identity need is justified |
| CAG + LIBERO-CF | official artifact verified at `8460457`; serial spatial and OOD runtime preflights pass | MIT | official benchmark/runtime retained; local CAG-TF remains a mechanism-faithful X-VLA port because the release has no X-VLA adapter |
| CAST | public code/data/checkpoint | MIT; generation needs Gemini/GCP | read-only comparator design; official generation blocked locally |
| OptimusVLA | source overlay and memory assets released | MIT | required base checkpoint identity unresolved; not a runnable fallback yet |
| RoVLA / ProGAL | paper-level design verified | announced/public paper; complete runnable identity not established | novelty comparator, not official reproduction |

## Audit conclusion

`ROTATION2_EXHAUSTED_FRESH_PORTFOLIO_ROTATION_REQUIRED`.

The language method space is saturated and the selected real-action mechanism failed, so no weaker language rescue is authorized. The latent-dynamics benchmark also closed before any policy rollout: its outcome-free simulator preflight passed all four tasks, but its frozen standard-only demonstration oracle established altered-condition success for only drawer opening and bowl placement. That is two tasks across two collapsed families, below the prespecified requirement of three tasks spanning all three families. The decision is `NO_LEGAL_HEADROOM`; it does not claim the failed altered tasks are impossible.

The prespecified policy-RNG reliability fallback also closed before outcomes. Its direct policy-seed factor is only one material change from the unresolved Epoch 6 schedule-invariant stochastic VLA thesis, which already established action-level schedule dependence. Reset-versus-policy variance decomposition is standard reporting, and a candidate-selection method collides with SDN. Candidate H is therefore `TOO_OVERLAPPING_OR_TRIVIAL` with zero rollouts. Rotation 2 is exhausted and a fresh method/benchmark/systems portfolio is required.

## Rotation 3: stability-qualified completion

The fresh scan found a narrower evaluation boundary around task completion. vla-eval reproduces host benchmarks and documents protocol ambiguity, but its published metrics remain native task success. PhAIL treats time to first success as a distribution. VLA-SCT adds a visual-memory termination detector intended to stop redundant actions. SafeVLA-Bench is the closest benchmark: it preserves native success and adds task-aware temporal safety constraints including object stability. None of these inspected primary artifacts defines native completion as persistence of the unchanged host goal predicate after the policy stops under a frozen neutral controller dwell, or uses unused expert follow-through as a recoverability oracle.

This boundary is only plausible. It collapses if SafeVLA-Bench's existing metrics fully explain every disagreement, if neutral holding is not controller-independent, if only one malformed LIBERO predicate is responsible, or if no multi-policy conclusion changes. The pre-policy ten-task expert gate is therefore frozen before any persistence outcome. The prior Epoch 6 persistent-success idea was never executed and was blocked only by the then-shared simulator resource rule; the current serial path and corrected pagefile policy make a fresh outcome-bearing gate legal without rewriting the archived record.

False-premise language grounding was reconsidered and rejected. DoWhat?, IVA, IGAR/ICBench, LIBERO-CF, and ProGAL-VLA now cover detection, rejection/correction, contradiction, and counterfactual grounding, while no retained policy exposes a competent clarification/refusal interface. Generic verification/recovery is likewise crowded by Pre-VLA and July 2026 agentic execution work.

### Rotation 3 primary adjudication

The ten-task official-demonstration gate was valid but returned `NO_REPEATABLE_GAP`. Native success occurred on all ten tasks; immediate neutral dwell failed on three, and the unused expert suffix recovered all three. Every failure was nevertheless an `On` placement predicate, so the effect covered one mechanism and one predicate family rather than the frozen cross-mechanism requirement. No policy was loaded or queried. The result is retained as an actionable LIBERO predicate observation, not promoted into a general VLA benchmark.

The prespecified typed non-gripper contact-transition fallback is now active only for an audit of its former operational blocker. Its scientific labels and thresholds remain frozen. Any repair must be outcome-free and limited to the resource rule; label extraction cannot begin until the amended rule is written and hashed.

### Contact-transition resource amendment freeze

The outcome-free amendment is frozen at
`reports/epoch7_contact_transition_topology/resource_rule_amendment.json`,
SHA-256
`7CCDCE5D9AA0B24C356AF873D0481AF76312D3C7FCF6871C4CA80FD6621ACFEB`.
It preserves the Epoch 6 scientific protocol hash and every label/headroom
gate. It changes only host qualification: an 85% peak-RAM ceiling, zero WSL
swap, no sampled paging writes, at most 16 MiB allocation-only pagefile
jitter, and bounded controlled cache release. No contact-label row or outcome
was accessed while writing or testing the amendment. A valid actual-path
resource smoke is required before Stage 0A can resume.
