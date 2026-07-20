# Epoch 7 Current Literature and Artifact Audit

Audit date: 2026-07-20 (Asia/Seoul)

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

`LITERATURE_REFRESH_COMPLETE_WITH_HIGH_NOVELTY_RISK`.

The current executable problem is real, but the method space is saturated. The later focused audit classifies real-action equivalence/selectivity ranking as `INCREMENTAL_BUT_DEFENSIBLE_WITH_STRONG_EVIDENCE`; every simpler canonicalization, augmentation-only, or guidance formulation is prohibited. A failed frozen Stage-0 test closes that mechanism rather than authorizing a weaker paraphrase adapter.
