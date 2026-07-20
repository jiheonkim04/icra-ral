# Epoch 8 Language-Grounding Primary-Literature and Artifact Audit

Audit cutoff: **2026-07-20 (Asia/Seoul)**
Machine-readable matrix: `reports/epoch8_language_artifact_matrix.json`

## Decision

The Base problem remains scientifically open, but most obvious solution families are occupied. The retained original X-VLA checkpoint has discovery evidence of **30/30 canonical versus 19/30 matched paraphrase success**. That does not authorize a generic consistency adapter: RobustVLA and RoVLA already cover meaning-preserving action consistency; RSS and CAG cover conditioned-versus-unconditioned action guidance; ProGAL-VLA and GuidedVLA cover explicit entity/object grounding coupled to action; direct grounded-point work covers lightweight target injection into an action head; GPLA covers language/trajectory ranking; CAST covers alternative feasible instructions paired with counterfactual action labels; Anchor-Align covers representation retention plus language/action direction alignment; VLA Grounder covers learned grounded rewriting.

The surviving narrow boundary is **paired causal action response**, not a new object head. A viable test must use real demonstration supervision to require (a) invariance for audited meaning-preserving instructions and (b) the correct structured action change for a genuine target/intent swap in the same or matched scene. It must preserve Base behavior at initialization and use only RGB, language, proprioception, and permitted history at inference. This is a conditional authorization to formulate and falsify candidates, not a novelty claim.

## Critical corrections to the inherited audit

- GuidedVLA is no longer merely "code coming soon." The current official repository, a three-file released LIBERO checkpoint, and the released training dataset all resolve at immutable revisions recorded in the matrix. The older project page is stale.
- RoVLA code now resolves, but no RoVLA checkpoint is released; the README recommends eight training GPUs and its RoboTwin evaluation is explicitly not directly executable.
- RSS code and a public pi0.5 checkpoint resolve. Its root `LICENSE` is Apache-2.0 while the README says MIT; the conflict is preserved.
- CAG v2 now reports X-VLA scientifically, but the official repository still releases adapters only for OpenPI and OpenVLA-OFT. The local X-VLA port is mechanism-faithful but empirically incompetent (14/30 canonical, 11/30 paraphrase), so it is a negative comparator rather than a positive Prior.
- July work adds two important collisions: Anchor-Align (released inference code/weights, training code pending) and VLA Grounder/CLAP (paper/project only, no current executable release).

## Artifact matrix

| ID | Paper/version | Official repository; revision; license | Local status | Collision |
|---|---|---|---|---|
| x_vla_base | 2509.21305 retained official release | https://github.com/2toinf/X-VLA; `6bc2513f5f1cbec715cc668b414392a6cae5c671`; Apache-2.0 | LOCAL_RETAINED_AND_CUDA_VERIFIED | BASE |
| libero_para | 2603.28301 v1 | https://github.com/cau-hai-lab/LIBERO-Para; `5a2198299a6d7a49bdb3cd519c7e92ed803adf5f`; MIT | LOCAL_SERIAL_SMOKE_PASS | PROBLEM_DEFINITION |
| libero_cf_cag | 2602.17659 v2 | https://github.com/yuffish/LIBERO-CF; `8460457bfca6e0ef2e856bc104e2c60b023ef2a7`; MIT | LOCAL_BENCHMARK_PASS_PRIOR_INCOMPETENT | HIGH |
| robustvla | 2510.00037 v4 | https://github.com/gakakulicc/RobustVLA; `4a80d0d759f465c94e1fd3f3823c498714e830d2`; NO LICENSE FILE DETECTED | REMOTE_SOURCE_INSPECTED_NOT_RUN | HIGH_FOR_EQUIVALENCE_ONLY |
| cast | 2508.13446 v2 | https://github.com/catglossop/CAST; `ec7a214e76167e0f844800ea91d6664863a3d9b1`; MIT | REMOTE_ARTIFACTS_VERIFIED_NOT_RUN | HIGHEST_FOR_TWO_SIDED_ACTION_SELECTIVITY |
| rovla | 2605.19678 v1 | https://github.com/HCPLab-SYSU/RoVLA; `dfa62b55980b052cedf4891f330194f37593b315`; custom non-commercial research-only license | REMOTE_SOURCE_INSPECTED_NOT_RUN | HIGH_FOR_POSITIVE_EQUIVALENCE |
| rss | 2601.04052 v2 | https://github.com/Doo-mon/RSS; `bf8ae69fee6ba97fd3c48335a69079da70f9de87`; Apache-2.0 LICENSE; README incorrectly says MIT | REMOTE_SOURCE_AND_CHECKPOINT_VERIFIED_NOT_RUN | HIGH |
| progal_vla | 2604.09824 v1 | none; `n/a`; n/a | PAPER_ONLY_ARTIFACT_ABSENT | VERY_HIGH_FOR_ENTITY_BINDING |
| guidedvla | 2605.12369 v2 | https://github.com/GuidedVLA/GuidedVLA; `04be059e0d6bd448be5cb45fdbafc775f7eb5e38`; Apache-2.0 plus Gemma/upstream terms | REMOTE_FULL_ARTIFACT_VERIFIED_NOT_RUN | VERY_HIGH_FOR_SUPERVISED_OBJECT_HEAD |
| direct_grounded_point_action_head | 2606.27663 v1 | none released | PAPER_ONLY_ARTIFACT_ABSENT | VERY_HIGH_FOR_TARGET_CONDITIONED_ACTION_INJECTION |
| robosemanticbench | 2606.02277 v1 | https://github.com/ZGC-EmbodyAI/RoboSemanticBench; `b996daf3e2b9d309e3e8ff94d466deba254ebc5c`; MIT plus bundled/upstream terms | REMOTE_FULL_BENCHMARK_VERIFIED_NOT_RUN | PROBLEM_AND_METRIC_PRECEDENT |
| igar_icbench | 2603.06001 v2 | none; `n/a`; n/a | PAPER_AND_PROJECT_ONLY_CODE_PENDING | HIGH_FOR_ATTENTION_RECALIBRATION |
| gpla | 2604.05614 v1 | https://github.com/TheodorWu/GPLA; `6f9eeeb9eca4d50043976cf1e1ac3db8e53c7668`; MIT | REMOTE_PARTIAL_SOURCE_VERIFIED_NOT_RUN | HIGH_FOR_SCALAR_TRAJECTORY_ALIGNMENT |
| anchor_align | 2607.13429 v1 | https://github.com/dwipddalal/Anchor-Align; `ee2919c5b47f037b606e2ef2f9f4deab43966644`; MIT | REMOTE_EVAL_SOURCE_AND_WEIGHTS_VERIFIED_NOT_RUN | HIGH_FOR_LANGUAGE_ACTION_ALIGNMENT |
| when_language_matters | 2606.11906 v1 | none released | PAPER_ONLY_ARTIFACT_ABSENT | MODERATE_TO_HIGH |
| strong_vla | 2604.10055 v2 | none released | PAPER_ONLY_ARTIFACT_ABSENT | MODERATE_FOR_GENERIC_AUGMENTATION |
| vla_grounder | 2607.04517 v1 | none; `n/a`; n/a | PAPER_AND_PROJECT_ONLY_NO_CODE | HIGH_FOR_LEARNED_REWRITING |
| clap | 2607.08974 v1 | none; `n/a`; n/a | PAPER_AND_MINIMAL_PROJECT_ONLY_RELEASE_PENDING | MODERATE_TO_HIGH_FOR_LANGUAGE_ACTION_PLAN |

Full checkpoints, dataset revisions, training/inference inputs, objectives, architectures, evaluated backbones/tasks, local details, and primary URLs are stored per entry in the JSON matrix.

## Closest mechanisms and exact boundary

### Meaning-preserving invariance

LIBERO-Para is evaluation-only and shows the problem at scale. RobustVLA directly constrains actions across semantic-preserving input transformations. RoVLA's Instructional Consistency is data-level augmentation: Qwen3-8B supplies about 15 paraphrases per trajectory and one is uniformly sampled, with no separate IC loss. RSS expands a syntactic neighborhood and pairs it with residual affordance steering. Therefore paraphrase augmentation, positive consistency, dense-neighborhood training, and a clean-realignment curriculum are controls or Priors, not Ours.

### True intent/target change

LIBERO-CF holds plausible scenes while changing feasible instructions, and CAG amplifies the conditional action relative to a visual-only branch. CAST goes further by generating alternative feasible instructions and synthetic atomic action branches for similar observations. RoboSemanticBench separately measures semantic target selection and grasp execution. A viable local candidate must therefore demonstrate a *specific structured response* to real target-swapped demonstrations; generic separation, counterfactual labels, or a benchmark-only decomposition is already occupied.

### Explicit target mediation

ProGAL-VLA binds symbolic subgoals to detector-derived 3D entities with a contrastive objective and conditions control only on the verified goal. GuidedVLA supervises action-query attention on object masks and fuses the specialized branch through a zero-initialized residual. Direct grounded-point injection sends gripper-relative 3D target displacement through DiT AdaLN. These works veto a generic target predictor, mask-aligned attention head, target-conditioned adapter, or zero-init injection claim. Training-only simulator masks remain legal supervision, but they are not by themselves novel.

### Language/action alignment and inference interventions

GPLA ranks language/trajectory pairs and uses preference learning, independently reinforcing the prohibition on the failed scalar action-energy formulation. Anchor-Align predicts motion-direction language from the same observation/action while anchoring the VLM representation. IGAR rebalances language attention at inference; When Does Language Matter gates alignment by step sensitivity; VLA Grounder rewrites instructions through a rollout-trained language policy; CLAP generates a language action description before numeric actions. Ranking, direction labels, attention recalibration, temporal gating, rewriting, and a language-plan prefix are consequently occupied.

## Executability decision

- **Direct local Base/benchmark path:** retained X-VLA + serial LIBERO/LIBERO-Para.
- **Counterfactual benchmark path:** local LIBERO-CF environments pass serial preflight.
- **Strong positive X-VLA Prior:** none is currently released and locally competent. CAG-TF is relevant but failed canonical retention; other released methods use pi0/OpenVLA/GR00T/VLA-Adapter families.
- **Allowed comparison use:** method papers remain novelty Priors; unavailable or incompatible artifacts are reported as comparator limits, never as method falsifications.
- **Resource implication:** RoVLA, GuidedVLA, ProGAL, and the direct grounded-point stack are not candidates for an identity-matched local X-VLA reproduction on this 24.87 GB/16 GB host. Their scientific overlap still controls positioning.

## Candidate gate

Two candidate formulations may now be written, but neither is named as Ours and no Ours outcome may be observed until the discovery, validation, confirmation, and generalization manifests are frozen. The first must explicitly test two-sided binding-to-action causality with a distinction from ProGAL/GuidedVLA/direct injection. The second must use a different causal mechanism -- most plausibly a structured counterfactual action-response objective -- and must distinguish itself from CAST, CAG/RSS, RobustVLA/RoVLA, Anchor-Align, and the failed scalar energy method.

## Epistemic limits

This is a bounded audit of the closest primary work through the stated cutoff, not a universal absence proof. A missing repository, checkpoint, license, or local run is an artifact limitation. It never closes the language problem or empirically falsifies a method. Repository heads and Hub revisions can change after the cutoff; all reported identities are immutable snapshots checked on the audit date.
