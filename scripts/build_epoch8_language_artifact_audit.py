#!/usr/bin/env python3
"""Build the immutable Epoch 8 language-grounding literature/artifact audit."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
GENERATED_AT = "2026-07-20T19:39:00+09:00"
CHECKED_ON = "2026-07-20"


def paper(title: str, arxiv: str, version: str, date: str, *, venue: str | None = None) -> dict:
    return {
        "title": title,
        "arxiv_id": arxiv,
        "version": version,
        "version_date": date,
        "venue": venue,
        "url": f"https://arxiv.org/abs/{arxiv}",
    }


def repo(url: str | None, revision: str | None, license_: str | None, **extra: object) -> dict:
    value = {
        "url": url,
        "immutable_revision": revision,
        "license": license_,
    }
    value.update(extra)
    return value


def entry(
    key: str,
    work: dict,
    artifacts: dict,
    training_inputs: list[str],
    inference_inputs: list[str],
    objective: str,
    architecture: str,
    backbones_tasks: list[str],
    intervention_scope: list[str],
    local_state: str,
    local_detail: str,
    collision_level: str,
    collision: str,
    comparator_role: str,
    sources: list[str],
) -> dict:
    return {
        "id": key,
        "paper": work,
        "official_artifacts": artifacts,
        "training_inputs": training_inputs,
        "inference_inputs": inference_inputs,
        "objective": objective,
        "architecture": architecture,
        "evaluated_backbones_and_tasks": backbones_tasks,
        "changes": intervention_scope,
        "local_run_status": {"state": local_state, "detail": local_detail},
        "novelty_collision": {"level": collision_level, "detail": collision},
        "epoch8_comparator_role": comparator_role,
        "primary_sources": sources,
        "checked_on": CHECKED_ON,
    }


entries = [
    entry(
        "x_vla_base",
        paper("X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model", "2509.21305", "retained official release", "2025"),
        {
            "repository": repo("https://github.com/2toinf/X-VLA", "6bc2513f5f1cbec715cc668b414392a6cae5c671", "Apache-2.0"),
            "checkpoint": repo("https://huggingface.co/2toINF/X-VLA-Libero", "129e71460678b7236cee6fc9707f09d9fa0c3590", "Apache-2.0"),
        },
        ["RGB", "language", "proprioception", "LIBERO demonstrations"],
        ["RGB", "language", "proprioception", "permitted history"],
        "Behavior cloning/action-chunk prediction with cross-embodiment soft prompts.",
        "Retained 0.9B X-VLA LIBERO policy; this is the frozen Base, not a novelty comparator.",
        ["LIBERO", "LIBERO-Para local discovery panel"],
        ["target binding", "action prediction"],
        "LOCAL_RETAINED_AND_CUDA_VERIFIED",
        "Source and checkpoint are retained locally. Epoch 7 executed official serial closed loop: 30/30 canonical and 19/30 matched paraphrase discovery episodes.",
        "BASE",
        "Defines the Base whose legal input and action interfaces Epoch 8 must preserve.",
        "BASE",
        ["https://github.com/2toinf/X-VLA", "https://huggingface.co/2toINF/X-VLA-Libero"],
    ),
    entry(
        "libero_para",
        paper("LIBERO-Para: A Diagnostic Benchmark and Metrics for Paraphrase Robustness in VLA Models", "2603.28301", "v1", "2026-03-30"),
        {
            "repository": repo("https://github.com/cau-hai-lab/LIBERO-Para", "5a2198299a6d7a49bdb3cd519c7e92ed803adf5f", "MIT"),
            "dataset": repo("https://huggingface.co/datasets/HAI-Lab/LIBERO-Para", "d306f66f8b441cad1155b21a3f69e440079c81c9", "dataset card/upstream terms"),
            "checkpoint": None,
        },
        ["canonical LIBERO-Goal instructions", "LLM-generated then audited paraphrases"],
        ["standard policy RGB", "paraphrased language", "proprioception"],
        "Evaluation-only controlled action/object/compositional paraphrase taxonomy plus PRIDE metric; no policy-training objective.",
        "Benchmark and evaluator integrations; it changes language only and keeps the simulator task fixed.",
        ["seven VLA configurations", "LIBERO-Goal", "4,092 paraphrases over 10 tasks"],
        ["evaluation"],
        "LOCAL_SERIAL_SMOKE_PASS",
        "Clean local clone at the immutable revision. One 360x360 official Goal environment, reset state, two cameras, and close lifecycle passed under the required serial adaptation.",
        "PROBLEM_DEFINITION",
        "Directly occupies benchmark-only paraphrase diagnosis and establishes object lexical variation as a major bottleneck; it does not solve binding.",
        "PRIMARY_BENCHMARK",
        ["https://arxiv.org/html/2603.28301", "https://github.com/cau-hai-lab/LIBERO-Para", "https://huggingface.co/datasets/HAI-Lab/LIBERO-Para"],
    ),
    entry(
        "libero_cf_cag",
        paper("When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs", "2602.17659", "v2", "2026-07-15"),
        {
            "repository": repo("https://github.com/yuffish/LIBERO-CF", "8460457bfca6e0ef2e856bc104e2c60b023ef2a7", "MIT"),
            "released_unconditioned_checkpoints": [
                repo("https://huggingface.co/yuffish/pi0_libero_unconditioned", "2f206050a9a459945ec7d8710d317c0a657ac23a", "model-card/upstream terms"),
                repo("https://huggingface.co/yuffish/pi05_libero_unconditioned", "7b7b6c2241245a15db80b0b4ff6a623ed329b7cc", "model-card/upstream terms"),
                repo("https://huggingface.co/yuffish/openvla_oft_libero_unconditioned", "f2a6083ffa1964797298065c266337b825a91b92", "model-card/upstream terms"),
            ],
        },
        ["standard demonstrations", "optionally language-dropped VA training"],
        ["RGB", "language for conditioned branch", "proprioception", "second language-unconditioned action branch"],
        "CAG combines an action-conditioned VLA with a language-unconditioned VA branch; TF drops language in the same model, while VA trains a separate visual prior.",
        "Dual-branch inference guidance; no target-binding head or architecture modification.",
        ["pi0", "pi0.5", "OpenVLA-OFT", "paper-only X-VLA appendix", "65 LIBERO-CF tasks", "real Franka"],
        ["inference guidance", "action prediction", "evaluation"],
        "LOCAL_BENCHMARK_PASS_PRIOR_INCOMPETENT",
        "Official serial spatial and OOD environment preflights pass. The release has no X-VLA adapter. The corrected local X-VLA CAG-TF port was action-connected but scored 14/30 canonical and 11/30 paraphrase, so it is not a competent positive Prior.",
        "HIGH",
        "Directly occupies language-conditioned minus visual-prior action guidance and counterfactual evaluation. Epoch 8 must not relabel residual guidance or the local incompetent port as a novel/positive method.",
        "BENCHMARK_AND_NEGATIVE_EXTERNAL_METHOD_COMPARATOR",
        ["https://arxiv.org/html/2602.17659", "https://github.com/yuffish/LIBERO-CF", "https://vla-cf.github.io/"],
    ),
    entry(
        "robustvla",
        paper("On Robustness of Vision-Language-Action Model against Multi-Modal Perturbations", "2510.00037", "v4", "2026-02-24"),
        {"repository": repo("https://github.com/gakakulicc/RobustVLA", "4a80d0d759f465c94e1fd3f3823c498714e830d2", "NO LICENSE FILE DETECTED"), "checkpoint": None},
        ["demonstrations", "semantic-preserving language/visual/environment perturbations", "adversarial action noise"],
        ["standard VLA inputs; no extra inference model"],
        "Consistent actions across semantic-preserving input perturbations plus worst-case flow/action-noise training selected with UCB.",
        "Training-time robust optimization on pi0 and OpenVLA action predictors.",
        ["pi0", "OpenVLA", "LIBERO", "FR5 real robot"],
        ["action prediction"],
        "REMOTE_SOURCE_INSPECTED_NOT_RUN",
        "Official source revision resolved, but no released checkpoint/license identity was found and no local environment was constructed.",
        "HIGH_FOR_EQUIVALENCE_ONLY",
        "Directly occupies action consistency under meaning-preserving lexical/syntactic variation. It does not impose true target-swap selectivity or explicit target binding.",
        "NOVELTY_PRIOR_ONLY",
        ["https://arxiv.org/abs/2510.00037", "https://github.com/gakakulicc/RobustVLA"],
    ),
    entry(
        "cast",
        paper("CAST: Counterfactual Labels Improve Instruction Following in Vision-Language-Action Models", "2508.13446", "v2", "2026-06-08"),
        {
            "repository": repo("https://github.com/catglossop/CAST", "ec7a214e76167e0f844800ea91d6664863a3d9b1", "MIT"),
            "dataset": repo("https://huggingface.co/datasets/catglossop/CAST-dataset", "fe9de38e1e519e649a40db20c9558f94e52887a0", "dataset-card/upstream terms"),
            "checkpoint": repo("https://huggingface.co/catglossop/CounterfactualVLA", "6da195e2dece9732266e5b144d8e5dd0ac50f6fe", "model-card/upstream terms"),
        },
        ["observations", "VLM-generated alternative feasible instructions", "synthetic atomic counterfactual action branches"],
        ["standard policy observation and language"],
        "Augments robot data with counterfactual language and actions so similar observations support different instruction-conditioned behavior.",
        "Data-generation method plus CounterfactualVLA policy, primarily evaluated in navigation with a manipulation extension.",
        ["CounterfactualVLA", "three navigation environments", "manipulation distractor tasks"],
        ["target binding", "action prediction"],
        "REMOTE_ARTIFACTS_VERIFIED_NOT_RUN",
        "Code/data/checkpoint revisions resolve. Reproducing the official generation pipeline requires Gemini/GCP credentials and an atomic policy; it is not credential-free locally.",
        "HIGHEST_FOR_TWO_SIDED_ACTION_SELECTIVITY",
        "Closest precedent to learning different actions for alternative feasible instructions under similar observations. A local candidate must use a materially different causal objective and cannot claim counterfactual label diversity itself.",
        "NOVELTY_PRIOR_ONLY",
        ["https://arxiv.org/html/2508.13446", "https://github.com/catglossop/CAST", "https://cast-vla.github.io/"],
    ),
    entry(
        "rovla",
        paper("RoVLA: Multi-Consistency Constraints for Robust Vision-Language-Action Models", "2605.19678", "v1", "2026-05-19"),
        {"repository": repo("https://github.com/HCPLab-SYSU/RoVLA", "dfa62b55980b052cedf4891f330194f37593b315", "custom non-commercial research-only license"), "checkpoint": None},
        ["RGB views", "language", "proprioception", "actions", "about 15 Qwen3-8B paraphrases per trajectory", "adversarial observation/state perturbations"],
        ["RGB", "language", "robot state", "standard Gaussian action noise for flow integration"],
        "Instructional Consistency uniformly samples paraphrases at the data level and adds no explicit IC loss; EC and OC constrain flow evolution and perturbed observations.",
        "InternVL3.5 semantic encoder plus 32-layer DiT derived from GR00T-N1.6.",
        ["LIBERO", "LIBERO-Plus", "RoboTwin 2.0", "five real tabletop tasks"],
        ["action prediction"],
        "REMOTE_SOURCE_INSPECTED_NOT_RUN",
        "Full source/training/evaluation tree resolves; README recommends eight GPUs, publishes no RoVLA checkpoint, and RoboTwin scripts are not directly executable. It is not compatible with the retained X-VLA path.",
        "HIGH_FOR_POSITIVE_EQUIVALENCE",
        "Directly occupies paraphrase-based positive instructional invariance; it does not enforce minimal true intent/target changes to separate bindings.",
        "NOVELTY_PRIOR_ONLY",
        ["https://arxiv.org/html/2605.19678", "https://github.com/HCPLab-SYSU/RoVLA"],
    ),
    entry(
        "rss",
        paper("Stable Language Guidance for Vision-Language-Action Models", "2601.04052", "v2", "2026-04-20", venue="ACL 2026"),
        {
            "repository": repo("https://github.com/Doo-mon/RSS", "bf8ae69fee6ba97fd3c48335a69079da70f9de87", "Apache-2.0 LICENSE; README incorrectly says MIT"),
            "checkpoint": repo("https://huggingface.co/doomon/RSS_pi05_cfg_libero_caption", "d992b3fe7b203b65aa0d2f2d8a3db61320536e58", "model-card/upstream terms"),
        },
        ["standard demonstrations", "Qwen2.5-VL dense syntactic neighborhoods"],
        ["RGB", "language", "proprioception", "conditioned and unconditioned action/logit forwards"],
        "Monte Carlo Syntactic Integration minimizes expected semantic loss over paraphrases; Residual Affordance Steering subtracts the visual-only action prior.",
        "Training-time language-neighborhood expansion plus inference-time dual-stream residual action steering.",
        ["OpenVLA-family and pi0/pi0.5-style policies", "LIBERO", "LIBERO-Plus", "linguistic corruption suites"],
        ["action prediction", "inference guidance"],
        "REMOTE_SOURCE_AND_CHECKPOINT_VERIFIED_NOT_RUN",
        "Source and checkpoint resolve. No X-VLA adapter is released. The repository's README/license conflict is preserved rather than silently resolved.",
        "HIGH",
        "Occupies dense paraphrase neighborhoods and conditional-minus-affordance action guidance; it does not provide explicit target identities or counterfactual target separation.",
        "NOVELTY_PRIOR_ONLY",
        ["https://arxiv.org/html/2601.04052", "https://github.com/Doo-mon/RSS", "https://huggingface.co/doomon/RSS_pi05_cfg_libero_caption"],
    ),
    entry(
        "progal_vla",
        paper("ProGAL-VLA: Grounded Alignment through Prospective Reasoning in Vision-Language-Action Models", "2604.09824", "v1", "2026-04-10", venue="CVPR Findings 2026"),
        {"repository": repo(None, None, None, project_page="https://nstrndrbi.github.io/ProGAL", project_page_status="404 on 2026-07-20"), "checkpoint": None},
        ["RGB/depth", "language", "proprioception", "teacher-VLM subgoals", "detector-derived 3D entities", "offline symbolic-entity matches"],
        ["OpenVLA observation/state", "Qwen planner once per episode", "YOLO-World detections", "3D entity graph", "verified goal embedding"],
        "Grounding Alignment Contrastive InfoNCE aligns symbolic goals with entity nodes; SACA forms a verified goal bottleneck and entropy-based ambiguity signal.",
        "Hierarchical planner + 3D Grounded State Module + cross-attention verifier + action policy conditioned only on verified goal.",
        ["OpenVLA-7B", "LIBERO-Plus", "Custom Ambiguity Benchmark"],
        ["target binding", "action prediction", "inference guidance"],
        "PAPER_ONLY_ARTIFACT_ABSENT",
        "No official repository/checkpoint was linked or found; the paper's project URL returned 404. This is an artifact limitation, not a scientific falsification.",
        "VERY_HIGH_FOR_ENTITY_BINDING",
        "Directly occupies entity-level contrastive grounding causally conditioning action. A patch/target contrastive bottleneck without stronger distinctions is not novel.",
        "NOVELTY_PRIOR_ONLY",
        ["https://arxiv.org/html/2604.09824", "https://openaccess.thecvf.com/content/CVPR2026F/html/Darabi_ProGAL-VLA_Grounded_Alignment_through_Prospective_Reasoning_in_Vision-Language-Action_Models_CVPRF_2026_paper.html"],
    ),
    entry(
        "guidedvla",
        paper("GuidedVLA: Specifying Task-Relevant Factors via Plug-and-Play Action Attention Specialization", "2605.12369", "v2", "2026-06-01"),
        {
            "repository": repo("https://github.com/GuidedVLA/GuidedVLA", "04be059e0d6bd448be5cb45fdbafc775f7eb5e38", "Apache-2.0 plus Gemma/upstream terms"),
            "checkpoint": repo("https://huggingface.co/ybwowen/pi0-libero-object-depth-skill", "524da8325864da21cd7dfd7c6456d72d84fed090", "model-card/Gemma/upstream terms"),
            "dataset": repo("https://huggingface.co/datasets/ybwowen/libero", "477f79595e4bc55829706fc655419523ae1da3b9", "dataset-card/upstream terms"),
        },
        ["RGB views", "language", "proprioception", "actions", "stage-aware object masks", "skill phases", "Depth Anything features"],
        ["RGB", "language", "proprioception", "frozen depth encoder"],
        "Specializes object, skill, and depth attention heads; object loss concentrates action-query attention on task-object masks.",
        "pi0 action decoder with ControlNet-style factor branch and zero-initialized fusion.",
        ["pi0", "LIBERO-Plus", "RoboTwin 2.0", "real robots"],
        ["target binding", "action prediction"],
        "REMOTE_FULL_ARTIFACT_VERIFIED_NOT_RUN",
        "Contrary to the stale project-page 'coming soon' label, current official source, checkpoint, and dataset all resolve. No X-VLA adapter exists; the extra depth model and pi0 path were not installed locally.",
        "VERY_HIGH_FOR_SUPERVISED_OBJECT_HEAD",
        "Directly occupies training-mask-supervised object attention in the action decoder and zero-init residual fusion. A generic object head or mask-aligned adapter is rejected.",
        "NOVELTY_PRIOR_ONLY",
        ["https://arxiv.org/html/2605.12369", "https://github.com/GuidedVLA/GuidedVLA", "https://huggingface.co/ybwowen/pi0-libero-object-depth-skill"],
    ),
    entry(
        "direct_grounded_point_action_head",
        paper("Direct Action-Head Injection of A Grounded 3D Point Unlocks Spatial and Task Generalization", "2606.27663", "v1", "2026-06-26"),
        {"repository": None, "checkpoint": None},
        ["demonstrations", "oracle segmentation-derived 2D target point in simulation", "depth/camera calibration", "gripper pose"],
        ["RGB/language to an external grounder", "depth", "camera calibration", "gripper state", "grounded target point"],
        "Lifts one target point to 3D, encodes gripper-relative displacement with a two-layer MLP, and adds it to timestep conditioning.",
        "Zero-final-layer MLP injecting 3D displacement through DiT AdaLN; no backbone change.",
        ["GR00T-N1.6", "pi0.5", "LIBERO", "LIBERO-PRO", "real robot with Qwen3-VL-4B grounding"],
        ["target binding", "action prediction", "inference guidance"],
        "PAPER_ONLY_ARTIFACT_ABSENT",
        "No official source/checkpoint was linked or found. Simulation uses oracle target masks; real deployment uses an additional Qwen3-VL-4B grounder.",
        "VERY_HIGH_FOR_TARGET_CONDITIONED_ACTION_INJECTION",
        "Directly occupies lightweight target-conditioned action-head injection, relative geometry, AdaLN fusion, and Base-exact zero initialization.",
        "NOVELTY_PRIOR_ONLY",
        ["https://arxiv.org/html/2606.27663"],
    ),
    entry(
        "robosemanticbench",
        paper("RoboSemanticBench: Diagnosing Semantic Grounding in Action Prediction for VLA Models", "2606.02277", "v1", "2026-06-01"),
        {
            "repository": repo("https://github.com/ZGC-EmbodyAI/RoboSemanticBench", "b996daf3e2b9d309e3e8ff94d466deba254ebc5c", "MIT plus bundled/upstream terms"),
            "datasets": [
                repo("https://huggingface.co/datasets/VLyb/RSB-Math", "99b90fea13d308a1b1d26929340a2ceecf594823", "dataset-card/upstream terms"),
                repo("https://huggingface.co/datasets/VLyb/RSB-Math-10blocks", "e83bfe76555fb22c7e344ebf2d20ee40b564e17c", "dataset-card/upstream terms"),
            ],
            "checkpoint": None,
        },
        ["multiple-choice math/knowledge questions", "answer blocks", "RoboTwin demonstrations"],
        ["policy RGB", "question/instruction", "proprioception"],
        "Evaluation benchmark separates ability to grasp a candidate from semantic selection of the correct physical answer target.",
        "RoboTwin-derived benchmark/data/policy harness; not a mitigation method.",
        ["GO1", "OpenVLA", "DexVLA", "TinyVLA", "PD-VLA", "pi0/pi0.5", "GR00T/QwenGR00T", "RSB 4/10-choice suites"],
        ["evaluation"],
        "REMOTE_FULL_BENCHMARK_VERIFIED_NOT_RUN",
        "Source and two public dataset revisions resolve. The bimanual RoboTwin stack and checkpoints were not installed; only part of the benchmark data is currently pre-collected.",
        "PROBLEM_AND_METRIC_PRECEDENT",
        "Occupies decomposed semantic target selection versus motor grasp success; Epoch 8 cannot claim that decomposition as new.",
        "NOVELTY_AND_GENERALIZATION_BENCHMARK",
        ["https://arxiv.org/html/2606.02277", "https://github.com/ZGC-EmbodyAI/RoboSemanticBench"],
    ),
    entry(
        "igar_icbench",
        paper("Restoring Linguistic Grounding in VLA Models via Train-Free Attention Recalibration", "2603.06001", "v2", "2026-07-02", venue="ECCV 2026"),
        {"repository": repo(None, None, None, project_page="https://ray-nh.github.io/igar/", project_status="Code coming soon on 2026-07-20"), "checkpoint": None},
        ["none; train-free"],
        ["standard VLA inputs", "attention sink statistics inside the forward pass"],
        "Detects sink tokens/imbalanced grounding heads and redistributes attention toward instruction tokens at inference.",
        "Train-free transformer attention intervention plus 30-task ICBench contradiction benchmark.",
        ["pi0", "pi0.5", "OpenVLA-OFT", "LIBERO contradictions", "real Franka"],
        ["inference guidance", "evaluation"],
        "PAPER_AND_PROJECT_ONLY_CODE_PENDING",
        "Official project explicitly says code coming soon; no executable source/checkpoint was available.",
        "HIGH_FOR_ATTENTION_RECALIBRATION",
        "Occupies train-free language-token attention reweighting and contradiction suppression, not learned target binding.",
        "NOVELTY_PRIOR_ONLY",
        ["https://arxiv.org/html/2603.06001", "https://ray-nh.github.io/igar/"],
    ),
    entry(
        "gpla",
        paper("Grounding Hierarchical Vision-Language-Action Models Through Explicit Language-Action Alignment", "2604.05614", "v1", "2026-04-07", venue="CVPR Findings 2026"),
        {"repository": repo("https://github.com/TheodorWu/GPLA", "6f9eeeb9eca4d50043976cf1e1ac3db8e53c7668", "MIT"), "checkpoint": None},
        ["LanguageTable observations", "generated language/action pairs", "contrastive grounding labels", "preference pairs"],
        ["hierarchical VLA observation/instruction"],
        "Trains a contrastive language-trajectory grounding model, ranks candidate language-action outputs, then uses offline preference learning.",
        "Hierarchical VLA with a separate action-conditioned grounding scorer and preference loop.",
        ["LanguageTable", "hierarchical VLA"],
        ["action prediction", "offline scoring"],
        "REMOTE_PARTIAL_SOURCE_VERIFIED_NOT_RUN",
        "MIT source resolves, but README contains no setup/reproduction path and no checkpoint is released; not a turnkey comparator.",
        "HIGH_FOR_SCALAR_TRAJECTORY_ALIGNMENT",
        "Directly reinforces the prohibition on renaming Candidate A's scalar action-energy/ranking formulation; it does not supply explicit scene target binding.",
        "NOVELTY_PRIOR_ONLY",
        ["https://arxiv.org/html/2604.05614", "https://github.com/TheodorWu/GPLA", "https://openaccess.thecvf.com/content/CVPR2026F/html/Wulff_Grounding_Hierarchical_Vision-Language-Action_Models_Through_Explicit_Language-Action_Alignment_CVPRF_2026_paper.html"],
    ),
    entry(
        "anchor_align",
        paper("Generalizable VLA Finetuning via Representation Anchoring and Language-Action Alignment", "2607.13429", "v1", "2026-07-15"),
        {
            "repository": repo("https://github.com/dwipddalal/Anchor-Align", "ee2919c5b47f037b606e2ef2f9f4deab43966644", "MIT"),
            "checkpoint": repo("https://huggingface.co/Dwipz/Anchor-Align", "8897bfd48a90180e6965629a43ff6a7736495430", "model-card/upstream terms"),
            "training_code_released": False,
        },
        ["robot demonstrations", "frozen VLM teacher features", "motion-direction labels derived from actions"],
        ["RGB", "language", "proprioception"],
        "Anchors student non-action representations to a frozen VLM and predicts discrete motion direction jointly with continuous actions.",
        "VLA-Adapter-family action head plus representation distillation and direction-alignment projection.",
        ["two VLA architectures", "LIBERO", "LIBERO-PRO", "LIBERO-Plus", "CALVIN", "xArm7"],
        ["action prediction"],
        "REMOTE_EVAL_SOURCE_AND_WEIGHTS_VERIFIED_NOT_RUN",
        "Current inference/evaluation code and four checkpoints resolve. Training code/loss implementation is explicitly deferred; no X-VLA adapter exists.",
        "HIGH_FOR_LANGUAGE_ACTION_ALIGNMENT",
        "Occupies representation preservation and absolute motion-direction language/action alignment. It does not use equivalence classes, target-swap pairs, or an explicit target mediator.",
        "NOVELTY_PRIOR_ONLY",
        ["https://arxiv.org/html/2607.13429", "https://github.com/dwipddalal/Anchor-Align", "https://huggingface.co/Dwipz/Anchor-Align"],
    ),
    entry(
        "when_language_matters",
        paper("When Does Language Matter? Multilingual Instructions Reveal Step-wise Language Sensitivity in Vision-Language-Action Models", "2606.11906", "v1", "2026-06-10"),
        {"repository": None, "checkpoint": None},
        ["multilingual LIBERO instructions", "step-wise gradient sensitivity"],
        ["standard VLA inputs", "retrieved reference representations at language-critical steps"],
        "Identifies language-critical steps and selectively aligns representations at inference.",
        "Step-wise sensitivity diagnostic and temporal inference intervention.",
        ["OpenVLA/OFT-style VLAs", "LIBERO in ten languages"],
        ["inference guidance", "evaluation"],
        "PAPER_ONLY_ARTIFACT_ABSENT",
        "No official repository/checkpoint was linked or found.",
        "MODERATE_TO_HIGH",
        "Occupies temporal gating of language influence; a step gate alone is not a distinct Epoch 8 mechanism.",
        "NOVELTY_PRIOR_ONLY",
        ["https://arxiv.org/html/2606.11906"],
    ),
    entry(
        "strong_vla",
        paper("STRONG-VLA: Decoupled Robustness Learning for Vision-Language-Action Models under Multimodal Perturbations", "2604.10055", "v2", "2026-04-14"),
        {"repository": None, "checkpoint": None},
        ["clean demonstrations", "curriculum of 28 visual/text perturbations"],
        ["standard VLA inputs"],
        "Separates perturbation-robustness acquisition from later clean-task realignment.",
        "Two-stage fine-tuning recipe without a target-specific mediator.",
        ["OpenVLA", "OpenVLA-OFT", "pi0", "LIBERO", "AIRBOT"],
        ["action prediction"],
        "PAPER_ONLY_ARTIFACT_ABSENT",
        "No official code/checkpoint was linked or found.",
        "MODERATE_FOR_GENERIC_AUGMENTATION",
        "Occupies curriculum perturbation training plus canonical realignment; generic paraphrase augmentation/retention cannot be Ours.",
        "NOVELTY_PRIOR_ONLY",
        ["https://arxiv.org/html/2604.10055"],
    ),
    entry(
        "vla_grounder",
        paper("VLA Grounder: Language-Conditioning Space Optimization for Black-Box VLA Models", "2607.04517", "v1", "2026-07-05"),
        {"repository": repo(None, None, None, project_page="https://tttonyalpha.github.io/vla_grounder/"), "checkpoint": None},
        ["scene image", "human instruction", "failure-derived command prior", "sparse rollout reward", "GRPO"],
        ["scene image", "human instruction", "trained command-generating VLM", "frozen downstream VLA"],
        "Optimizes a scene-conditioned language policy that rewrites human intent into VLA-compatible grounded commands using rollout reward.",
        "Upstream language-conditioning policy; downstream pi0/OpenVLA remains frozen.",
        ["pi0", "OpenVLA", "VL-Think", "RL4VLA"],
        ["target binding", "inference guidance"],
        "PAPER_AND_PROJECT_ONLY_NO_CODE",
        "Project page and paper resolve, but no source/checkpoint is linked.",
        "HIGH_FOR_LEARNED_REWRITING",
        "Directly occupies learned grounded command rewriting. Prompt/canonicalization/RL language-space optimization is not a distinct Epoch 8 contribution.",
        "NOVELTY_PRIOR_ONLY",
        ["https://arxiv.org/html/2607.04517", "https://tttonyalpha.github.io/vla_grounder/"],
    ),
    entry(
        "clap",
        paper("CLAP: Direct VLM-to-VLA Adaptation via Language-Action Grounding", "2607.08974", "v1", "2026-07-09"),
        {"repository": repo(None, None, None, project_page="https://omron-sinicx.github.io/clap/"), "checkpoint": None, "release_statement": "paper says weights will be released"},
        ["robot demonstrations", "natural-language action descriptions prepended to numeric action sequences"],
        ["RGB", "instruction", "autoregressive language plan then numeric action tokens"],
        "Causally generates a natural-language action description before numeric action tokens to keep action prediction near the VLM language distribution.",
        "Unmodified VLM backbone converted to an autoregressive VLA through output formatting and single-epoch fine-tuning.",
        ["0.8B/2B/4B planned family", "LIBERO", "LIBERO-PRO"],
        ["action prediction"],
        "PAPER_AND_MINIMAL_PROJECT_ONLY_RELEASE_PENDING",
        "The project page currently exposes no code/weights and the paper promises a future open-weight release.",
        "MODERATE_TO_HIGH_FOR_LANGUAGE_ACTION_PLAN",
        "Occupies language-plan-before-action causal alignment, but not explicit target binding or paired equivalence/selectivity.",
        "NOVELTY_PRIOR_ONLY",
        ["https://arxiv.org/html/2607.08974", "https://omron-sinicx.github.io/clap/"],
    ),
]


payload = {
    "schema_version": "epoch8.language_artifact_matrix.v1",
    "generated_at": GENERATED_AT,
    "audit_cutoff": "2026-07-20 Asia/Seoul",
    "scope": "Current primary papers and official artifacts closest to paraphrase robustness, counterfactual target selectivity, object grounding, language-action alignment, and action-chunk adaptation.",
    "epistemic_rules": [
        "Paper-level overlap is positioning evidence, not empirical falsification of a local mechanism.",
        "Absent, incompatible, or unrun code is an artifact/comparator limitation.",
        "Only official paper, project, repository, and model/dataset artifacts are used for scientific claims.",
        "Remote availability is never relabeled as a local run.",
    ],
    "host_constraint": {
        "physical_ram_bytes": 24871014400,
        "gpu": "NVIDIA GeForce RTX 5080",
        "vram_mib": 16303,
        "policy": "one live simulator environment and one resident backbone; serial; no CPU/disk model offload; no swap/pagefile growth caused by experiment",
    },
    "entries": entries,
    "summary": {
        "entry_count": len(entries),
        "official_repository_urls_resolved": sum(
            1
            for x in entries
            if (x["official_artifacts"].get("repository") or {}).get("url")
        ),
        "locally_executed_or_smoked": ["x_vla_base", "libero_para", "libero_cf_cag"],
        "artifact_absence_does_not_close_science": True,
        "novelty_vetoes": [
            "scalar action-energy or language-trajectory ranking",
            "positive-only paraphrase/action consistency",
            "generic paraphrase augmentation or curriculum",
            "conditioned-minus-unconditioned residual action guidance",
            "generic supervised object-attention head",
            "external grounded point injected into an action head",
            "entity-level contrastive verified-goal bottleneck without a material distinction",
            "learned instruction rewriting/canonicalization",
            "absolute motion-direction language-action alignment",
        ],
        "remaining_testable_boundary": "A paired, two-sided causal intervention must jointly preserve action behavior within true semantic equivalence classes and enforce the correct signed/action-structured response under genuine target swaps, using real demonstration supervision and legal inference inputs. It must beat one-sided paraphrase training and cannot reduce to a target head, scalar score, residual guidance, rewriting, or synthetic CAST relabeling.",
    },
}


REPORTS.mkdir(parents=True, exist_ok=True)
matrix_path = REPORTS / "epoch8_language_artifact_matrix.json"
matrix_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fmt_artifact(item: dict | None) -> str:
    if not item:
        return "none released"
    url = item.get("url") or "none"
    revision = item.get("immutable_revision") or "n/a"
    license_ = item.get("license") or "n/a"
    return f"{url}; `{revision}`; {license_}"


rows = []
for item in entries:
    artifacts = item["official_artifacts"]
    rows.append(
        "| {id} | {paper} | {repo} | {local} | {collision} |".format(
            id=item["id"],
            paper=f"{item['paper']['arxiv_id']} {item['paper']['version']}",
            repo=fmt_artifact(artifacts.get("repository")),
            local=item["local_run_status"]["state"],
            collision=item["novelty_collision"]["level"],
        )
    )

md = f"""# Epoch 8 Language-Grounding Primary-Literature and Artifact Audit

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
{chr(10).join(rows)}

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
"""

(REPORTS / "epoch8_language_primary_overlap_audit.md").write_text(md, encoding="utf-8")

print(json.dumps({"matrix": str(matrix_path), "entries": len(entries), "markdown": str(REPORTS / "epoch8_language_primary_overlap_audit.md")}, indent=2))
