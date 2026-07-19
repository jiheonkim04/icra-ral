# Epoch 6 RA-L Venue and Evidence Calibration

Audit date: 2026-07-19 KST
Evidence rule: current official venue pages and primary paper/proceedings/project
sources only. Internal study-design defaults below are not represented as IEEE
requirements.

## Current official rules

| Area | Verified rule | Epoch 6 consequence |
|---|---|---|
| Scope and review | RA-L asks for timely, concise, innovative, significant, and technically sound robotics/automation contributions. Review is double-anonymous and uses at least two independent reviewers. [Author guidance](https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/) and [reviewer guidance](https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-reviewers/) | There is no official universal minimum for seeds, episodes, backbones, or physical trials. The evidence burden follows the exact claim. |
| Length and format | Six pages include figures, tables, appendices, and references. At most two paid overlength pages are permitted at USD 175 each. Initial submission is US Letter, 10 pt, two-column IEEE conference format, without author information. [Author guidance](https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/) | Design for six pages. Do not depend on supplemental prose or figures to make the paper judgeable. Paid overlength is not authorized by this campaign. |
| Multimedia | The paper must be self-contained. At most one video is permitted; the complete multimedia ZIP is at most 50 MB. [Author guidance](https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/) | Keep one anonymous, evidence-linked video only if frozen clips exist; audit every ZIP member and metadata field. |
| Anonymity | Since 2025-01-01, fully sponsored RAS journals use double-anonymous review. Names, affiliations, acknowledgments/funding, lab names/logos, faces and identifying metadata or links must be removed; self-citation must be neutral. ArXiv posting is allowed. [Double-anonymous rules](https://www.ieee-ras.org/publications/rules-for-the-double-anonymous-review-process/) | Audit the PDF, source, video frames, filenames, ZIP metadata, and URLs. A distinctive robot image is used only if scientifically necessary. |
| Generative AI | AI-generated text, figures, images, or code requires disclosure identifying the system, affected sections, and level of use; humans retain responsibility and AI cannot be an author. AI illustrations require tool and prompt in the caption. Original literature sources must be checked. [RAS generative-AI guidance](https://www.ieee-ras.org/publications/guidelines-for-generative-ai-usage/) | Maintain an AI-use ledger from the first paper-writing action. Do not present generated illustration as experimental evidence. |

Two submission-stage instruction collisions remain for human/editor confirmation
before external submission: the AI page places disclosure in acknowledgments
while the anonymous rules prohibit submission-stage acknowledgments, and the
multimedia instructions request contact information in `ReadMe.txt` while the
anonymity rules prohibit identifying supplemental information. Epoch 6 will
not guess at either interpretation.

## Evidence calibration

`NR` means the primary source did not report, or this audit could not verify, a
number. It is not interpreted as zero.

| Paper and status | Contribution and closest comparison | Tasks / policies / repetitions | Evidence, ablations, generalization | Resources and limitations relevant here |
|---|---|---|---|---|
| [BYOVLA](https://arxiv.org/abs/2410.01971), ICRA 2025 ([DOI](https://doi.org/10.1109/ICRA55743.2025.11128017)) | Inference-only visual observation interventions; compares unmodified policies, a no-sensitivity variant, and Grad-CAM where applicable. | OpenVLA 7B and Octo-Base 93M; one WidowX 250S; two tasks; normally 15 physical trials per baseline/condition and 30 for one Octo distraction evaluation; training seeds NR. | Object/background distractions; per-step versus initialization-only sensitivity and observation/action sampling ablations. | RTX 4090; reported OpenVLA/Octo rates about 6/13 Hz, with a roughly 2 s intervention pipeline. One robot and two simple/static tasks; no uncertainty intervals. |
| [iRe-VLA](https://arxiv.org/abs/2501.16664), ICRA 2025 ([DOI](https://doi.org/10.1109/ICRA55743.2025.11127299)) | Alternates online RL and supervised post-training; compares SFT, task-by-task PPO replay, and learning from scratch. | BLIP-2-based 3B policy; 25 MetaWorld tasks with 50 expert trajectories each; five Franka Kitchen tasks; 2,000 real Panda trajectories across five skills; independent physical evaluation denominator and training seeds NR. | Frozen-VLM ablation; simulation and hardware, including unseen objects/vegetables. | Local RL used RTX 4090; supervised stages used 4 A100 GPUs. Sparse reward did not acquire entirely new skill types. Full training is outside this machine envelope. |
| [OpenVLA-OFT](https://www.roboticsproceedings.org/rss21/p017.html), RSS 2025 ([project/artifacts](https://openvla-oft.github.io/)) | LoRA recipe with parallel decoding, action chunking, continuous L1 actions, and optional FiLM; broad contemporary comparator set. | OpenVLA 7B; all 40 LIBERO tasks, 50 episodes per task (2,000 simulation trials); four ALOHA tasks and 56 physical trials per method; repeated training seeds NR. | Recipe-element, VLA-pretraining, and FiLM ablations; one physical task contains 12 ID and 12 OOD trials. | Reported inference is about 16 GB on LIBERO and 18 GB on ALOHA; training used 8x80 GB GPUs for 1–2 days per job. Frozen local inference is plausible; official training is not. |
| [CompoSuite](https://proceedings.mlr.press/v199/mendez22a.html), CoLLAs 2022 | A compositional manipulation benchmark comparing single-task, shared multitask, and modular PPO policies. | 256 simulated combinations from four robots, objects, obstacles, and objectives; 10M steps/task; three seeds; 56/224-task training sets with held-out compositions. | Zero-shot compositions, smaller/restricted settings, and incorrect task descriptors. Simulation-only by design. | Accepted precedent that simulation-only robotics evidence can be appropriate when the benchmark/compositional protocol is the contribution. It does not justify omitting hardware for an ordinary deployment claim. |
| [ActionMap](https://arxiv.org/abs/2606.06904) (2026 preprint, [official repository](https://github.com/showlab/ActionMap)) | Voxel translation/rotation heatmaps, Gaussian targets, and top-k soft-argmax; directly collides with target/action-map methods. | OpenVLA-OFT and a second backbone; LIBERO and real Franka evaluation; exact training seeds/trials require manuscript-table extraction before citation in a paper. | Cross-backbone and real-robot evidence. The public repository exposes the core head but not a full reproducible training/evaluation stack or checkpoints at audit time. | Reported training used 2 H200s for OpenVLA-OFT and 8 H200s for the second backbone. Not a locally runnable Prior and not a novelty opening. |
| [Action Map Policy](https://arxiv.org/abs/2607.10706) (2026 preprint) | Projects 3-D gripper keypoints to multi-view pixel heatmaps with triangulation and equivariant augmentation; directly collides with heatmap/keypoint action decoders. | 41.2M policy; evaluates selected checkpoints on 50 tests, with best-of-20 selection described in the paper; seeds NR. | Multi-view closed-loop manipulation; checkpoint selection itself creates a reporting-risk motivation for an outcome-separated protocol. | Trained on one RTX 5090, but no public code link was found at audit time. Artifact fidelity is inadequate for a comparator reproduction. |
| [vla-eval](https://arxiv.org/abs/2603.13966) (2026 preprint, [official repository](https://github.com/allenai/vla-evaluation-harness)) | A model/benchmark-decoupled evaluation harness addressing dependency, normalization, and termination ambiguity. | Supports 13 simulation benchmarks and six model servers; reports a 2,000-episode LIBERO run and a three-benchmark reproducibility audit; seed structure NR. | Reproduces three published values and exposes hidden normalization and ambiguous termination requirements. | Reports 47x sharded/batched throughput and about 18 minutes for 2,000 LIBERO episodes. It does not itself correct adaptive checkpoint-selection optimism. |
| [VLA-Arena](https://arxiv.org/abs/2512.22539) (2025 preprint, [official repository](https://github.com/PKU-Alignment/VLA-Arena)) | Open data/evaluation framework whose preprocessing preserves selected no-op actions near transitions because deleting every no-op harms replay. | Public datasets and simulator tasks; the fixed neighborhoods 4/8/12/16 are heuristic candidates rather than causal labels. Exact seeds/trials require source-table extraction before paper use. | Demonstrates a direct conflict with wholesale no-op filtering, motivating a predeclared causal replay test. | Useful runnable starting point, but its heuristic preservation rule is a strong simple control and closest Prior, not evidence for the proposed residual mechanism. |

## Calibration decision

A simulation-only paper is permissible but high-risk. Its claim must be
intrinsically measurable in simulation and supported by official closed-loop
success, diverse tasks, untouched generalization, the closest runnable Prior,
the strongest simple control, a mechanism ablation, paired uncertainty,
independent replication when training is involved, and honest latency/RAM/VRAM
reporting. No Epoch 6 result may claim physical safety, hardware robustness,
sim-to-real transfer, or deployment readiness.

The current machine favors an inference/replay-heavy or small-module study.
OpenVLA-OFT is credible as a frozen Base but not as a locally trainable 7B
method. A smaller official policy may be selected only after checkpoint,
license, normalization, and resource verification.
