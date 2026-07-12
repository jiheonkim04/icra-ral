# Epoch 1 Candidate Discovery

Date: 2026-07-12 KST

Campaign branch: `codex/autonomous-until-ral-evidence-ready`

Epoch 1 search shift:

- away from post-hoc action damping, scalar trust heads, candidate ranking, and unavailable human intervention chunks;
- toward locally testable deployment robustness, representation changes, simulator counterfactual signals, and action-generation mechanisms.

Latest primary sources checked:

- DEFLECT, delay-robust VLA execution: https://arxiv.org/abs/2605.19294
- TIC-VLA, latency-consistent slow/fast VLA control: https://arxiv.org/html/2602.02459v2
- LIBERO-Occ, occlusion and viewpoint imagination: https://arxiv.org/html/2606.10862v2
- visual corruption restoration transformer for VLAs: https://arxiv.org/html/2602.01158v1
- RobustVLA robustness-aware post-training: https://arxiv.org/abs/2511.01331
- Counterfactual Action Guidance / LIBERO-CF: https://arxiv.org/html/2602.17659v1
- PAPO-VLA planning-aware policy optimization: https://arxiv.org/html/2605.19580v1
- EventVLA visual evidence memory: https://arxiv.org/html/2606.20092v1
- KEMO keyframe memory: https://arxiv.org/html/2606.23589v1
- LaMem-VLA dual latent memory: https://arxiv.org/html/2607.07608v1

## Candidate 1: DICD-VLA

Name: `Delay-Indexed Chunk Distillation for VLA Policies`

Research problem: VLA policies emit action chunks from an observation that may be stale by the time the low-level robot executes the action. The local official stack is fast, but deployment delay is a realistic second condition and can be simulated exactly.

Hidden assumption in current work: the first action in a predicted chunk is the correct command even when execution is delayed.

Technical novelty: learn a delay-indexed chunk adapter that maps the frozen policy's current action chunk, recent executed actions, proprioceptive timing features, and declared delay to the action that should be executed at the delayed physical time. This is not a scalar hold, damping, or candidate ranking rule; it changes which element/residual of the action-generation chunk is deployed under known delay.

Mathematical formulation: for context `x_t`, frozen chunk `A_t = [a_t^0, ..., a_t^{H-1}]`, recent executed actions `h_t`, and delay `d`, train `g_theta(A_t, h_t, d)` to approximate the future expert/control target `a_{t+d}` while regularizing clean-delay `d=0` to preserve the first chunk element. The supervised loss is:

`L(theta) = ||g_theta(A_t, h_t, d) - a_{t+d}||_1 + lambda ||g_theta(A_t, h_t, 0) - a_t^0||_2^2`

Training supervision: simulator/demo action sequences and frozen-policy chunks under known artificial delay. No human intervention chunks.

Data source: existing LIBERO demonstrations and/or exact-state frozen-policy traces generated locally.

Policy component changed: action chunk deployment head / execution-index adapter.

Inference algorithm: run frozen SmolVLA normally, pass its action chunk and recent executed-action history to the delay-indexed adapter, execute one calibrated action, and update history.

Closest five papers: DEFLECT, TIC-VLA, RobustVLA, PAPO-VLA, OpenVLA-OFT.

Exact difference: DEFLECT uses flow-matching likelihood-estimated counterfactual tuning for stale/fresh pairs; DICD uses an explicit delay index and supervised chunk-index distillation that can be tested on SmolVLA without flow likelihoods. TIC-VLA targets slow reasoning/fast control architecture; DICD is a small deployment adapter for existing chunk emitters.

Direct baseline: frozen SmolVLA under the same artificial action delay.

Simple killer baseline: execute chunk index `d` directly without learning.

Main ablation: learned adapter without executed-action history.

Prototype tasks: `libero_spatial/task_4` and `libero_10/task_4`, because they are documented closed-loop weak slices for SmolVLA.

Expected compute: small offline trace generation and lightweight adapter training; Stage A closed-loop `2` tasks x fixed reset identities x policies.

Expected closed-loop mechanism: under a declared delay, the full adapter changes executed actions relative to frozen and relative to simple chunk-indexing, while retaining clean `d=0` behavior.

Novelty risk: high due DEFLECT and TIC-VLA, but not near-exact if implemented as explicit delay-indexed chunk distillation rather than flow likelihood post-training or slow/fast architecture.

Implementation risk: moderate. The local runner must expose or reconstruct action chunks rather than only one postprocessed action.

Likely failure mode: SmolVLA action chunks may not contain useful delayed targets, so the simple chunk-index baseline may match or beat the adapter.

Scale-up route: if Stage A passes, expand paired delayed rollouts and then test clean retention and quantized OpenVLA-OFT INT4 with disclosed quantization.

Second-backbone route: implement the same delay-index adapter for OpenVLA-OFT INT4 action chunks or, if chunk access is incompatible, run the exact artificial-delay condition with an equivalent output-index adapter.

Second-condition route: controlled action delay is already the primary claim condition; clean retention is the second condition.

## Candidate 2: SCOP-VLA

Name: `Semantic Counterfactual Object-Persistence VLA`

Research problem: VLA actions fail when task-relevant objects become hidden or partially observable.

Hidden assumption in current work: the current image contains enough target evidence for every action.

Technical novelty: maintain a language-conditioned object-persistence memory from previously visible target evidence and use it as compact policy-side context under occlusion.

Mathematical formulation: learn a target memory state `m_t = f_theta(m_{t-1}, o_t, l)` and action adapter `pi(a_t | o_t, l, m_t)`, supervised by simulator-generated visibility changes and clean actions.

Training supervision: simulator-generated occlusion/camera-corruption traces with target visibility labels available only for training diagnostics.

Data source: LIBERO exact-state rollouts and controlled visual occlusion.

Policy component changed: visual-temporal representation.

Inference algorithm: update compact target memory from observations and instruction, then condition action generation on memory.

Closest five papers: LIBERO-Occ, EventVLA, KEMO, LaMem-VLA, Multi-Scale Embodied Memory.

Direct baseline: frozen SmolVLA under controlled occlusion.

Simple killer baseline: feed the last unoccluded image or a fixed frame stack.

Main ablation: memory without language-conditioned target selection.

Expected compute: moderate, but visual adapter integration into SmolVLA may be nontrivial.

Reviewer attack: high near-prior-art risk. LIBERO-Occ already targets VLA occlusion using generated complementary views, and EventVLA/KEMO/LaMem occupy memory-augmented VLA under occlusion/partial observability. This is not selected for cycle 1.

## Candidate 3: CLC-VLA

Name: `Counterfactual Language-Control Consistency VLA`

Research problem: VLA policies can follow visual shortcuts over language intent.

Hidden assumption in current work: the language-conditioned action is sufficiently different from a vision-only/default-scene action.

Technical novelty: train a language-control consistency objective so action changes under instruction counterfactuals are concentrated on semantically relevant degrees of freedom.

Mathematical formulation: for same observation `o` and counterfactual instructions `l` and `l'`, enforce a margin between `pi(a|o,l)` and `pi(a|o,l')` only on action components predicted to be relevant to the changed semantic relation.

Training supervision: counterfactual instructions generated from LIBERO BDDL/task templates and clean trajectories.

Data source: existing LIBERO tasks and language templates.

Policy component changed: language-action coupling.

Inference algorithm: standard single-branch VLA; no test-time candidate ranking.

Closest five papers: CAG/LIBERO-CF, CAST, PAPO-VLA, ACoT-VLA, action-chain/semantic-kinematic VLA work.

Direct baseline: CAG-style language guidance proxy when locally feasible.

Simple killer baseline: instruction dropout or negative-prompt action differencing.

Main ablation: consistency loss without semantic action-component masking.

Reviewer attack: high overlap risk with CAG and CAST. It also requires robust counterfactual instruction labels; poor labels would repeat the CensorCredit supervision collapse pattern. This is not selected for cycle 1.

## Reviewer B Attack

Candidate 1 has the best time-to-evidence profile but must survive DEFLECT/TIC-VLA overlap. It is allowed to proceed only if the proposal freezes an exact distinction: explicit delay-indexed chunk distillation for locally available action chunks, not flow-matching likelihood preference tuning, not slow/fast VLA architecture, and not adaptive chunk length alone. The strongest simple baseline is direct chunk-index execution.

Candidate 2 is not selected because the current literature already contains occlusion-specific viewpoint imagination and several memory-augmented VLA systems. A first local implementation would likely be a weaker version of those methods.

Candidate 3 is not selected because language counterfactual failures, dual-branch guidance, and counterfactual label augmentation are already directly occupied. It also risks requiring synthetic labels whose validity would be difficult to prove before training.

## Researcher A Selection

Selected method for cycle 1:

`DICD-VLA`

Reason:

- it changes the deployment condition and action-generation mechanism rather than rescuing post-hoc correction;
- it uses locally feasible simulator traces and known artificial delay rather than unavailable human correction chunks;
- it has a decisive cheap baseline, direct chunk-index execution;
- if the learned adapter cannot beat direct chunk-indexing, the method can be killed quickly with real closed-loop evidence.

Next required artifacts:

- child branch `codex/auto-method-20260712-01-dicd-vla`;
- independent `researcher_proposal.md`;
- proposal content hash;
- independent `reviewer_attack.md`;
- one rebuttal;
- frozen preregistration/protocol;
- implementation and mechanism smoke.
