# Epoch 2 Cycle 3 Candidate Generation

Date: 2026-07-12 KST

Governance: `reports/current_research_governance.md`

Cycle 3 pivots away from `PTC-VLA` and `SACF-VLA`. PTC was a direct transition-conditioned action generator and SACF was a same-scene semantic-prefix action generator. Both were active and both were clearly worse than frozen SmolVLA in Stage A. Cycle 3 must therefore avoid another direct small action head, another demonstration-only prefix, post-hoc residual correction, candidate ranking, verification, generic progress/value/confidence, barriers, filters, damping, and generic DPO.

Latest primary-source pressure points:

- SmolVLA uses a flow-matching action expert that starts from Gaussian noise and denoises action chunks conditioned on visual, language, and state features: https://arxiv.org/pdf/2506.01844
- Guided Action Flow keeps SmolVLA frozen and guides its reverse flow with a learned action-chunk critic: https://arxiv.org/html/2607.02092v1
- CF-VLA changes flow action generation by constructing a coarse action-aware starting point before refinement: https://arxiv.org/html/2604.24622v1
- ACG uses training-free action-coherence guidance for flow-based VLAs: https://arxiv.org/html/2510.22201v2
- VLS steers frozen diffusion/flow robot policies with VLM-generated differentiable rewards at inference: https://arxiv.org/html/2602.03973v1
- VOTE uses trajectory ensemble voting for VLA optimization: https://arxiv.org/html/2507.05116v3
- VLA-OPD uses expert teacher supervision on student on-policy trajectories: https://arxiv.org/html/2603.26666v1
- HiPolicy and AAC-style work crowd the adaptive chunking/frequency route: https://arxiv.org/html/2604.06067v1

Local feasibility facts:

- The official SmolVLA runner accepts an optional `noise` tensor in `policy.select_action(batch, noise=noise)`.
- Prior local ECHO evidence killed online candidate ranking/headroom for action alternatives, not the latent sampling prior itself.
- Stage A can run a small train/eval split of fixed noise identities without downloads, full-model fine-tuning, OpenVLA training, or keeping two large backbones resident.
- The selected route must record a second-backbone risk honestly because Quantized OpenVLA-OFT INT4 uses an L1 action head rather than SmolVLA-style flow sampling.

## Candidate 1: OCFN-VLA

Name: `OCFN-VLA`, Outcome-Conditioned Flow-Noise Prior VLA

Hidden assumption: some SmolVLA closed-loop failures are caused by an uncalibrated latent sampling prior rather than missing action competence. Because the frozen flow policy maps different initial noise tensors to different action chunks, a task-conditioned prior over initial flow noise can move sampling toward more successful modes while preserving the frozen VLA denoising field.

Precise novelty: learn a small closed-loop outcome-conditioned prior over SmolVLA's initial flow noise. Unlike Q-guided or VLM-guided methods, OCFN does not inject gradients, compute reward functions, score action proposals, or rank candidate actions at inference. It changes the initial distribution of the frozen reverse-flow sampler and then executes the single denoised action emitted by the frozen policy.

Equations:

- fixed noise bank: `Z = {z_j}_{j=1..K}`, `z_j in R^{H x D}`
- frozen flow rollout action: `A_t(j) = F_theta(o_t, l, z_j)`
- closed-loop train label: `y_{task,reset,j} in {0,1}`
- task-conditioned prior score: `s_phi(task,j) = w_task^T e_j`
- prior objective: `L = - sum_{task,reset,j} [y log softmax_j s_phi + (1-y) log(1-softmax_j s_phi)] + lambda KL(q_phi(.|task) || uniform)`
- Stage A deterministic deployment: `j*(task) = argmax_j q_phi(j | task)`, `a_t = first(F_theta(o_t, l, z_{j*(task)}))`

Representation: fixed flow-noise tensors, task key, suite family, and closed-loop success labels from training reset identities. No simulator state or reward is used at inference.

Objective: learn a task-conditioned latent sampling prior from closed-loop outcomes; not behavior cloning, not action residual correction, not value/progress prediction, and not online candidate ranking.

Supervision: self-generated official SmolVLA-LIBERO rollouts on train identities, labeled only by task success. The frozen VLA is never fine-tuned.

Inference: one frozen SmolVLA call per control step with a preregistered noise tensor supplied to `select_action`. The output action is not postprocessed beyond the normal official postprocessor.

Required data: no downloads. Use standard official LIBERO tasks and reset identities already supported by the WSL runner.

Closest five papers:

| Paper | Overlap | Difference |
| --- | --- | --- |
| Guided Action Flow / QGF | frozen SmolVLA flow sampler plus learned success-related signal | OCFN changes only the initial noise prior; no critic gradients through the sampler and no action-value head at inference |
| CF-VLA | modifies the start of VLA action generation | OCFN does not train a coarse action generator or local refinement head; it learns a task-conditioned prior over existing noise identities from closed-loop outcomes |
| VLS | inference-time steering of frozen diffusion/flow policies | OCFN has no VLM-generated constraints, no online reward gradients, no particle resampling, and no stage switching |
| ACG | flow-based VLA guidance without backbone fine-tuning | OCFN uses outcome-supervised latent prior selection, not action-coherence smoothing or gradient-free coherence guidance |
| VOTE | uses multiple trajectories from a VLA | OCFN does not vote or rank trajectories online; the prior is fixed before held-out rollout |

Exact overlap matrix:

| Axis | OCFN-VLA | Closest overlap? |
| --- | --- | --- |
| problem | latent sampling prior miscalibration in frozen flow VLA deployment | partial with QGF/CF-VLA/VLS |
| representation | fixed flow-noise bank plus task-conditioned prior | distinct from action chunks, rewards, values, or language factors |
| supervision | closed-loop outcome labels for noise identities on train resets | distinct from critic gradients, VLM rewards, BC, or teacher actions |
| objective | success-conditioned noise-prior likelihood with uniform regularization | partial with policy-distribution learning, not value learning |
| policy component | frozen flow sampler initial condition | distinct from action head, residual, prefix, or barrier |
| inference | one frozen denoising call with preselected noise tensor | distinct from online action candidate ranking or verifier filtering |
| data | locally generated official SmolVLA rollouts | distinct from large-scale pretraining or human correction data |
| claim | closed-loop outcome-conditioned latent prior can improve frozen flow policy sampling | narrower than generic VLA guidance or distillation |

Direct baseline: `zero_noise_smolvla`, a deterministic fixed-zero flow start.

Simple killer baseline: `global_success_noise_prior`, the single best noise identity on all training tasks combined.

Ablation: `task_shuffled_noise_prior`, a task-conditioned prior trained with task labels shuffled before fitting.

Implementation plan: add `tca_map/smolvla/ocfn_vla.py`, `scripts/run_ocfn_vla_prototype.py`, tests for deterministic noise bank construction/no privileged inference/no test leakage, a synthetic mechanism smoke, a train-noise-bank acquisition pass, and Stage A evaluation.

Prototype tasks: `libero_spatial/task_4` and `libero_10/task_4`. Training identities: `20260711`, `20260712`. Held-out Stage A identities: `20260713` through `20260717`.

Second-backbone path: if OCFN reaches Stage B/GO, first test whether another locally available flow-action VLA exposes the same initial-noise interface. Quantized OpenVLA-OFT INT4 must still be used as a same-task non-flow comparator, but the paper claim may need to narrow to flow-matching VLAs unless an OpenVLA-compatible analogue is found.

Second-condition path: controlled action-sampling robustness or perturbation of initial noise distribution on held-out LIBERO tasks; not occlusion repair or action fault correction.

Compute estimate: synthetic under 1 minute; noise-bank train acquisition about `2 tasks * 2 train identities * K=4` episodes; Stage A about `5 variants * 2 tasks * 5 identities = 50` held-out episodes.

Failure risk: high. If all noise identities behave equivalently, the method is trivial. If global noise matches task-conditioned noise, the task prior is not useful. If frozen random/default is already best, OCFN is killed.

## Candidate 2: DARF-VLA

Name: `DARF-VLA`, Demonstration-Anchored Retrieval Flow VLA

Hidden assumption: frozen flow policies fail because Gaussian initial noise ignores the local action manifold available in nearby demonstrations; retrieving a demonstration action chunk as the initial flow point may keep denoising inside feasible action support.

Precise novelty: build a lightweight retrieval memory over demonstration state/action chunks and feed the retrieved action chunk as the flow initial condition, allowing SmolVLA to refine the retrieved chunk rather than replacing the policy action.

Equations:

- memory: `M = {(r_i, A_i)}`
- query representation: `r_t = h(s_t, l, suite)`
- retrieved chunk: `A_ret = A_argmin_i ||r_t - r_i||`
- flow action: `A_t = F_theta(o_t, l, alpha A_ret + (1-alpha) z_0)`

Representation: robot state, task text hash, suite code, and demonstration action chunk memory.

Objective: metric learning or nearest-neighbor retrieval into action-chunk support.

Supervision: official LIBERO HDF5 demonstrations.

Inference: one frozen SmolVLA call initialized from retrieved action support; no online candidate scoring.

Required data: local official LIBERO HDF5 demonstrations.

Closest five papers:

| Paper | Overlap | Difference |
| --- | --- | --- |
| CF-VLA | action-aware starting point for flow generation | DARF uses nonparametric local retrieval, not a trained coarse generator plus refinement architecture |
| FAST / FAST+ | action sequence representation | DARF does not tokenize or train an autoregressive VLA |
| VQ-VLA | vector-quantized action tokens | DARF retrieves continuous chunks without scaling a tokenizer |
| VOTE | multiple VLA trajectories | DARF uses one retrieved prior, not ensemble voting |
| ActionMap / retrieval-style action priors | support-aware action generation | DARF keeps frozen SmolVLA refinement rather than a learned heatmap/action head |

Exact overlap matrix:

| Axis | DARF-VLA | Closest overlap? |
| --- | --- | --- |
| problem | Gaussian flow start ignores local action support | partial with CF-VLA |
| representation | retrieval memory of demo chunks | partial with action-token/retrieval methods |
| supervision | demonstrations only | common |
| objective | retrieval metric and frozen flow refinement | distinct from direct BC prefix |
| policy component | flow initial condition | partial with CF-VLA |
| inference | one retrieved initialization then frozen denoising | distinct from ranking but close to retrieval |
| data | HDF5 demos | common |
| claim | demo-supported flow starts improve closed loop | high prior-art risk |

Direct baseline: CF-VLA-style learned coarse initializer proxy.

Simple killer baseline: nearest-neighbor demonstration action without SmolVLA refinement.

Ablation: shuffled retrieval memory.

Implementation plan: build action chunk memory from HDF5, adapt `select_action(noise=...)`, compare retrieval-only, shuffled retrieval, and refined retrieval.

Prototype tasks: same two hard tasks as OCFN.

Second-backbone path: weak. Only applies cleanly to flow/diffusion policies exposing an initial latent or action trajectory input.

Second-condition path: held-out task-family retrieval with train/test demo separation.

Compute estimate: low for memory build, Stage A similar to OCFN.

Failure risk: very high. This is close to CF-VLA and may collapse to nearest-neighbor BC or repeat SACF's demo-prefix failure.

## Candidate 3: CAD-VLA

Name: `CAD-VLA`, Cross-Architecture Differential Distillation VLA

Hidden assumption: Quantized OpenVLA-OFT INT4 solves some SmolVLA hard slices because it encodes a better action manifold; a small student adapter can absorb only the teacher-student action difference on train states without full VLA fine-tuning.

Precise novelty: collect teacher actions from Quantized OpenVLA-OFT INT4 on train identities, compare them to frozen SmolVLA actions, and train a trust-region student adapter only on the teacher-student differential direction for the hard slice.

Equations:

- student action: `a_s = pi_s(o,l)`
- teacher action: `a_t = pi_T(o,l)`
- differential target: `d = clip(a_t - a_s, -rho, rho)`
- adapter action: `a = clip(a_s + g_phi(s,l) d_hat, -1, 1)`
- loss: `||g_phi(s,l) - d||^2 + beta ||g_phi||^2`

Representation: frozen student action, policy-input state, task code, and teacher-student action differential.

Objective: trust-region distillation of cross-backbone differential actions.

Supervision: Quantized OpenVLA-OFT INT4 teacher actions on train states, plus frozen SmolVLA actions.

Inference: SmolVLA plus a small differential adapter. Teacher is not loaded at inference.

Required data: no new downloads, but requires running the existing OpenVLA-OFT INT4 stack for train-state labeling and managing environments across two model stacks.

Closest five papers:

| Paper | Overlap | Difference |
| --- | --- | --- |
| VLA-OPD | teacher supervision on student trajectories | CAD uses local cross-backbone differentials, not token-level reverse-KL on student on-policy trajectories |
| VITA-VLA | action expert distillation | CAD distills only a bounded residual direction, not action ability into a VLM |
| OpenVLA-OFT | strong fine-tuned teacher | CAD is a small student adapter, not OpenVLA fine-tuning |
| SDP / correction learning | uses alternative action chunks | CAD has no human correction or set-valued action data |
| MiniVLA / distillation family | compresses policy knowledge | CAD is claim-specific differential transfer for paired hard slices |

Exact overlap matrix:

| Axis | CAD-VLA | Closest overlap? |
| --- | --- | --- |
| problem | cross-backbone gap on SmolVLA hard slices | partial with distillation |
| representation | teacher-student action differential | partial |
| supervision | OpenVLA teacher labels | common in distillation |
| objective | trust-region differential regression | partial |
| policy component | small student adapter | common |
| inference | no teacher, one adapted SmolVLA action | distinct from online teacher |
| data | paired train states and teacher actions | partial with VLA-OPD |
| claim | compact transfer of hard-slice teacher action manifold | high overlap risk |

Direct baseline: direct teacher-action BC adapter.

Simple killer baseline: frozen Quantized OpenVLA-OFT INT4 itself on the same tasks.

Ablation: no trust region / plain residual adapter.

Implementation plan: use existing OpenVLA-OFT INT4 environment to label train states, then train a tiny SmolVLA wrapper and evaluate.

Prototype tasks: hard-slice tasks where OpenVLA-OFT INT4 was `20/20` and SmolVLA was weaker.

Second-backbone path: conceptually awkward because the method already uses the second backbone as teacher; a third backbone or symmetric teacher would be needed for final paper strength.

Second-condition path: another SmolVLA hard-slice task family with a validated teacher-student gap.

Compute estimate: moderate to high due OpenVLA INT4 labeling and two-stack orchestration.

Failure risk: high prior-art risk with VLA-OPD/VITA/distillation and weak final-paper second-backbone story.

## Selection

Selected method: `OCFN-VLA`.

Reason: OCFN changes at least four core dimensions relative to PTC and SACF: the core problem becomes latent sampling prior miscalibration, the representation becomes flow noise rather than state/action prefixes, the supervision becomes closed-loop outcome labels rather than demonstrations or transition traces, and the action-generation mechanism changes the frozen flow sampler's initial condition instead of training a direct action head. It is locally implementable, has strong simple baselines, and will fail cleanly if the effect is merely global-noise selection or no latent headroom.
