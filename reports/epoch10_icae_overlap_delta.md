# Epoch 10 ICAE-VLA novelty and overlap delta audit

Date: 2026-07-21 KST

Decision: `NO_EXACT_PRIOR_COLLISION_FOUND`

## Exact proposed combination

The collision test is intentionally narrow and mechanical. A prior collides only if it queries each candidate checkpoint through its deployment adapter at an exact held-out demonstration state, restores a simulator twin, executes the checkpoint action unit versus the expert action unit, uses the same short checkpoint-independent continuation to measure paired task-relative deterioration, aggregates those consequences to rank held-out VLA checkpoints, and prospectively validates checkpoint-selection utility against closed-loop outcomes.

No audited primary source implements that complete combination.

## Local delta

The local Epoch 4 Cycle 21 ledger rejected TASF-VLA because it remained in SDP/FAN-style set-valued training supervision and lacked observed correction sets. ICAE does not train a policy and does not estimate a desired action set. The existing official scale-up completed 400/400 episodes with no infrastructure failures and found success rates of 74%, 74%, 68%, and 66% for four genuine SmolVLA identities. Raw offline L2 had Spearman correlation about `-0.632456` with success. This establishes the evaluation problem and a trusted deployment route, but the four known outcomes are development-only anchors.

## Primary-source overlap matrix

| Work | Input and supervision | Simulator intervention | Learned component | Output and selection endpoint | Cost tier | Collision finding |
| --- | --- | --- | --- | --- | --- | --- |
| Critical Interval MSE (arXiv:2606.29898) | Expert demonstrations; critical segments; action alignment | None | None required for metric | Offline error ranking across checkpoints; reported Spearman `-0.87` vs raw MSE `-0.61` | Offline queries only | Closest equal-input baseline, but no physical branch or restored twin. |
| Discounted Liveness OPE (arXiv:2605.11479, RSS 2026) | Candidate-policy episode data and sparse rewards | None at evaluation time | Conservative liveness value function | Finite-horizon OPE and task-progress value | Candidate rollout dataset required | Different data tier and Bellman/value formulation; no demonstration-state splice. |
| FAN (arXiv:2604.01570, CVPR 2026) | Training data plus a Gaussian feasible-action-neighborhood prior | None | VLA finetuning regularizer | Improved trained policy success/sample efficiency | Training | Training method, not checkpoint evaluator. |
| Set-Supervised Diffusion Policy (arXiv:2606.01865) | Paired undesired and human-corrected action chunks | None for evaluation | Diffusion-policy training objective over desired sets | Improved trained policy and data aggregation | Human correction plus training | Set supervision, not prospective checkpoint ranking. |
| Per-Group Error, Not Total MSE (arXiv:2606.00253) | Expert actions decomposed by joint group | None | None required for metric | Checkpoint selection; 60 real-robot trials | Offline plus limited robot validation | Important new action-error baseline; no state intervention. |
| SIMPLER (arXiv:2405.05941) | Policy plus recreated simulation tasks | Full policy rollouts | No learned dynamics required | Sim-to-real policy ranking and MMRV | Full simulated rollouts | Physics-based evaluator, but executes full candidate trajectories rather than short demonstration-state splices. |
| RoboLab (arXiv:2604.09860) | Policies in high-fidelity generated tasks and controlled perturbations | Full policy rollouts and environment perturbations | Optional neural posterior analysis | Capability and sensitivity benchmarking | Full simulated rollouts | Perturbs environments, not one candidate action at a restored demonstration state. |
| RoboDojo (arXiv:2607.04434) | Integrated policies across 42 simulated and 18 real tasks | Full rollouts | Benchmark infrastructure | Broad capability ranking | Full sim/real evaluation | New comprehensive benchmark, not rollout-light intervention scoring. |
| WorldEval (arXiv:2505.19017) | Initial observation and policy actions | Learned imagined rollouts | Policy2Vec video world model | Policy and checkpoint ranking | Learned world-model rollouts | Same ranking endpoint, but learned long imagined rollouts rather than exact physics twins. |
| dWorldEval (arXiv:2604.22152) | Vision, language, actions | Learned closed-loop imagined rollouts | Discrete diffusion, sparse memory, progress token | Success and ranking on LIBERO/RoboTwin/real tasks | Learned world model | No exact restored simulator intervention. |
| PiL-World (arXiv:2606.05773) | VLA actions and generated multi-view observations | Learned chunk-wise imagined rollouts | Video world model with latent history | Closed-loop success estimation | Learned full policy-in-loop rollout | Full learned closed loop, not short paired physics consequence. |
| SC3-Eval (arXiv:2606.18610) | Multi-view observations and policy action chunks | Learned imagined rollouts | Forward-inverse, cross-view, and test-time consistency | Real-policy ranking; Pearson `0.929` across seven VLAs | Learned world-model rollouts | Strong evaluator but no demonstration-state physics splice. |
| RoboWorld (arXiv:2607.01060) | Policy actions and initial video | Learned autoregressive rollouts | Step Forcing plus task-progress VLM scoring | Ranking; reported Spearman `0.970` | Learned world-model rollouts | Current strong world-model comparator, not exact-state physical micro-intervention. |
| EValueAction (INDIN 2023, DOI:10.1109/INDIN51400.2023.10218251) | Interactive-imitation policy and simulated state risk | Simulation estimates policy value to request demonstrations | Policy/value machinery | Safe informative demonstration acquisition | Simulation search/rollout | Adjacent use of simulation for policy assessment, but not VLA checkpoint ranking or paired candidate/expert one-action twins. |

## Exact query audit

The execution-date searches were:

- `"intervention-calibrated robot policy evaluation"`
- `"counterfactual action-effect metric" robot policy evaluation`
- `"VLA checkpoint selection" robot`
- `"offline robot policy ranking"`
- `robot policy evaluation "short-horizon" intervention simulator action checkpoint ranking`
- `robot policy evaluation "exact state" action intervention checkpoint selection`
- `robot checkpoint ranking simulator "demonstration states" action consequences`
- `counterfactual robot policy evaluation action splice simulator demonstration`
- `simulator micro intervention robot policy evaluation checkpoint ranking`

These searches recovered offline action-error metrics, full physics-simulator evaluation, learned world-model evaluators, interactive-imitation risk estimation, and broad new benchmarks. They did not recover the exact collision defined above.

## Claim boundary

The novelty delta is the use of a small, common, exact-state physical intervention panel to measure the immediate task-relative consequence of each checkpoint's deployment action and use that score prospectively for VLA checkpoint ranking and selection. This is not claimed to be the first state-dependent metric, anisotropic metric, counterfactual evaluator, or causal action metric. A paper claim remains unauthorized until the prospective evidence gates pass.

## Primary and official sources

- https://arxiv.org/abs/2606.29898
- https://ci-mse.github.io/
- https://arxiv.org/abs/2605.11479
- https://arxiv.org/abs/2604.01570
- https://arxiv.org/abs/2606.01865
- https://arxiv.org/abs/2606.00253
- https://github.com/paumontagut/per-group-mse-vla
- https://arxiv.org/abs/2405.05941
- https://simpler-env.github.io/
- https://arxiv.org/abs/2604.09860
- https://research.nvidia.com/labs/srl/projects/robolab/
- https://arxiv.org/abs/2607.04434
- https://arxiv.org/abs/2505.19017
- https://worldeval.github.io/
- https://arxiv.org/abs/2604.22152
- https://dworldeval.github.io/
- https://arxiv.org/abs/2606.05773
- https://arxiv.org/abs/2606.18610
- https://arxiv.org/abs/2607.01060
- https://repository.tudelft.nl/record/uuid:2d1e10b9-d435-4e4d-9867-de1ccd864d55
