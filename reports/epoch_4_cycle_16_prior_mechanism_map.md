# Epoch 4 Cycle 16 Prior Mechanism Map

Date: 2026-07-15 KST

Purpose: select the first post-LIFT method. LIFT remains closed as
`LIFT_COMPUTE_INFEASIBLE` under its fixed Stage 0 protocol. Nothing in this
cycle clips LIFT actions, changes its guidance scales, reopens headroom, decodes
confirmatory policy records, or reinterprets its result.

## Local Constraints From Closed Results

The next method must not cosmetically re-enter a closed local axis:

- action history, residual correction, EMA, chunk scheduling, queue control,
  or output-action repair;
- progress, future-state, waypoint, action-latent, or residual predictability;
- 3D point labels without a new noncollapsed supervision source;
- occlusion completion, complementary-view prediction, or COVI objective
  repair;
- counterfactual field guidance, action mixing, or a LIFT scale/clipping rescue;
- LoRA or QLoRA presented as the scientific contribution.

Cycle 12 generated but did not select `DCR-VLA`, a generic local
STRONG-VLA-style adapter with clean refinement. Cycle 16 may use STRONG-VLA only
if the technical difference is more specific than identity preservation or
ordinary two-stage fine-tuning. The new mechanism must directly alter the
robustness/clean optimization interaction and be independently ablatable.

## Close Sources

### STRONG-VLA

Full title: STRONG-VLA: Decoupled Robustness Learning for Vision-Language-Action
Models under Multimodal Perturbations.

Paper: https://arxiv.org/abs/2604.10055

AUTHOR_STATED:

- STRONG-VLA defines `28` textual and visual perturbation types.
- Stage I learns robustness through a severity-aware curriculum; Stage II
  fine-tunes on clean task data to restore nominal execution fidelity.
- The paper motivates stage separation by clean/perturbed gradient conflict,
  but its method changes the training distribution rather than directly
  constraining the Stage II update direction.
- It reports gains up to `12.60 / 7.77` points on OpenVLA under seen/unseen
  perturbations, `14.48 / 13.81` on OpenVLA-OFT, and `16.49 / 5.58` on pi0.
- OpenVLA and OpenVLA-OFT use LoRA as the parameter-efficient implementation;
  pi0 uses its native direct fine-tuning path. LoRA is therefore infrastructure,
  not the cross-backbone scientific mechanism.
- Clean success changes are small in the main table while several textual,
  geometric, and multimodal perturbations show large gains.

OFFICIAL ARTIFACT STATUS:

- No official code or checkpoint link is present in the arXiv v2 HTML, and an
  exact-name GitHub search did not identify an author repository during this
  audit.
- The local comparison must be called a transparent STRONG-VLA proxy. It can
  reproduce the published distribution schedule and action objective, but it
  cannot be labeled an official reproduction.

LOCAL OPPORTUNITY:

- The paper identifies clean/robust gradient conflict but does not protect the
  robust objective explicitly during Stage II clean refinement.
- A minimum-sufficient extension can pair each clean Stage II batch with a
  semantics-preserving perturbed replay batch and project the clean gradient
  only when it would increase robust loss to first order.
- This changes one training rule, adds no inference module, uses the same action
  supervision, and can be compared under one fixed SmolVLA LoRA scaffold.
- The selected claim would be conflict-aware robustness consolidation, not
  generic PEFT and not a claim to invent gradient projection.

### Gradient Episodic Memory

Full title: Gradient Episodic Memory for Continual Learning.

Paper: https://arxiv.org/abs/1706.08840

Code: https://github.com/facebookresearch/GradientEpisodicMemory

AUTHOR_STATED:

- GEM constrains a current-task update using gradients from remembered examples
  so that loss on previous tasks does not increase to first order.
- It reports positive continual-learning results and provides official code.

CROSS-PAPER SYNTHESIS:

- STRONG-VLA supplies the positive VLA robustness prior, perturbation structure,
  and the specific clean-versus-robust conflict.
- GEM supplies a reproducible constrained-update mechanism.
- Cycle 16 does not claim gradient projection itself as novel. The proposed
  difference is a VLA-specific Stage II consolidation rule that protects a
  perturbation-replay action objective while clean fidelity is restored.

### AffordanceVLA

Full title: AffordanceVLA: A Vision-Language-Action Model Empowering Action
Generation through Affordance-Aware Understanding.

Paper: https://arxiv.org/abs/2606.06155

Code: https://github.com/Skywalker-yqz/AffordanceVLA/

Project: https://skywalker-yqz.github.io/AffordanceVLA/

AUTHOR_STATED:

- AffordanceVLA uses Which2Act, Where2Act, and How2Act intermediate predictions
  for object grounding, 2D interaction localization, and 3D geometry.
- It reports `95.8%` average LIBERO success, `4.33` CALVIN ABCD average chain
  length, and positive real-world results.
- Its labels are produced by a multi-stage pipeline using an LLM, a VLM,
  grounding, depth, and affordance generation.

LOCAL LIMITATION:

- Cycle 11 already considered structured affordance representation, and G3P
  later exposed a collapsed local material-point label.
- A new 2D-only bottleneck is plausible but has weaker prior fidelity than the
  full three-part representation and requires a fresh label-health audit.
- Selecting it now would carry meaningful supervision and prior-proxy risk.

### Self-Correcting VLA

Full title: Self-Correcting VLA: Online Action Refinement via Sparse World
Imagination.

Paper: https://arxiv.org/abs/2602.21633

Code: https://github.com/Kisaragi0/SC-VLA

AUTHOR_STATED:

- SC-VLA predicts task progress and future trajectory trends, then trains an
  online residual policy with a shaped reward.
- It reports `9%` higher success than the strongest compared baseline, `16%`
  fewer steps, and a `14%` real-world gain.

LOCAL LIMITATION:

- The positive method requires predictive heads, online residual RL, and
  substantial training rather than one minimum-sufficient local intervention.
- EvoState, CALA, and RAR already found local progress/future-state or residual
  predictability failures, while DICD/FEDO covered online correction.
- A lightweight proxy would have weak fidelity and high cosmetic-reentry risk.

## Cycle 16 Opportunity

The strongest bounded opportunity is `IARC-VLA`, Interference-Aware Robustness
Consolidation for VLA policies.

Its scientific method is one constrained Stage II update. After a transparent
STRONG-style Stage I robustness curriculum, let `g_c` be the action-loss
gradient on a clean batch and `g_r` the action-loss gradient on a paired
semantics-preserving perturbed replay batch. The clean update is

`g_iarc = g_c - min(0, <g_c,g_r>) * g_r / (||g_r||_2^2 + epsilon)`.

When the gradients agree, IARC is ordinary clean refinement. When they
conflict, the projected update is first-order non-increasing for the replay
robustness loss because `<g_iarc,g_r> = 0` up to `epsilon`. The mechanism adds no
inference-time input, branch, gate, head, memory, or latency.

The low-compute parameterization is a fixed SmolVLA LoRA scaffold with a frozen
base. The same rank, target modules, optimizer, demonstrations, perturbations,
steps, and selection rule must be used for Prior, Ours, ablation, and standard
LoRA wherever scientifically applicable. Removing the word LoRA leaves the
constrained update and scientific claim intact.

