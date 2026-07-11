# Autonomous Cycle 03 - Contact Barrier And Irreversibility Boundaries

Date: 2026-07-11 KST

Cycle branch target: `codex/ral-cycle-03-contact-barrier`

Final cycle decision: `KILL_BASELINE_PRIOR_ART_AND_EVIDENCE_COLLAPSE`

## Researcher A Proposal

Cycle 03 changed at least two axes relative to Cycles 01 and 02:

- core problem: contact/constraint/irreversibility failures;
- representation: phase-conditioned barrier or recoverability margin;
- closed-loop intervention: prevent or truncate actions near irreversible physical boundaries.

Proposed method family:

`BARRIER-IRIS-VLA`: learn a visual/proprioceptive margin for phase-conditioned contact, support, collision, or irreversible-effect boundaries, then use it as a lightweight action-generation penalty or mid-chunk intervention trigger.

Three concrete variants were considered:

1. `PhaseBarrier`: signed visual/proprioceptive barrier margin for chunk prefixes, trained from simulator constraint labels.
2. `RecoverabilityBoundary`: binary or calibrated margin predicting whether continuing a chunk makes the phase unrecoverable within a bounded horizon.
3. `ContactProxy`: RGB/proprioceptive proxy for contact sufficiency trained from simulator labels and used only at deployment from non-privileged inputs.

## Reviewer B Search

Closest current papers:

- VeriSpace, recorded in `reports/latest_vla_method_landscape_2026.md`, occupies spatially grounded action verification.
- Pre-VLA, https://arxiv.org/abs/2605.22446, occupies safety confidence and critic-derived advantage for candidate action chunks.
- VLA-Corrector, https://www.alphaxiv.org/abs/2607.01804, occupies event-triggered truncation and corrective replanning during action-chunk execution.
- SEAM, https://arxiv.org/html/2607.04609v1, occupies training-free chunk-boundary smoothing.
- AAC and Legato, recorded in `reports/latest_vla_method_landscape_2026.md`, occupy adaptive chunking and continuation/smooth execution.
- TORL-VLA, https://arxiv.org/html/2606.09337v3, occupies contact-rich online adaptation with tactile/wrench feedback.

Closest local negative evidence:

- `reports/openvla_oft_quantized_cross_backbone_decision.md`: the drawer stable-grasp and long-horizon failures were not cross-backbone VLA failures; Quantized OpenVLA-OFT INT4 solved all `20/20` matched hard-slice episodes.
- `reports/contactset_vla_kill_summary.md`: richer contact-set geometry lost to active single-point, destination-only, and no-geometry baselines.
- `reports/contacttube_aug_kill_summary.md`: contact-tube augmentation lost to simple object-relative retargeting and had controller-validity problems.
- `reports/closed_loop_method_gate_decision.md`: the local stable-grasp mechanism had only two independent rerun-failure reset seeds and did not survive the method-ready gate.

## Rebuttal

Researcher A could narrow the claim to vision-only contact proxies with simulator labels and no tactile hardware. Reviewer B rejects this as insufficient because the strongest contact-adaptation papers use richer physical feedback, while the local vision-only contact/geometry routes have repeatedly lost to simple baselines or failed cross-backbone generality.

## Kill Reason

The family is killed before implementation:

- generic safety/spatial validity is too close to VeriSpace and Pre-VLA;
- mid-chunk truncation/recovery is too close to VLA-Corrector/AAC/SEAM/Legato;
- true contact-rich adaptation would require unavailable tactile/force/robot hardware;
- local contact/geometry evidence has already been baseline-killed;
- the specific SmolVLA stable-grasp failure is not cross-backbone.

Implementation is not authorized.

