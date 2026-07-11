# Autonomous Cycle 01 - Action Conditioning And Action Representation

Date: 2026-07-11 KST

Cycle branch target: `codex/ral-cycle-01-action-conditioning`

Final cycle decision: `KILL_NOVELTY_AND_LOCAL_HEADROOM_COLLAPSE`

## Researcher A Proposal

Cycle 01 tested the strongest remaining action-generation family after ECHO:

`LATENT-EFFECT-VLA`: learn an intermediate latent action/effect representation that conditions a continuous action expert, with effect-equivalence classes used to distinguish action chunks that are raw-action different but physically equivalent.

Three concrete variants were considered:

1. `LatentEffect-CAC`: VLM-predicted latent action codes gated into the action expert, with an auxiliary effect-equivalence loss.
2. `EffectActionMap`: a heatmap or candidate action representation where neighboring bins are trained by physical-effect equivalence rather than raw action proximity.
3. `HistoryEffect-AEM`: a compact action-effect memory over image/action history used to condition the next action chunk.

## Reviewer B Search

Closest current papers:

- CAC-VLA, https://arxiv.org/abs/2607.04816, already proposes context-gated latent-action conditioning for continuous expert control and reports LIBERO/LIBERO-Plus gains.
- ACoT-VLA, https://arxiv.org/abs/2601.11404, formulates action-space reasoning as coarse action intents that condition the downstream action head.
- LaRA-VLA, https://arxiv.org/html/2602.01166v1, internalizes textual and visual chain-of-thought into continuous latent representations for action generation.
- ActionMap, https://arxiv.org/abs/2606.06904, makes action representation itself the method contribution through a voxel heatmap action head and reports cross-backbone LIBERO gains.
- LARA, https://arxiv.org/html/2606.07100v1, and LAWM, https://arxiv.org/html/2509.18428v2, occupy latent-action representation learning from visual dynamics/world modeling.
- Action-Effect Memory, https://arxiv.org/abs/2606.12499, explicitly pretrains action-conditioned temporal representations for manipulation.

Closest local evidence:

- `reports/echo_final_headroom_decision.md`: ECHO same-state action-effect candidate headroom is absent under downstream task success.
- `reports/actionmap_mini_anchor_kill_summary.md`: local ActionMap-style candidate learning lost to mean-action and cheap MLP and collapsed.
- `reports/openvla_oft_quantized_cross_backbone_decision.md`: the strongest SmolVLA hard-slice failures did not reproduce in Quantized OpenVLA-OFT INT4.

## Rebuttal

Researcher A could argue that effect-equivalence is not identical to CAC-VLA latent actions or ActionMap raw action heatmaps. However, the only locally executable evidence for effect-equivalence action candidates is ECHO, and the final gate showed no downstream oracle headroom. A revised representation would first need a new candidate-generation contribution; that would be an ActionMap/CAC/ACoT-adjacent paper, not a locally supported ECHO successor.

## Kill Reason

The family fails both novelty and local headroom gates:

- latent action conditioning is now directly occupied by CAC-VLA, ACoT-VLA, and LaRA-VLA;
- action representation is directly occupied by ActionMap;
- action-effect temporal representation is directly occupied by AEM/LAWM/LARA-style work;
- local ECHO action-effect headroom is `0.0` percentage points;
- local ActionMap approximation already failed simple baselines.

Implementation is not authorized. A prototype here would either duplicate recent latent-action/action-representation papers or rescue ECHO without headroom.

