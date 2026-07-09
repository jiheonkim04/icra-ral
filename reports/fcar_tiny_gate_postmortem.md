# FCAR Tiny Gate Postmortem

Date: 2026-07-10 KST

Final FCAR status: `killed`
Reference decision: `FCAR_KILLED_BY_STATIC_BASELINE`

## Why FCAR Was Killed

- FCAR vs frozen/base: `{'fcar_action_l2': 0.100144625, 'frozen_base_action_l2': 0.123998278, 'gain': {'absolute': 0.023853653, 'relative': 0.192370841}}`
- FCAR vs rank-4 LoRA: `{'fcar_action_l2': 0.100144625, 'rank4_lora_action_l2': 0.076191123, 'fcar_minus_lora': 0.023953502}`
- FCAR vs static merge: `{'fcar_action_l2': 0.100144625, 'static_merge_action_l2': 0.091179973, 'fcar_minus_static': 0.008964652}`
- FCAR vs frame oracle: `{'fcar_action_l2': 0.100144625, 'frame_oracle_action_l2': 0.066124022, 'fcar_minus_frame_oracle': 0.034020603}`
- alpha collapse: `{'fraction_routed_to_base_alpha_lt_0_5': 1.0, 'fraction_routed_to_lora_alpha_ge_0_5': 0.0, 'max': 0.493465692, 'mean': 0.443432957, 'min': 0.320281953, 'std': 0.02648654}`
- behaved like near-static mixture: `True`

## Interpretation

A static base/LoRA mixture has no learned frame gate, no method novelty, no inference-time oracle input, and still beat FCAR on the held-out FCAR test split.

The FCAR kill criteria were fixed before seeing results; tuning FCAR now would be post-hoc test-set adaptation.

Useful remaining evidence:

- official per-frame base/LoRA prediction artifact
- frame oracle still measures possible routing headroom
- static mixture and rank-4 LoRA are mandatory reviewer baselines for any future frame-level method
