# AWF-XVLA Stage 0 Preregistration

- Decision: `AWF_XVLA_STAGE0_PREREGISTERED`
- Method: Agentview-Wrist Fill for X-VLA (`AWF-XVLA`)
- Class: no-training deterministic inference module.
- Selected only after wrist-camera dropout was verified as a claim-specific condition.

## Mechanism

If the current wrist RGB has mean pixel value <= 1.0, AWF-XVLA fills the wrist input slot with the same flipped agentview RGB that X-VLA receives as its policy-input agentview image. It uses no reward, done/success oracle, simulator object/contact state, privileged pose, future observation, training, optimizer, or checkpoint.

## Stage 0

- Condition: `wrist_blackout`.
- Task: `libero_spatial/task5`.
- Discovery identities: `20260731`, `20260732`.
- Clean baseline: `2/2`.
- Frozen-prior wrist-dropout baseline: `0/2`.
- GO if AWF-XVLA succeeds on at least `1/2` discovery episodes under wrist dropout.

Held-out identities remain unused.
