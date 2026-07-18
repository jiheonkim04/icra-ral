# CVLR-XVLA Stage 0 Preregistration

- Decision: `CVLR_XVLA_STAGE0_PREREGISTERED_NOT_LAUNCHED`
- Execution classification: `OURS_VLA_TRAINING`
- Frozen contract SHA-256: `6767e529bd43a61760cd75ae8e4b05d235946fcbf4c5f8e05dbae5e35aa72746`
- CVLR training, checkpoint, and rollout: none.

One `422,144`-parameter predictor will receive current agent-view Florence2 tokens, language embeddings, and X-VLA proprioception and reconstruct the synchronized clean wrist token block. X-VLA remains frozen. Training is exactly `96` AdamW steps on 24 fixed records; validation uses 12 disjoint records and no checkpoint selection.

Controls are zero-fill/no-reconstruction and deterministic AWF agent-token fill. Full validation MSE must improve by at least `5%` over the better control overall and beat both controls on at least two tasks. A meaningful full-versus-zero action effect must exceed a separately frozen translation, rotation, or raw-gripper threshold.

Safety is action-semantic-aware: translation, rotation, raw pre-discretization gripper values, and final binary gripper flips have separate thresholds. Clean bypass must be exact. No universal max-absolute gate combines continuous motion with the binary gripper.

Stage 0 contains no closed-loop rollout. A complete pass requires a separately frozen bounded Stage A contract; an implementation failure permits only one narrow repair; a valid nonacting or unsafe mechanism is archived without Stage A.
