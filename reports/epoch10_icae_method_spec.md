# ICAE-VLA method specification

Status: frozen before checkpoint comparison or prospective closed-loop labels.

ICAE-VLA is an evaluator, not a policy-training method. At a prospectively registered raw LIBERO demonstration state, it reconstructs the action that a saved SmolVLA checkpoint would actually execute from its 50-step action queue. It restores the identical physics state twice: the nominal branch executes the recorded expert action, while the candidate branch executes the checkpoint action. Both then receive the same recorded expert continuation for the selected short horizon. The signed difference in normalized native-goal error is the row score; lower is better.

The four frozen tasks cover `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`. Whole demonstrations define mechanics, metric-development, and metric-holdout state partitions. Whole training seeds define checkpoint development (`101`, `202`) and checkpoint holdout (`303`, `404`). Adjacent frames or snapshots never cross their respective split boundary.

## Deployment equivalence

The locked model has `chunk_size = n_action_steps = 50`. For frame `t`, the queue origin is `floor(t/50)*50` and the executed unit is queue offset `t mod 50`. The exact origin state is rendered through both 256×256 simulator cameras, then the official LeRobot environment preprocessor, SmolVLA preprocessor, policy queue, policy postprocessor, and LIBERO environment postprocessor are used. The candidate is one 7-D environment action—not a whole predicted chunk compared with one expert action. Raw out-of-range values and the bound-enforced executed action are both logged.

## Physics score

The dense task error uses the task's native BDDL goal predicates and their operand states. For each demonstration, the target operand configuration is captured after its final recorded expert action and native success is audited. Available relative position, relative orientation, articulated-joint, and native predicate-satisfaction components are scaled by fixed physical units in the protocol. No pixel or embedding distance is used. Task normalization is fitted only from mechanics-calibration nominal rows.

For checkpoint `c`, task `t`, and state `i`:

`d[c,i] = normalized_goal_error(candidate at H) - normalized_goal_error(nominal at H)`

`ICAE(c,t)` is the mean signed row score, and the overall score is an equal-task macro-average. Invalid actions or branches receive the predeclared maximum-harm score and remain separately reported.

## Assay controls

Horizon candidates are `4`, `8`, and `16`; the shortest horizon with stable nominal twins and monotone pooled response to fixed small/medium action perturbations is selected. Exact shams, randomized branch order, deterministic cache duplicates, duplicate state subsets, adapter round trips, clipping rates, and one safety-limited outcome-blind simulator step per adapter are audited. The uniform outer-wrapper accessor defect in the first preflight consumed the single repair allowance; its 128 invalid originals are retained, and every row was rerun in Attempt 2.

The synthetic perturbations only establish that the measurement apparatus responds in the expected direction. They are never treated as checkpoints and never enter the prospective scientific ranking.
