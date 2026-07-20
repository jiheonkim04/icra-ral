# Epoch 7 contact-transition Stage 0B execution contract

This contract fixes operational details that the unchanged Epoch 6 scientific
protocol left implicit. It was frozen after Stage 0A GO but before any visual
or ridge model was fit. The machine-readable JSON is authoritative.

Rows use state index `t >= 3`. Exact state-`t` contact transitions are paired
with `actions[t-1,0:6]`; causal nonvisual history ends at state `t-1` and
action `t-2`. The oracle evaluates rows within two frames of a boundary, while
its binary and typed contact features remain the exact-`t` vectors. Thus the
binary control can explain transition timing and topology must add contact
type, rather than inheriting a window aggregation advantage.

The visual probe uses the two allowed cameras at `t-1` and `t`, four frozen
convolution blocks, a 64-unit fusion layer, AdamW, and at most 20 epochs. A
same-width nonvisual control receives identical causal robot/action/time
features. Epochs and ridge alphas are selected only on the single tune task;
the four validation tasks remain untouched until final evaluation.

The oracle compares base, gripper/stage, binary-contact, and typed-contact
ridge models. The full model extends the strongest tune-selected augmented
control. Fifty independently derived within-demo typed-vector permutations
must erase at least 80% of any full-model gain. All original thresholds and
decisions remain binding.

Stage 0B is pre-method discovery. Even a GO authorizes only a separately
frozen method and evaluation contract—not training, rollout, or a paper.

Frozen JSON SHA-256:
`D50FE6287B68BEF93F292F6AD9C207740F1B56DD599D6B1004B5E984D2764A20`.
