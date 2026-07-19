# Schedule-Invariant Evaluation Problem-Verification Protocol

Protocol class: `DISCOVERY_PROBLEM_VERIFICATION`, no Ours.
Freeze boundary: before any Epoch 6 X-VLA action or LIBERO outcome is read.
Primary machine-readable contract: `problem_verification_protocol.json`.

## Stage 0: outcome-suppressed execution-semantics gate

Use the pinned local X-VLA source/checkpoint on the current fixed hardware and
software stack. Capture one ordinary LIBERO initial observation from Spatial
task 0, official initial-state index 0. Persist and hash RGB, language, and
proprioceptive tensors before model inference. Do not execute any predicted
action and do not read or persist reward, success, or done.

The capture path must not call LIBERO's `set_init_state`, because that helper
internally evaluates success. Restore the official flattened state directly,
call simulator forward, the environment post-process hook, a forced observable
update, and ordinary observation generation while instrumenting both success
check entry points to fail closed. Required success-check call count is zero.

Create 20 stable logical request keys from benchmark revision, task ID, episode
index, action-query index, and sample index. Every key uses the same fixed input
tensor. From the same cold-start root seed, run:

1. canonical key order `0..19`;
2. a cold restart of canonical order;
3. reversed key order `19..0`;
4. an independent-root-seed reference in canonical order.

Clear session-local state before each first-chunk call but seed the process only
once per sequence. Record raw 20-D X-VLA chunks, processed 7-D LIBERO chunks,
binary SHA-256 hashes, normalized RMS differences, gripper disagreement,
latency, VRAM, RAM, and exact source/checkpoint hashes. Stage-0 gates are applied
to the raw 20-D chunks; processed chunks are an execution-facing diagnostic.

Each of the four sequences runs in a fresh operating-system process. Seed
Python, NumPy, CPU Torch, and CUDA Torch exactly once before model load. Load the
official local `models.modeling_xvla.XVLA` class in offline mode as float32,
with LIBERO domain ID `3`, 10 denoising steps, and the native 30-by-20 action
chunk. Before every query, rebuild the same processor output and proprio tensor;
no predicted proprioception or action-plan state carries across logical keys.
After every newly completed key, persist the completed prefix plus Python,
NumPy, CPU Torch, and CUDA Torch RNG states. An interrupted sequence must load
the identical model, restore those states, and execute only the missing suffix;
completed model queries must not be replayed or overwritten.

The fixed policy input uses the ordinary 256-by-256 observation: rotate the
agent-view RGB image by 180 degrees, leave the wrist RGB image unchanged, and
construct 20-D proprioception from controller end-effector position, the first
two rotation-matrix columns in contiguous 6-D form, a zero gripper value, and a
zero second arm. Persist both raw captured arrays and exact policy-input arrays
before inference.

For arrays `x` and `y`, normalized RMS is
`sqrt(mean((x-y)^2)) / max(sqrt(mean(x^2)), sqrt(mean(y^2)), 1e-12)`.
Chunk SHA-256 is computed over an ASCII dtype tag, a little-endian int64 shape
vector, and C-contiguous little-endian float32 bytes. Gripper disagreement is
the fraction of the 30 processed rows with different thresholded gripper
commands. All contrasts pair the same logical key: A versus B for order and A
versus C for independent-root reference.

Before the four sequences, run one fresh-process actual-path resource smoke
using the same fixture and root seed. It performs exactly one X-VLA forward,
persists only raw-chunk shape, finiteness, SHA-256, and resource telemetry, and
is excluded from A/A-repeat/B/C and every scientific gate statistic. It must
show zero WSL swap use, zero Windows pagefile growth or write activity relative
to a stable baseline, host RAM below the frozen ceiling, released GPU/process
memory after exit, and zero telemetry exceptions. It still executes no action
and reads no reward, success, or done.

Required GO facts:

- cold canonical restart reproduces all `20/20` raw chunk hashes or has
  normalized RMS error at most `1e-6` for every key;
- reversing request order changes at least `19/20` raw action-chunk
  hashes;
- the median order-induced normalized RMS is at least `0.10` of the median
  independent-root-seed normalized RMS reference;
- inputs, checkpoint, preprocessing, postprocessing, and root seed are identical
  between the two order arms; there are zero exceptions and no action is stepped
  in the simulator.

Decision order:

1. cold restart failure -> `EVALUATION_INVALID_CANNOT_ISOLATE_SCHEDULE`;
2. implementation/resource/integrity failure ->
   `PROBLEM_GATE_IMPLEMENTATION_OR_RESOURCE_FAILURE`;
3. any schedule-effect threshold failure ->
   `NO_MATERIAL_ACTION_LEVEL_SCHEDULE_DEPENDENCE`;
4. all facts pass -> `ACTION_LEVEL_SCHEDULE_DEPENDENCE_GO`.

Only the fourth decision authorizes the closed-loop problem gate. None authorizes
the episode-addressed method.

## Closed-loop problem gate

Freeze 20 reset identities as four task families times five official initial
states:

- `libero_spatial/task_4`, indices `0..4`;
- `libero_object/task_4`, indices `0..4`;
- `libero_goal/task_4`, indices `0..4`;
- `libero_10/task_4`, indices `0..4`.

These task/reset choices are fixed by a suite-balanced deterministic rule before
new outcomes. Historical results remain discovery evidence and are not used in
the gate statistic.

Run the identical pinned X-VLA checkpoint, preprocessing, postprocessing,
chunking, root seed, environment reset states, and episode limits under the
primary contrast only:

1. single-lane canonical serial schedule;
2. four harness shards with predeclared reversed launch offsets.

The sharded implementation must record actual inference arrival order and the
predeclared launch offsets. Run one actual-path resource smoke first; stop rather
than rely on swap/offload. Preserve per policy-call noise position, raw and
processed chunk hashes, first-chunk traces, executed actions,
reward/success/done, timeouts, and environment states needed for integrity—not
for policy inputs.

Closed-loop problem status is
`PROBLEM_VERIFIED_METHOD_DESIGN_AUTHORIZED` only when both are true:

- at least `4/20` paired identities (`>=0.20`) disagree in binary success or
  timeout/failure status between canonical serial and four-shard execution;
- the absolute success-rate difference is at least `0.10` (at least 2/20
  episodes), with trace divergence beginning at the first schedule-remapped
  stochastic query for at least `0.80` of discordant pairs and identical state
  immediately before that query.

The 20% paired-disagreement and 10-point difference are minimum practical effects:
smaller differences would not justify a new evaluation method or a six-page
RA-L systems/evaluation claim under this campaign. The trace requirement ties
the outcome difference to stochastic sample assignment rather than a simulator
or reset mismatch.

Otherwise return `NO_REPEATABLE_PROBLEM` and archive the thesis. Any reset,
action-semantic, checkpoint, scheduler, or simulator defect returns
`EVALUATION_INVALID`; resource failure returns
`INFRASTRUCTURE_OR_RESOURCE_BLOCKED`. No threshold may change after outcomes.

## Evidence partition and prohibitions

- Stage 0 and the 20-reset panel are `DISCOVERY_PROBLEM_VERIFICATION`.
- No validation, confirmatory, or replication identity is defined yet.
- Do not implement keyed noise, Ours, ablations, training, or manuscript tables
  before `PROBLEM_VERIFIED_METHOD_DESIGN_AUTHORIZED`.
- Do not use reward/success to choose tasks, resets, schedules, trace examples,
  or thresholds.
- No historical outcome is fresh confirmation.
