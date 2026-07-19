# Epoch 6 Schedule-Invariance Closed-Loop Execution Manifest

Frozen before any closed-loop outcome exposure. The machine-readable JSON is
authoritative.

The frozen 20 identities are suite-balanced task-4/reset-0-through-4 rows.
Both arms use the same pinned X-VLA checkpoint, float32 CUDA execution, root
seed, official observation/action semantics, 10 settle steps, and official
suite horizons. Each schedule starts in a fresh OS process with one shared
model instance.

The serial arm uses one canonical shard over identities 0 through 19. The
four-shard arm assigns identities round-robin and uses launch offsets of 3, 2,
1, and 0 seconds for shards 0, 1, 2, and 3. Requests enter one shared FIFO
inference queue, and actual arrival/service order is recorded. The resulting
global service ordinal is the process-global stochastic-noise position.

Before either scientific arm, a fresh resource smoke must hold four LIBERO env
instances together with one X-VLA model, perform one outcome-suppressed forward,
and pass the calibrated host gate. Every scientific policy query and episode is
transactionally persisted with RNG, simulator, session, raw/processed action,
arrival-order, executed-action, and outcome evidence needed for exact suffix
resume and adjudication.

The closed-loop gate and decision order are unchanged from the frozen problem
protocol. No method, Ours, training, or paper work is authorized yet.
