# ECHO Final Headroom Plan

Status: final bounded candidate-headroom adjudication. No ECHO, SmolVLA, OpenVLA, effect, phase, ranking, or value head is trained.

## Frozen Scope

- official stochastic candidates: frozen official SmolVLA-LIBERO only
- task count: `3`
- phases per task: `4`
- same-state intervention groups: `12`
- candidate count K: `8`
- effect horizons: `[4, 8, 16]`
- continuation intervention horizon: `16`
- OpenVLA-OFT: `not used`
- full benchmark: `not run`
- downloads: `forbidden/offline env vars set by launcher`

## Predeclared Near-Identical Thresholds

- exact identical pair: full postprocessed chunk L2 `<=1e-9`
- nearly identical pair: full postprocessed chunk L2 `<=1e-3`, or translation/rotation/gripper component L2 all `<=1e-4`
- impoverished state: effective distinct candidates `<2`, mean pairwise action L2 `<0.01`, or nearly-identical pair fraction `>=0.75`
- impoverished policy candidate set: at least two-thirds of the 12 states are impoverished

## Non-Relaxed Headroom Criteria

- final task-success oracle improvement over default candidate must be at least `10` absolute percentage points
- at least `15%` of default-failure states must contain another official policy candidate that succeeds
- recoveries must span at least two tasks and more than one phase/state
- structured perturbations are diagnostic only and are not official VLA candidates
