# Autonomous RA-L Researcher Log

## 2026-07-12 KST

Cycle 1 method: `DICD-VLA`

Researcher position:

- ECHO showed no recoverable candidate headroom, so the next method must change the deployed action distribution.
- DICD-VLA targets a physically meaningful delay axis: action chunks are generated at one observation time but executed after deployment delay.
- The adapter uses frozen SmolVLA chunks, declared delay, step fraction, and recent executed actions.
- Real SmolVLA traces trained full and no-history adapters with finite gradients and checkpoint reload identity.

Evidence ready before Stage A:

- synthetic mechanism smoke passed
- real SmolVLA chunk smoke passed
- real trace training passed
- Stage A implementation compiles
- focused DICD unit tests pass

## 2026-07-12 KST Cycle 2

Cycle 2 method: `FEDO-VLA`

Researcher position:

- DICD-VLA is closed and cannot be rescued.
- FEDO-VLA targets a different deployment axis: low-level action-realization disturbance rather than observation/action delay.
- The method uses command/realized-action feedback and task phase to emit residual commands.
- Synthetic smoke passed and real SmolVLA trace training produced reloadable full and no-feedback checkpoints.

Evidence ready before Stage A:

- proposal frozen and hashed
- reviewer attack completed with APEX/static baselines required
- preregistration frozen
- FEDO module and runner compile
- focused unit tests pass
- real trace training passed
