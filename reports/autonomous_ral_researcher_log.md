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
