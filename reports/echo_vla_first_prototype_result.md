# ECHO-VLA First Prototype Result

- final decision: `NO_ECHO_CANDIDATE_HEADROOM`
- novelty gate: `ECHO_NOVELTY_SURVIVES_TARGETED_GATE`
- candidate headroom ran: `True`
- candidate headroom passed: `False`
- oracle improvement pp: `0.0`
- default-failure recoverable rate: `0.0`
- data generated groups: `4`
- components trained: `none_headroom_gate_first`
- closed-loop evaluation run: `False`
- latency/VRAM: `{'elapsed_seconds': 140.301, 'cuda_memory': {'allocated_bytes': 935880704, 'max_allocated_bytes': 971650560, 'allocated_mb': 892.525, 'max_allocated_mb': 926.638}}`

## Blocker Or Kill Reason

oracle improvement <10pp or fewer than 15% of default-failure states contain a successful/materially better candidate

## Interpretation

No lightweight ECHO phase/effect/ranking heads were trained because the predeclared oracle candidate-headroom gate failed. The run produced same-state intervention data and verified exact start-state identity, but the candidate set did not contain recoverable improvement over the frozen/default candidate.
