# SmolVLA LoRA Next Decision

Final decision: `ACTION_INTERFACE_BUG`

Exact next step: Fix the SmolVLA/LIBERO action interface before any method work: the local data is 7D LIBERO action space while the checkpoint action head and normalizer are 6D SO100-style, and overfit sanity did not clear the action metric gate.

Do not propose a new paper method unless the decision is `READY_FOR_REAL_METHOD_AFTER_BASELINE`.
