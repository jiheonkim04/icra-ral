# TG-7D Adapter Failure Tree

- Root failure: `KILL_CANONICALIZATION_DOMINATED`.
- Baseline dominance: canonicalization-only `0.587661` beats TG-7D `0.740922`.
- Standard adaptation dominance: standard SmolVLA 7D LoRA `0.600887` beats TG-7D.
- Simple model dominance: MLP `0.619985` beats TG-7D.
- Clean retention failure: TG-7D clean L2 `0.735738` versus standard LoRA `0.600887`.
- Headroom failure: oracle target upper bound `0.724674` is worse than canonicalization-only.
- Residual interpretation: no method-worthy post-canonicalization language/target gap was proven.
