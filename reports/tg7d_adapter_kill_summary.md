# TG-7D Adapter Kill Summary

Original hypothesis: target/object semantic grounding injected into the fixed SmolVLA LIBERO_7D action pathway would improve paraphrase/object lexical robustness while preserving clean action quality and counterfactual target sensitivity.

Strongest positive evidence:

- fixed LIBERO 7D interface existed,
- standard SmolVLA 7D LoRA/adapter baseline worked,
- leakage-safe LIBERO-Para group split existed,
- target-prior audit used instruction text plus visible object-candidate names only.

Decisive negative evidence:

- canonicalization-only held-out paraphrase L2: `0.587661`,
- standard LoRA held-out paraphrase L2: `0.600887`,
- TG-7D held-out paraphrase L2: `0.740922`,
- TG-7D clean retention failed: clean L2 `0.735738` versus standard LoRA clean L2 `0.600887`.

Exact kill criterion triggered: canonicalization-only matched or beat TG-7D on the target/paraphrase metric.

TG-7D should not continue because the proposed target adapter worsened clean action quality and was beaten by canonicalization, standard LoRA, MLP, and even the oracle target upper-bound diagnostic.

Reusable artifacts: fixed LIBERO_7D interface, SmolVLA 7D LoRA baseline, leakage-safe LIBERO-Para/group split, canonicalization baseline, target-prior audit, and counterfactual sensitivity records.

Revival requirement: the broader language/target family would need an official or clearly named benchmark slice where canonicalization-only still has a large structured residual, standard LoRA and MLP/ridge do not solve it, a non-leaking method signal exists, and oracle/headroom diagnostics are positive.
