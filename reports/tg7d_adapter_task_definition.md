# TG-7D Adapter Task Definition

TG-7D Adapter tests target/object semantic grounding injected into the fixed SmolVLA LIBERO_7D action pathway.

LoRA and adapters are training tools only. The novelty claim is valid only if target/object priors from instruction text and visible object names improve paraphrase/object lexical robustness while preserving clean action quality and counterfactual target sensitivity.

Forbidden inference sources: BDDL target labels, eval labels, task IDs, filenames, reward/success labels, future actions, old 6D/SO100 labels, and hard-coded gripper fill.
