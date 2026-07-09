# TG-7D Adapter Kill Criteria

Final decision must be one of `READY_FOR_TG7D_SCALE_UP`, `KILL_BASELINE_DOMINATED`, `KILL_CANONICALIZATION_DOMINATED`, `KILL_LEAKAGE_RISK`, `NO_TARGET_GROUNDING_EVAL_PATH`, or `TOO_HEAVY_LOCAL`.

Kill immediately if standard LoRA, canonicalization-only, or MLP/ridge matches or beats TG-7D on the claimed target/paraphrase/object metric; if target prior requires leakage; if clean action quality collapses; if counterfactual sensitivity fails; or if no meaningful target/paraphrase/counterfactual metric exists.
