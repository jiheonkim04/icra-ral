# TG-7D Adapter Risk Register

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Novelty collapses to standard LoRA capacity | High | Keep standard rank-4/8 SmolVLA 7D LoRA as primary baseline. |
| Canonicalization dominates | High | Treat canonicalization-only as a kill baseline. |
| Target prior leaks through BDDL/task IDs/filenames | High | Use only instruction text plus visible object-candidate names for inference prior. |
| Consistency objective ignores target changes | High | Require counterfactual sensitivity. |
| Local action metric overclaim | Medium | Mark as bounded method gate, not rollout or paper evidence. |
