# Outcome-free canonicalizer preflight

Date: 2026-07-20

This diagnostic was run before any Epoch 7 closed-loop outcome. It tests a cheap falsifier, not a method.

The legal control lowercases the observed instruction, retains alphanumeric tokens, forms character-trigram count vectors, and selects the highest-cosine instruction from the fixed catalog of ten canonical LIBERO-Goal instructions. Ties follow descending numeric task ID in the frozen prototype and must be made explicit in the evaluator.

Results:

- complete LIBERO-Para metadata: 3,476/4,092 correct task mappings (84.95%);
- frozen hard discovery panel: 24/30 correct mappings (80.0%);
- no rollout reward, success, simulator state, BDDL filename, or eval ID was used by the mapper.

This does not saturate the text-mapping problem, but it is strong enough that any later method must beat it or establish an open-vocabulary residual that closed-set retrieval cannot address.
