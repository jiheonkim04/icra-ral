# Epoch 7 Base action-energy falsifier adjudication

Decision date: 2026-07-20

Status: `STAGE0_MECHANISM_NOT_SUPPORTED` (`BASE_ENERGY_FALSIFIER_FAIL`).

The valid corrected execution completed all 180 frozen X-VLA forwards across 30 real demonstration samples, three instruction conditions, and two paired time/noise seeds. Every value was finite, WSL swap remained zero, no simulator outcome was read, no optimizer step occurred, and the confirmatory partition remained sealed.

## Frozen result

The diagnostic confirms a real language interface: canonical selectivity was 30/30, canonical-to-paraphrase energy drift occurred on 25/30 samples, and the canonical-versus-negative prediction mean absolute delta was 0.1043. Those checks rule out an instruction-disconnected action head.

The claim-defining headroom check failed. Only 2/30 paraphrases had real-action energy at least as high as the frozen distinct feasible instruction, versus the preregistered minimum of 6/30. The two violations span two tasks but only the compositional family; the gate required at least three tasks and two families. X-VLA already rejected the frozen hard negative on 28/30 samples.

## Internal review

This is not a resource or implementation failure. It is a valid negative result for the exact formulation. The rate gate misses by four pairs, and task and family coverage also fail, so the protocol's one-fixed-expansion exception does not apply.

Paraphrase-only positive adaptation could still reduce the observed within-intent drift, but that would collapse the contribution into augmentation or instructional consistency already occupied by RobustVLA and RoVLA. Changing the negative sampler after observing the frozen result would redefine the method around validation evidence. Neither rescue is authorized.

The closure is deliberately narrow: equivalence-selective action-energy ranking v1, with token-Jaccard feasible negatives and the exact X-VLA clean-action energy, is closed. Language robustness and two-sided equivalence/counterfactual evaluation remain open questions and require a fresh novelty, artifact, and paperability audit before any empirical execution.

## Execution-repair provenance

Attempt 1 loaded the model but performed no sample forwards because importing a dataset utility executed an unrelated missing dependency. Attempt 2 completed 180 forwards but was invalid because raw `{-1,1}` gripper values were passed to BCE. The outcome-independent repair applies the released X-VLA LIBERO handler threshold `(gripper > 0)`; samples, texts, actions `0:9`, images, seeds, noise/time pairs, metrics, and thresholds were unchanged. Both attempts are retained as audit artifacts.
