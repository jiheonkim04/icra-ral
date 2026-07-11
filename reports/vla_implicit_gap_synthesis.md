# VLA Implicit Gap Synthesis

Date: 2026-07-11 KST

This synthesis derives opportunities from shared assumptions, objective-evaluation mismatch, missing interactions, cross-paper synthesis, extrapolation, and adjacent-field transfer. No implementation or experiment occurred.

## Ten Implicit Opportunities

1. Counterfactual action-effect credit for action chunks.
   - Why it exists: OpenVLA-OFT, pi0, and SmolVLA optimize action regression/flow/chunk likelihood, while LIBERO/LIBERO-Plus measure closed-loop success. Pre-VLA, CoVer, and VeriSpace add verifiers after proposals but do not train a causal effect representation.
   - Not copied from limitations: the gap is implied by objective mismatch across action-learning and verification papers, not by a single future-work paragraph.

2. Phase-conditioned predicate effects instead of scalar progress.
   - Why it exists: ProgressVLA, ProgVLA, and SPR show progress helps, but manipulation phases require different physical effects such as grasp, lift, transport, support, insert, and release.
   - Not copied: these papers motivate progress, but the opportunity is to replace scalar progress with a vector of language-bound physical predicate deltas.

3. Effect-equivalence classes for candidate action diversity.
   - Why it exists: CoVer, VeriSpace, DREAM-Chunk, and Pre-VLA rely on candidates, but multiple candidates can be the same intervention in physical terms.
   - Not copied: the opportunity uses causal abstraction of actions, not another verifier.

4. Training-time privileged effect labels with non-privileged deployment.
   - Why it exists: LIBERO provides success predicates and simulator states during training/evaluation, while deployment can use learned visual predicate estimators.
   - Not copied: VIM removes cameras at deployment; this opportunity removes simulator predicates at deployment while keeping them for effect supervision.

5. Effect-aware closed-loop calibration.
   - Why it exists: VLAConf calibrates task-success confidence, but confidence can be high for familiar wrong actions.
   - Not copied: calibrating `will this chunk cause the required effect` is different from calibrating `will the task succeed`.

6. Nonmonotonic long-horizon credit.
   - Why it exists: long-horizon tasks can require temporary decreases in visible progress, while progress-guided methods generally push forward.
   - Not copied: this derives from interaction between progress papers and manipulation physics, not from their stated limitations.

7. Contact-phase effect learning from visual proxies.
   - Why it exists: Gemini Robotics-ER, VeriSpace, and LIBERO-Occ strengthen spatial reasoning, but contact stability remains only partly visible.
   - Not copied: the method would use simulator contact/predicate labels only to train visual effect proxies, not add a post-hoc controller.

8. Risk-sensitive effect selection under intervention cost.
   - Why it exists: VLA-Corrector and AAC change when to call or truncate a policy, while CoVer and Pre-VLA spend test-time samples.
   - Not copied: the opportunity is to trade expected predicate effect against policy-call and latency cost, borrowing risk-sensitive control structure.

9. Counterfactual language-control coupling.
   - Why it exists: LIBERO-CF/CAG isolates vision-over-language shortcuts, while CoVer verifies instruction-action alignment. Neither asks whether the action's physical effect is the one the language phase requires.
   - Not copied: the proposed relation is language-to-effect, not language-to-action score.

10. Robustness as effect invariance, not observation invariance.
    - Why it exists: CRT, STRONG-VLA, LIBERO-Plus, and LIBERO-Occ treat robustness via visual/language perturbations or restoration. A policy may choose different raw actions under shifts while preserving the same desired effect.
    - Not copied: this reframes robustness around invariant action-induced effects, not input restoration or perturbation curricula.

11. Closed-loop credit from short-horizon simulator interventions.
    - Why it exists: AFIL uses failure rollouts as negative guidance, while Pre-VLA uses critic-derived advantage. Neither isolates `do(action_chunk)` under matched context.
    - Not copied: the label is interventional effect difference, not failure class or critic scalar.

12. Cross-backbone effect heads as method portability.
    - Why it exists: SmolVLA and quantized OpenVLA-OFT both work locally, but their action interfaces differ. A small effect module over candidate chunks and visual predicates can be shared conceptually across both.
    - Not copied: portability comes from the representation, not adapter routing, LoRA rank, or quantization.

## Selected Opportunity Class

The strongest synthesis is opportunity 1 plus 2 plus 4: counterfactual, phase-conditioned action-effect credit using privileged training labels and non-privileged deployment. It attacks the deepest shared mismatch without being a generic verifier, confidence head, progress predictor, replanner, chunking rule, or adapter route.
