# VLA Shared Assumption Analysis

Date: 2026-07-11 KST

No experiments, inference, simulator execution, training, download, or implementation occurred.

## Shared Assumptions

1. Current observation is sufficient.
   - Appears in: OpenVLA-OFT, SmolVLA, pi0, CoVer, VLAConf, Pre-VLA.
   - Failure mode: the current image may omit contact state, object support, occluded target geometry, or whether the previous chunk had the intended physical effect.
   - Implication: a method should represent the effect a chunk is expected to cause, not only the action chosen from the present image.

2. Action uncertainty is calibrated.
   - Appears in: action-token probability methods, confidence heads, entropy chunking, verifier sampling.
   - Failure mode: the policy can be confident in a high-likelihood demonstration action that is physically wrong under a shifted reset.
   - Implication: uncertainty alone is weak unless tied to downstream effect on task predicates.

3. Task progress is monotonic.
   - Appears in: ProgressVLA, ProgVLA, SPR, milestone planners.
   - Failure mode: manipulation often requires temporary regressions, contact regrasp, or object motion away from a goal before placement.
   - Implication: progress should be decomposed into phase-conditioned physical effects instead of a scalar.

4. A chosen action chunk remains valid while executing.
   - Appears in: action-chunked VLAs, OpenVLA-OFT, SmolVLA, pi0.
   - Recent responses: AAC, FASTER, VLA-Corrector, DREAM-Chunk, SEAM, Legato.
   - Remaining gap: these methods adapt horizon/smoothness/reactivity, but usually do not train the policy to know which chunk effects matter for success.

5. Visual state contains all physical information.
   - Appears in: VLAConf, VeriSpace, CoVer, CRT, LIBERO-Occ/VIM.
   - Failure mode: tactile/contact/force stability and support relationships can be visually latent.
   - Implication: simulator-derived training labels can teach visual proxies of hidden effect states, while deployment remains non-privileged.

6. Model confidence correlates with physical success.
   - Appears in: VLAConf, Pre-VLA, verification/reranking work.
   - Failure mode: confidence can reflect familiarity, not whether a chunk changes the world in the required way.
   - Implication: confidence needs to be subordinate to action-effect estimates, not the research contribution itself.

7. Candidate actions differ meaningfully.
   - Appears in: CoVer, VeriSpace, DREAM-Chunk, Pre-VLA.
   - Failure mode: repeated samples from a narrow policy can be action-equivalent or share the same physical error.
   - Implication: training should contrast actions by effect equivalence classes, not raw action distance.

8. Replanning frequency solves execution error.
   - Appears in: adaptive horizon, corrective inference, progress/recovery methods.
   - Failure mode: frequent replanning can repeatedly choose the same high-likelihood but low-effect action.
   - Implication: replanning needs an effect-credit target, otherwise it is only more opportunities to repeat the wrong local behavior.

9. Action loss aligns with task success.
   - Appears in: OpenVLA-OFT L1, flow matching, token likelihood, BC-style fine-tuning.
   - Failure mode: small action errors can be harmless, while action-likelihood-perfect chunks can fail if they do not cause the required predicate transition.
   - Implication: the paper opportunity is a closed-loop effect objective, not another decoder or PEFT recipe.

10. Language grounding and physical feasibility can be verified independently.
    - Appears in: CAG for language shortcuts, VeriSpace for geometry, CoVer for alignment.
    - Failure mode: an action can be semantically aligned yet physically ineffective, or physically valid yet semantically irrelevant.
    - Implication: the effect representation must bind language phase, current physical state, and action-induced predicate change.

## Derived Design Constraint

The strongest opportunity is to estimate `P(effect | do(action_chunk), observation, instruction, phase)` and use that estimate as a training and inference signal. This differs from confidence, verification, and progress because the object being learned is the causal effect of a chunk on required task predicates under matched context.
