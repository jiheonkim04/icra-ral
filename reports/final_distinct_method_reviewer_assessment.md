# Final Distinct Method Reviewer Assessment

Date: 2026-07-12 KST

Reviewed proposal: `Intervention-Set Action-Chunk Fine-Tuning (ISAC-VLA)`

Reviewer decision: `FINAL_METHOD_KILLED_BEFORE_IMPLEMENTATION`

## Grounds

Ground 1: `NEAR_EXACT_PRIOR_ART_DUPLICATION`

ISAC-VLA's central mechanism is to train an action-chunk policy from paired negative policy chunks and positive corrective chunks. Set-Supervised Diffusion Policy already frames human corrections as paired undesired and corrective action chunks and trains a policy with a set-supervised/contrastive action-chunk objective. TORL-VLA and ConRFT further occupy the intervention-censored and human-intervention VLA fine-tuning space.

Ground 2: `HARD_UNAVAILABLE_RESOURCE`

A faithful ISAC-VLA run requires paired intervention/correction chunks from a human, robot, or high-fidelity intervention simulator. The local evidence artifacts contain weak effect-score rows, frozen-policy rollouts, and post-hoc wrappers, but no paired correction chunk dataset. Creating that dataset would require new resources outside the branch scope.

## Consequence

No final-method code, training, or rollout is allowed. The final method is killed at the reviewer gate.
