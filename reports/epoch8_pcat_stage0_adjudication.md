# Epoch 8 PCAT Stage 0 Adjudication

Decision: **valid scientific failure; exact PCAT mechanism closed** (`closure_class=2`).

## Outcome

The intended execution occurred: 974 real CUDA Base forwards, 1,200 real CUDA optimizer steps across four 254,090-parameter adapters, finite positive gradients, nonzero learned PCAT output, zero swap use, and no simulator or confirmation access. Every checkpoint and the raw result are hash-preserved in `reports/epoch8_pcat_stage0_adjudication.json`.

PCAT failed four preregistered gates:

- canonical official energy was `0.03463` versus Base `0.01271` (2.73x; allowed at most 1.05x);
- paraphrase-equivalence drift was `0.27876` versus Base `0.27226`, so it did not improve equivalence;
- target-swap transport cosine fell from Base `0.80093` to PCAT `0.75445` rather than gaining at least 0.10 over every control;
- normalized transport error rose from Base `0.43012` to PCAT `0.44348`.

PCAT did change actions (`0.08234` normalized RMS), kept all tested actions legal, retained positive cosine in every target-pair group, and produced a noncollapsed response magnitude. The failure is therefore not a dead adapter. Base already has strong signed action-response alignment, and the new constraint worsened that pathway while damaging canonical competence.

## Validity and repair decision

The protocol and script hashes match. A focused sign oracle gives zero transport loss for the declared right-minus-left response and `0.58674` for its reversal. Dataset/action hashes, shapes, checkpoint hashes, gradients, steps, and resources all pass.

No semantically null repair is justified. Changing the transport weight, match phase, architecture, budget, or checkpoint selection after these metrics were observed would tune the mechanism, not repair execution. The raw attempt is preserved and no confirmatory evidence was opened.

## Closure boundary

This result closes the exact PCAT residual adapter, initial-pose pairing, real 30-step transport target, frozen losses, 300-step budget, and direct reparameterizations of the same causal mechanism. It does not close the verified 30/30-versus-19/30 language problem or independent causal routes.

The second language candidate, explicit binding-posterior mediation, was already rejected before outcomes as occupied/incremental relative to ProGAL, GuidedVLA, and direct grounded-point action injection. The next action is the mandated independent-route ranking, not closed-loop promotion of PCAT.
