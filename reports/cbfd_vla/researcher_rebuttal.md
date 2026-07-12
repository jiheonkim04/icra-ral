# CBFD-VLA Researcher Rebuttal

Date: 2026-07-12 KST

Reviewer B is correct that generic teacher distillation would not be enough for a method paper. The prototype claim is therefore narrowed:

`CBFD-VLA` claims only that heterogeneous successful teacher traces become useful to a compact student when concentrated on validated student failure sets and regularized by retention replay.

The prototype will not claim:

- generic VLA-OPD improvement;
- OpenVLA-OFT performance as our method;
- routing between SmolVLA and OpenVLA;
- online teacher control;
- nearest-neighbor memory retrieval;
- success from replaying train resets.

Required decisive comparison:

- If `direct_distill_proxy` matches `cbfd_full`, kill CBFD.
- If `teacher_trace_memory` matches `cbfd_full`, kill CBFD.
- If `cbfd_no_retention` matches `cbfd_full`, kill the retention-specific formulation and do not rescue by tuning lambda after held-out results.

The key measurement is whether failure-set weighting plus retention replay produces held-out closed-loop success that is not explained by generic teacher BC or train-trace replay.
