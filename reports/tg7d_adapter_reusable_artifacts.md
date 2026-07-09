# TG-7D Adapter Reusable Artifacts

- Fixed LIBERO_7D interface and train-split-only normalization.
- SmolVLA 7D LoRA/adapter baseline table.
- LIBERO-Para to local LIBERO-Goal HDF5 linking.
- Held-out paraphrase group split without group leakage.
- Object lexical subset and counterfactual instruction-swap records.
- Canonicalization-only baseline.
- Target-prior audit from instruction text plus visible object-candidate names.

Revival requirement: a future language/target family would need a large structured residual after canonicalization, standard LoRA, and MLP, plus a non-leaking signal and oracle/headroom evidence. This run did not find that.
