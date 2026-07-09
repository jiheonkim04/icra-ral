# Next Actions

Date: 2026-07-09 KST

Current decision:

`KILL_CANONICALIZATION_DOMINATED`

## Immediate Next Action

Stop the local language/target robustness route after TG-7D and post-canonicalization residual mining.

## Why

The bounded method gate produced real fixed-interface 7D adapter evidence, and TG-7D did not clear the anti-baseline requirements:

- canonicalization-only held-out paraphrase L2: `0.587661`,
- standard SmolVLA 7D LoRA/adapter held-out paraphrase L2: `0.600887`,
- MLP held-out paraphrase L2: `0.619985`,
- TG-7D held-out paraphrase L2: `0.740922`,
- TG-7D clean L2: `0.735738`,
- standard LoRA clean L2: `0.600887`.

TG-7D improved same-target consistency, but that is not enough. The method lost clean action quality and was dominated by canonicalization-only and standard LoRA on the claimed target/paraphrase metric.

Post-canonicalization residual mining found no method-worthy remaining gap:

- canonicalization clean-to-paraphrase delta: `-0.000748`,
- object lexical L2 under canonicalization: `0.587388`,
- largest residual subgroup: gripper error `0.389255`, not a target/language-specific slice,
- oracle/headroom evidence: no.

## Reusable Artifacts

- LIBERO-Para to local LIBERO-Goal HDF5 linking.
- Held-out paraphrase group split with no group leakage.
- Object lexical subset.
- Counterfactual instruction-swap sensitivity audit.
- Target prior from instruction text plus visible object-candidate names.
- Fixed SmolVLA/LIBERO_7D adapter gate with standard LoRA, canonicalization, paraphrase augmentation, TG-7D, and oracle target upper bound arms.

## Disallowed Next Work

Do not:

- continue TG-7D Adapter training,
- tune TG-7D until it wins by seed/overfitting,
- make a paper claim,
- run OpenVLA-OFT,
- download large assets,
- run a full benchmark,
- use old TCA-Select,
- use the old 6D/SO100 action path,
- use hard-coded gripper fill,
- use BDDL/eval labels/task IDs/filenames as inference target labels.

## Next Valid Step

Archive TG-7D as canonicalization-dominated. Future method work must start from a new predeclared hypothesis that cannot be explained by canonicalization-only, standard SmolVLA 7D LoRA, or simple MLP/ridge baselines.
