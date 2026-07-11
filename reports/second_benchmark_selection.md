# Second Benchmark Selection

Date: 2026-07-11 KST

Selected second benchmark: `LIBERO-PRO`

## Selection Rationale

LIBERO-PRO is selected because it:

- is an official LIBERO extension with code at `https://github.com/Zxy-MLlab/LIBERO-PRO`
- has paper source `https://arxiv.org/abs/2510.03827`
- provides public non-gated MIT metadata and a small Hugging Face BDDL/init dataset
- uses the same LIBERO runtime family rather than a different robot/action interface
- supports object, position, semantic, task, and environment perturbations
- can test both the drawer/bowl stable-grasp route and the LIBERO-10 long-horizon route

## Selected Perturbation Families

If the cross-backbone test later reproduces `stable_grasp`, use LIBERO-PRO object/position perturbations first.

If the cross-backbone test later reproduces `long_horizon_compounding`, use LIBERO-PRO task/semantic/position perturbations first.

Environment perturbations are lower priority because the LIBERO-PRO README notes that some environment replacements can move table objects unexpectedly.

## Current Selection Status

Selection status: `selected_but_not_run`

Active blocker: selected second VLA is not downloaded or validated.
