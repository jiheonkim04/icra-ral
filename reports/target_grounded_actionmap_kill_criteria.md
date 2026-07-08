# Target-Grounded ActionMap Kill Criteria

Date: 2026-07-08

These criteria must be applied before any implementation scaling. No experiment, training, rollout, download, GPU job, OpenVLA-OFT execution, or local proxy diagnostic happened in this pass.

## Continue Criteria

Continue only if all are true:

- ActionMap-style head beats mean action, linear/L1, and simple MLP on held-out action quality.
- ActionMap-style heatmap/candidate predictions do not collapse.
- Target/object prior is obtained without inference-time leakage.
- Target-grounded conditioning beats ActionMap alone on wrong-target, target-consistency, object lexical, or paraphrase subset metrics.
- Canonicalization-only does not explain the gain.
- Single-point and destination-only grounding do not explain the gain.
- Clean retention is preserved.
- Counterfactual target sensitivity is preserved or improved.
- A concrete path exists to real VLA, LoRA/adapter, or official benchmark evaluation.

## Kill Criteria

Kill or reframe immediately if any are true:

- ActionMap-style head fails against mean action.
- ActionMap-style head fails against linear/L1.
- ActionMap-style head fails against simple MLP.
- Heatmap or candidate prediction collapses to trivial translation, rotation, or gripper bins.
- Target grounding does not beat ActionMap alone.
- Canonicalization-only explains the paraphrase/object lexical gain.
- Generic paraphrase consistency explains the gain.
- Single grounded 3D point explains the gain.
- Destination-only point explains the gain.
- Object-relative retargeting explains the gain.
- The target prior uses BDDL target labels, eval labels, task IDs, filenames, reward/success labels, same/future HDF5 actions, or any other inference-time leakage.
- No wrong-target, paraphrase, or object lexical metric can be constructed.
- The method improves only a local proxy and has no path to real VLA or official benchmark evaluation.
- LoRA/adapter becomes the claimed novelty rather than an implementation tool.
- The first evidence requires OpenVLA-OFT, full VLA training, large downloads, GPU work, or benchmark-scale rollouts before the anchor/baseline gate is green.

## Required Baseline Suite

Minimum baseline suite:

- mean action;
- linear/L1 action head;
- simple MLP action head;
- ActionMap-style heatmap;
- ActionMap plus canonicalization;
- canonicalization-only for language robustness;
- direct single 3D point;
- destination-only point;
- no-geometry action head;
- object-relative retargeting if retarget claims appear;
- oracle target upper bound, clearly labeled invalid as method evidence.

## Decision Boundary

Current state:

`NEED_ACTIONMAP_ANCHOR_REPRO_FIRST`

Target-Grounded ActionMap should not enter method implementation until the ActionMap anchor clears the simple-head baseline gate.
