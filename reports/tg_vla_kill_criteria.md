# TG-VLA Kill Criteria

Date: 2026-07-09 KST

## Immediate Kill Criteria

Kill or do not train if any of these are true:

- only tiny NumPy/local proxy evidence is possible,
- no real SmolVLA/OpenVLA adapter path exists,
- canonicalization-only is expected to dominate or already dominates the same metric,
- standard LoRA or action imitation adapter is expected to match the method,
- no held-out paraphrase/object lexical/target variation metric can be constructed,
- no counterfactual target sensitivity metric can be constructed,
- no official or standard benchmark path exists,
- target prior requires eval labels, BDDL target labels, task IDs, filenames, or simulator privileged state at inference.

## STATE 2 Hard Kill Criteria

Kill after a smoke if:

- canonicalization-only matches or beats TG-VLA,
- standard LoRA matches or beats TG-VLA,
- simple paraphrase augmentation matches TG-VLA,
- a single 3D point or destination-only point matches TG-VLA where relevant,
- TG-VLA improves consistency by ignoring true target changes,
- counterfactual sensitivity fails,
- clean performance collapses,
- no real VLA adapter metric appears,
- OOM or runtime makes the approach impractical on RTX 5080 16GB,
- evidence remains offline proxy only,
- inference target priors require eval labels, BDDL target metadata, task IDs, or filenames.

## Route-Level Kill Criteria

Kill as an RA-L route if:

- method novelty reduces to standard LoRA, QLoRA, canonicalization, prompt engineering, generic DPO/ORPO, or generic paraphrase augmentation,
- direct grounded 3D point action-head injection explains the gain,
- ActionMap plus a standard adapter explains the gain,
- the method cannot produce rollout/control evidence after a bounded smoke path,
- improvements appear only on cherry-picked local tasks.

## Continue Criteria

Continue only with real evidence that TG-VLA beats strong simple baselines on target/object grounding robustness while preserving clean retention and counterfactual sensitivity.
