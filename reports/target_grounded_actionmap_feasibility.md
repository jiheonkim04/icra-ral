# Target-Grounded ActionMap Feasibility

Date: 2026-07-08

Scope: feasibility analysis only. No implementation, experiment, training, rollout, download, GPU job, OpenVLA-OFT execution, local proxy diagnostic, or new NumPy surrogate happened.

## Candidate

Working names:

- Target-Grounded ActionMap
- Language-Grounded Action Heatmap for VLA Manipulation

Core idea:

Combine an ActionMap-style voxel action heatmap decoder with explicit semantic target/object conditioning derived from the instruction and object candidates. The target prior conditions the heatmap through FiLM, AdaLN, residual gating, or equivalent lightweight conditioning. Add counterfactual target consistency and paraphrase/object lexical robustness objectives. Use LoRA or adapters only as later training tools.

## Feasibility Questions

| Question | Status | Evidence | Required before GO |
| --- | --- | --- | --- |
| Can a local ActionMap-style baseline be approximated? | red/yellow | Local mini-anchor ran, but failed mean-action and cheap-MLP gates; oracle candidate headroom was strong. | Reproduce a stronger ActionMap-style anchor that beats mean action, linear/L1, and simple MLP without candidate collapse. |
| Can target/object prior be obtained without leakage? | yellow/green | Prior-source audits, instruction/object-key resolvers, and prior target diagnostics previously avoided eval-label/task-id/filename leakage. | Re-audit under the ActionMap setup; forbid BDDL target labels, eval labels, task IDs, filenames, future actions, and reward labels as inference target sources. |
| Can LIBERO-Para paraphrase/object lexical split connect to local LIBERO tasks? | yellow | PRISM-VLA integrated official LIBERO-Para metadata and built held-out paraphrase group splits with PRIDE metrics. | Map target/object lexical variants to the selected ActionMap/LIBERO tasks and preserve canonicalization-only baseline. |
| Can old fixed-prior TCA evidence map to this design? | yellow | Fixed-prior target conditioning gave positive offline signals and wrong-target proxy gains; old head failed action-quality gate. | Treat old evidence as motivation only. It cannot substitute for heatmap-head evidence. |
| Can LoRA/adapter be used later as a tool? | yellow | SmolVLA local readiness looked plausible; SafeLoRA found LoRA plausible but no clear official safety LoRA path; QLoRA tooling missing. | Use adapters only after anchor and bounded diagnostics pass. Do not claim LoRA novelty. |
| Is there a real VLA or official benchmark path? | yellow | ActionMap, OpenVLA-OFT, LIBERO, LIBERO-Para, and SmolVLA provide plausible anchors. | Identify exact model/task/assets and compute constraints before Stage 3. No OpenVLA-OFT local run without separate approval. |
| Is method-level novelty still plausible? | yellow | ActionMap lacks explicit semantic target heatmap conditioning; direct 3D point injection lacks heatmap distribution; LIBERO-Para is diagnostic. | Must differentiate against w2 VLA, GuidedVLA, CAC-VLA, RoVLA, single-point injection, and canonicalization. |

## Main Feasibility Blocker

The blocker is not whether target priors are useful. The blocker is whether the ActionMap-style decoder substrate is strong enough.

Current blocker:

`actionmap_anchor_not_reproduced_against_simple_heads`

## Continue Conditions For Feasibility

Continue only if:

- ActionMap-style anchor beats mean action, linear/L1, and simple MLP on held-out action quality;
- heatmap/candidate predictions do not collapse;
- target/object prior is non-leaking;
- LIBERO-Para or equivalent object lexical/paraphrase subset can be mapped;
- target conditioning is testable against ActionMap alone, canonicalization-only, single-point, and destination-only baselines;
- a real VLA/LoRA/official benchmark path is concrete.

## Feasibility Verdict

The candidate is the only salvageable research direction, but it is not ready for method implementation.

Decision:

`NEED_ACTIONMAP_ANCHOR_REPRO_FIRST`
