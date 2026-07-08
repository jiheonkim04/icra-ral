# ActionMap Mini-Anchor STATE 1 Result

Bounded local HDF5 action-head diagnostic only. This is not a full VLA reproduction, standard LIBERO success, rollout evidence, or a paper-grade claim.

- decision: `kill`
- final decision: `KILL_ACTIONMAP_ANCHOR`
- reason: ActionMap-style head collapsed to too few action candidates; mean-action baseline matches or beats the ActionMap-style heatmap head; cheap MLP action head matches or beats the ActionMap-style heatmap head
- training happened: `True`
- loss computed: `True`
- replay/control happened: `False`
- GPU/download/OpenVLA-OFT: `False` / `False` / `False`
- official ActionMap reproduction / extension / failure mining: `False` / `False` / `False`
- dataset/split: `8` demos, `deterministic_per_demo_time_holdout`
- oracle gate passed: `True`
- mean-action action L2: `0.466767673`
- linear/L1 action L2: `0.812610317`
- simple MLP action L2: `0.501926707`
- ActionMap-style action L2: `0.529931357`
- oracle candidate action L2: `0.065653208`
- ActionMap beats mean/linear: `False` / `True`
- simple MLP matches or beats ActionMap: `True`
- next state: `kill_or_reframe_anchor_reproduction`
- exact next step: Stop; do not proceed to Target-Grounded ActionMap from this mini-anchor result.

Triggered kill criteria:

- ActionMap-style head collapsed to too few action candidates
- mean-action baseline matches or beats the ActionMap-style heatmap head
- cheap MLP action head matches or beats the ActionMap-style heatmap head

## Variants

| variant | action L2 | translation L2 | rotation L2 | gripper error | action L1 | top-k / NLL | collapse notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| mean_action_baseline | 0.466768 | 0.38741 | 0.050504 | 0.998778 | 0.289159 | n/a | n/a |
| linear_l1_action_head | 0.81261 | 0.965434 | 0.328985 | 0.994695 | 0.586692 | n/a | n/a |
| oracle_nearest_action_candidate_upper_bound | 0.065653 | 0.091242 | 0.041621 | 0.0 | 0.043004 | oracle upper bound | invalid as method evidence |
| simple_mlp_action_head | 0.501927 | 0.426866 | 0.079105 | 0.934607 | 0.301072 | n/a | n/a |
| actionmap_heatmap_candidate_head | 0.529931 | 0.501792 | 0.047267 | 0.601852 | 0.25374 | top1=0.018519, trans@3=0.111111, rot@3=0.981481, nll=8.41813 | uniq trans/rot/grip=5/1/2 |

## Evidence Boundary

- The ActionMap-style head is a local CPU approximation of the voxel heatmap decoder idea, not the official VLA training recipe.
- The oracle candidate row is a discretization upper bound and is invalid as method evidence.
- Failed or weak reproduction should be reported before any new extension is proposed.
