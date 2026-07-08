# ActionMap Mini-Anchor Risk Register

Date: 2026-07-08

| Risk | Status | Mitigation / outcome |
| --- | --- | --- |
| Local mini-anchor is mistaken for official ActionMap reproduction. | controlled | Reports state this is a CPU NumPy HDF5 action-head diagnostic only. |
| Oracle candidate upper bound is mistaken for method evidence. | controlled | Oracle row is labeled invalid as method evidence. |
| Heatmap result looks promising while simple baselines are stronger. | triggered | Mean action and cheap MLP matched or beat the learned ActionMap-style head, causing `KILL_ACTIONMAP_ANCHOR`. |
| Candidate predictions collapse. | triggered | Unique translation/rotation/gripper bins were `5 / 1 / 2`; rotation collapsed. |
| Run drifts into Target-Grounded ActionMap implementation. | avoided | No target-grounded method code was added. |
| Run drifts into heavy assets, GPU, OpenVLA-OFT, or rollout. | avoided | Script refuses dangerous gates and uses local HDF5 plus tiny CPU NumPy training only. |

## Residual Risk

This result does not disprove the official ActionMap paper. It only kills the local mini-anchor as a sufficient substrate for Target-Grounded ActionMap. Any future revival would need an official-style ActionMap reproduction/source gate, not another local proxy extension.
