# Phase-Locked Retiming Related Work Matrix

| Area | Local Reading | Gap For This Route | Baseline Risk |
| --- | --- | --- | --- |
| VLA safety/evaluation | Recent work emphasizes broad safety risks and evaluation mechanisms. | This route isolates temporal chunk phase mismatch as an executable replay/control failure. | A safety-only or clipping-only rule must not explain recovery. |
| Action-space design | Recent action-space work shows representation and calibration choices are often decisive. | This route must show timing value beyond action scaling or diagonal affine calibration. | Global scale or diagonal affine can kill the claim. |
| Action chunking/latency | Recent chunking work studies fast execution and timing robustness. | This route targets event-locked retiming under task phase slippage in LIBERO replay. | Fixed shift or linear time warp can kill the claim. |
| Diffusion/action policies | Chunk policies can be corrected or guided, but first evidence here is replay/control, not training. | No full policy training is needed for the first result. | Offline-only action matching is invalid. |
| Demo/progress matching | Demo progress can be a strong trivial baseline. | Event anchors must add value beyond nearest-progress demo lookup. | Nearest-progress demo can kill the claim. |

Source pointers are carried from `reports\next_topic_candidates_v2.md`; this STATE 0 is not a paper claim.
