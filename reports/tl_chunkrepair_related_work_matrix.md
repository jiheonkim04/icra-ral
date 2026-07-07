# TL-ChunkRepair Related Work Matrix

| Area | What It Covers | Gap TL-ChunkRepair Tests |
| --- | --- | --- |
| VLA action chunking | Chunk latency, horizon, reactivity, visual feedback | Does not explicitly repair temporal safety/property violations inside a proposed chunk. |
| Action retiming | Fixed shift, time warp, phase alignment | Often recoverable by simple timing baselines; TL-ChunkRepair must beat these. |
| Runtime shielding | Clip or block unsafe actions | Can preserve safety but may destroy utility; TL-ChunkRepair must preserve progress. |
| Task monitors / temporal logic | Finite-state property tracking | Needs a concrete replay/control repair result, not only symbolic satisfaction. |
| Gripper timing fixes | Close/open phase patching | Strong baseline; TL-ChunkRepair must add value beyond gripper-only repairs. |
| Abort/replan/hold | Conservative fallback safety | Strong utility-cost baseline; TL-ChunkRepair must be safer with better progress or lower edit cost. |

Initial novelty test: not a paper claim. The only question in STATE 1 is whether temporal monitor-guided chunk edits produce real replay/control gains that simple baselines do not already explain.
