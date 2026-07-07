# AMP-GD Related Work Matrix

This is a concise positioning note, not a broad survey.

| area | examples | relation to AMP-GD | first gate |
| --- | --- | --- | --- |
| VLA execution and adaptation | OpenVLA, SmolVLA, OpenVLA-OFT | strong policy/action baselines; AMP-GD must not assume native competence | avoid model loading in State 1 |
| fine-grained instruction following | FineVLA-style language grounding | improves passive interpretation; AMP-GD tests active evidence gathering before commitment | compare against no-probe/nearest |
| runtime safety filtering | SafeVLA, path-consistent filters, CSS-Shield-style controls | can reduce unsafe actions but may not disambiguate targets | compare against safety-only/clipping-only |
| action chunk correction | RTC, A2C2, VLA-Corrector-like correction | reacts to stale or bad chunks; AMP-GD acts before target commitment | compare against random probe |
| active perception/control | active sensing, information gain, tactile/visual probing | closest mechanism: choose a safe action for information, not task progress | measure entropy reduction and wrong-target rate |

Novelty risk: if random probes, nearest-target, or safety-only controls match AMP-GD on rollout/control metrics, the route should be killed or reframed.

