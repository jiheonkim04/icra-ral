# AMP-GD Risk Register

| risk | severity | mitigation |
| --- | --- | --- |
| Toy diagnostic overstates value | high | label toy evidence, require clear LIBERO/RoboSuite path before State 2/3 claims |
| Random probes match planned probes | high | predeclare random-probe baseline and kill if matched |
| Safety-only explains improvement | high | include safety-only/clipping-only and kill if matched |
| Probe helps by stopping | high | require post-probe commitment and report probe/commit rates |
| Hidden labels leak into policy | high | use labels only for evaluation; policy uses instruction, geometry, and probe observation |
| Native VLA competence blocks first metric | medium | State 1 avoids VLA loading and uses scripted control |
| Broad report sprawl before metric | medium | cap State 0 docs and immediately run State 1 |

