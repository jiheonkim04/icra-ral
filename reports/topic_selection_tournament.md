# Topic Selection Tournament

Date: 2026-07-07

Purpose: choose the next RA-L candidate after archiving Target-Prior TCA-Map, CSS-Shield, and ExecSpec-Repair.

No experiments, training, rollout, replay, downloads, GPU jobs, heavy VLA imports, OpenVLA-OFT execution, loss computation, or paper claims happened during this topic-selection step.

## Hard Entry Constraints

Every candidate must satisfy these before implementation:

- produce a rollout, replay, or direct control metric within 48 hours,
- beat a simple baseline within 72 hours,
- avoid offline-only proxy evidence as the primary result,
- avoid relying on native VLA competence unless verified first,
- include clipping-only, safety-only, mean-action, and diagonal-affine baselines where relevant,
- define kill criteria before code is written,
- provide a clear robotics evidence path.

## Recent Paper Pressure Points

The literature check makes three traps obvious:

- Strong VLA and action-decoding work already exists: OpenVLA, SmolVLA, and OpenVLA-OFT show that modern VLAs and fine-tuning recipes are strong baselines, not empty strawmen. Sources: [OpenVLA](https://arxiv.org/abs/2406.09246), [SmolVLA](https://arxiv.org/abs/2506.01844), [OpenVLA-OFT](https://arxiv.org/abs/2502.19645).
- Action chunk reactivity is crowded: RTC, A2C2, and VLA-Corrector already target stale action chunks and adaptive correction. Sources: [RTC](https://arxiv.org/abs/2506.07339), [A2C2](https://arxiv.org/abs/2509.23224), [VLA-Corrector](https://arxiv.org/abs/2607.01804).
- Safety and recovery are also crowded: SafeVLA, path-consistent safety filtering, and retry-supervised value learning cover major parts of safe deployment and recovery. Sources: [SafeVLA](https://arxiv.org/abs/2503.03480), [PACS](https://arxiv.org/abs/2511.06385), [ReTVL](https://arxiv.org/abs/2606.24633).

## Candidate Ranking

| rank | candidate | 48h metric likelihood | 72h baseline gap likelihood | novelty risk | RA-L stability | recommendation |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Active Micro-Probe Goal Disambiguation | high | medium-high | medium | high if probe beats no-probe/random-probe | recommended |
| 2 | Demonstration-Tube Recovery Library | high | medium | medium | medium-high if recovery beats nearest/mean baselines | backup |
| 3 | Path-Consistent Event-Triggered Chunk Guard | medium | medium | high | medium if it beats RTC-like and safety-only controls | risky due crowded literature |

## Recommended Next Topic

Recommended: Active Micro-Probe Goal Disambiguation.

Why: it is the most different from the killed routes. It is an active control topic, not an offline proxy, not a semantic shield, and not another calibration layer. It can generate failures quickly by constructing ambiguous-target scenes or object-pair tasks, it has obvious simple baselines, and it can use scripted probes before relying on any VLA competence.

First implementation gate, if selected later: produce a bounded simulator/control metric comparing no-probe, random-probe, heuristic-probe, and active micro-probe variants. Kill immediately if random-probe or no-probe matches it within tolerance.

## Tournament Decision

Start with Candidate 1 unless a later pre-implementation literature pass finds a directly overlapping active-probing VLA/control paper. Keep Candidate 2 as the fallback if micro-probes cannot be made safe or measurable in the local simulator within 48 hours. Do not start Candidate 3 first unless the novelty can be narrowed sharply against RTC, A2C2, VLA-Corrector, and PACS.

