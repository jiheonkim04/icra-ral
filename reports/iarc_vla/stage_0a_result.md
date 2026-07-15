# IARC-VLA Stage 0A Result

Date: 2026-07-15 KST

Decision: `IARC_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`

Proposal hash: `A1B0CF8BCBCF6A88F27B31EF5E38BAF408A3E62BB34206A1AC9F051EA6B57408`

## Frozen Audit

- micro-fit steps: `20 / 20`
- fixed-subset loss: `0.0959141970379278` -> `0.09489042789209634`
- conflict pairs: `18 / 40`
- activating families: `['context_wrapper', 'gaussian_sensor_noise', 'image_translation', 'instruction_repetition']`
- projection constraints passed: `18 / 18`
- validation dataset-range action validity: `0.3`
- checkpoint reload error: `0.0`
- confirmatory observations/actions: `0 / 0`
- peak CUDA GiB: `1.0882797241210938`
- timing paper eligible: `True`

## Boundary

Stage 0A did not satisfy its frozen gate; no threshold, perturbation, rank, optimizer, or row rescue is allowed.

Next command: `Adjudicate the frozen failure under the false-negative safeguard and continue to the next method cycle.`
