# ContactSet-VLA Kill Summary

Decision: kill ContactSet-VLA before replay scale-up or full VLA fine-tuning.

## Original Hypothesis

ContactSet-VLA proposed that a structured source/destination/support/safety/normal contact set injected into a VLA action head would improve contact-rich and multi-stage manipulation beyond a single grounded 3D point. The route was anchored on recent direct 3D point action-head injection evidence, but extended the representation from one target point to a role-tagged contact set.

## Strongest Positive Evidence

- The local diagnostic extracted source object, destination/support, safety, and normal proxy points from `6` local LIBERO HDF5 demos.
- Geometry extraction used HDF5 observations, HDF5 simulator state, and embedded MuJoCo XML without reward, success, eval-label, or task-id target leakage.
- All required variants ran: no geometry, active single point, source-only, destination-only, source+destination, and full contact set.
- A real dataset-backed held-out action-head metric was produced.
- Tiny CPU NumPy action-head fitting ran and loss was computed.

## Decisive Negative Evidence

- Full contact-set action L2 was `1.105028754`.
- Active single-point action L2 was better at `0.930495702`.
- No-geometry action L2 was better at `0.851451`.
- Destination-only action L2 was better at `0.86372`.
- Simple point/no-geometry baselines matched or beat the full contact set.
- Replay/control metric did not happen.

## Exact Kill Criterion Triggered

The anchor-baseline gate failed: full contact-set injection did not beat active single-3D-point injection, and simple point/no-geometry baselines matched or beat it on the first held-out action metric.

## Why Single-Point And Destination-Only Baselines Kill The Novelty

The proposed novelty was not merely to expose geometry to the action head; it was that a structured contact set provides useful extra information beyond the single-point action-head injection anchor. When active single-point and destination-only injection perform better, the extra source/support/safety/normal roles add complexity without measurable local action-quality value. Scaling this to VLA fine-tuning would test whether a larger learner can ignore or absorb a weak geometry encoding, not whether ContactSet-VLA has a method-level advantage.

## Why Not RA-L-Stable

A RA-L-stable route needs an early baseline gap before compute-heavy training or replay scale-up. ContactSet-VLA produced executable infrastructure, but the main method lost to simpler baselines before any simulator metric. This fails the repository baseline-first rule and should not continue as the main research direction.

Execution boundary for this archive: documentation only. No new experiment, replay, rollout, training, loss computation, GPU job, download, heavy VLA import/model load, OpenVLA-OFT execution, or paper claim occurred.

