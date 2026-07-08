# ContactSet-VLA Failure Tree

Root failure: ContactSet-VLA is not a valid continuation route before full VLA fine-tuning or replay scale-up.

## Branch 1: Anchor Baseline Failure

- Full contact-set action L2: `1.105028754`.
- Active single-point action L2: `0.930495702`.
- Triggered kill criterion: full contact set did not beat active single-point injection.

## Branch 2: Simple Baseline Dominance

- No-geometry action L2: `0.851451`.
- Destination-only action L2: `0.86372`.
- Source+destination and source-only were also tested.
- Triggered kill criterion: simple point/no-geometry baselines matched or beat the proposed richer point set.

## Branch 3: No Replay Confirmation

- Replay/control metric happened: no.
- Exact-init replay/progress was intentionally not started because the offline anchor-baseline gate already failed.
- Triggered risk: no downstream control evidence exists to rescue the weaker action metric.

## Branch 4: Method Complexity Without Local Gain

- Geometry was observable and leakage-audited, so the failure was not simply missing local data.
- The richer set encoder added source/support/safety/normal roles but did not improve held-out action quality.
- A larger VLA fine-tuning run would be premature because the first local action-head diagnostic already favored simpler baselines.

## Surviving Positive Signal

The extraction and diagnostic path is reusable. It can audit future action-head geometry ideas against active single-point, source-only, destination-only, source+destination, and no-geometry baselines before any expensive training.

