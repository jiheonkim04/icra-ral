# RA-L Strategy Reset

Decision: stop local proxy-first method invention as the default route.

Local proxy-first method invention has repeatedly failed. Recent routes produced plausible diagnostics, local proxy metrics, or working scaffolds, then collapsed under stronger simple baselines such as mean-action, safety-only, diagonal affine, global scale, canonicalization-only, no-geometry, active single-point, cheap MLP, and generic preference labels.

No new method topic should be started from proxy diagnostics. A local proxy may be useful as a smoke test only after an official anchor has been reproduced; it is not a topic generator.

RA-L-stable work now requires official benchmark/source reproduction first. The next step should be exactly one official anchor reproduction route:

| Option | Route | Purpose |
| --- | --- | --- |
| A | SafeManip official benchmark reproduction | Reproduce temporal safety monitor evaluation on official SafeManip assets before any temporal-safety method. |
| B | LIBERO-Safety official benchmark reproduction | Reproduce the official LIBERO-Safety evaluation/data path before any safety-alignment method. |
| C | ForesightSafety-VLA reproduction | Reproduce process-level risk metrics such as cumulative safety cost and risk exposure time. |
| D | ActionMap reproduction | Reproduce the official action-decoder anchor before any action-head extension or failure mining. |
| E | VLA-Corrector reproduction | Reproduce an official action-chunk correction baseline before any correction/repair method. |

Recommended first anchor: SafeManip official benchmark reproduction, because the current killed route targeted temporal safety and SafeManip is the most direct official temporal-safety monitor anchor. If SafeManip setup is source-blocked, the next fallback is LIBERO-Safety official benchmark reproduction.

No custom method should be proposed, implemented, trained, or benchmarked until one official anchor baseline is reproduced with its own documented source, license/access status, expected size, local availability, tasks, metrics, and simple baselines.

Execution boundary for this reset: documentation only. No experiment, training, rollout, download, GPU job, OpenVLA-OFT, new method topic, or paper claim occurred.

