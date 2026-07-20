# Epoch 9B Metric Integrity Audit

Date: 2026-07-21T00:52:21+09:00

Decision: `VISUAL_METRIC_TARGET_CROP_DEFECT_CONFIRMED`

## Finding

The Epoch 9 visual return metric did not observe the manipulated front bowl. Its fixed 17x17 template was centered at `(95, 62)`, while the preserved moved-bowl support is in `[81, 15, 102, 39]`. The two regions do not overlap. The metric therefore returned a near-perfect zero shift on static pixels even though the evaluation-only body displacement was `0.041935 m`.

The saved initial and final frames have distinct SHA-256 digests and contain `422` changed workspace pixels above the 10-level threshold. This rules out cached-frame reuse and confirms that the runner used distinct before/after images. The defect is the target crop, not dtype conversion, end-effector return, or a rewritten historical result.

Historical visual-return values remain unchanged in their original files. Any front-slot value computed with the legacy crop is labelled unreliable for object return.

## Deliberate translation test

Development identity 37 was reset, each target bowl free joint was translated along world +x by 0, 1, 2, 3, and 4 cm, and the ordinary 128x128 agent-view image was rerendered. No sealed identity or controller outcome was accessed.

| slot | audited pixel magnitudes for 0/1/2/3/4 cm | monotonic | responsive |
|---|---:|---:|---:|
| front | 0.000, 2.000, 3.162, 5.099, 6.083 | True | True |
| back | 0.000, 1.000, 2.000, 3.000, 4.123 | True | True |

The audited centers are `(92, 28)` for the front candidate and `(71, 60)` for the back candidate in the unmodified image-array orientation. The tracker records its center, template bounds, effective search radius, template texture, and confidence on every call.

## Distribution reporting

The machine-readable result contains per-episode, per-probe, and per-identity displacement distributions for the key preserved Epoch 9 regimes. Each includes count above the unchanged 3 cm legacy reference, mean, median, standard deviation, 5/25/50/75/95% quantiles, extrema, and a deterministic bootstrap 95% interval for the mean.

## Integrity boundary

- Validation identities 40..44: not accessed.
- Confirmation identities 45..49: not accessed.
- Old reports and frame files: read only.
- Simulator pose: used only to impose known evaluation translations and to retain the historical continuous displacement reference; never exposed to a controller or learned inference path.
