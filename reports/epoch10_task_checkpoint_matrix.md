# Epoch 10 task and checkpoint matrix

## Checkpoints

| Whole seed | Partition | Predeclared stages | Saved identities | Fresh disk action reload |
|---:|---|---|---:|---|
| 101 | development | 10, 30, 100 | 3 | 3/3 pass |
| 202 | development | 10, 30, 100 | 3 | 3/3 pass |
| 303 | official holdout | 10, 30, 100 | 3 | 3/3 pass |
| 404 | official holdout | 10, 30, 100 | 3 | 3/3 pass |

The frozen base is a named reference and not counted as an independent training run. Adjacent snapshots remain clustered by whole-seed lineage. Prospective simulator outcomes were never used to create or curate this panel.

## Tasks and state partitions

| Suite | Task | Family | Mechanics states | Development states | Held-out states | Exact preflight restore |
|---|---|---|---:|---:|---:|---:|
| LIBERO-Spatial | black bowl between plate/ramekin → plate | spatial pick-place | 8 | 12 | 12 | 32/32 |
| LIBERO-Object | alphabet soup → basket | object pick-place | 8 | 12 | 12 | 32/32 |
| LIBERO-Goal | open middle drawer | articulated fixture | 8 | 12 | 12 | 32/32 |
| LIBERO-10 | turn on stove and place moka pot | long-horizon multi-goal | 8 | 12 | 12 | 32/32 |

Whole demonstration IDs `0/1`, `2/3/4`, and `5/6/7` define mechanics, development, and held-out state partitions. Frames at 15%, 35%, 55%, and 75% of each trajectory stay with their whole demonstration. Closed-loop initial-state indices `20–34` and `35–49` were frozen for development and official evaluation but never opened because the mechanics assay failed first.
