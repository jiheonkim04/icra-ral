# Official SmolVLA Canonical Checkpoint Proposal

Date: 2026-07-10 KST

Accepted as canonical: `False`
Reason: Do not canonicalize until the in-memory historical evaluation path, persisted PEFT reload path, and unpinned evaluation RNG state are fixed or explicitly re-baselined.

## Side-By-Side Old Vs Candidate Canonical Metrics

| seed | historical rank4 | historical static | regenerated rank4 | regenerated static | selected alpha |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 0.084128699 | 0.077354597 | 0.087213856 | 0.078911385 | 0.5 |
| 22 | 0.090162398 | 0.080789904 | 0.088713382 | 0.080982228 | 0.25 |
| 33 | 0.090426934 | 0.083704791 | 0.085934428 | 0.078716617 | 0.5 |

Policy if canonicalization is later approved:

- preserve old results as historical
- do not overwrite historical metrics
- use only explicitly adopted canonical checkpoint metrics in future rollout reports
- update the reproducibility lock with checkpoint hashes
- do not claim exact replication of historical ephemeral runs
