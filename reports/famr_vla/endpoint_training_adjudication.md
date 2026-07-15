# FAMR-VLA Endpoint Training Adjudication

Date: 2026-07-15 KST

Decision: `FAMR_ENDPOINT_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE`

## Integrity

- the detached worker completed normally with exit code `0`;
- optimizer steps completed/planned: `300 / 300`;
- microbatches completed/planned: `2400 / 2400`;
- exceptions: `0`;
- source-key and row-key duplicates: `0 / 0`;
- task counts: `800 / 800 / 800`;
- source episodes: discovery `0-34` only;
- validation and test decodes: `0 / 0`;
- confirmatory observations and actions: `0 / 0`;
- checkpoint file-hash mismatches: `0`.

## Passed Gates

The fixed-subset loss decreased from `0.7321685557253659` to
`0.1628471128642559`, a relative reduction of `0.7775824820789773`.
All `300` optimizer steps had finite nonzero gradients. The endpoint differed
from Base above `1e-4` on `24 / 24` fixed discovery rows. Base parameters were
unchanged, disk reload error was `0.0`, peak CUDA allocation was
`1.0808053016662598 GiB`, and the frozen schedule and checkpoint hashes passed.

## Failed Gate

The frozen Base-relative numerical action-validity gate failed:

- endpoint outside-`[-1,1]` fraction: `0.1130952380952381`;
- permitted fraction: `0.08738095238095238`;
- endpoint p99 exceedance: `0.09376012921333322`;
- permitted p99 exceedance: `0.04096377015113834`.

Finite fraction `1.0` and absolute maximum `1.1066012382507324 <= 1.25`
passed, but every action-validity component was required. Simulator acceptance
was not attempted because the offline numerical gate had already failed.

## Scientific Classification

This is an `IMPLEMENTATION_OR_DATA_FAILURE` under the preregistered
false-negative calibration, not a closed-loop scientific kill. It does not
show that function-aware model retention is scientifically ineffective.

No clipping, action postprocessing change, threshold change, coefficient
search, headroom rollout, validation search, or confirmatory evaluation is
allowed as a rescue of this endpoint. Preserve the result and continue to the
next method cycle.
