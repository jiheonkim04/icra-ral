# A2C2 Panel Host-Telemetry Failed Attempt

Date: 2026-07-19 KST

The first 45-row official-semantics panel attempt completed internally, but its
scientific decision is quarantined and not adopted. The inherited host monitor
returned `A2C2_CORRECTED_HOST_FAIL_MEMORY_OR_PAGEFILE` solely because the
system-wide Windows pagefile `CurrentUsage` reservation counter rose from 72
to a peak of 78 MiB.

Across all 872 host samples, both `PageWrites/sec` and `PagesOutput/sec` peaked
at zero. WSL swap, model offload, kernel OOM, and the 82% host ceiling were
also zero/false; memory release succeeded. Thus the monitor conflated a
system-wide reservation-counter drift with actual paging. The active resource
contract freezes the verified 12 GB WSL cap, swap zero, one residency, and no
offload; it does not freeze the unrelated Windows `CurrentUsage` counter.

This failed attempt remains preserved with its exact raw hashes and its
internal candidate decision `CORRECTED_A2C2_NO_REPEATABLE_DELAY_GAP`. That
candidate is not counted until an identical rerun completes under the repaired
telemetry classifier.

Exactly one root-cause-bounded logging/telemetry repair records `CurrentUsage`
drift separately and detects pagefile activity from sampled `PageWrites/sec`
or `PagesOutput/sec`. No checkpoint, action path, task, identity, delay,
timeout, success rule, outcome threshold, or adjudicator changes. The full
45-row panel must rerun from zero rows under a new run id.
