# Epoch 7 contact-transition resource-rule amendment

Frozen: 2026-07-20 17:00 KST

Scope: `OUTCOME_FREE_HOST_RESOURCE_QUALIFICATION_ONLY`

The scientific protocol remains exactly
`reports/epoch6_contact_transition_topology/problem_verification_protocol.json`
at SHA-256
`7FA28AAEEAC9886F36DD5CCD059CA7AC4CD65B21FABFBBCA4AFFA53B0A256240`.
No task, demo, contact definition, split, label gate, headroom gate, or decision
threshold changes.

## Why the archived resource decision is reopenable

All four Epoch 6 smokes executed the actual one-environment simulator path,
used zero WSL swap, exposed zero scientific label rows and no outcomes, and
reported no pagefile write activity. Their peak host-memory fractions were
0.591--0.610. The failures instead came from 0--1 MiB pagefile-allocation
jitter, PowerShell exit-code retention, or a strict immediate cache-release
bound. Those observations do not scientifically adjudicate contact-label
prevalence, predictability, or action headroom.

## Amended host qualification

- Baseline physical-memory use must be at most 70%; peak use must be at most
  85%.
- WSL swap must remain exactly zero and free disk must remain at least 60 GB.
- Allocation-only pagefile growth up to 16 MiB is nonfatal. Any sampled
  positive `PageWritesPersec` or `PagesOutputPersec` value is fatal.
- After the child exits, the monitor may drop reclaimable WSL caches without
  shutting down WSL. Used physical memory must return to within 2 GiB and GPU
  use to within 256 MiB of baseline.
- The internal actual-path smoke must pass with zero simulator actions, zero
  success checks, no reward/done read, and zero contact-label gate rows.

The machine-readable JSON is authoritative. Its hash is recorded by the
monitor and by the Stage 0A runner. A passing smoke authorizes only the
already-frozen Stage 0A label extraction; it does not authorize Stage 0B,
training, rollout, or paper work.

Frozen JSON SHA-256:
`7CCDCE5D9AA0B24C356AF873D0481AF76312D3C7FCF6871C4CA80FD6621ACFEB`.
