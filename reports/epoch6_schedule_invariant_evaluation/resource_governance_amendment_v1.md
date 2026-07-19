# Epoch 6 Schedule Stage 0 Resource-Governance Amendment v1

Frozen: 2026-07-20 03:48:53 KST
Scope: `RESOURCE_ONLY_OUTCOME_FREE`

The clean-host resource smoke completed its one frozen X-VLA forward with a
finite `[30, 20]` result, zero WSL swap use, zero simulator actions, zero
outcome reads, and no telemetry exception. Host RAM peaked at 71.53%, below
the frozen 82% ceiling. Windows reported no page writes and no pages output.
The preserved host wrapper nevertheless returned
`EPOCH6_STAGE0_RESOURCE_SMOKE_FAIL_PAGEFILE_ACTIVITY` because pagefile
`CurrentUsage` changed from 4 MiB to 25 MiB after child teardown, while the
old bounded memory-release comparison also remained false.

No scientific gate row was produced or inspected. The failed run and its
decision remain immutable evidence; this amendment does not reinterpret it as
a pass. It prospectively governs a new immutable qualification run.

## Calibrated outcome-free qualification

Before the model child starts, Windows must observe a 60-second idle control
with WSL stopped, one sample per second, RAM no higher than 65%, and fewer than
three consecutive samples with nonzero `Page Writes/sec` or
`Pages Output/sec`. During the model child, RAM remains capped at 82% and three
consecutive paging-active half-second samples constitute sustained pressure
and terminate the run.

`Win32_PageFileUsage.CurrentUsage` is retained as diagnostic allocation
telemetry. A change in that allocation field alone is not paging evidence and
does not fail the calibrated gate. A pass still requires zero WSL swap use,
one finite CUDA forward, no OOM or kill signature, no CPU/disk model offload,
zero telemetry exceptions, zero simulator actions, and zero outcome access.

After the child exits, controlled cache drop is permitted. If the old bounded
memory-release test remains false, `wsl --shutdown` is permitted and the host
may wait up to 60 seconds for a clean next-run state. That state requires WSL
stopped, host RAM at or below 65%, GPU use within 256 MiB of the idle baseline,
and no sustained paging during restoration. Failure of this restoration is a
resource failure.

## Scientific invariants

The scientific protocol remains byte-for-byte unchanged at SHA-256
`E5BA74354A1947A00045879A4815CCD09856F127E6809CF8BF649F10E2359946`.
The model/checkpoint, task/reset identity, fixture, seeds, sequence orders,
action processing, metrics, scientific thresholds, A/A-repeat/B/C rules, and
downstream authorization gates are unchanged.

Machine-readable authority:
`reports/epoch6_schedule_invariant_evaluation/resource_governance_amendment_v1.json`.
