# A2C2 Resource Smoke Environment Failed Attempt 1

Decision: `A2C2_RESOURCE_SMOKE_FAIL_UNRELATED_IMPLEMENTATION`

The resource-only wrapper used the lightweight `tca_map_sim` test environment,
where `peft` is absent, rather than the previously accepted
`official-smolvla-libero` environment. The model and simulator did not load,
no forward or step occurred, and no task outcome was exposed.

The one repair for this root changes only the wrapper executable to the
verified existing environment, which reports PyTorch `2.10.0+cu128` and PEFT
`0.19.1`. The frozen runner and scientific protocol remain byte-identical.
