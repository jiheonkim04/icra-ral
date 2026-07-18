# Action-Consistent Missing-View Telemetry Device Repair

Decision: `TELEMETRY_DEVICE_REPAIR_SMOKE_VALID`

Classification: `EXCEPTIONAL_TELEMETRY_DEVICE_REPAIR`

The patch is limited to CUDA telemetry device normalization. It calls
`torch.cuda.current_device()` to initialize CUDA and obtain the actual integer
index, constructs the model device from that index, and uses the same index
for reset, synchronization, device properties, and peak-memory telemetry. It
is neither a `METHOD_REPAIR` nor a `SCIENTIFIC_REDESIGN`, and it does not reset
the general repair budget.

The telemetry-only smoke ran on CUDA device `0`, `NVIDIA GeForce RTX 5080`,
under PyTorch `2.10.0+cu128` / CUDA `12.8`. It allocated a live 4,096-byte CUDA
tensor, verified no CPU fallback, read current and peak allocated/reserved
memory, freed the tensor, emptied the cache, and exited with code `0`.

- Peak allocated: `4,608` bytes
- Peak reserved: `2,097,152` bytes
- Allocated after free: `0` bytes
- Reserved after empty-cache: `0` bytes
- X-VLA loaded: `False`
- Discovery/validation/confirmatory output accessed: `False`
- Optimizer steps: `0`
- Targeted tests: `10 passed`

The historical `STAGE0_IMPLEMENTATION_OR_RESOURCE_FAILURE` and paper-level
`IMPLEMENTATION_DATA_OR_RESOURCE_FAILURE` remain preserved as the
pre-resumption outcome. The next authorized action is only the unchanged
numerical-noise calibration.
