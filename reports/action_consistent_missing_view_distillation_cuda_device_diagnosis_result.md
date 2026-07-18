# Action-Consistent Missing-View CUDA Device Diagnosis

Decision: `CUDA_TELEMETRY_DEVICE_DEFECT_DIAGNOSED`

Classification: `EXCEPTIONAL_TELEMETRY_DEVICE_REPAIR`

The defect was reproduced in the actual WSL environment without loading
X-VLA. The environment uses PyTorch `2.10.0+cu128`, CUDA `12.8`, one visible
CUDA device, and current device `0`, named `NVIDIA GeForce RTX 5080`.
`CUDA_VISIBLE_DEVICES` is unset.

The frozen preflight passed `torch.device("cuda:0")`, a `torch.device` with
type `cuda` and index `0`. After `torch.cuda.is_available()`,
`torch.cuda.device_count()`, and `torch.cuda.empty_cache()`, the CUDA allocator
remained uninitialized. In that exact state,

```text
torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
```

reproduced `RuntimeError: Invalid device argument` and left the allocator
uninitialized.

Calling `torch.cuda.current_device()` returned `0` and initialized CUDA. After
that, all three requested forms succeeded:

- no explicit reset argument;
- integer `torch.cuda.current_device()`;
- `torch.device("cuda", torch.cuda.current_device())`.

The minimal permitted patch is therefore to verify CUDA, obtain the actual
current device index first, construct the model device from that index, and
use the validated integer index for reset and peak-memory telemetry. No
method, data, loss, threshold, optimizer, identity, comparator, or decision
rule is changed. No X-VLA, discovery output, validation outcome, confirmatory
outcome, or optimizer was accessed during diagnosis.
