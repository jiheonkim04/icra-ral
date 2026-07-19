# A2C2 fidelity-corrected actual-path smoke: attempt 1

Date: `2026-07-19 KST`

Classification: `PRIOR_ACTUAL_PATH_PREFLIGHT`

Fidelity label: `A2C2_FIDELITY_CORRECTED_LOCAL_PORT`

Decision: `A2C2_CORRECTED_ACTUAL_PATH_SMOKE_FAIL`

The first actual-path smoke stopped during strict prior-checkpoint loading. It
persisted and counted no task-success outcome. The public prior contains a
`[512,512]` linear image projection, while the later frozen repository source
constructs `[512,512,1,1]` after changing that projection to a 1x1 convolution.

Repository history provides a unique, outcome-independent serializer repair.
The public checkpoint strict-loads against the author's immediately preceding
commit `c197a011aabf070cf2c0b2b0705be5f33d178ad7`; the next author commit
`75fa9d2` introduces the incompatible projection change. The rerun therefore
uses that exact author tree with `strict=True`. It does not reshape tensors,
use non-strict loading, replace the checkpoint, or alter the frozen scientific
protocol.

Resource evidence was clean: CUDA PID `289`, peak allocated VRAM `1532.542`
MiB, peak process RSS `2941.32` MiB, Windows peak physical-use fraction
`0.6626`, zero WSL swap, zero pagefile growth, no kernel OOM, and verified
post-shutdown memory release.

The complete immutable attempt is preserved under
`runs/a2c2_fidelity_corrected/a2c2_corrected_smoke_20260719t1807`.
