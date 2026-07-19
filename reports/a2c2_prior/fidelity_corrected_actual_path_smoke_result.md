# A2C2 fidelity-corrected actual-path smoke result

Date: `2026-07-19 KST`

Execution: `PRIOR_ACTUAL_PATH_PREFLIGHT`

Implementation: `A2C2_FIDELITY_CORRECTED_LOCAL_PORT`

Final decision: `CORRECTED_A2C2_EVALUATION_INVALID`

The exact author base and public six-layer prior strict-loaded on CUDA and
completed both outcome-suppressed development traces. The Base ran 10 model
forwards; the delayed trace ran three Base forwards and 94 live prior
forwards. The prior produced a nonzero mean absolute correction of
`0.091438221`. Both views were rotated on every live query, no expert or
future action was used, and no success outcome was persisted or counted.

The frozen raw-action legality gate failed. Base reached maximum absolute
action `1.024949789`; the corrected Prior reached `1.000505567`, both beyond
the frozen `[-1,1]` action bounds. The released evaluator sends actions
without an added clip; the simulator controller internally clips, but the
frozen corrected protocol explicitly requires legal raw actions and forbids
a clipping rescue. The threshold is therefore not relaxed after observing
the development trace.

The new 45-row verification panel was not started and contains zero
scientific rows. This is evaluation invalidity, not evidence that A2C2 does
or does not improve success. It does not alter the preserved v1 result
`A2C2_PRIOR_NO_LOCAL_IMPROVEMENT` and does not disprove the paper.

Two earlier non-scientific failures remain preserved. First, the public
checkpoint's `[512,512]` image projection did not load against the later
`[512,512,1,1]` author source. It strict-loaded against the immediately
preceding author commit `c197a011...`; no tensor was reshaped and no
non-strict load was used. Second, the historical config required the author's
dataclass serialization path. Neither repair changed the scientific graph or
protocol.

Peak allocated VRAM was `1532.542 MiB`, peak process RSS `4063.348 MiB`, and
Windows physical use peaked at `65.09%`; swap, pagefile growth, offload, and
OOM were zero. The temporary `.wslconfig` was removed, WSL was shut down, and
no worker remains.

Verification completed with 40 focused A2C2 tests, the current-governance
script, scaffold tree check, and `git diff --check` passing. The separate
historical governance pytest remains at five passes and one pre-existing stale
assertion that expects epoch 4 although the authoritative campaign state was
already epoch 5 before this stage.

Under the frozen route rules, corrected evaluation invalidity authorizes
neither the additional Prior nor Ours. Do not clip/retune, launch the panel,
select another Prior, design Ours, or generate a paper package without new
explicit authority.
