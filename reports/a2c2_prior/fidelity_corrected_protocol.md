# Frozen Corrected A2C2 Protocol

Date: `2026-07-19 KST`

Label: `A2C2_FIDELITY_CORRECTED_LOCAL_PORT`

Audit gate: `A2C2_OBJECTIVE_FIDELITY_DEFECT_FOUND`

This is the single permitted correction. It does not alter or reinterpret v1.
It pins the author's Spatial-scratch base at
`caa0efcb24e261574c824366526c5775d3664cac`, the six-layer
`add_vlm_context` residual checkpoint at
`9c89cca4aae8eecc42a20084ef414ff74f94ba05`, and the frozen official source
at `54dd088302a0ef3f50c4add3ec927ab94d76a406`. It also applies the official
180° rotation to both live RGB views. No correction-head training is allowed.

The technical smoke uses task `2`, official init state `10`, and is not a
scientific result. The verification panel preserves tasks `0,4,8` but uses
new official init states `5,6,7,8,9`, with no overlap with v1 or the smoke.
Its three 15-row arms remain Base `e=10,d=0`, Base `e=40,d=10`, and corrected
Prior `e=40,d=10`; max episode length remains 220 and success remains official
LIBERO `check_success`.

Execution is sequential with one full base residency, WSL memory 12 GB,
`swap=0`, batch/task parallelism 1, atomic row persistence, and missing-key
resume only. The previous 8/10/12 GB qualification is not repeated; the new
paired loader receives one bounded compatibility/resource smoke.

The allowed final decisions are exactly:

- `CORRECTED_A2C2_PRIOR_IMPROVES_AND_LEAVES_RESIDUAL`
- `CORRECTED_A2C2_PRIOR_SATURATES_DELAY`
- `CORRECTED_A2C2_PRIOR_NO_IMPROVEMENT`
- `CORRECTED_A2C2_BASE_NOT_COMPETENT`
- `CORRECTED_A2C2_EVALUATION_INVALID`
- `CORRECTED_A2C2_IMPLEMENTATION_OR_RESOURCE_FAILURE`

There is no second fidelity correction. Additional-Prior selection, Ours, and
paper generation remain forbidden until the corrected decision permits their
respective gate.
