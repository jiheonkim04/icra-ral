# Decision Log

## 2026-07-08: Research Reset And Target-Grounded ActionMap Scout

Decision: `NEED_ACTIONMAP_ANCHOR_REPRO_FIRST`

Reason: the only salvageable family was Target-Prior TCA reframed as Target-Grounded ActionMap / Language-Grounded Action Heatmap, but the local ActionMap substrate had not cleared mean-action, linear/L1, and cheap-MLP gates.

## 2026-07-08: ActionMap Mini-Anchor Gate

Decision: `KILL_ACTIONMAP_ANCHOR`

Reason: the bounded LIBERO/HDF5 mini-anchor produced real metrics but failed the hard gate.

Key metrics:

- dataset/split: `8` local LIBERO HDF5 demos, `deterministic_per_demo_time_holdout`
- train/eval records: `1008 / 432`
- mean-action action L2: `0.466767673`
- linear/L1 action L2: `0.812610317`
- simple MLP action L2: `0.501926707`
- ActionMap-style action L2: `0.529931357`
- oracle candidate action L2: `0.065653208`

Triggered kill criteria:

- mean-action baseline matches or beats the ActionMap-style heatmap head;
- cheap MLP action head matches or beats the ActionMap-style heatmap head;
- ActionMap-style head collapsed to too few candidates.

Consequence: do not proceed to Target-Grounded ActionMap in this run. The exact next step is to archive this local mini-anchor result or plan an official-style ActionMap reproduction/source gate as a separate task.

Execution boundary:

- experiments happened: yes, bounded mini-anchor diagnostic only;
- training happened: yes, tiny CPU NumPy heads only;
- loss computed: yes;
- rollout/replay happened: no;
- downloads/GPU/OpenVLA-OFT happened: no / no / no;
- full official ActionMap reproduction happened: no;
- Target-Grounded ActionMap implementation happened: no.
