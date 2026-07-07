# ContactTube-Aug Kill Criteria

Kill or reframe before STATE 2 if any condition holds:

- object/contact state is unavailable from both HDF5 and bounded replay traces,
- contact-tube extraction is unreliable,
- augmented trajectories are not controller-valid or replay-valid,
- no real replay/control metric is produced,
- random action jitter matches or beats ContactTube-Aug on tube preservation,
- random pose jitter matches or beats ContactTube-Aug on tube preservation,
- simple object-relative translation retargeting matches or beats ContactTube-Aug,
- raw/default replay already explains the result,
- the evidence is only image augmentation, metadata relabeling, or offline planning.

Continue only if:

- a contact tube is extractable,
- ContactTube-Aug replay executes without controller/action validity failure,
- replay/control metrics are written,
- ContactTube-Aug beats random jitter, random pose jitter, and simple object-relative retargeting on the predeclared tube-preservation metrics.

