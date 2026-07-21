# Epoch 9E Fail-Fast Root Cause and Scope Correction

Prospective state: `EPOCH9E_JOINT_EXECUTION_INTERRUPTED_BY_FAIL_FAST_RUNNER_DEFECT`; `ACTIVE_ROUTE_NOT_YET_ADJUDICABLE`; `PAPER_NOT_AUTHORIZED`.

The immutable scientific miss is `primary:epoch9e_joint_base_20261134_assignment_B`, back probe. Physical contact occurred, but ordinary RGB verification stayed at `0.546384--0.548244 px`, below the frozen `0.55 px` transition threshold, so the trace contains zero frozen response-window steps.

The separate implementation defect is at `scripts/run_epoch9e_joint_certification.py:489--490`: every failed row raised and aborted the batch even though the aggregate gates had remaining miss allowance. Historical artifacts remain untouched. Base `20261134` is never rerun or replaced; Assignment B stays failed/incomplete, and the pair is adverse/nonflip for binary evidence but absent from physical-contrast means and intervals.
