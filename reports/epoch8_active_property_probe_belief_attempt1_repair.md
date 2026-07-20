# Probe-Belief Stage-0 Attempt 1 Repair

Attempt 1 decision: `PROBE_RESPONSE_BELIEF_STAGE0_INVALID`.

All 18 selected pairs completed with target contact and at least 30 response
frames, but training `middle/demo_3` had unequal executed-action hashes. The
implementation stopped each condition 30 steps after its *own* first contact;
that contact occurred at a different step under heavy mass. This contradicted
the protocol's explicit requirement that paired probes use identical action
hashes.

The repair is semantically null with respect to the intended standardized
probe: select the action-prefix length from the light replay, then execute that
exact prefix under heavy mass. No feature, label, split, mass factor,
classifier, threshold, or scientific gate changes. The invalid result remains
preserved at `reports/epoch8_active_property_probe_belief_stage0.json`; the
repair writes a new result file.

Scientific outcomes were visible before repair: the invalid run's absolute
validation accuracy was 0.50, while all six paired heavy scores exceeded their
light scores. Those observations cannot authorize a classifier or gate change.
