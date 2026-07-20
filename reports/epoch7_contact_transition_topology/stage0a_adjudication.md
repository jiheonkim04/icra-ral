# Epoch 7 contact-transition Stage 0A adjudication

Decision: `CONTACT_TOPOLOGY_LABEL_GATE_GO`

All 18 frozen label, fidelity, determinism, and scientific-firewall gates
passed on the ten-task, six-demo panel. The four validation tasks contained
181 debounced transition frames among 3,239 eligible frame pairs (5.59%). All
four validation tasks had transitions in all six demos, eight typed
birth/death bins had validation support, and 90.06% of transition frames fell
outside a plus-or-minus-two-frame window around gripper changes.

Exact replay was clean: state roundtrip error was zero, robot contact-geometry
resolution and selected-state restore fractions were 100%, retained edges
touching the robot were zero, and cold-repeat graph hashes matched 100%.
There were zero exceptions, simulator actions, forbidden dataset reads,
success checks, and outcome reads.

This result establishes a noncollapsed, deterministic training signal; it
does not establish visual predictability, action headroom, or policy benefit.
It authorizes only the already-frozen Stage 0B visual and oracle probes. No
method, training, policy rollout, or paper generation is authorized.
