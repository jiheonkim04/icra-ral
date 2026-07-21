# Epoch 9D bounded controller failure

Terminal state: `ACTIVE_DYNAMIC_PROBE_SIGNAL_CONFIRMED_TASK_PRESERVATION_NOT_ACHIEVED`.

The causal mass signal remains established, but Variant 1 did not preserve the task under its sealed pilot gate. It achieved 24/24 finite bounded probes, 24/24 intended contacts or excitations, 12/12 scenes with both candidates excited, 12/12 oracle completions, and zero collision, identity-swap, fall, or workspace-exit events. It failed lane/reachability at 20/24, overall ranking at 9/12, and back-heavy ranking at 3/6.

The four failed lane audit rows reduce to two causal exits and two inherited failures. Both causal exits first crossed a lane boundary during `contact_verify_retract`, before the fixed response window. Their triggering RGB signed margins were below 10 mm, not in the frozen `(10, 14]` mm adjustment interval. The adjustment is therefore ineligible.

Variant 2 is also unauthorized. The evidence does not show post-probe pose/restoration as the limiting cause, and a recovery stage after the response window cannot repair the two failed ranking gates without changing the frozen score path. Unused numerical search budget does not create authority to rotate controllers.

No fresh 24-scene feasibility panel, estimator training, validation, official closed loop, confirmation, or paper package is authorized. Validation and confirmation identities remain untouched.
