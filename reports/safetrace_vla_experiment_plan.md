# SafeTrace-VLA Experiment Plan

STATE 1A: audit official safety sources and run the smallest temporal monitor available. Prefer SafeManip or LIBERO-Safety if local; otherwise use local standard LIBERO HDF5 only as a non-paper-grade proxy.

STATE 1B: generate temporal preference pairs from monitor risk, progress, and action magnitude. Report pair count, nontriviality, no-op/stop collapse, utility loss under stop-on-risk, and whether safety-only or generic preference scoring already solves the labels.

STATE 1C: compare base/no optimization, safety-only, stop-on-risk, clipping proxy, generic preference proxy, and SafeTrace temporal preference proxy. Compute preference loss only; no weight update is needed for this feasibility gate.

Continue only if a real safety benchmark path is available, temporal metrics are observable, nontrivial pairs exist, simple baselines do not explain the effect, and utility retention is plausible.

