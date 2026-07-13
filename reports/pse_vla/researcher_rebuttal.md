# PSE-VLA Researcher A Rebuttal

Date: 2026-07-12 KST

Reviewer B is correct that PSE is only valuable if ensemble averaging itself helps closed-loop control. The experiment therefore accepts the best single transform and duplicate-clean ensemble as decisive baselines.

PSE remains worth one prototype because it tests a different axis from SCVC. SCVC attempted to recover a canonical clean image distribution under a fixed shifted condition. PSE does not canonicalize, calibrate, train, or invert a shift; it asks whether the frozen action generator has useful nuisance-view diversity that can be aggregated directly in official 7D action space.

The implementation will use stateless first-action-chunk prediction for every transform. This prevents action-queue artifacts from masquerading as an ensemble effect.

Decision: proceed to preregistered prototype. Any match by `bright_single`, `dark_single`, or `pse_duplicate_clean` kills the method.
