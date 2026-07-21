# Epoch 9E Joint Seal Pre-outcome Git Boundary Repair

The first joint-seal build attempt stopped before producing a seal or any joint result. The WSL Git client interpreted the Windows worktree with different file-mode and line-ending rules and therefore reported repository-wide changes even though Windows Git reported the tracked worktree clean.

The repair changes only the builder's precondition: every exact file entering the seal is now compared byte-for-byte against its `HEAD` blob by SHA-256. This is stricter for the scientific execution surface and independent of cross-client metadata interpretation.

No controller, protocol, runner, adjudicator, host wrapper, threshold, scientific field, validation identity, or confirmation identity changed. No resume or rerun authority was added.
