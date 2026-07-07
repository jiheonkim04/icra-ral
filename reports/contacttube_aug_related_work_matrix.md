# ContactTube-Aug Related Work Matrix

| Area | What It Explains | ContactTube-Aug Difference | STATE 1 Baseline Pressure |
| --- | --- | --- | --- |
| Image-only augmentation | Visual robustness without action physics | Must produce valid 7D action trajectories | not enough without replay metric |
| Random action jitter | Local action noise robustness | Should preserve contact timing and object-motion events | direct baseline |
| Random pose jitter | Pose-shift diversity | Should preserve EEF-object tube, not just perturb commands | direct baseline |
| Object-relative retargeting | Translate EEF path by object delta | Contact phases use current object-relative tube and gripper timing | must beat or kill |
| Dataset relabeling/object swap | Novel language/object labels | No physical contact validity by itself | logged as not run in STATE 1 |
| DAgger/new teleop | Adds real corrected demonstrations | ContactTube-Aug avoids new teleoperation | future comparison, not STATE 1 |
| Residual repair/runtime wrappers | Online correction after failure | This route is data-centric augmentation before policy training | not continued from killed routes |

