# ContactSet-VLA Kill Criteria

## STATE 1 Decision

Decision: kill ContactSet-VLA as the current main route before full VLA fine-tuning or replay scale-up.

Reason: the bounded local HDF5 action-head diagnostic computed held-out losses over `6` local LIBERO demos and found that full contact-set injection did not beat active single-3D-point injection. Full contact-set action L2 was `1.105028754`, active single-point action L2 was `0.930495702`, destination-only action L2 was `0.86372`, and no-geometry action L2 was `0.851451`.

Execution boundary: tiny CPU NumPy action-head training happened and loss was computed. No simulator rollout/replay, GPU job, download, heavy VLA import/model load, OpenVLA-OFT execution, token access, or paper-grade claim occurred.

## STATE 1 Continue Gate

Continue only if all are true:

- a bounded local HDF5 action-head loss is computed,
- source object, destination/placement, support/contact, and normal/contact-set cues are observable without eval-label leakage,
- `full_contact_set_injection` beats `single_3d_point_injection` on held-out 7D action L2 by at least a small practical margin,
- source-only, destination-only, and source+destination baselines do not match the full contact set,
- geometry selection uses instruction text plus visible/observable HDF5 or XML state, not reward, success, eval labels, or task-id/filename target labels.

## Kill Or Block Conditions

Kill if any are true:

- active single 3D point injection matches or beats full contact-set injection,
- source-only or destination-only geometry matches full contact-set injection,
- source+destination two-point injection matches full contact-set injection,
- contact-set features require oracle/eval target labels,
- object/placement/contact points are not observable from local infrastructure,
- the method improves only a contact proxy while action L2 or later replay/progress does not improve,
- a simple baseline explains the result.

Block if:

- local HDF5 files are missing,
- `h5py` is missing,
- the diagnostic cannot compute a bounded loss,
- a later replay/progress metric is needed but simulator risk assessment is not green.

## Evidence Boundary

STATE 1 can support an exploratory offline action-head decision only. It cannot support SOTA, standard LIBERO success, real VLA competence, simulator rollout success, or paper-grade claims.
