# AMP-GD Kill Summary

AMP-GD is killed as the main RA-L-stable route.

Reason:
- Toy AMP-GD evidence was matched by deterministic informative-probe and entropy-greedy probe heuristics.
- The LIBERO/RoboSuite port exposed non-leaking object and EEF state, but not a useful active ambiguity signal.
- In the tiny LIBERO diagnostic, no-probe, safety-only, and nearest-target all had wrong-target movement `0.0` and target movement `0.000109378`.
- AMP-GD had wrong-target movement `1.0`, target movement `0.000044975`, and reward/success `0.0 / false`.
- AMP-GD did not beat safety-only, and random-probe matched AMP-GD on wrong-target movement.

Execution boundary:
- toy rollout/control audit happened: yes.
- LIBERO/RoboSuite micro-probe diagnostic happened: yes, bounded tiny diagnostic only.
- training, LoRA training, loss, GPU, downloads, heavy VLA imports, OpenVLA-OFT, benchmark rollout, and paper-grade claims: no.

Reusable evidence:
- object/EEF observability path,
- safe WSL LIBERO/RoboSuite runner,
- instruction-text plus visible-object-key target resolver,
- baseline-first kill discipline.

Do not continue AMP-GD as the current main route.
