# Closed-loop process-start repair audit

Decision: `NO_SCIENTIFICALLY_EQUIVALENT_LOCAL_PROCESS_START_REPAIR`

An outcome-free import probe tested the last plausible narrow repair: share
read-only LIBERO/Torch runtime pages through a pre-CUDA Linux fork while
retaining four real processes, four simultaneous environments, one model, and
the frozen actual-arrival schedule.

A fresh process used 32,378,880 bytes after importing the runner and
652,017,664 bytes after the pinned LIBERO import; LIBERO also imported Torch.
Even granting an unrealistically perfect collapse of five parent/worker
runtime copies to one saves at most 2,478,555,136 bytes. That reduces the
observed four-environment phase from 21,178,970,112 to 18,700,414,976 bytes,
leaving only 1,693,816,832 bytes below the frozen 82% ceiling.

The independently measured model-active increment was 6,923,243,520 bytes.
After subtracting one complete import increment, 6,303,604,736 bytes remain.
Thus perfect import sharing is still several gigabytes short of the ceiling;
the deliberately favorable additive bound is 25,004,019,712 bytes.

Forking after CUDA or EGL initialization is unsafe. Pre-forking a live MuJoCo
environment would also cease to be four independently constructed official
environments. Reducing the live environment count or multiplexing simulator
states changes the preregistered four-shard intervention. No code was changed,
no environment or model was created by the probe, and no action or outcome was
read.

The existing `INFRASTRUCTURE_OR_RESOURCE_BLOCKED` decision is therefore
preserved. Resume on a clean 48 GB-or-larger equivalent host, or only under
explicit authority for a genuinely new lower-concurrency scientific study.
