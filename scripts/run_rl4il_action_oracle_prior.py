#!/usr/bin/env python3
"""Entry point for the mechanism-faithful RL4IL local prior port."""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.rl4il_prior.mechanism_port import main


if __name__ == "__main__":
    raise SystemExit(main())
