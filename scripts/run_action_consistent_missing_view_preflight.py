#!/usr/bin/env python3
"""Run the frozen X-VLA numerical-noise or microbatch preflight."""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.action_consistent_missing_view_distillation.preflight import main


if __name__ == "__main__":
    raise SystemExit(main())
