"""Script wrapper for MARC-VLA Stage A manifest freezing."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.marc_vla_stage_a import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
