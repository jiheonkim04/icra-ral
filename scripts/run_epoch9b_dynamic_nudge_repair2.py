#!/usr/bin/env python3
"""Run the isolated Epoch 9B inward-contact repair campaign."""

from __future__ import annotations

from pathlib import Path

import run_epoch9b_dynamic_nudge as campaign


ROOT = Path(__file__).resolve().parents[1]
campaign.PROTOCOL_PATH = ROOT / "reports/epoch9b_v2_task_preservation_protocol_repair2.json"
campaign.FREEZE_PATH = campaign.OUTPUT_ROOT / "controller_freeze_repair2.json"
campaign.PANEL_PATH = campaign.OUTPUT_ROOT / "feasibility_panel_repair2_result.json"


if __name__ == "__main__":
    raise SystemExit(campaign.main())
