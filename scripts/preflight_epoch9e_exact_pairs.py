#!/usr/bin/env python3
"""Outcome-suppressed exact-state A/B audit for Epoch 9E."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_epoch9b_dynamic_nudge as campaign
from scripts.run_epoch9_probe_controller_development import BDDL_ROOT
from scripts.run_epoch9d_causal_panel import memory_sample, sha256
from tca_map.epoch7_latent_dynamics import apply_intervention, atomic_write_json
from tca_map.epoch9b_metrics import rgb_sha256


REPORTS = ROOT / "reports"
PROTOCOL_PATH = REPORTS / "epoch9e_joint_certification_protocol.json"
CALIBRATION_PATH = REPORTS / "epoch9b_dynamic_nudge/controller_calibration_repair1.json"
OUTPUT = REPORTS / "epoch9e_exact_pair_preflight.json"


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def localizations(observation: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    frame = np.asarray(observation["agentview_image"], dtype=np.uint8)
    result = {}
    for slot in ("front", "back"):
        _, _, metric = campaign.localize_candidate(frame, slot, calibration)
        result[slot] = {"subpixel_dx": float(metric["subpixel_dx"]), "subpixel_dy": float(metric["subpixel_dy"]), "quality": float(metric["quality"])}
    return result


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError("refusing to overwrite exact-pair preflight")
    protocol = load(PROTOCOL_PATH)
    calibration = load(CALIBRATION_PATH)
    bases = {row["base_identity_id"]: row for row in protocol["base_states"]}
    env_class = campaign.load_env_class()
    rows = []
    process_peak = 0
    wsl_peak = 0
    swap_peak = 0
    for assignment in protocol["assignments"]:
        base = bases[assignment["base_identity_id"]]
        env = None
        try:
            env = env_class(bddl_file_name=str(BDDL_ROOT / assignment["task_bddl"]), camera_heights=128, camera_widths=128)
            env.seed(int(base["generator_seed"]))
            env.reset()
            env.sim.set_state_from_flattened(np.asarray(base["base_state_vector_float64"], dtype=np.float64))
            env.sim.forward()
            observation = campaign.forced_observation(env)
            before = rgb_sha256(np.asarray(observation["agentview_image"], dtype=np.uint8))
            before_localization = localizations(observation, calibration)
            for slot, factor in assignment["mass_factor"].items():
                if float(factor) != 1.0:
                    apply_intervention(env.sim.model, {"axis": "target_mass", "body_name": campaign.BODY_BY_SLOT[slot], "arrays": ["body_mass", "body_inertia"], "factor": float(factor)})
            env.sim.forward()
            observation = campaign.forced_observation(env)
            after = rgb_sha256(np.asarray(observation["agentview_image"], dtype=np.uint8))
            after_localization = localizations(observation, calibration)
            rows.append({
                "scene_id": assignment["scene_id"], "base_identity_id": base["base_identity_id"], "assignment": assignment["assignment"],
                "task_bddl": assignment["task_bddl"], "expected_first_rgb_sha256": base["first_agentview_rgb_sha256"],
                "before_mass_rgb_sha256": before, "after_mass_rgb_sha256": after,
                "first_rgb_exact": before == after == base["first_agentview_rgb_sha256"],
                "initial_localization_exact": before_localization == after_localization == base["initial_rgb_localization_audit"],
                "mass_assignment_only_mutation": True, "actions_executed": 0, "reward_done_success_accessed": False,
            })
        finally:
            if env is not None:
                env.close()
        sample = memory_sample()
        process_peak = max(process_peak, sample["process_max_rss_bytes"])
        wsl_peak = max(wsl_peak, sample["wsl_mem_used_bytes"])
        swap_peak = max(swap_peak, sample["wsl_swap_used_bytes"])
        if swap_peak != 0:
            raise RuntimeError("WSL swap used during exact-pair preflight")
    pair_map: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        pair_map.setdefault(row["base_identity_id"], {})[row["assignment"]] = row
    pair_exact = []
    for identity, pair in sorted(pair_map.items()):
        pair_exact.append({"base_identity_id": identity, "a_b_rgb_exact": pair["A"]["after_mass_rgb_sha256"] == pair["B"]["after_mass_rgb_sha256"], "a_b_localization_exact": pair["A"]["initial_localization_exact"] and pair["B"]["initial_localization_exact"]})
    result = {
        "schema_version": "epoch9e.exact_pair_preflight.v1", "completed_at": timestamp(),
        "protocol_path": "reports/epoch9e_joint_certification_protocol.json", "protocol_sha256": sha256(PROTOCOL_PATH),
        "rows": rows, "pair_rows": pair_exact,
        "summary": {"assignment_rows": len(rows), "first_rgb_exact_rows": sum(row["first_rgb_exact"] for row in rows), "initial_localization_exact_rows": sum(row["initial_localization_exact"] for row in rows), "a_b_rgb_exact_pairs": sum(row["a_b_rgb_exact"] for row in pair_exact), "a_b_localization_exact_pairs": sum(row["a_b_localization_exact"] for row in pair_exact)},
        "resource": {"process_max_rss_bytes": process_peak, "wsl_mem_used_peak_bytes": wsl_peak, "wsl_swap_used_peak_bytes": swap_peak},
        "scientific_outcomes_accessed": False, "validation_accessed": False, "confirmation_accessed": False,
    }
    if result["summary"] != {"assignment_rows": 24, "first_rgb_exact_rows": 24, "initial_localization_exact_rows": 24, "a_b_rgb_exact_pairs": 12, "a_b_localization_exact_pairs": 12}:
        raise RuntimeError("exact-pair preflight failed")
    atomic_write_json(OUTPUT, result)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
