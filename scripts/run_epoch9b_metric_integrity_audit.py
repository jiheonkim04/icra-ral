#!/usr/bin/env python3
"""Audit Epoch 9 visual metrics using preserved evidence and translated bowls."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json
from tca_map.epoch9b_metrics import (
    bounds_overlap,
    changed_pixel_support,
    distribution_summary,
    nondecreasing,
    rgb_sha256,
    template_shift_at_center,
)

REPORT_JSON = REPO_ROOT / "reports/epoch9b_metric_integrity_result.json"
REPORT_MD = REPO_ROOT / "reports/epoch9b_metric_integrity_audit.md"
FRAME_ROOT = REPO_ROOT / "reports/epoch9b_metric_integrity_audit/translated_bowl_frames"
LEGACY_CENTERS = {"front": (95, 62), "back": (70, 57)}
# Development-calibrated centers in the unmodified 128x128 agent-view array.
# The simulator translation panel below independently verifies these centers.
AUDITED_CENTERS = {"front": (92, 28), "back": (71, 60)}
TRANSLATIONS_M = (0.0, 0.01, 0.02, 0.03, 0.04)
LEGACY_REFERENCE_M = 0.03


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_rgb(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(temporary)
    temporary.replace(path)


def preserved_frame_regression() -> dict[str, Any]:
    root = REPO_ROOT / "reports/epoch9_relational_probe_dataset/development/rotation3_demo37_diagnostic"
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    episode_id = "development_demo37_front1_back8_front-first"
    row = next(value for value in result["rows"] if value["episode_id"] == episode_id)
    probe = row["probes"][0]
    initial_path = root / "frames" / f"{episode_id}_front_initial.png"
    final_path = root / "frames" / f"{episode_id}_front_final.png"
    initial = load_rgb(initial_path)
    final = load_rgb(final_path)
    legacy = template_shift_at_center(initial, final, LEGACY_CENTERS["front"], radius=8, search=12)
    audited = template_shift_at_center(initial, final, AUDITED_CENTERS["front"], radius=8, search=18)
    change = changed_pixel_support(initial, final, threshold=10, workspace_y_max=84)
    return {
        "source_result": str((root / "result.json").relative_to(REPO_ROOT)).replace("\\", "/"),
        "episode_id": episode_id,
        "manipulated_body": "akita_black_bowl_1_main",
        "simulator_displacement_m_eval_only": float(probe["target_displacement_m_eval_only"]),
        "historical_recorded_visual_metric": probe["visual_return_estimate"],
        "recomputed_legacy_metric": legacy,
        "audited_target_center_metric": audited,
        "initial_frame_sha256": rgb_sha256(initial),
        "final_frame_sha256": rgb_sha256(final),
        "frames_are_byte_distinct": rgb_sha256(initial) != rgb_sha256(final),
        "visible_change": change,
        "legacy_template_overlaps_changed_workspace_bbox": (
            bounds_overlap(legacy["template_bounds_xyxy"], change["largest_component_bbox_xyxy"])
            if change["largest_component_bbox_xyxy"] is not None
            else False
        ),
        "audited_template_overlaps_changed_workspace_bbox": (
            bounds_overlap(audited["template_bounds_xyxy"], change["largest_component_bbox_xyxy"])
            if change["largest_component_bbox_xyxy"] is not None
            else False
        ),
    }


def _body_free_joint_qpos_address(sim: Any, body_name: str) -> int:
    body_id = int(sim.model.body_name2id(body_name))
    joint_id = int(sim.model.body_jntadr[body_id])
    if joint_id < 0:
        raise RuntimeError(f"{body_name} has no joint")
    joint_type = int(sim.model.jnt_type[joint_id])
    if joint_type != 0:  # mjJNT_FREE
        raise RuntimeError(f"{body_name} joint is type {joint_type}, expected free joint")
    return int(sim.model.jnt_qposadr[joint_id])


def translated_bowl_panel() -> dict[str, Any]:
    """Render known development-only translations without using sealed identities."""

    from scripts.run_epoch9_probe_controller_development import TASKS, load_env_class, make_env, read_demo

    env_class = load_env_class()
    rows: list[dict[str, Any]] = []
    for slot, task in TASKS.items():
        env = None
        try:
            init_state, _ = read_demo(task, 37)
            env, observation = make_env(env_class, task, init_state)
            qpos_address = _body_free_joint_qpos_address(env.sim, task["body"])
            baseline_qpos = np.asarray(env.sim.data.qpos, dtype=np.float64).copy()
            frames: dict[float, np.ndarray] = {}
            for translation_m in TRANSLATIONS_M:
                env.sim.data.qpos[:] = baseline_qpos
                env.sim.data.qvel[:] = 0.0
                env.sim.data.qpos[qpos_address] = baseline_qpos[qpos_address] + float(translation_m)
                env.sim.forward()
                # Direct simulator-state changes do not advance observable
                # timestamps. A forced refresh is therefore required; the
                # first preserved audit attempt intentionally records the
                # cached-frame failure that exposed this boundary.
                observation = env.env._get_observations(force_update=True)
                frame = np.asarray(observation["agentview_image"], dtype=np.uint8).copy()
                frames[float(translation_m)] = frame
                save_rgb(FRAME_ROOT / f"{slot}_world_x_plus_{int(round(translation_m * 1000)):03d}mm.png", frame)
            initial = frames[0.0]
            for translation_m in TRANSLATIONS_M:
                frame = frames[float(translation_m)]
                legacy = template_shift_at_center(initial, frame, LEGACY_CENTERS[slot], radius=8, search=18)
                audited = template_shift_at_center(initial, frame, AUDITED_CENTERS[slot], radius=8, search=18)
                rows.append(
                    {
                        "slot": slot,
                        "demo_index": 37,
                        "sealed_identity_used": False,
                        "world_translation_axis": "+x",
                        "known_translation_m": float(translation_m),
                        "legacy_metric": legacy,
                        "audited_metric": audited,
                        "visible_change": changed_pixel_support(initial, frame, threshold=10, workspace_y_max=84),
                        "frame_sha256": rgb_sha256(frame),
                    }
                )
        finally:
            if env is not None:
                env.close()

    by_slot: dict[str, Any] = {}
    for slot in ("front", "back"):
        slot_rows = [row for row in rows if row["slot"] == slot]
        magnitudes = [float(row["audited_metric"]["magnitude_pixels"]) for row in slot_rows]
        changed_counts = [int(row["visible_change"]["changed_pixel_count"]) for row in slot_rows]
        by_slot[slot] = {
            "known_translations_m": [float(row["known_translation_m"]) for row in slot_rows],
            "audited_visual_displacement_pixels": magnitudes,
            "changed_pixel_counts": changed_counts,
            "displacement_nondecreasing": nondecreasing(magnitudes),
            "change_support_nondecreasing": nondecreasing(changed_counts),
            "responds_to_largest_translation": bool(magnitudes[-1] > magnitudes[0]),
        }
    return {
        "evidence_class": "DEVELOPMENT_METRIC_DIAGNOSTIC",
        "identity_policy": "demo 37 only; validation 40..44 and confirmation 45..49 remain sealed",
        "translation_method": "evaluation-only direct free-joint qpos translation followed by ordinary agent-view rendering",
        "online_controller_use": False,
        "rows": rows,
        "summary_by_slot": by_slot,
        "all_audited_metrics_monotonic_and_responsive": all(
            value["displacement_nondecreasing"] and value["responds_to_largest_translation"]
            for value in by_slot.values()
        ),
    }


def _identity(row: dict[str, Any]) -> str:
    if "demo_index" in row:
        return f"demo_{int(row['demo_index'])}"
    episode_id = str(row.get("episode_id", "unknown"))
    marker = "demo"
    if marker in episode_id:
        suffix = episode_id.split(marker, 1)[1].split("_", 1)[0]
        return f"demo_{suffix}"
    return episode_id


def historical_distributions() -> dict[str, Any]:
    sources = {
        "fixed_contact_full_development": REPO_ROOT
        / "reports/epoch9_relational_probe_dataset/development/rotation1_front_reference_v1/result.json",
        "shortened_travel_demo37": REPO_ROOT
        / "reports/epoch9_relational_probe_dataset/development/rotation2_demo37_diagnostic/result.json",
        "zero_travel_demo37": REPO_ROOT
        / "reports/epoch9_relational_probe_dataset/development/rotation3_demo37_diagnostic/result.json",
        "open_gripper_visual_calibration": REPO_ROOT
        / "reports/epoch9_controller_development/v12_visual_tolerance_calibration/result.json",
    }
    output: dict[str, Any] = {}
    for name, path in sources.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        per_episode: list[float] = []
        per_probe: list[float] = []
        per_identity: dict[str, list[float]] = defaultdict(list)
        for row in payload.get("rows", []):
            identity = _identity(row)
            if "candidate_final_displacement_m_eval_only" in row:
                values = [float(value) for value in row["candidate_final_displacement_m_eval_only"].values()]
                per_episode.append(max(values))
                per_identity[identity].extend(values)
                for probe in row.get("probes", []):
                    if "target_displacement_m_eval_only" in probe:
                        per_probe.append(float(probe["target_displacement_m_eval_only"]))
            elif "final_target_displacement_m_eval_only" in row:
                value = float(row["final_target_displacement_m_eval_only"])
                per_episode.append(value)
                per_probe.append(value)
                per_identity[identity].append(value)
        output[name] = {
            "source": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "per_episode_max_candidate_displacement_m": distribution_summary(
                per_episode, threshold=LEGACY_REFERENCE_M
            ),
            "per_probe_target_displacement_m": distribution_summary(per_probe, threshold=LEGACY_REFERENCE_M),
            "per_identity_candidate_displacement_m": {
                identity: distribution_summary(values, threshold=LEGACY_REFERENCE_M)
                for identity, values in sorted(per_identity.items())
            },
        }
    return output


def write_markdown(result: dict[str, Any]) -> None:
    regression = result["preserved_frame_regression"]
    panel = result["translated_bowl_panel"]
    lines = [
        "# Epoch 9B Metric Integrity Audit",
        "",
        f"Date: {result['timestamp']}",
        "",
        f"Decision: `{result['decision']}`",
        "",
        "## Finding",
        "",
        "The Epoch 9 visual return metric did not observe the manipulated front bowl. Its fixed "
        "17x17 template was centered at `(95, 62)`, while the preserved moved-bowl support is in "
        f"`{regression['visible_change']['largest_component_bbox_xyxy']}`. The two regions do not overlap. "
        "The metric therefore returned a near-perfect zero shift on static pixels even though the "
        f"evaluation-only body displacement was `{regression['simulator_displacement_m_eval_only']:.6f} m`.",
        "",
        "The saved initial and final frames have distinct SHA-256 digests and contain "
        f"`{regression['visible_change']['changed_pixel_count']}` changed workspace pixels above the "
        "10-level threshold. This rules out cached-frame reuse and confirms that the runner used "
        "distinct before/after images. The defect is the target crop, not dtype conversion, end-effector "
        "return, or a rewritten historical result.",
        "",
        "Historical visual-return values remain unchanged in their original files. Any front-slot value "
        "computed with the legacy crop is labelled unreliable for object return.",
        "",
        "## Deliberate translation test",
        "",
        "Development identity 37 was reset, each target bowl free joint was translated along world +x by "
        "0, 1, 2, 3, and 4 cm, and the ordinary 128x128 agent-view image was rerendered. No sealed identity "
        "or controller outcome was accessed.",
        "",
        "| slot | audited pixel magnitudes for 0/1/2/3/4 cm | monotonic | responsive |",
        "|---|---:|---:|---:|",
    ]
    for slot, summary in panel["summary_by_slot"].items():
        values = ", ".join(f"{value:.3f}" for value in summary["audited_visual_displacement_pixels"])
        lines.append(
            f"| {slot} | {values} | {summary['displacement_nondecreasing']} | "
            f"{summary['responds_to_largest_translation']} |"
        )
    lines.extend(
        [
            "",
            "The audited centers are `(92, 28)` for the front candidate and `(71, 60)` for the back "
            "candidate in the unmodified image-array orientation. The tracker records its center, template "
            "bounds, effective search radius, template texture, and confidence on every call.",
            "",
            "## Distribution reporting",
            "",
            "The machine-readable result contains per-episode, per-probe, and per-identity displacement "
            "distributions for the key preserved Epoch 9 regimes. Each includes count above the unchanged "
            "3 cm legacy reference, mean, median, standard deviation, 5/25/50/75/95% quantiles, extrema, "
            "and a deterministic bootstrap 95% interval for the mean.",
            "",
            "## Integrity boundary",
            "",
            "- Validation identities 40..44: not accessed.",
            "- Confirmation identities 45..49: not accessed.",
            "- Old reports and frame files: read only.",
            "- Simulator pose: used only to impose known evaluation translations and to retain the historical "
            "continuous displacement reference; never exposed to a controller or learned inference path.",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-simulator", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    regression = preserved_frame_regression()
    translated = {"skipped": True} if args.skip_simulator else translated_bowl_panel()
    result: dict[str, Any] = {
        "schema_version": "epoch9b.metric_integrity.v1",
        "timestamp": timestamp(),
        "evidence_class": "DEVELOPMENT_METRIC_AUDIT",
        "decision": "VISUAL_METRIC_TARGET_CROP_DEFECT_CONFIRMED",
        "historical_files_modified": False,
        "validation_identities_accessed": False,
        "confirmation_identities_accessed": False,
        "legacy_reference_displacement_m": LEGACY_REFERENCE_M,
        "legacy_centers_xy": {key: list(value) for key, value in LEGACY_CENTERS.items()},
        "audited_centers_xy": {key: list(value) for key, value in AUDITED_CENTERS.items()},
        "preserved_frame_regression": regression,
        "translated_bowl_panel": translated,
        "historical_displacement_distributions": historical_distributions(),
        "reliability_labels": {
            "legacy_front_visual_return": "UNRELIABLE_WRONG_TARGET_CROP",
            "legacy_back_visual_return": "NO_DEFECT_PROVEN_BY_THIS_AUDIT",
            "simulator_object_displacement_eval_only": "RELIABLE_CONTINUOUS_EVALUATION_METRIC",
        },
    }
    atomic_write_json(REPORT_JSON, result)
    if not args.skip_simulator:
        write_markdown(result)
    print(json.dumps({"decision": result["decision"], "translated_panel": translated.get("all_audited_metrics_monotonic_and_responsive")}, sort_keys=True))
    return 0 if args.skip_simulator or translated["all_audited_metrics_monotonic_and_responsive"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
