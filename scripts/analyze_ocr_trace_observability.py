#!/usr/bin/env python
"""Analyze the bounded OCR-XVLA legal trace acquisition pass.

The script tests whether a no-progress trigger is identifiable from legal
deployment-time inputs above a trivial action-history-only baseline. It does
not train or tune a policy, write checkpoints, run a simulator, or execute
Ours. Success/failure labels are used only as offline observability labels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


RESIDUAL_FAILURE_IDENTITIES = {20260727, 20260730, 20260733}
CLEAN_RETENTION_IDENTITIES = {20260731, 20260732}
HELD_OUT_CONFIRMATORY_IDENTITIES = [20260734, 20260735, 20260736, 20260737]
ATTEMPT_WINDOW_STEPS = 120
EPS = 1e-6


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dark_centroid_features(frames: np.ndarray) -> dict[str, float | None]:
    """Compute a legal RGB-only dark-object proxy for black-bowl movement."""

    if frames.size == 0:
        return {"max_centroid_shift_px": None, "max_area_delta_frac": None}
    rgb = frames.astype(np.float32)
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    dark = luminance < 70.0
    centroids: list[tuple[float, float] | None] = []
    area_fracs: list[float] = []
    yy, xx = np.mgrid[0 : dark.shape[1], 0 : dark.shape[2]]
    for mask in dark:
        area = float(mask.mean())
        area_fracs.append(area)
        if int(mask.sum()) < 4:
            centroids.append(None)
            continue
        centroids.append((float(yy[mask].mean()), float(xx[mask].mean())))
    anchor_index = next((i for i, item in enumerate(centroids) if item is not None), None)
    if anchor_index is None:
        return {
            "max_centroid_shift_px": None,
            "max_area_delta_frac": float(max(area_fracs) - min(area_fracs)) if area_fracs else None,
        }
    anchor = centroids[anchor_index]
    shifts = [
        math.hypot(item[0] - anchor[0], item[1] - anchor[1])
        for item in centroids[anchor_index:]
        if item is not None
    ]
    return {
        "max_centroid_shift_px": float(max(shifts)) if shifts else 0.0,
        "max_area_delta_frac": float(max(area_fracs) - min(area_fracs)) if area_fracs else 0.0,
    }


def first_attempt_index(actions: np.ndarray, eef_position: np.ndarray) -> int:
    """Use action/proprio only to identify the first grasp/lift-attempt window."""

    if len(actions) == 0:
        return 0
    gripper = actions[:, -1]
    close = np.flatnonzero(gripper > 0.0)
    if close.size:
        return int(close[0])
    z_delta = eef_position[:, 2] - float(eef_position[0, 2])
    lift = np.flatnonzero((np.arange(len(z_delta)) >= 20) & (z_delta > 0.015))
    if lift.size:
        return int(lift[0])
    return 0


def separation_score(rows: list[dict[str, Any]], feature: str, *, success_should_be_higher: bool) -> dict[str, Any]:
    residual = [
        float(row["features"][feature])
        for row in rows
        if row["role"] == "residual_failure" and row["features"].get(feature) is not None
    ]
    clean = [
        float(row["features"][feature])
        for row in rows
        if row["role"] == "clean_retention_success" and row["features"].get(feature) is not None
    ]
    if not residual or not clean:
        return {
            "feature": feature,
            "success_should_be_higher": success_should_be_higher,
            "strictly_separates": False,
            "gap": None,
            "normalized_gap": None,
        }
    if success_should_be_higher:
        gap = min(clean) - max(residual)
    else:
        gap = min(residual) - max(clean)
    all_values = residual + clean
    normalized_gap = gap / (max(all_values) - min(all_values) + EPS)
    return {
        "feature": feature,
        "success_should_be_higher": success_should_be_higher,
        "strictly_separates": bool(gap > 0.0),
        "gap": float(gap),
        "normalized_gap": float(normalized_gap),
        "residual_failure_values": residual,
        "clean_retention_success_values": clean,
    }


def row_from_result(result_path: Path) -> dict[str, Any]:
    result = load_json(result_path)
    episode = (result.get("episodes") or [{}])[0]
    identity = int(episode["reset_identity"])
    trace_artifact = episode.get("trace_artifact") or {}
    trace_path = Path(trace_artifact["trace_npz"])
    if not trace_path.exists():
        trace_path = result_path.parent / trace_path
    with np.load(trace_path) as trace:
        step_index = trace["step_index"]
        actions = trace["executed_env_action_7d"]
        eef_position = trace["eef_position"]
        chunk_index = trace["chunk_index"]
        new_chunk_started = trace["new_chunk_started"]
        agentview = trace["policy_input_agentview_rgb"]
        wrist = trace["wrist_rgb"]
        attempt = first_attempt_index(actions, eef_position)
        end = min(len(step_index), attempt + ATTEMPT_WINDOW_STEPS)
        window = slice(attempt, end)
        pos_window = eef_position[window]
        action_window = actions[window]
        chunk_window = chunk_index[window]
        new_chunk_window = new_chunk_started[window]
        agent_window = agentview[window]
        wrist_window = wrist[window]
        anchor = eef_position[attempt]
        displacement = np.linalg.norm(pos_window - anchor, axis=1) if len(pos_window) else np.asarray([0.0])
        vertical_lift = pos_window[:, 2] - float(anchor[2]) if len(pos_window) else np.asarray([0.0])
        action_l2 = np.linalg.norm(action_window[:, :6], axis=1) if len(action_window) else np.asarray([0.0])
        agent_dark = dark_centroid_features(agent_window)
        wrist_dark = dark_centroid_features(wrist_window)
        if len(agent_window) > 1:
            agent_frame_delta = float(np.mean(np.abs(agent_window.astype(np.float32)[1:] - agent_window[0].astype(np.float32))) / 255.0)
            wrist_frame_delta = float(np.mean(np.abs(wrist_window.astype(np.float32)[1:] - wrist_window[0].astype(np.float32))) / 255.0)
        else:
            agent_frame_delta = 0.0
            wrist_frame_delta = 0.0

    if identity in RESIDUAL_FAILURE_IDENTITIES:
        role = "residual_failure"
    elif identity in CLEAN_RETENTION_IDENTITIES:
        role = "clean_retention_success"
    else:
        role = "unexpected_identity"
    features = {
        "attempt_index_action_proprio_only": float(attempt),
        "window_step_count": float(end - attempt),
        "action_history_chunk_count_window": float(len(set(int(x) for x in chunk_window.tolist()))) if len(chunk_window) else 0.0,
        "action_history_new_chunk_count_window": float(int(np.sum(new_chunk_window))) if len(new_chunk_window) else 0.0,
        "action_history_mean_delta_action_l2_window": float(np.mean(action_l2)) if len(action_l2) else 0.0,
        "action_history_gripper_close_fraction_window": float(np.mean(action_window[:, -1] > 0.0)) if len(action_window) else 0.0,
        "proprio_max_eef_displacement_cm": float(np.max(displacement) * 100.0),
        "proprio_max_vertical_lift_cm": float(np.max(vertical_lift) * 100.0),
        "agentview_dark_object_max_centroid_shift_px": agent_dark["max_centroid_shift_px"],
        "agentview_dark_object_max_area_delta_frac": agent_dark["max_area_delta_frac"],
        "wrist_dark_object_max_centroid_shift_px": wrist_dark["max_centroid_shift_px"],
        "wrist_dark_object_max_area_delta_frac": wrist_dark["max_area_delta_frac"],
        "agentview_mean_frame_delta_from_attempt": agent_frame_delta,
        "wrist_mean_frame_delta_from_attempt": wrist_frame_delta,
    }
    return {
        "identity": identity,
        "role": role,
        "result_path": str(result_path),
        "trace_npz": str(trace_path),
        "trace_sha256": trace_artifact.get("trace_sha256"),
        "completed": bool(episode.get("completed")),
        "success_label_for_offline_observability_only": bool(episode.get("success")),
        "steps": int(episode.get("steps", 0)),
        "action_chunk_count": int(episode.get("action_chunk_count", 0)),
        "attempt_window": {"start_step": int(attempt), "end_step_exclusive": int(end)},
        "features": features,
    }


def build_report(run_dir: Path) -> dict[str, Any]:
    result_paths = sorted(run_dir.glob("identity_*/result.json"))
    rows = [row_from_result(path) for path in result_paths]
    action_features = [
        "action_history_chunk_count_window",
        "action_history_new_chunk_count_window",
        "action_history_mean_delta_action_l2_window",
        "action_history_gripper_close_fraction_window",
    ]
    obs_features = [
        "proprio_max_eef_displacement_cm",
        "proprio_max_vertical_lift_cm",
        "agentview_dark_object_max_centroid_shift_px",
        "agentview_dark_object_max_area_delta_frac",
        "wrist_dark_object_max_centroid_shift_px",
        "wrist_dark_object_max_area_delta_frac",
        "agentview_mean_frame_delta_from_attempt",
        "wrist_mean_frame_delta_from_attempt",
    ]
    separations = []
    for feature in action_features:
        for success_should_be_higher in (True, False):
            separations.append(separation_score(rows, feature, success_should_be_higher=success_should_be_higher))
    for feature in obs_features:
        for success_should_be_higher in (True, False):
            separations.append(separation_score(rows, feature, success_should_be_higher=success_should_be_higher))
    action_best = max(
        (item for item in separations if item["feature"] in action_features),
        key=lambda item: -1.0 if item["normalized_gap"] is None else float(item["normalized_gap"]),
    )
    obs_best = max(
        (item for item in separations if item["feature"] in obs_features),
        key=lambda item: -1.0 if item["normalized_gap"] is None else float(item["normalized_gap"]),
    )
    trace_valid = (
        len(rows) == 5
        and {row["identity"] for row in rows} == RESIDUAL_FAILURE_IDENTITIES | CLEAN_RETENTION_IDENTITIES
        and all(row["completed"] for row in rows)
    )
    action_baseline_separates = bool(action_best.get("strictly_separates"))
    obs_separates = bool(obs_best.get("strictly_separates"))
    obs_above_action = (
        obs_best.get("normalized_gap") is not None
        and action_best.get("normalized_gap") is not None
        and float(obs_best["normalized_gap"]) > float(action_best["normalized_gap"]) + 0.05
    )
    passed = bool(trace_valid and obs_separates and (not action_baseline_separates or obs_above_action))
    return {
        "schema_version": "2026-07-18.epoch5_ocr_trigger_observability_analysis.v1",
        "stage": "epoch_5_ocr_xvla_task5_bounded_trace_observability_analysis",
        "decision": "OCR_TRIGGER_OBSERVABILITY_PASS" if passed else "OCR_TRIGGER_OBSERVABILITY_FAIL",
        "trace_run_dir": str(run_dir),
        "candidate_id": "OCR-XVLA",
        "candidate_name": "Observation-Consistency Retry for X-VLA",
        "frozen_ocr_mechanism_changed": False,
        "policy_optimizer_used": False,
        "policy_checkpoint_written": False,
        "ours_rollout_happened": False,
        "control_rollout_happened": False,
        "training_happened": False,
        "discovery_identities_only": True,
        "held_out_confirmatory_identities_not_used": HELD_OUT_CONFIRMATORY_IDENTITIES,
        "inference_feature_policy": {
            "allowed_trace_inputs": [
                "per-step RGB observations",
                "per-step proprioception",
                "per-step issued env actions",
                "chunk indices",
                "step/timestamps",
            ],
            "forbidden_inference_inputs_not_used": [
                "reward",
                "done",
                "success oracle",
                "simulator object state",
                "privileged contact/pose",
                "future observation",
                "reset identity label as a trigger feature",
            ],
            "success_labels_used_only_for_offline_observability_scoring": True,
        },
        "preregistered_test": {
            "attempt_detector": "first step with positive gripper command; fallback first post-step20 +1.5cm EEF z rise",
            "attempt_window_steps": ATTEMPT_WINDOW_STEPS,
            "action_history_only_baseline_features": action_features,
            "legal_observation_proprio_features": obs_features,
            "pass_condition": (
                "all five discovery traces complete, and at least one legal observation/proprio feature strictly separates "
                "residual failures from clean-retention successes in the attempt window, with separation not explained by "
                "a trivial action-history-only feature set"
            ),
        },
        "trace_valid": trace_valid,
        "action_history_only_best_separation": action_best,
        "legal_observation_proprio_best_separation": obs_best,
        "action_history_only_baseline_separates": action_baseline_separates,
        "legal_observation_proprio_separates": obs_separates,
        "legal_observation_proprio_above_action_history_baseline": obs_above_action,
        "rows": rows,
        "bounded_conclusion": {
            "ocr_trigger_observability_passed": passed,
            "ocr_candidate_can_advance_to_action_bounds": passed,
            "ocr_candidate_archived_if_failed": not passed,
            "result_interpretation": (
                "OCR trigger observability survived the one permitted trace test."
                if passed
                else "The one permitted trace test did not identify a legal no-progress trigger above the action-history-only baseline; OCR is archived under the user steer."
            ),
        },
    }


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    action = report["action_history_only_best_separation"]
    obs = report["legal_observation_proprio_best_separation"]
    rows = report["rows"]
    lines = [
        "# OCR-XVLA Bounded Trace Observability Result",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Trace run: `{report['trace_run_dir']}`",
        "- Policy: frozen official X-VLA prior only; no Ours rollout, no optimizer, no checkpoint.",
        "- Discovery identities only: residual `20260727/20260730/20260733`; clean-retention `20260731/20260732`.",
        "- Held-out identities not used: `20260734/20260735/20260736/20260737`.",
        "",
        "## Preregistered test",
        "",
        f"- Attempt window: first gripper-close step, fallback post-step20 +1.5cm EEF z-rise; `{ATTEMPT_WINDOW_STEPS}` steps.",
        "- PASS requires a legal RGB/proprio feature to strictly separate residual failures from clean-retention successes above the action-history-only baseline.",
        "- Reward, done, success, simulator object/contact state, privileged pose, and future observations were not trigger features.",
        "",
        "## Best separations",
        "",
        f"- Action-history-only best: `{action['feature']}`, strict={action['strictly_separates']}, normalized_gap={action['normalized_gap']}",
        f"- Legal observation/proprio best: `{obs['feature']}`, strict={obs['strictly_separates']}, normalized_gap={obs['normalized_gap']}",
        f"- Observation/proprio above action baseline: `{report['legal_observation_proprio_above_action_history_baseline']}`",
        "",
        "## Per-identity trace summary",
        "",
        "| identity | role | completed | success label for offline scoring only | steps | chunks | attempt window | trace |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        win = row["attempt_window"]
        lines.append(
            f"| {row['identity']} | {row['role']} | {row['completed']} | {row['success_label_for_offline_observability_only']} | "
            f"{row['steps']} | {row['action_chunk_count']} | {win['start_step']}..{win['end_step_exclusive']} | `{row['trace_npz']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["bounded_conclusion"]["result_interpretation"],
            "",
            "This result does not reopen SGL-XVLA and does not execute OCR-XVLA as an intervention policy.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(args.run_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, args.output_md)
    print(json.dumps({"decision": report["decision"], "output_json": str(args.output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
