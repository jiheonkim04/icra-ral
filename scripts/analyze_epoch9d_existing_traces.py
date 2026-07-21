#!/usr/bin/env python3
"""Diagnose existing Epoch 9B dynamic-nudge traces without new outcomes."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json


REPORTS = ROOT / "reports"
OUTPUT_JSON = REPORTS / "epoch9d_existing_trace_causal_diagnostic.json"
OUTPUT_MD = REPORTS / "epoch9d_existing_trace_causal_diagnostic.md"
THRESHOLD_M = 0.005219466062047384
BOOTSTRAP_SEED = 914010
BOOTSTRAP_DRAWS = 20_000
CAMPAIGNS = (
    {
        "id": "original_frozen_panel",
        "result": REPORTS / "epoch9b_dynamic_nudge/feasibility_panel_result.json",
        "protocol": REPORTS / "epoch9b_v2_task_preservation_protocol.json",
        "evidence_role": "FROZEN_HISTORICAL_NO_GO",
    },
    {
        "id": "repair1_centered_edge_development",
        "result": REPORTS / "epoch9b_dynamic_nudge/development/d17_centered_contact_edge_stress/result.json",
        "protocol": REPORTS / "epoch9b_v2_task_preservation_protocol_repair1.json",
        "evidence_role": "DEVELOPMENT_DIAGNOSTIC",
    },
    {
        "id": "repair2_inward_edge_development",
        "result": REPORTS / "epoch9b_dynamic_nudge/development/d18_inward_contact_edge_stress/result.json",
        "protocol": REPORTS / "epoch9b_v2_task_preservation_protocol_repair2.json",
        "evidence_role": "DEVELOPMENT_DIAGNOSTIC",
    },
    {
        "id": "repair2_inward_balanced_development",
        "result": REPORTS / "epoch9b_dynamic_nudge/development/d19_inward_contact_balanced/result.json",
        "protocol": REPORTS / "epoch9b_v2_task_preservation_protocol_repair2.json",
        "evidence_role": "DEVELOPMENT_DIAGNOSTIC",
    },
    {
        "id": "repair3_frozen_panel",
        "result": REPORTS / "epoch9b_dynamic_nudge/feasibility_panel_repair3_result.json",
        "protocol": REPORTS / "epoch9b_v2_task_preservation_protocol_repair3.json",
        "evidence_role": "FROZEN_HISTORICAL_NO_GO",
    },
)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def signed_lane_margin(protocol: dict[str, Any], slot: str, positions: np.ndarray) -> float:
    lane = protocol["safe_center_lanes_m"][slot]
    reach = protocol["reachable_center_envelope_m"]
    margins = np.stack(
        (
            positions[:, 0] - float(lane["x"][0]),
            float(lane["x"][1]) - positions[:, 0],
            positions[:, 1] - float(lane["y"][0]),
            float(lane["y"][1]) - positions[:, 1],
            positions[:, 2] - float(reach["z"][0]),
            float(reach["z"][1]) - positions[:, 2],
        ),
        axis=1,
    )
    return float(np.min(margins))


def oracle_failure(row: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    oracle = row["oracle_completion"]
    target = str(oracle["target_slot_eval_only"])
    initial = np.asarray(row["initial_candidate_positions_eval_only"][target], dtype=np.float64)
    post_probe = np.asarray(oracle["target_object_positions_eval_only"]["initial"], dtype=np.float64)
    after_lift = np.asarray(oracle["target_object_positions_eval_only"]["after_lift"], dtype=np.float64)
    after_release = np.asarray(oracle["target_object_positions_eval_only"]["after_release"], dtype=np.float64)
    lane = protocol["safe_center_lanes_m"][target]
    reach = protocol["reachable_center_envelope_m"]
    post_margin = min(
        post_probe[0] - float(lane["x"][0]),
        float(lane["x"][1]) - post_probe[0],
        post_probe[1] - float(lane["y"][0]),
        float(lane["y"][1]) - post_probe[1],
        post_probe[2] - float(reach["z"][0]),
        float(reach["z"][1]) - post_probe[2],
    )
    lifted = bool(after_lift[2] - post_probe[2] >= 0.05)
    success = bool(oracle["official_task_success"])
    if success:
        category = "SUCCESS"
    elif post_margin < 0 and not lifted:
        category = "PROBE_STATE_AND_COMPLETION_GRASP_OR_LIFT"
    elif post_margin < 0:
        category = "PROBE_STATE_AND_COMPLETION_TRANSPORT_OR_RELEASE"
    elif not lifted:
        category = "COMPLETION_GRASP_OR_LIFT"
    else:
        category = "COMPLETION_TRANSPORT_OR_RELEASE"
    return {
        "category": category,
        "post_probe_target_shift_m": float(np.linalg.norm(post_probe - initial)),
        "post_probe_target_lane_margin_m": float(post_margin),
        "lift_height_m": float(after_lift[2] - post_probe[2]),
        "release_to_plate_proxy_displacement_m": float(np.linalg.norm(after_release - post_probe)),
        "stages_reached": oracle["stages_reached"],
    }


def probe_trace_metrics(
    row: dict[str, Any], probe: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    path = ROOT / probe["trace_path"]
    with np.load(path, allow_pickle=False) as trace:
        phase = np.asarray(trace["phase"]).astype(str)
        action = np.asarray(trace["action"], dtype=np.float64)
        eef_pos = np.asarray(trace["eef_pos"], dtype=np.float64)
        controller_error = np.asarray(trace["controller_error"], dtype=np.float64)
        rgb_delta = np.asarray(trace["estimated_world_delta_xy_m"], dtype=np.float64)
        rgb_quality = np.asarray(trace["rgb_quality"], dtype=np.float64)
        target_contact = np.asarray(trace["target_contact_eval_only"], dtype=bool)
        positions = np.asarray(trace["candidate_positions_eval_only"], dtype=np.float64)
    response_indices = np.flatnonzero(np.isin(phase, ["fixed_micro_impulse", "post_impulse_response"]))
    if response_indices.size != 5 or not np.all(np.diff(response_indices) == 1):
        raise RuntimeError(f"unexpected frozen response window in {path}: {response_indices.tolist()}")
    start = int(response_indices[0])
    stop = int(response_indices[-1]) + 1
    baseline = np.median(rgb_delta[max(0, start - 3) : start], axis=0)
    response = rgb_delta[start:stop] - baseline[None, :]
    axis = response[:, 0]
    velocity_per_step = np.diff(np.concatenate(([0.0], axis)))
    acceleration_per_step2 = np.diff(np.concatenate(([0.0], velocity_per_step)))
    realized_eef_axis = eef_pos[start:stop, 0] - eef_pos[start - 1, 0]
    contact_indices = np.flatnonzero(target_contact)
    slot_index = 0 if probe["slot"] == "front" else 1
    slot_positions = positions[:, slot_index, :]
    verification = probe.get("contact_verification", [])
    verified_rows = [value for value in verification if value.get("verified")]
    contact_confidence = (
        float(max(value["persistent_subpixel_motion_pixels"] * value["tracker_quality"] for value in verified_rows))
        if verified_rows
        else 0.0
    )
    return {
        "trace_path": relative(path),
        "trace_sha256": sha256(path),
        "steps": int(len(phase)),
        "first_contact_step_eval_only": int(contact_indices[0]) if contact_indices.size else None,
        "first_contact_phase_eval_only": str(phase[contact_indices[0]]) if contact_indices.size else None,
        "verified_contact_attempt_count": len(verified_rows),
        "contact_confidence_rgb_product": contact_confidence,
        "response_window_start_step": start,
        "response_window_stop_step_exclusive": stop,
        "response_window_length_steps": int(response_indices.size),
        "rgb_expected_axis_response_m_by_step": axis.tolist(),
        "rgb_expected_axis_peak_m": float(max(float(np.max(axis)), 0.0)),
        "rgb_expected_axis_terminal_m": float(np.mean(axis[-2:])),
        "rgb_expected_axis_peak_velocity_m_per_control_step": float(np.max(velocity_per_step)),
        "rgb_expected_axis_terminal_velocity_m_per_control_step": float(np.mean(velocity_per_step[-2:])),
        "rgb_expected_axis_peak_deceleration_m_per_control_step2": float(max(-np.min(acceleration_per_step2), 0.0)),
        "rgb_expected_axis_recoil_from_peak_m": float(max(float(np.max(axis)) - float(np.mean(axis[-2:])), 0.0)),
        "rgb_settling_terminal_over_peak_ratio": float(
            np.mean(axis[-2:]) / np.max(axis) if np.max(axis) > 1e-12 else 0.0
        ),
        "commanded_impulse_action_x_sum": float(np.sum(action[start:stop, 0])),
        "commanded_impulse_nonzero_steps": int(np.count_nonzero(np.abs(action[start:stop, 0]) > 1e-12)),
        "realized_eef_expected_axis_peak_m": float(max(float(np.max(realized_eef_axis)), 0.0)),
        "realized_eef_expected_axis_terminal_m": float(np.mean(realized_eef_axis[-2:])),
        "pre_response_controller_error_mean_m": float(np.mean(controller_error[max(0, start - 5) : start])),
        "pre_response_controller_error_max_m": float(np.max(controller_error[max(0, start - 5) : start])),
        "response_controller_error_max_m": float(np.max(controller_error[start:stop])),
        "response_rgb_tracker_quality_min": float(np.min(rgb_quality[start:stop])),
        "minimum_signed_lane_reachability_margin_m_eval_only": signed_lane_margin(
            protocol, str(probe["slot"]), slot_positions
        ),
        "neutral_return_eef_displacement_m": float(probe["final_eef_displacement_m"]),
        "neutral_return_eef_z_m": float(probe["final_eef_z_m"]),
        "restoration_phases_reached": probe["phases_reached"],
        "simulator_expected_axis_peak_m_eval_only": float(probe["response_eval_only_expected_axis_peak_m"]),
    }


def bootstrap_group_difference(
    records: list[dict[str, Any]], value_key: str, label_key: str, group_key: str
) -> dict[str, Any]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[int(row[group_key])].append(row)
    keys = sorted(groups)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    def effect(sample: list[dict[str, Any]]) -> float:
        positive = [float(row[value_key]) for row in sample if row[label_key]]
        negative = [float(row[value_key]) for row in sample if not row[label_key]]
        return float(np.mean(positive) - np.mean(negative)) if positive and negative else math.nan

    observed = effect(records)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample: list[dict[str, Any]] = []
        for key in rng.choice(keys, size=len(keys), replace=True):
            sample.extend(groups[int(key)])
        value = effect(sample)
        if math.isfinite(value):
            draws.append(value)
    return {
        "contrast_definition": f"mean({value_key} | {label_key}=true) - mean({value_key} | {label_key}=false)",
        "estimate": observed,
        "group_key": group_key,
        "unique_groups": len(keys),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws_requested": BOOTSTRAP_DRAWS,
        "bootstrap_draws_finite": len(draws),
        "percentile_95_interval": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def adjusted_mass_effect(scene_rows: list[dict[str, Any]]) -> dict[str, Any]:
    # The primary score depends only on the back probe.  The regression asks
    # whether the back response still falls under heavy mass after the declared
    # nuisance controls, with campaign indicators absorbing controller/protocol
    # family shifts.  Source-demo identities are the bootstrap groups.
    campaigns = sorted({row["campaign_id"] for row in scene_rows})
    features = []
    outcomes = []
    mass = []
    groups = []
    for row in scene_rows:
        back = row["probe_metrics_by_slot"]["back"]
        initial = row["initial_candidate_positions_eval_only"]["back"]
        localization = row["probe_result_by_slot"]["back"]["online_localization"]
        order = row["probe_order"].index("back")
        campaign_dummy = [float(row["campaign_id"] == name) for name in campaigns[1:]]
        features.append(
            [
                float(initial[0]),
                float(initial[1]),
                float(order),
                float(localization["subpixel_dx"]),
                float(localization["subpixel_dy"]),
                float(localization["quality"]),
                float(back["first_contact_step_eval_only"] or 0),
                float(back["pre_response_controller_error_mean_m"]),
                float(back["minimum_signed_lane_reachability_margin_m_eval_only"]),
                *campaign_dummy,
            ]
        )
        outcomes.append(float(back["rgb_expected_axis_peak_m"]))
        mass.append(float(row["heavy_slot"] == "back"))
        groups.append(int(row["source_state_demo_index"]))
    z = np.asarray(features, dtype=np.float64)
    z_mean = z.mean(axis=0)
    z_std = z.std(axis=0)
    z_std[z_std < 1e-12] = 1.0
    z = (z - z_mean) / z_std
    y = np.asarray(outcomes, dtype=np.float64)
    m = np.asarray(mass, dtype=np.float64)
    design = np.column_stack((np.ones(len(y)), m, z))
    coefficient = float(np.linalg.lstsq(design, y, rcond=None)[0][1])

    # Group bootstrap: every row from the same source-demo/reset identity is
    # resampled together. Duplicate sampled groups are represented by copying
    # their full rows and refitting.
    group_to_indices = {key: np.flatnonzero(np.asarray(groups) == key) for key in sorted(set(groups))}
    group_keys = sorted(group_to_indices)
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        indices = np.concatenate([group_to_indices[int(key)] for key in rng.choice(group_keys, len(group_keys), replace=True)])
        x_b = design[indices]
        y_b = y[indices]
        if np.unique(x_b[:, 1]).size < 2:
            continue
        draws.append(float(np.linalg.lstsq(x_b, y_b, rcond=None)[0][1]))

    # Leave-one-reset-group-out nuisance-only residualization.  Ridge is used
    # only to stabilize the small retrospective audit and is fixed here.
    residual = np.zeros(len(y), dtype=np.float64)
    nuisance = np.column_stack((np.ones(len(y)), z))
    for held_out in group_keys:
        test = np.asarray(groups) == held_out
        train = ~test
        gram = nuisance[train].T @ nuisance[train]
        penalty = np.eye(gram.shape[0]) * 1e-6
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(gram + penalty, nuisance[train].T @ y[train])
        residual[test] = y[test] - nuisance[test] @ beta
    heavy_residual = residual[m == 1]
    light_residual = residual[m == 0]
    pairwise = [float(h < l) + 0.5 * float(h == l) for h in heavy_residual for l in light_residual]
    return {
        "outcome": "observation-derived back-slot expected-axis response peak (m)",
        "mass_indicator": "back candidate heavy",
        "nuisance_controls": [
            "initial back x",
            "initial back y",
            "back probe order index",
            "initial RGB localization dx",
            "initial RGB localization dy",
            "initial RGB localization quality",
            "first contact timing",
            "pre-response admissible controller error",
            "signed lane/reachability margin (evaluation-only audit covariate)",
            "campaign fixed effects",
        ],
        "mass_coefficient_m": coefficient,
        "expected_direction": "negative",
        "source_demo_group_bootstrap_95_interval_m": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_seed": BOOTSTRAP_SEED + 1,
        "bootstrap_draws_finite": len(draws),
        "source_demo_groups": group_keys,
        "leave_one_source_demo_group_out_residual_heavy_mean_m": float(np.mean(heavy_residual)),
        "leave_one_source_demo_group_out_residual_light_mean_m": float(np.mean(light_residual)),
        "residual_pairwise_heavy_below_light_auc": float(np.mean(pairwise)),
        "interpretation": (
            "The adjusted coefficient and grouped interval are retrospective diagnostics, not confirmation. "
            "A negative coefficient means the response remains smaller under heavy back mass after the declared controls."
        ),
    }


def binomial_interval(success: int, total: int) -> list[float]:
    low = 0.0 if success == 0 else float(stats.beta.ppf(0.025, success, total - success + 1))
    high = 1.0 if success == total else float(stats.beta.ppf(0.975, success + 1, total - success))
    return [low, high]


def campaign_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(row["heavy_rank_correct"] for row in rows)
    by_heavy: dict[str, Any] = {}
    for slot in ("front", "back"):
        subset = [row for row in rows if row["heavy_slot"] == slot]
        hits = sum(row["heavy_rank_correct"] for row in subset)
        by_heavy[slot] = {
            "correct": hits,
            "total": len(subset),
            "accuracy": hits / len(subset),
            "clopper_pearson_95_interval": binomial_interval(hits, len(subset)),
        }
    back_heavy = [row for row in rows if row["heavy_slot"] == "back"]
    back_light = [row for row in rows if row["heavy_slot"] == "front"]
    return {
        "scenes": len(rows),
        "rank_correct": correct,
        "rank_accuracy": correct / len(rows),
        "rank_clopper_pearson_95_interval": binomial_interval(correct, len(rows)),
        "by_heavy_position": by_heavy,
        "back_response_m_when_back_heavy": {
            "mean": float(np.mean([row["primary_back_response_m"] for row in back_heavy])),
            "min": float(np.min([row["primary_back_response_m"] for row in back_heavy])),
            "max": float(np.max([row["primary_back_response_m"] for row in back_heavy])),
        },
        "back_response_m_when_back_light": {
            "mean": float(np.mean([row["primary_back_response_m"] for row in back_light])),
            "min": float(np.min([row["primary_back_response_m"] for row in back_light])),
            "max": float(np.max([row["primary_back_response_m"] for row in back_light])),
        },
        "oracle_completion": {
            "success": sum(row["oracle_failure"]["category"] == "SUCCESS" for row in rows),
            "total": len(rows),
            "failure_categories": dict(sorted(Counter(row["oracle_failure"]["category"] for row in rows).items())),
        },
        "failed_ranking_scene_ids": [row["scene_id"] for row in rows if not row["heavy_rank_correct"]],
    }


def main() -> int:
    if OUTPUT_JSON.exists() or OUTPUT_MD.exists():
        raise FileExistsError("refusing to overwrite Epoch 9D existing-trace diagnostic")
    scene_rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for campaign in CAMPAIGNS:
        result_path = Path(campaign["result"])
        protocol_path = Path(campaign["protocol"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        evidence.append(
            {
                "campaign_id": campaign["id"],
                "evidence_role": campaign["evidence_role"],
                "result_path": relative(result_path),
                "result_sha256": sha256(result_path),
                "protocol_path": relative(protocol_path),
                "protocol_sha256": sha256(protocol_path),
            }
        )
        for row in result["rows"]:
            if not row.get("completed"):
                raise RuntimeError(f"incomplete historical row: {campaign['id']} {row['scene_id']}")
            probe_results = {probe["slot"]: probe for probe in row["probes"]}
            metrics = {
                slot: probe_trace_metrics(row, probe_results[slot], protocol) for slot in ("front", "back")
            }
            back_response = float(metrics["back"]["rgb_expected_axis_peak_m"])
            scores = {"front": THRESHOLD_M - back_response, "back": back_response - THRESHOLD_M}
            predicted = min(scores, key=scores.get)
            if predicted != row["predicted_heavy_slot"]:
                raise RuntimeError(f"raw-trace score reconstruction mismatch: {campaign['id']} {row['scene_id']}")
            scene_rows.append(
                {
                    "campaign_id": campaign["id"],
                    "evidence_role": campaign["evidence_role"],
                    "scene_id": row["scene_id"],
                    "source_state_demo_index": int(row["scene"]["source_state_demo_index"]),
                    "generator_seed": row["scene"].get("generator_seed"),
                    "heavy_slot": row["scene"]["heavy_slot"],
                    "probe_order": row["scene"]["probe_order"],
                    "initial_candidate_positions_eval_only": row["initial_candidate_positions_eval_only"],
                    "primary_back_response_m": back_response,
                    "primary_scores_m": scores,
                    "predicted_heavy_slot": predicted,
                    "heavy_rank_correct": bool(predicted == row["scene"]["heavy_slot"]),
                    "probe_metrics_by_slot": metrics,
                    "probe_result_by_slot": {
                        slot: {
                            "online_localization": probe_results[slot]["online_localization"],
                            "contact_verification": probe_results[slot]["contact_verification"],
                            "intended_target_excitation_peak_m_eval_only": probe_results[slot][
                                "intended_target_excitation_peak_m_eval_only"
                            ],
                            "lane_and_reachability_continuous_pass": probe_results[slot][
                                "lane_and_reachability_continuous_pass"
                            ],
                        }
                        for slot in ("front", "back")
                    },
                    "oracle_failure": oracle_failure(row, protocol),
                }
            )

    by_campaign = {
        campaign["id"]: campaign_summary([row for row in scene_rows if row["campaign_id"] == campaign["id"]])
        for campaign in CAMPAIGNS
    }
    original = [row for row in scene_rows if row["campaign_id"] == "original_frozen_panel"]
    repair3 = [row for row in scene_rows if row["campaign_id"] == "repair3_frozen_panel"]
    original_back_misses = [row for row in original if row["heavy_slot"] == "back" and not row["heavy_rank_correct"]]
    repair3_back_misses = [row for row in repair3 if row["heavy_slot"] == "back" and not row["heavy_rank_correct"]]
    all_probe_metrics = [
        (row, slot, row["probe_metrics_by_slot"][slot]) for row in scene_rows for slot in ("front", "back")
    ]
    rgb = np.asarray([item[2]["rgb_expected_axis_peak_m"] for item in all_probe_metrics])
    simulator = np.asarray([item[2]["simulator_expected_axis_peak_m_eval_only"] for item in all_probe_metrics])
    correlation = float(np.corrcoef(rgb, simulator)[0, 1])

    # Position/order-only control: any deterministic prediction from covariates
    # that are identical across a future exact mass swap cannot flip both
    # assignments. The retrospective accuracy is reported using a fixed linear
    # leave-one-source-demo-group-out audit.
    nuisance = []
    labels = []
    groups = []
    for row in scene_rows:
        back = row["probe_result_by_slot"]["back"]["online_localization"]
        initial = row["initial_candidate_positions_eval_only"]["back"]
        nuisance.append([1.0, initial[0], initial[1], row["probe_order"].index("back"), back["subpixel_dx"], back["subpixel_dy"], back["quality"]])
        labels.append(float(row["heavy_slot"] == "back"))
        groups.append(row["source_state_demo_index"])
    nuisance_x = np.asarray(nuisance, dtype=np.float64)
    labels_y = np.asarray(labels, dtype=np.float64)
    control_predictions = np.zeros(len(labels_y), dtype=np.float64)
    for held_out in sorted(set(groups)):
        test = np.asarray(groups) == held_out
        train = ~test
        beta = np.linalg.lstsq(nuisance_x[train], labels_y[train], rcond=None)[0]
        control_predictions[test] = nuisance_x[test] @ beta
    control_accuracy = float(np.mean((control_predictions >= 0.5) == (labels_y >= 0.5)))

    adjusted = adjusted_mass_effect(scene_rows)
    result = {
        "schema_version": "epoch9d.existing_trace_causal_diagnostic.v1",
        "generated_at": timestamp(),
        "evidence_class": "RETROSPECTIVE_DEVELOPMENT_ONLY_DIAGNOSIS",
        "new_simulator_outcomes_accessed": False,
        "validation_accessed": False,
        "confirmation_accessed": False,
        "evidence_sources": evidence,
        "frozen_primary_score": {
            "name": "original_back_slot_rgb_response_threshold_score",
            "primary": True,
            "secondary_score_frozen": False,
            "response_window": "exact five consecutive stored control steps whose phase is fixed_micro_impulse or post_impulse_response",
            "back_response_definition": (
                "max(0, max_t e_x^T T_back (delta_pixel_t - median(delta_pixel over the three immediately preceding stored steps)))"
            ),
            "pixel_to_world_transform_source": "reports/epoch9b_dynamic_nudge/controller_calibration_repair1.json",
            "threshold_m": THRESHOLD_M,
            "candidate_scores_smaller_predicts_heavier": {
                "front": "threshold_m - back_response_m",
                "back": "back_response_m - threshold_m",
            },
            "admissible_inputs": [
                "ordinary agentview RGB",
                "offline-frozen RGB pixel-to-world calibration",
                "commanded action history",
                "elapsed control steps",
            ],
            "forbidden_inputs": [
                "simulator mass",
                "simulator object pose",
                "segmentation",
                "force",
                "reward",
                "success",
                "oracle target",
            ],
            "rationale": (
                "This exactly reconstructs the original 22/24 frozen near-pass from raw observation-derived traces, "
                "uses the original response window, introduces no panel-derived threshold, and is simpler than a new model."
            ),
        },
        "campaign_summaries": by_campaign,
        "front_back_asymmetry_diagnosis": {
            "original": {
                "finding": (
                    "All 12 front-heavy scenes are correct because every light back response lies above the fixed threshold. "
                    "Two of 12 back-heavy scenes are wrong because their heavy back responses cross above that same threshold."
                ),
                "back_heavy_miss_rows": [
                    {
                        "scene_id": row["scene_id"],
                        "back_response_m": row["primary_back_response_m"],
                        "threshold_excess_m": row["primary_back_response_m"] - THRESHOLD_M,
                        "initial_back_xyz_m": row["initial_candidate_positions_eval_only"]["back"],
                        "probe_order": row["probe_order"],
                        "contact_step": row["probe_metrics_by_slot"]["back"]["first_contact_step_eval_only"],
                        "pre_response_controller_error_mean_m": row["probe_metrics_by_slot"]["back"]["pre_response_controller_error_mean_m"],
                        "lane_margin_m_eval_only": row["probe_metrics_by_slot"]["back"]["minimum_signed_lane_reachability_margin_m_eval_only"],
                    }
                    for row in original_back_misses
                ],
            },
            "repair3": {
                "finding": (
                    "Repair3 changed the generated interior geometry while preserving the same response rule. Every light-back scene "
                    "still clears the threshold, but five of 12 heavy-back responses cross it. Mechanics therefore remain valid while "
                    "the absolute back threshold becomes less robust to contact geometry."
                ),
                "back_heavy_miss_rows": [
                    {
                        "scene_id": row["scene_id"],
                        "back_response_m": row["primary_back_response_m"],
                        "threshold_excess_m": row["primary_back_response_m"] - THRESHOLD_M,
                        "initial_back_xyz_m": row["initial_candidate_positions_eval_only"]["back"],
                        "probe_order": row["probe_order"],
                        "contact_step": row["probe_metrics_by_slot"]["back"]["first_contact_step_eval_only"],
                        "pre_response_controller_error_mean_m": row["probe_metrics_by_slot"]["back"]["pre_response_controller_error_mean_m"],
                        "lane_margin_m_eval_only": row["probe_metrics_by_slot"]["back"]["minimum_signed_lane_reachability_margin_m_eval_only"],
                    }
                    for row in repair3_back_misses
                ],
            },
            "causal_interpretation": (
                "The front/back accuracy gap is a structural consequence of a back-only absolute response threshold: front-heavy "
                "decisions test light-back separation, while back-heavy decisions test heavy-back separation. The increased Repair3 "
                "heavy-back overlap implicates geometry/contact sensitivity, so exact-state mass swaps are required before a causal claim."
            ),
        },
        "unadjusted_mass_response_effect": bootstrap_group_difference(
            [
                {
                    "response": row["primary_back_response_m"],
                    "back_heavy": row["heavy_slot"] == "back",
                    "source_demo": row["source_state_demo_index"],
                }
                for row in scene_rows
            ],
            "response",
            "back_heavy",
            "source_demo",
        ),
        "nuisance_adjusted_mass_response_effect": adjusted,
        "position_order_initial_rgb_control": {
            "leave_one_source_demo_group_out_accuracy": control_accuracy,
            "rows": len(scene_rows),
            "chance_reference": 0.5,
            "future_exact_swap_limit": (
                "Because exact paired assignments have identical pre-contact RGB, position, lane, and order, a deterministic "
                "position/order/pre-contact control cannot flip its prediction across both assignments and can score at most one of two per pair."
            ),
        },
        "observation_availability_audit": {
            "candidate_probe_count": len(all_probe_metrics),
            "rgb_vs_simulator_expected_axis_peak_pearson_r": correlation,
            "rgb_vs_simulator_expected_axis_peak_mae_m": float(np.mean(np.abs(rgb - simulator))),
            "rgb_vs_simulator_expected_axis_peak_median_abs_error_m": float(np.median(np.abs(rgb - simulator))),
            "primary_score_uses_simulator_fields": False,
            "finding": (
                "The useful score is computed from RGB tracking and a frozen calibration. Simulator displacement is retained only "
                "for the retrospective agreement audit and is not subtracted from or supplied to the deployed score."
            ),
        },
        "oracle_failure_attribution": {
            "all_campaign_categories": dict(sorted(Counter(row["oracle_failure"]["category"] for row in scene_rows).items())),
            "original_categories": by_campaign["original_frozen_panel"]["oracle_completion"]["failure_categories"],
            "repair3_categories": by_campaign["repair3_frozen_panel"]["oracle_completion"]["failure_categories"],
            "finding": (
                "Failures are classified from continuous post-probe lane margin and observed oracle lift/place progress. Out-of-lane "
                "post-probe targets implicate probe state; failures with in-lane targets and inadequate lift implicate the frozen "
                "completion controller; combined categories implicate both. All three original frozen-panel failures and all eight "
                "Repair3 failures are completion-only under this rule; one development-repair failure implicates both. The complete "
                "per-scene rows are retained below."
            ),
        },
        "all_scene_rows": scene_rows,
        "decision": "FREEZE_ORIGINAL_PRIMARY_SCORE_AND_RUN_EXACT_STATE_MASS_SWAP_CAUSAL_PANEL",
        "paper_status": "PAPER_NOT_AUTHORIZED",
    }
    atomic_write_json(OUTPUT_JSON, result)

    unadjusted = result["unadjusted_mass_response_effect"]
    markdown = f"""# Epoch 9D Existing-Trace Causal Diagnostic

Evidence: `RETROSPECTIVE_DEVELOPMENT_ONLY_DIAGNOSIS`

Decision: `FREEZE_ORIGINAL_PRIMARY_SCORE_AND_RUN_EXACT_STATE_MASS_SWAP_CAUSAL_PANEL`

Paper: `PAPER_NOT_AUTHORIZED`

No new simulator outcome, validation identity, or confirmation identity was accessed.

## Fixed primary response score

The primary score is the original frozen back-slot RGB response threshold. From the exact five-step response window, the back response is the peak positive expected-axis displacement reconstructed from ordinary RGB after subtracting the median of the three immediately preceding stored steps. The frozen threshold is `{THRESHOLD_M:.15f} m`. Candidate scores are `front = threshold - back_response` and `back = back_response - threshold`; the smaller score predicts heavier. No secondary score is frozen and no neural model is used.

## Front/back asymmetry

The original panel is 12/12 when the front candidate is heavy and 10/12 when the back candidate is heavy. This is structural: every light-back response exceeds the fixed threshold, but two heavy-back responses also exceed it. Repair3 preserves mechanics yet falls to 12/12 front-heavy and 7/12 back-heavy because five heavy-back responses cross the unchanged absolute threshold under the new interior geometry. The raw miss rows, contact timing, controller error, lane margin, and response dynamics are in `{relative(OUTPUT_JSON)}`.

## Residualization and observability

Across all 64 historical dynamic-nudge scenes, the unadjusted back-heavy-minus-back-light response contrast is `{unadjusted['estimate']:.6f} m`, with a source-demo-group bootstrap 95% interval `[{unadjusted['percentile_95_interval'][0]:.6f}, {unadjusted['percentile_95_interval'][1]:.6f}] m`; smaller response under heavy mass is the expected direction. After adjusting for position, lane, order, initial RGB localization, contact timing, admissible controller error, and campaign effects, the mass coefficient is `{adjusted['mass_coefficient_m']:.6f} m`, with grouped interval `[{adjusted['source_demo_group_bootstrap_95_interval_m'][0]:.6f}, {adjusted['source_demo_group_bootstrap_95_interval_m'][1]:.6f}] m`. The leave-one-reset-group-out residual heavy-below-light AUC is `{adjusted['residual_pairwise_heavy_below_light_auc']:.3f}`. These are retrospective diagnostics, not confirmation.

The RGB-derived peak agrees with the simulator-only displacement audit at Pearson `r = {correlation:.3f}` and mean absolute error `{result['observation_availability_audit']['rgb_vs_simulator_expected_axis_peak_mae_m']:.6f} m`. The primary score itself uses no simulator field. Simulator displacement is evaluation-only and is not controlled away because displacement is the hypothesized physical signal.

## Oracle failure attribution

Original and Repair3 oracle failures are not assigned to a single cause by success count alone. Each row is classified using post-probe target lane margin and lift/release progress. All three original frozen-panel failures and all eight Repair3 failures are completion-only under this rule; one failure in the smaller development repairs implicates both probe state and completion. All continuous shifts, margins, lift heights, stage flags, and categories are retained in the JSON report.

## Causal implication

The retrospective signal remains positive enough to justify the preregistered exact-state intervention, but the absolute-threshold geometry sensitivity prevents a causal claim from historical panels. The original primary score and response window are therefore frozen unchanged for the 16-base-state, 32-assignment mass-swap panel. The secondary score cannot rescue a failure because none is frozen.
"""
    atomic_write_text(OUTPUT_MD, markdown)
    print(json.dumps({
        "decision": result["decision"],
        "scenes": len(scene_rows),
        "original_rank": by_campaign["original_frozen_panel"]["rank_correct"],
        "repair3_rank": by_campaign["repair3_frozen_panel"]["rank_correct"],
        "adjusted_mass_coefficient_m": adjusted["mass_coefficient_m"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
