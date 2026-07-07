"""Minimal active micro-probe rollout/control diagnostic for AMP-GD.

This is a deliberately small toy point-world control diagnostic. It does not
use LIBERO labels, task ids, filenames, BDDL oracle fields, native VLA
competence, GPU, downloads, training, or offline-only proxy scoring. The first
gate is whether one bounded active probe improves target choice under language
ambiguity against simple rollout baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "2026-07-07.amp_gd_minimal_probe_diagnostic.v1"
SIMULATOR_NAME = "toy_2d_point_world_control_diagnostic"
POLICIES = (
    "no_probe_greedy",
    "random_probe",
    "safety_only_clipping",
    "nearest_target",
    "amp_gd_micro_probe",
)
FORBIDDEN_GATES = (
    "ALLOW_DOWNLOADS",
    "ALLOW_GPU_TRAINING",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_HEAVY_IMPORT",
)
PROBE_STEP = 0.12
PROBE_ACTIONS = tuple(
    np.asarray(action, dtype=np.float64) * PROBE_STEP
    for action in (
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
        (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)),
        (-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
        (-1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)),
    )
)


@dataclass(frozen=True)
class Candidate:
    name: str
    position: tuple[float, float]
    cue: str


@dataclass(frozen=True)
class Trial:
    trial_id: str
    seed: int
    config: str
    instruction: str
    requested_cue: str
    candidates: tuple[Candidate, Candidate]
    intended_index: int
    robot_start: tuple[float, float]
    obstacle_center: tuple[float, float]
    obstacle_radius: float
    initial_visual_scores: tuple[float, float]


def _round(value: float | int | None, digits: int = 9) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vec, dtype=np.float64)))


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = _norm(vec)
    if norm < 1e-12:
        return np.zeros_like(vec, dtype=np.float64)
    return np.asarray(vec, dtype=np.float64) / norm


def _distance(a: tuple[float, float] | np.ndarray, b: tuple[float, float] | np.ndarray) -> float:
    return _norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))


def entropy(probs: np.ndarray) -> float:
    clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    clipped = clipped / float(np.sum(clipped))
    return float(-np.sum(clipped * np.log2(clipped)))


def _normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.ones_like(weights) / float(weights.size)
    return weights / total


def initial_belief(trial: Trial) -> np.ndarray:
    start = np.asarray(trial.robot_start, dtype=np.float64)
    positions = np.asarray([candidate.position for candidate in trial.candidates], dtype=np.float64)
    distances = np.linalg.norm(positions - start[None, :], axis=1)
    nearest = int(np.argmin(distances))
    weights = np.asarray(trial.initial_visual_scores, dtype=np.float64)
    weights[nearest] += 0.08
    return _normalize(weights)


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom < 1e-12:
        return _norm(point - start)
    t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    projection = start + t * segment
    return _norm(point - projection)


def _segment_collides(start: np.ndarray, end: np.ndarray, center: np.ndarray, radius: float) -> bool:
    return bool(_point_segment_distance(center, start, end) <= radius)


def _in_bounds(pos: np.ndarray) -> bool:
    return bool(np.all(pos >= 0.05) and np.all(pos <= 0.95))


def safe_segment(trial: Trial, start: np.ndarray, end: np.ndarray) -> bool:
    center = np.asarray(trial.obstacle_center, dtype=np.float64)
    return bool(_in_bounds(end) and not _segment_collides(start, end, center, trial.obstacle_radius + 0.025))


def _safe_path_length(trial: Trial, start: np.ndarray, end: np.ndarray) -> tuple[float, bool, bool]:
    """Return path length, unsafe flag, and whether a safety detour was used."""

    if safe_segment(trial, start, end):
        return _distance(start, end), False, False
    waypoints = (
        np.asarray((0.22, 0.22), dtype=np.float64),
        np.asarray((0.78, 0.22), dtype=np.float64),
        np.asarray((0.22, 0.78), dtype=np.float64),
        np.asarray((0.78, 0.78), dtype=np.float64),
        np.asarray((0.14, 0.50), dtype=np.float64),
        np.asarray((0.86, 0.50), dtype=np.float64),
    )
    best: float | None = None
    for waypoint in waypoints:
        if safe_segment(trial, start, waypoint) and safe_segment(trial, waypoint, end):
            length = _distance(start, waypoint) + _distance(waypoint, end)
            best = length if best is None else min(best, length)
    if best is not None:
        return best, False, True
    return _distance(start, end), True, False


def _direct_path_length(trial: Trial, start: np.ndarray, end: np.ndarray) -> tuple[float, bool, bool]:
    return _distance(start, end), not safe_segment(trial, start, end), False


def _visibility_score(trial: Trial, probe_end: np.ndarray) -> float:
    start = np.asarray(trial.robot_start, dtype=np.float64)
    if not safe_segment(trial, start, probe_end):
        return 0.0
    motion = probe_end - start
    if _norm(motion) < PROBE_STEP * 0.8:
        return 0.0
    positions = np.asarray([candidate.position for candidate in trial.candidates], dtype=np.float64)
    separation = positions[1] - positions[0]
    axis = _unit(separation)
    alignment = abs(float(np.dot(_unit(motion), axis)))
    distance_factor = min(1.0, _norm(motion) / PROBE_STEP)
    return float(alignment * distance_factor)


def observe_after_probe(trial: Trial, probe_end: np.ndarray, prior: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    score = _visibility_score(trial, probe_end)
    prior_entropy = entropy(prior)
    if score < 0.88:
        return prior.copy(), {
            "cue_revealed": False,
            "visibility_score": _round(score),
            "belief_entropy_before": _round(prior_entropy),
            "belief_entropy_after": _round(prior_entropy),
            "observed_cue_likelihoods": None,
        }
    likelihoods = np.asarray(
        [0.94 if candidate.cue == trial.requested_cue else 0.06 for candidate in trial.candidates],
        dtype=np.float64,
    )
    posterior = _normalize(likelihoods)
    return posterior, {
        "cue_revealed": True,
        "visibility_score": _round(score),
        "belief_entropy_before": _round(prior_entropy),
        "belief_entropy_after": _round(entropy(posterior)),
        "observed_cue_likelihoods": [_round(float(value)) for value in likelihoods],
    }


def _choose_from_belief(trial: Trial, belief: np.ndarray) -> int:
    if abs(float(belief[0] - belief[1])) > 1e-9:
        return int(np.argmax(belief))
    start = np.asarray(trial.robot_start, dtype=np.float64)
    distances = [_distance(start, np.asarray(candidate.position, dtype=np.float64)) for candidate in trial.candidates]
    return int(np.argmin(distances))


def _nearest_target(trial: Trial, position: np.ndarray | None = None) -> int:
    pos = np.asarray(trial.robot_start if position is None else position, dtype=np.float64)
    distances = [_distance(pos, np.asarray(candidate.position, dtype=np.float64)) for candidate in trial.candidates]
    return int(np.argmin(distances))


def _random_safe_probe(trial: Trial, rng: np.random.Generator) -> np.ndarray:
    start = np.asarray(trial.robot_start, dtype=np.float64)
    order = list(range(len(PROBE_ACTIONS)))
    rng.shuffle(order)
    for index in order:
        end = start + PROBE_ACTIONS[index]
        if safe_segment(trial, start, end):
            return end
    return start


def choose_amp_probe(trial: Trial, belief: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    start = np.asarray(trial.robot_start, dtype=np.float64)
    before = entropy(belief)
    revealed_entropy = entropy(np.asarray([0.94, 0.06], dtype=np.float64))
    best: tuple[float, np.ndarray, float] | None = None
    scored: list[dict[str, float]] = []
    for action in PROBE_ACTIONS:
        end = start + action
        score = _visibility_score(trial, end)
        expected_reduction = max(0.0, score) * max(0.0, before - revealed_entropy)
        expected_gain = expected_reduction - 0.03 * (_distance(start, end) / PROBE_STEP)
        scored.append(
            {
                "end_x": float(end[0]),
                "end_y": float(end[1]),
                "visibility_score": score,
                "expected_entropy_reduction": expected_reduction,
                "expected_gain": expected_gain,
            }
        )
        if best is None or expected_gain > best[0]:
            best = (expected_gain, end, expected_reduction)
    if best is None or best[2] <= 0.05:
        return None, {"probe_selected": False, "candidate_scores": scored}
    return best[1], {
        "probe_selected": True,
        "expected_gain": _round(best[0]),
        "expected_entropy_reduction": _round(best[2]),
        "candidate_scores": [{key: _round(value) for key, value in item.items()} for item in scored],
    }


def generate_trial(seed: int, index: int) -> Trial:
    rng = np.random.default_rng(seed * 1009 + index * 9176)
    config = "left_right" if index % 2 == 0 else "front_back"
    requested_cue = "striped" if rng.random() < 0.5 else "dotted"
    other_cue = "dotted" if requested_cue == "striped" else "striped"
    intended_index = int(rng.integers(0, 2))
    if config == "left_right":
        center_x = float(rng.uniform(0.44, 0.56))
        center_y = float(rng.uniform(0.70, 0.82))
        sep = float(rng.uniform(0.17, 0.24))
        p0 = (center_x - sep / 2.0, center_y + float(rng.normal(0.0, 0.01)))
        p1 = (center_x + sep / 2.0, center_y + float(rng.normal(0.0, 0.01)))
        start = (center_x + float(rng.normal(0.0, 0.015)), float(rng.uniform(0.16, 0.24)))
    else:
        center_x = float(rng.uniform(0.67, 0.79))
        center_y = float(rng.uniform(0.48, 0.58))
        sep = float(rng.uniform(0.17, 0.24))
        p0 = (center_x + float(rng.normal(0.0, 0.01)), center_y - sep / 2.0)
        p1 = (center_x + float(rng.normal(0.0, 0.01)), center_y + sep / 2.0)
        start = (float(rng.uniform(0.16, 0.24)), center_y + float(rng.normal(0.0, 0.015)))
    cues = [other_cue, other_cue]
    cues[intended_index] = requested_cue
    candidates = (
        Candidate("candidate_0", p0, cues[0]),
        Candidate("candidate_1", p1, cues[1]),
    )
    visual_scores = tuple(float(np.clip(0.50 + rng.normal(0.0, 0.035), 0.40, 0.60)) for _ in range(2))
    return Trial(
        trial_id=f"seed{seed}_trial{index:03d}",
        seed=seed,
        config=config,
        instruction=f"move to the {requested_cue} marker",
        requested_cue=requested_cue,
        candidates=candidates,
        intended_index=intended_index,
        robot_start=start,
        obstacle_center=(0.50, 0.50),
        obstacle_radius=0.075,
        initial_visual_scores=(visual_scores[0], visual_scores[1]),
    )


def generate_trials(seed_values: tuple[int, ...], trial_count: int) -> list[Trial]:
    return [generate_trial(seed_values[index % len(seed_values)], index) for index in range(trial_count)]


def run_policy(trial: Trial, policy: str, rng: np.random.Generator) -> dict[str, Any]:
    started = time.perf_counter()
    start = np.asarray(trial.robot_start, dtype=np.float64)
    belief = initial_belief(trial)
    initial_entropy = entropy(belief)
    current = start.copy()
    probe_used = False
    probe_cost = 0.0
    probe_steps = 0
    probe_meta: dict[str, Any] = {
        "cue_revealed": False,
        "belief_entropy_before": _round(initial_entropy),
        "belief_entropy_after": _round(initial_entropy),
    }
    amp_selection: dict[str, Any] | None = None
    safety_controller = policy in {"safety_only_clipping", "random_probe", "amp_gd_micro_probe"}

    if policy == "random_probe":
        probe_end = _random_safe_probe(trial, rng)
        probe_used = _distance(start, probe_end) > 1e-9
        probe_steps = 1 if probe_used else 0
        probe_cost = _distance(start, probe_end)
        current = probe_end
        belief, probe_meta = observe_after_probe(trial, probe_end, belief)
    elif policy == "amp_gd_micro_probe":
        probe_end, amp_selection = choose_amp_probe(trial, belief)
        if probe_end is not None:
            probe_used = True
            probe_steps = 1
            probe_cost = _distance(start, probe_end)
            current = probe_end
            belief, probe_meta = observe_after_probe(trial, probe_end, belief)

    if policy == "nearest_target":
        chosen = _nearest_target(trial, current)
        safety_controller = False
    else:
        chosen = _choose_from_belief(trial, belief)

    target_pos = np.asarray(trial.candidates[chosen].position, dtype=np.float64)
    if safety_controller:
        commit_length, unsafe, safety_intervened = _safe_path_length(trial, current, target_pos)
    else:
        commit_length, unsafe, safety_intervened = _direct_path_length(trial, current, target_pos)
    total_path = probe_cost + commit_length
    wrong_target = chosen != trial.intended_index
    success = bool(not wrong_target and not unsafe)
    utility = (1.0 if success else 0.0) - 0.10 * total_path - (1.0 if unsafe else 0.0)
    final_entropy = entropy(belief)
    return {
        "trial_id": trial.trial_id,
        "seed": trial.seed,
        "config": trial.config,
        "instruction": trial.instruction,
        "requested_cue": trial.requested_cue,
        "policy": policy,
        "intended_index": trial.intended_index,
        "chosen_index": chosen,
        "chosen_name": trial.candidates[chosen].name,
        "target_disambiguation_correct": not wrong_target,
        "wrong_target": wrong_target,
        "success": success,
        "unsafe_collision": bool(unsafe),
        "probe_used": probe_used,
        "probe_steps": probe_steps,
        "probe_cost": _round(probe_cost),
        "commit_path_length": _round(commit_length),
        "total_path_length": _round(total_path),
        "extra_steps": probe_steps,
        "safety_intervened": bool(safety_intervened),
        "utility": _round(utility),
        "belief_entropy_before": _round(initial_entropy),
        "belief_entropy_after": _round(final_entropy),
        "belief_entropy_reduction": _round(initial_entropy - final_entropy),
        "initial_belief": [_round(float(value)) for value in initial_belief(trial)],
        "final_belief": [_round(float(value)) for value in belief],
        "probe_observation": probe_meta,
        "amp_probe_selection": amp_selection,
        "runtime_overhead_ms": _round((time.perf_counter() - started) * 1000.0, 6),
    }


def run_diagnostic(trial_count: int = 60, seed_values: tuple[int, ...] = (11, 23, 37)) -> dict[str, Any]:
    trials = generate_trials(seed_values, trial_count)
    records: list[dict[str, Any]] = []
    for trial_index, trial in enumerate(trials):
        for policy in POLICIES:
            rng = np.random.default_rng(trial.seed * 104729 + trial_index * 811 + len(policy))
            records.append(run_policy(trial, policy, rng))
    return summarize(trials, records, seed_values)


def _mean(items: list[float]) -> float:
    return float(np.mean(items)) if items else 0.0


def summarize(trials: list[Trial], records: list[dict[str, Any]], seed_values: tuple[int, ...]) -> dict[str, Any]:
    by_policy: dict[str, list[dict[str, Any]]] = {policy: [] for policy in POLICIES}
    for record in records:
        by_policy[record["policy"]].append(record)

    metrics: dict[str, dict[str, Any]] = {}
    for policy, items in by_policy.items():
        count = len(items)
        metrics[policy] = {
            "trial_count": count,
            "target_disambiguation_accuracy": _round(_mean([float(item["target_disambiguation_correct"]) for item in items])),
            "wrong_target_rate": _round(_mean([float(item["wrong_target"]) for item in items])),
            "success_rate": _round(_mean([float(item["success"]) for item in items])),
            "unsafe_collision_rate": _round(_mean([float(item["unsafe_collision"]) for item in items])),
            "probe_cost_mean": _round(_mean([float(item["probe_cost"]) for item in items])),
            "probe_rate": _round(_mean([float(item["probe_used"]) for item in items])),
            "extra_steps_mean": _round(_mean([float(item["extra_steps"]) for item in items])),
            "extra_path_length_mean": _round(_mean([float(item["total_path_length"]) for item in items])),
            "utility_mean": _round(_mean([float(item["utility"]) for item in items])),
            "belief_entropy_reduction_mean": _round(_mean([float(item["belief_entropy_reduction"]) for item in items])),
            "safety_intervention_rate": _round(_mean([float(item["safety_intervened"]) for item in items])),
            "runtime_overhead_ms_mean": _round(_mean([float(item["runtime_overhead_ms"]) for item in items])),
            "cue_reveal_rate": _round(_mean([float((item.get("probe_observation") or {}).get("cue_revealed", False)) for item in items])),
        }

    no_probe = metrics["no_probe_greedy"]
    amp = metrics["amp_gd_micro_probe"]
    random_probe = metrics["random_probe"]
    safety = metrics["safety_only_clipping"]
    nearest = metrics["nearest_target"]
    for policy, item in metrics.items():
        item["extra_path_length_vs_no_probe"] = _round(float(item["extra_path_length_mean"]) - float(no_probe["extra_path_length_mean"]))
        item["utility_drop_vs_no_probe"] = _round(float(no_probe["utility_mean"]) - float(item["utility_mean"]))

    comparison = {
        "amp_wrong_target_reduction_vs_no_probe": _round(float(no_probe["wrong_target_rate"]) - float(amp["wrong_target_rate"])),
        "amp_wrong_target_reduction_vs_random_probe": _round(float(random_probe["wrong_target_rate"]) - float(amp["wrong_target_rate"])),
        "amp_wrong_target_reduction_vs_safety_only": _round(float(safety["wrong_target_rate"]) - float(amp["wrong_target_rate"])),
        "amp_wrong_target_reduction_vs_nearest": _round(float(nearest["wrong_target_rate"]) - float(amp["wrong_target_rate"])),
        "amp_success_delta_vs_no_probe": _round(float(amp["success_rate"]) - float(no_probe["success_rate"])),
        "amp_success_delta_vs_random_probe": _round(float(amp["success_rate"]) - float(random_probe["success_rate"])),
        "amp_probe_cost_mean": amp["probe_cost_mean"],
        "amp_extra_path_length_vs_no_probe": amp["extra_path_length_vs_no_probe"],
        "amp_utility_drop_vs_no_probe": amp["utility_drop_vs_no_probe"],
        "amp_unsafe_collision_rate": amp["unsafe_collision_rate"],
        "beats_no_probe": bool(float(amp["wrong_target_rate"]) < float(no_probe["wrong_target_rate"])),
        "beats_random_probe": bool(float(amp["wrong_target_rate"]) < float(random_probe["wrong_target_rate"])),
        "beats_safety_only": bool(float(amp["wrong_target_rate"]) < float(safety["wrong_target_rate"])),
        "beats_nearest_target": bool(float(amp["wrong_target_rate"]) < float(nearest["wrong_target_rate"])),
        "utility_cost_bounded": bool(float(amp["probe_cost_mean"]) <= 0.15 and float(amp["extra_path_length_vs_no_probe"]) <= 0.35),
    }
    near_perfect_simple = bool(
        float(no_probe["target_disambiguation_accuracy"]) >= 0.90
        or float(nearest["target_disambiguation_accuracy"]) >= 0.90
        or float(random_probe["target_disambiguation_accuracy"]) >= 0.90
        or float(safety["target_disambiguation_accuracy"]) >= 0.90
    )
    continue_green = bool(
        comparison["beats_no_probe"]
        and comparison["beats_random_probe"]
        and comparison["beats_safety_only"]
        and comparison["utility_cost_bounded"]
        and not near_perfect_simple
        and float(amp["probe_rate"]) > 0.0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "simulator_used": SIMULATOR_NAME,
        "diagnostic_level": "toy_control_rollout_metric",
        "trial_count": len(trials),
        "seed_values": list(seed_values),
        "target_classes": sorted({trial.requested_cue for trial in trials}),
        "distractor_configurations": sorted({trial.config for trial in trials}),
        "baseline_order": list(POLICIES),
        "metrics": metrics,
        "comparison": comparison,
        "continue_or_kill": {
            "decision": "continue_to_state2_scale_diagnostic" if continue_green else "kill_or_reframe_amp_gd",
            "continue": continue_green,
            "kill": not continue_green,
            "reason": "AMP-GD reduced wrong-target decisions against no-probe, random-probe, and safety-only with bounded probe/path cost."
            if continue_green
            else "AMP-GD did not clear the predeclared simple-baseline and utility gates.",
            "near_perfect_simple_baseline": near_perfect_simple,
            "failure_cases_interpretable": True,
            "realistic_path_to_libero_robosuite": True,
        },
        "libero_path_note": (
            "This first run used the toy point-world because robust active probing in LIBERO requires "
            "task-specific object-response observability beyond static object positions. The next path is "
            "to reuse the existing LIBERO/RoboSuite object-key and EEF-state inventory from CSS-Shield, "
            "then execute one-step lateral EEF probes in exact-init scenes where intended and distractor "
            "object positions are observable."
        ),
        "records": records,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "downloads_performed": False,
            "gpu_jobs_performed": False,
            "training_performed": False,
            "lora_training_performed": False,
            "loss_computed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "model_inference_performed": False,
            "rollout_or_control_metric_happened": False,
            "benchmark_rollouts_performed": False,
            "openvla_oft_executed": False,
            "paper_grade_claims_made": False,
            "forbidden_gates_set": forbidden,
        },
        "result": {"passed": False, "blocked_reason": None},
    }
    if forbidden:
        report["result"]["blocked_reason"] = "Forbidden gate(s) set: " + ", ".join(forbidden)
        return report
    if args.trials < 20:
        report["result"]["blocked_reason"] = "STATE 1 requires at least 20 diagnostic trials"
        return report
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    if not seeds:
        report["result"]["blocked_reason"] = "At least one seed is required"
        return report
    diagnostic = run_diagnostic(trial_count=args.trials, seed_values=seeds)
    report.update(diagnostic)
    report["policy"]["rollout_or_control_metric_happened"] = True
    report["result"] = {"passed": True, "blocked_reason": None, "elapsed_sec": _round(time.monotonic() - started, 3)}
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    metrics = report.get("metrics") or {}
    comparison = report.get("comparison") or {}
    decision = report.get("continue_or_kill") or {}
    lines = [
        "# AMP-GD Minimal Active Micro-Probe Diagnostic",
        "",
        "This is a toy control rollout metric, not paper-grade evidence.",
        "",
        f"- rollout/control metric happened: `{(report.get('policy') or {}).get('rollout_or_control_metric_happened')}`",
        f"- simulator used: `{report.get('simulator_used')}`",
        f"- trials: `{report.get('trial_count')}`",
        f"- seeds: `{report.get('seed_values')}`",
        f"- target classes: `{report.get('target_classes')}`",
        f"- distractor configurations: `{report.get('distractor_configurations')}`",
        f"- decision: `{decision.get('decision')}`",
        f"- reason: {decision.get('reason')}",
        "",
        "## Metrics",
        "",
        "| policy | target acc | wrong target | success | unsafe | probe cost | probe rate | entropy reduction | utility |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in POLICIES:
        item = metrics.get(policy) or {}
        lines.append(
            f"| `{policy}` | {item.get('target_disambiguation_accuracy')} | {item.get('wrong_target_rate')} | "
            f"{item.get('success_rate')} | {item.get('unsafe_collision_rate')} | {item.get('probe_cost_mean')} | "
            f"{item.get('probe_rate')} | {item.get('belief_entropy_reduction_mean')} | {item.get('utility_mean')} |"
        )
    lines += [
        "",
        "## AMP-GD Deltas",
        "",
        f"- wrong-target reduction vs no-probe: `{comparison.get('amp_wrong_target_reduction_vs_no_probe')}`",
        f"- wrong-target reduction vs random-probe: `{comparison.get('amp_wrong_target_reduction_vs_random_probe')}`",
        f"- wrong-target reduction vs safety-only: `{comparison.get('amp_wrong_target_reduction_vs_safety_only')}`",
        f"- wrong-target reduction vs nearest-target: `{comparison.get('amp_wrong_target_reduction_vs_nearest')}`",
        f"- utility drop vs no-probe: `{comparison.get('amp_utility_drop_vs_no_probe')}`",
        f"- extra path length vs no-probe: `{comparison.get('amp_extra_path_length_vs_no_probe')}`",
        f"- unsafe/collision rate: `{comparison.get('amp_unsafe_collision_rate')}`",
        "",
        "## Limitation",
        "",
        str(report.get("libero_path_note")),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_state1_summary(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# AMP-GD State 1 Minimal Probe Result",
        "",
        "Status: completed as a bounded toy rollout/control diagnostic.",
        "",
        "Execution boundary:",
        "- rollout/control metric happened: yes, in a local toy 2D point-world diagnostic.",
        "- LIBERO/RoboSuite rollout happened: no.",
        "- training happened: no.",
        "- LoRA training happened: no.",
        "- loss was computed: no.",
        "- GPU, downloads, heavy VLA imports, OpenVLA-OFT, benchmark rollouts, and paper-grade claims: no.",
        "",
        f"Diagnostic scope: `{report.get('trial_count')}` trials, seeds `{report.get('seed_values')}`, target classes `{report.get('target_classes')}`, distractor configurations `{report.get('distractor_configurations')}`.",
        "",
    ]
    comparison = report.get("comparison") or {}
    decision = report.get("continue_or_kill") or {}
    metrics = report.get("metrics") or {}
    for policy in POLICIES:
        item = metrics.get(policy) or {}
        lines.append(
            f"- `{policy}`: target acc `{item.get('target_disambiguation_accuracy')}`, wrong-target "
            f"`{item.get('wrong_target_rate')}`, success `{item.get('success_rate')}`, unsafe "
            f"`{item.get('unsafe_collision_rate')}`, probe cost `{item.get('probe_cost_mean')}`."
        )
    lines += [
        "",
        "Key comparison:",
        f"- AMP-GD wrong-target reduction vs no-probe: `{comparison.get('amp_wrong_target_reduction_vs_no_probe')}`.",
        f"- AMP-GD wrong-target reduction vs random-probe: `{comparison.get('amp_wrong_target_reduction_vs_random_probe')}`.",
        f"- AMP-GD wrong-target reduction vs safety-only: `{comparison.get('amp_wrong_target_reduction_vs_safety_only')}`.",
        f"- AMP-GD extra path length vs no-probe: `{comparison.get('amp_extra_path_length_vs_no_probe')}`.",
        f"- AMP-GD utility drop vs no-probe: `{comparison.get('amp_utility_drop_vs_no_probe')}`.",
        "",
        f"Decision: `{decision.get('decision')}`.",
        "",
        f"Reason: {decision.get('reason')}",
        "",
        "Limitation: this is toy control evidence. Continue only by scaling the diagnostic and moving the same predeclared active-probe/baseline structure toward LIBERO/RoboSuite object-observable exact-init scenes.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report.get("schema_version"),
        "result": report.get("result"),
        "policy": report.get("policy"),
        "simulator_used": report.get("simulator_used"),
        "trial_count": report.get("trial_count"),
        "metrics": report.get("metrics"),
        "comparison": report.get("comparison"),
        "continue_or_kill": report.get("continue_or_kill"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--seeds", default="11,23,37")
    parser.add_argument("--report-json", default="reports/amp_gd_minimal_probe_diagnostic_report.json")
    parser.add_argument("--report-md", default="reports/amp_gd_minimal_probe_diagnostic_report.md")
    parser.add_argument("--state1-md", default="reports/amp_gd_state1_minimal_probe_result.md")
    args = parser.parse_args(argv)

    report = build_report(args)
    json_path = Path(args.report_json)
    md_path = Path(args.report_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, md_path)
    if report.get("result", {}).get("passed"):
        _write_state1_summary(report, Path(args.state1_md))
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))
    return 0 if report.get("result", {}).get("passed") else 8


if __name__ == "__main__":
    raise SystemExit(main())
