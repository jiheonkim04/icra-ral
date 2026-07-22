"""Execute and adjudicate the frozen Epoch 10B development Stage 0 panel.

The runner has an outcome-blind preregistration mode and three outcome-opening
modes.  It deliberately keeps checkpoint action inference, fresh-controller
simulation, and official closed-loop evaluation in separate processes/jobs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_epoch10b_fresh_controller_assay as assay


SCHEMA_VERSION = 1
CAMPAIGN = "epoch10b_icae_fresh_controller"
HORIZON = 4
ENDPOINT = "bounded_expert_recovery_cost"
RESET_SEEDS = tuple(range(20, 35))
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_722
SUITE_STEP_CAPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
EXPECTED_ACTION_ROWS = 3_840
EXPECTED_BRANCHES = EXPECTED_ACTION_ROWS * 3
EXPECTED_ROLLOUT_ROWS = 16 * 4 * len(RESET_SEEDS)
STRONG_BASELINES = (
    "raw_mse",
    "raw_mae",
    "action_dimension_normalized_mse",
    "arm_gripper_equal_weight_mse",
    "phase_state_criticality_weighted_normalized_mse",
    "response_magnitude_control",
)
METRICS = (
    "icae",
    *STRONG_BASELINES,
    "unpaired_icae",
    "state_shuffled_icae",
)


class Stage0Error(RuntimeError):
    """A bounded, reportable Stage 0 integrity or execution failure."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, rows: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [rows] if isinstance(rows, Mapping) else list(rows)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in materialized:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), default=_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise Stage0Error("INVALID_JSONL", f"{path}:{line_number}: {exc}") from exc
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _git_head() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _windows_to_wsl(raw: str) -> str:
    path = PureWindowsPath(raw)
    drive = path.drive.rstrip(":").lower()
    if not drive:
        return raw.replace("\\", "/")
    return "/mnt/" + drive + "/" + "/".join(path.parts[1:])


def _policy_state_key(row: Mapping[str, Any]) -> str:
    return f"{row['policy_identity']}|{row['state_id']}"


def _branch_key(policy: str, state_id: str, kind: str) -> str:
    return f"stage0|{policy}|{state_id}|{kind}"


def _load_freeze(args: argparse.Namespace, *, verify_runner: bool = True) -> dict[str, Any]:
    path = Path(args.stage0_freeze)
    freeze = _read_json(path)
    stored = str(freeze.get("canonical_payload_sha256") or "")
    payload = dict(freeze)
    payload.pop("canonical_payload_sha256", None)
    if stored != _canonical_sha256(payload):
        raise Stage0Error("STAGE0_FREEZE_DRIFT", str(path))
    if verify_runner and freeze.get("runner_sha256") != _sha256_file(Path(__file__)):
        raise Stage0Error("STAGE0_RUNNER_DRIFT", str(Path(__file__)))
    action_manifest = Path(args.action_manifest)
    raw_actions = Path(args.action_rows)
    if freeze.get("action_manifest_sha256") != _sha256_file(action_manifest):
        raise Stage0Error("ACTION_MANIFEST_DRIFT", str(action_manifest))
    if freeze.get("action_rows_sha256") != _sha256_file(raw_actions):
        raise Stage0Error("ACTION_ROWS_DRIFT", str(raw_actions))
    return freeze


def _permutation_maps(states: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    by_task: dict[str, list[str]] = defaultdict(list)
    for state in states:
        by_task[str(state["suite"])].append(str(state["state_id"]))
    unpaired: dict[str, str] = {}
    for suite, state_ids in sorted(by_task.items()):
        ordered = sorted(
            state_ids,
            key=lambda item: hashlib.sha256(f"epoch10b-unpaired|{suite}|{item}".encode()).hexdigest(),
        )
        for index, target in enumerate(ordered):
            unpaired[target] = ordered[(index + 1) % len(ordered)]
    ordered_global = sorted(
        [str(row["state_id"]) for row in states],
        key=lambda item: hashlib.sha256(f"epoch10b-state-shuffle|{item}".encode()).hexdigest(),
    )
    shuffled = {
        target: ordered_global[(index + 1) % len(ordered_global)]
        for index, target in enumerate(ordered_global)
    }
    if any(target == source for target, source in unpaired.items()):
        raise Stage0Error("UNPAIRED_FIXED_POINT", "within-task permutation contains a fixed point")
    if any(target == source for target, source in shuffled.items()):
        raise Stage0Error("SHUFFLE_FIXED_POINT", "global permutation contains a fixed point")
    return unpaired, shuffled


def preregister(args: argparse.Namespace) -> dict[str, Any]:
    action_freeze_path = Path(args.action_freeze)
    action_manifest_path = Path(args.action_manifest)
    action_rows_path = Path(args.action_rows)
    action_freeze = _read_json(action_freeze_path)
    action_manifest = _read_json(action_manifest_path)
    actions = _read_jsonl(action_rows_path)
    if len(actions) != EXPECTED_ACTION_ROWS or len({_policy_state_key(row) for row in actions}) != EXPECTED_ACTION_ROWS:
        raise Stage0Error("ACTION_CARDINALITY", f"expected {EXPECTED_ACTION_ROWS} unique actions")
    if action_manifest.get("status") != "EPOCH10B_DEVELOPMENT_ACTION_CACHE_COMPLETE":
        raise Stage0Error("ACTION_CACHE_INCOMPLETE", str(action_manifest.get("status")))
    if int(action_manifest.get("heldout_checkpoint_actions_queried", -1)) != 0:
        raise Stage0Error("HELDOUT_ACTION_LEAKAGE", "held-out checkpoint actions were queried")
    unpaired, shuffled = _permutation_maps(action_freeze["development_states"])
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "FROZEN_BEFORE_STAGE0_DEVELOPMENT_OUTCOME_OPENING",
        "source_commit": _git_head(),
        "runner_sha256": _sha256_file(Path(__file__)),
        "host_guard_sha256": _sha256_file(Path(args.host_guard)),
        "fresh_branch_constructor_sha256": _sha256_file(Path(assay.__file__)),
        "action_freeze_sha256": _sha256_file(action_freeze_path),
        "action_manifest_sha256": _sha256_file(action_manifest_path),
        "action_rows_sha256": _sha256_file(action_rows_path),
        "selected_assay": {
            "constructor": "prefix_reconstructed_fresh_controller",
            "horizon": HORIZON,
            "endpoint": ENDPOINT,
            "score": "candidate bounded recovery cost minus independent nominal bounded recovery cost",
            "continuation": "registered expert actions t+1 through t+4; the splice step plus four continuations are scored",
        },
        "branch_panel": {
            "roles": ["nominal", "candidate", "unpaired_candidate"],
            "expected_primary_pairs": EXPECTED_ACTION_ROWS,
            "expected_branches": EXPECTED_BRANCHES,
            "fresh_environment_per_role": True,
            "common_registered_seed_within_policy_state": True,
            "unpaired_action_source_by_target_state": unpaired,
            "state_shuffled_score_source_by_target_state": shuffled,
            "unpaired_mapping_sha256": _canonical_sha256(unpaired),
            "state_shuffled_mapping_sha256": _canonical_sha256(shuffled),
            "infrastructure_retry": "one identical retry only when no valid materialized score exists",
        },
        "development_rollouts": {
            "checkpoint_count": 16,
            "tasks": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
            "task_ids": [0],
            "common_reset_seeds": list(RESET_SEEDS),
            "episodes_per_task_checkpoint": len(RESET_SEEDS),
            "expected_episode_rows": EXPECTED_ROLLOUT_ROWS,
            "official_success": "LeRobot eval_policy_all successes/native LIBERO is_success",
            "suite_step_caps": SUITE_STEP_CAPS,
            "timeout_is_failure": True,
            "unexecuted_or_invalid_is_missing_never_zero": True,
        },
        "aggregation": {
            "checkpoint_task_metric": "mean across selected states, with states equally weighted",
            "checkpoint_task_performance": "mean native success over the 15 common resets",
            "gate_concordance": "equal-task macro-average cross-lineage checkpoint-pair concordance; performance ties excluded and predictor ties score 0.5",
            "nested_stages": "both retained as repeated measures inside a whole-seed lineage; same-lineage checkpoint pairs excluded",
            "competitive_subset": "optimizer step 100 only, one checkpoint per development lineage",
            "task_centered_rank": "Spearman and Kendall tau-b between success centered within task and negative metric centered within task",
            "top_k": "top-1 and top-3 overlap per task; empirical success ties included",
            "selection_regret": "empirical best task success minus empirical success of the metric-selected checkpoint",
        },
        "performance_identifiability": {
            "fixed_quality_bands": [[0.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0], [2.0 / 3.0, 1.000000000001]],
            "three_band_rule": "all three fixed macro-success bands contain at least one checkpoint",
            "distinguishable_rule": "hierarchical paired 95% bootstrap interval for empirical best-minus-worst macro success is strictly above zero",
            "gate": "three-band rule OR distinguishable rule",
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "units": ["task", "whole_seed_lineage", "whole_demo_cluster", "common_reset"],
            "resampling": "tasks and whole lineages sampled with replacement; whole demos sampled within task; common resets sampled within task; both nested stages remain in each sampled lineage occurrence",
        },
        "stage0_gate": {
            "minimum_icae_concordance": 0.60,
            "minimum_gain_over_normalized_mse": 0.08,
            "minimum_bootstrap_probability_gain_positive": 0.90,
            "strong_baseline_clear_margin": 0.02,
            "non_domination": "pass unless every available strong equal-input baseline exceeds ICAE concordance by at least 0.02",
            "negative_control_reproduction": "a control reproduces the gain only if concordance >=0.60, gain over normalized MSE >=0.08, and grouped-bootstrap P(gain>0) >=0.90",
            "maximum_icae_step_fraction_of_exhaustive_rollout": 0.20,
            "cost_numerator": "all nominal, paired-candidate, and unpaired-candidate prefix plus scored simulator control steps; absorbing scored steps are also reported",
            "cost_denominator": "16 checkpoints * 15 resets * sum of official suite step caps",
        },
        "baselines": {
            "strong": list(STRONG_BASELINES),
            "negative_controls": ["unpaired_icae", "state_shuffled_icae"],
            "ci_mse": "NOT_IMPLEMENTED_NO_PROXY_UNDER_FROZEN_ACTION_PROTOCOL",
        },
        "leakage_at_freeze": {
            "development_simulator_scores_opened": False,
            "development_success_labels_opened": False,
            "heldout_actions_or_outcomes_opened": False,
            "confirmation_outcomes_opened": False,
        },
        "post_freeze_scientific_change_allowed": False,
    }
    payload["canonical_payload_sha256"] = _canonical_sha256(payload)
    _write_json(Path(args.stage0_freeze), payload)
    return payload


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    freeze = _load_freeze(args)
    actions = _read_jsonl(Path(args.action_rows))
    action_freeze = _read_json(Path(args.action_freeze))
    checkpoints = {str(row["policy_identity"]) for row in action_freeze["development_checkpoints"]}
    states = {str(row["state_id"]) for row in action_freeze["development_states"]}
    checks = {
        "action_rows": len(actions) == EXPECTED_ACTION_ROWS,
        "unique_action_keys": len({_policy_state_key(row) for row in actions}) == EXPECTED_ACTION_ROWS,
        "checkpoint_count": len(checkpoints) == 16,
        "state_count": len(states) == 240,
        "all_development": all(row.get("partition") == "development" for row in actions),
        "no_outcomes_read_in_action_cache": all(
            not row.get("simulator_outcome_read") and not row.get("closed_loop_success_label_read")
            for row in actions
        ),
        "heldout_sealed": int(_read_json(Path(args.action_manifest))["heldout_checkpoint_actions_queried"]) == 0,
        "freeze_status": freeze.get("status") == "FROZEN_BEFORE_STAGE0_DEVELOPMENT_OUTCOME_OPENING",
    }
    status = "EPOCH10B_STAGE0_PREFLIGHT_PASS" if all(checks.values()) else "EPOCH10B_STAGE0_PREFLIGHT_FAIL"
    return {"schema_version": 1, "campaign": CAMPAIGN, "status": status, "checks": checks}


def _branch_score(row: Mapping[str, Any]) -> float:
    return float(row["bounded_recovery_cost_by_horizon"][str(HORIZON)])


def _branch_materialized(row: Mapping[str, Any]) -> bool:
    try:
        value = _branch_score(row)
    except Exception:
        return False
    return bool(row.get("valid")) and math.isfinite(value)


def _completed_branches(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    complete: dict[str, dict[str, Any]] = {}
    attempts: dict[str, int] = defaultdict(int)
    for row in _read_jsonl(path):
        key = str(row["branch_key"])
        attempts[key] += 1
        if _branch_materialized(row):
            if key in complete:
                raise Stage0Error("DUPLICATE_VALID_BRANCH", key)
            complete[key] = row
    return complete, dict(attempts)


def _record_interrupted_pending(run_dir: Path) -> None:
    pending_path = run_dir / "active_branch.json"
    if not pending_path.is_file():
        return
    raw = pending_path.read_bytes()
    record = {
        "classification": "INCOMPLETE_BRANCH_WITHOUT_MATERIALIZED_SCORE",
        "pending_sha256": hashlib.sha256(raw).hexdigest(),
        "pending_record": json.loads(raw.decode("utf-8")),
        "retry_authorized": True,
    }
    record["attempt_sha256"] = _canonical_sha256(record)
    log = run_dir / "infrastructure_attempts.jsonl"
    existing = {row.get("attempt_sha256") for row in _read_jsonl(log)}
    if record["attempt_sha256"] not in existing:
        _append_jsonl(log, record)


def _load_demo_arrays(action_freeze: Mapping[str, Any]) -> tuple[dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], dict[tuple[str, int], dict[str, Any]]]:
    import h5py

    tasks = {(str(row["suite"]), int(row["task_id"])): dict(row) for row in action_freeze["development_tasks"]}
    demos: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for state in action_freeze["development_states"]:
        key = (str(state["suite"]), str(state["demo_name"]))
        if key in demos:
            continue
        task = tasks[(str(state["suite"]), int(state["task_id"]))]
        hdf5_path = Path(str(task.get("wsl_hdf5_path") or task["windows_hdf5_path"]))
        with h5py.File(hdf5_path, "r") as handle:
            demo = handle["data"][state["demo_name"]]
            demos[key] = (
                np.asarray(demo["states"], dtype=np.float64),
                np.asarray(demo["actions"], dtype=np.float64),
            )
    return demos, tasks


def _execute_role(
    *,
    task: Mapping[str, Any],
    states: np.ndarray,
    expert_actions: np.ndarray,
    target: Mapping[str, Any],
    action_row: Mapping[str, Any],
    first_action: np.ndarray,
    role: str,
    source_state_id: str,
    attempt: int,
    camera_size: int,
) -> dict[str, Any]:
    policy = str(action_row["policy_identity"])
    state_id = str(action_row["state_id"])
    key = _branch_key(policy, state_id, role)
    registered_seed = assay._seed_from_text(f"epoch10b-stage0|{policy}|{state_id}")
    row = assay.execute_fresh_branch(
        task=task,
        states=states,
        actions=expert_actions,
        target=target,
        frame=int(action_row["frame"]),
        first_action=np.asarray(first_action, dtype=np.float64),
        design="prefix_reconstructed_fresh_controller",
        branch_key=key,
        camera_size=int(camera_size),
        horizon=HORIZON,
        registered_seed=registered_seed,
    )
    row.update(
        {
            "schema_version": SCHEMA_VERSION,
            "campaign": CAMPAIGN,
            "stage": "development_stage0",
            "branch_role": role,
            "attempt": int(attempt),
            "policy_identity": policy,
            "lineage_cluster": action_row["lineage_cluster"],
            "optimizer_step": int(action_row["optimizer_step"]),
            "suite": action_row["suite"],
            "task_id": int(action_row["task_id"]),
            "state_id": state_id,
            "source_state_id": source_state_id,
            "demo_name": action_row["demo_name"],
            "demo_cluster": action_row["demo_cluster"],
            "phase": action_row["phase"],
            "frame": int(action_row["frame"]),
            "action_cache_sha256": action_row["executed_action_sha256"],
            "development_outcome_authorized": True,
            "heldout_outcome_opened": False,
        }
    )
    return row


def run_interventions(args: argparse.Namespace) -> dict[str, Any]:
    freeze = _load_freeze(args)
    action_freeze = _read_json(Path(args.action_freeze))
    action_rows = sorted(_read_jsonl(Path(args.action_rows)), key=lambda row: (row["policy_identity"], row["state_id"]))
    action_lookup = {_policy_state_key(row): row for row in action_rows}
    unpaired = freeze["branch_panel"]["unpaired_action_source_by_target_state"]
    run_dir = Path(args.intervention_run_dir)
    raw_path = run_dir / "branches.jsonl"
    paired_path = run_dir / "paired_scores.jsonl"
    pending_path = run_dir / "active_branch.json"
    _record_interrupted_pending(run_dir)
    complete, attempts = _completed_branches(raw_path)
    demos, tasks = _load_demo_arrays(action_freeze)
    targets: dict[tuple[str, str], dict[str, Any]] = {}
    started = time.monotonic()
    new_count = 0
    peak_host = 0.0
    peak_wsl = 0
    peak_swap = 0
    for action_row in action_rows:
        policy = str(action_row["policy_identity"])
        state_id = str(action_row["state_id"])
        task = tasks[(str(action_row["suite"]), int(action_row["task_id"]))]
        states, expert_actions = demos[(str(action_row["suite"]), str(action_row["demo_name"]))]
        frame = int(action_row["frame"])
        if assay._array_sha256(states[frame]) != next(
            row["raw_state_sha256"] for row in action_freeze["development_states"] if row["state_id"] == state_id
        ):
            raise Stage0Error("REGISTERED_STATE_DRIFT", state_id)
        demo_key = (str(action_row["suite"]), str(action_row["demo_name"]))
        if demo_key not in targets:
            targets[demo_key] = assay._target_for_demo(task, states, expert_actions, int(args.camera_size))
        source_state_id = str(unpaired[state_id])
        source_row = action_lookup[f"{policy}|{source_state_id}"]
        roles = (
            ("nominal", np.asarray(action_row["expert_action"], dtype=np.float64), state_id),
            ("candidate", np.asarray(action_row["executed_action"], dtype=np.float64), state_id),
            ("unpaired_candidate", np.asarray(source_row["executed_action"], dtype=np.float64), source_state_id),
        )
        for role, first_action, source in roles:
            key = _branch_key(policy, state_id, role)
            if key in complete:
                continue
            prior = int(attempts.get(key, 0))
            if prior >= 2:
                raise Stage0Error("PERSISTENT_BRANCH_FAILURE", key)
            _write_json(
                pending_path,
                {
                    "branch_key": key,
                    "attempt": prior + 1,
                    "started_at_unix": time.time(),
                    "retry_rule": freeze["branch_panel"]["infrastructure_retry"],
                },
            )
            result = _execute_role(
                task=task,
                states=states,
                expert_actions=expert_actions,
                target=targets[demo_key],
                action_row=action_row,
                first_action=first_action,
                role=role,
                source_state_id=source,
                attempt=prior + 1,
                camera_size=int(args.camera_size),
            )
            _append_jsonl(raw_path, result)
            attempts[key] = prior + 1
            if pending_path.exists():
                pending_path.unlink()
            if _branch_materialized(result):
                complete[key] = result
            elif prior + 1 >= 2:
                raise Stage0Error("PERSISTENT_BRANCH_FAILURE", key)
            for sample in (result.get("resource_before") or {}, result.get("resource_after") or {}):
                peak_host = max(peak_host, float(sample.get("host_ram_percent") or 0.0))
                used = max(0, int(sample.get("wsl_mem_total_bytes") or 0) - int(sample.get("wsl_mem_available_bytes") or 0))
                peak_wsl = max(peak_wsl, used)
                peak_swap = max(peak_swap, int(sample.get("wsl_swap_used_bytes") or 0))
            if peak_host >= 90.0:
                raise Stage0Error("HOST_RAM_HARD_STOP", f"{peak_host:.3f}%")
            new_count += 1
            if len(complete) % 10 == 0:
                print(
                    json.dumps(
                        {
                            "completed_branches": len(complete),
                            "expected_branches": EXPECTED_BRANCHES,
                            "branch_key": key,
                            "peak_host_ram_percent": peak_host,
                            "peak_wsl_swap_used_bytes": peak_swap,
                        }
                    ),
                    flush=True,
                )
            if int(args.max_new_branches) > 0 and new_count >= int(args.max_new_branches):
                report = {
                    "schema_version": 1,
                    "campaign": CAMPAIGN,
                    "status": "EPOCH10B_STAGE0_INTERVENTION_BATCH_COMPLETE_RESUMABLE",
                    "completed_branch_count": len(complete),
                    "expected_branch_count": EXPECTED_BRANCHES,
                    "new_branch_count": new_count,
                    "raw_path": str(raw_path),
                    "raw_sha256": _sha256_file(raw_path),
                }
                _write_json(run_dir / "batch_state.json", report)
                return report
    if len(complete) != EXPECTED_BRANCHES:
        raise Stage0Error("BRANCH_CARDINALITY", f"{len(complete)}/{EXPECTED_BRANCHES}")
    paired_rows = []
    for action_row in action_rows:
        policy = str(action_row["policy_identity"])
        state_id = str(action_row["state_id"])
        nominal = complete[_branch_key(policy, state_id, "nominal")]
        candidate = complete[_branch_key(policy, state_id, "candidate")]
        unpaired_row = complete[_branch_key(policy, state_id, "unpaired_candidate")]
        nominal_score = _branch_score(nominal)
        candidate_score = _branch_score(candidate)
        unpaired_score = _branch_score(unpaired_row)
        paired_rows.append(
            {
                "schema_version": 1,
                "campaign": CAMPAIGN,
                "pair_key": _policy_state_key(action_row),
                "policy_identity": policy,
                "lineage_cluster": action_row["lineage_cluster"],
                "optimizer_step": int(action_row["optimizer_step"]),
                "suite": action_row["suite"],
                "task_id": int(action_row["task_id"]),
                "state_id": state_id,
                "demo_name": action_row["demo_name"],
                "demo_cluster": action_row["demo_cluster"],
                "phase": action_row["phase"],
                "frame": int(action_row["frame"]),
                "nominal_score": nominal_score,
                "candidate_score": candidate_score,
                "icae": candidate_score - nominal_score,
                "unpaired_candidate_score": unpaired_score,
                "unpaired_icae": unpaired_score - nominal_score,
                "nominal_branch_key": nominal["branch_key"],
                "candidate_branch_key": candidate["branch_key"],
                "unpaired_branch_key": unpaired_row["branch_key"],
                "prefix_steps": int(nominal["prefix"]["prefix_steps_completed"])
                + int(candidate["prefix"]["prefix_steps_completed"])
                + int(unpaired_row["prefix"]["prefix_steps_completed"]),
                "scored_control_steps": int(nominal["requested_control_steps"])
                + int(candidate["requested_control_steps"])
                + int(unpaired_row["requested_control_steps"]),
                "executed_control_steps": int(nominal["completed_control_steps"])
                + int(candidate["completed_control_steps"])
                + int(unpaired_row["completed_control_steps"]),
                "absorbing_scored_steps": int(nominal["absorbing_terminal_steps_scored_without_additional_env_step"])
                + int(candidate["absorbing_terminal_steps_scored_without_additional_env_step"])
                + int(unpaired_row["absorbing_terminal_steps_scored_without_additional_env_step"]),
                "heldout_outcome_opened": False,
            }
        )
    temporary = paired_path.with_suffix(".jsonl.tmp")
    if temporary.exists():
        temporary.unlink()
    _append_jsonl(temporary, paired_rows)
    temporary.replace(paired_path)
    report = {
        "schema_version": 1,
        "campaign": CAMPAIGN,
        "status": "EPOCH10B_STAGE0_DEVELOPMENT_INTERVENTIONS_COMPLETE",
        "branch_count": len(complete),
        "pair_count": len(paired_rows),
        "invalid_materialized_branch_count": 0,
        "raw_path": str(raw_path),
        "raw_sha256": _sha256_file(raw_path),
        "paired_path": str(paired_path),
        "paired_sha256": _sha256_file(paired_path),
        "prefix_simulator_steps": sum(int(row["prefix_steps"]) for row in paired_rows),
        "scored_simulator_steps": sum(int(row["scored_control_steps"]) for row in paired_rows),
        "executed_post_splice_steps": sum(int(row["executed_control_steps"]) for row in paired_rows),
        "absorbing_scored_steps": sum(int(row["absorbing_scored_steps"]) for row in paired_rows),
        "resource_telemetry": {
            "peak_host_ram_percent": peak_host,
            "peak_wsl_used_bytes": peak_wsl,
            "peak_wsl_swap_used_bytes": peak_swap,
            "wall_time_seconds_this_invocation": round(time.monotonic() - started, 3),
        },
        "development_outcomes_opened": True,
        "heldout_outcomes_opened": False,
    }
    report["canonical_payload_sha256"] = _canonical_sha256(report)
    _write_json(Path(args.intervention_manifest), report)
    return report


def _close_nested_envs(envs: Mapping[str, Mapping[int, Any]]) -> None:
    errors = []
    for task_map in envs.values():
        for env in task_map.values():
            try:
                env.close()
            except Exception as exc:  # pragma: no cover - runtime cleanup boundary
                errors.append(f"{type(exc).__name__}: {exc}")
    if errors:
        raise Stage0Error("ENVIRONMENT_CLOSE_FAILED", "; ".join(errors))


def _rollout_completed(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        key = str(row["episode_key"])
        if key in rows:
            raise Stage0Error("DUPLICATE_ROLLOUT_KEY", key)
        rows[key] = row
    return rows


def _extract_successes(metrics: Mapping[str, Any], suite: str) -> list[bool]:
    per_task = list(metrics.get("per_task") or [])
    if len(per_task) != 1:
        raise Stage0Error("ROLLOUT_TASK_CARDINALITY", f"{suite}: {len(per_task)}")
    item = per_task[0]
    if str(item.get("task_group")) != suite or int(item.get("task_id", -1)) != 0:
        raise Stage0Error("ROLLOUT_TASK_IDENTITY", f"{suite}: {item.get('task_group')} task {item.get('task_id')}")
    successes = list((item.get("metrics") or {}).get("successes") or [])
    if len(successes) != len(RESET_SEEDS):
        raise Stage0Error("ROLLOUT_EPISODE_CARDINALITY", f"{suite}: {len(successes)}/{len(RESET_SEEDS)}")
    return [bool(value) for value in successes]


def _unload_policy(loaded: dict[str, Any] | None) -> None:
    if loaded is not None:
        loaded.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_rollouts(args: argparse.Namespace) -> dict[str, Any]:
    _load_freeze(args)
    action_freeze = _read_json(Path(args.action_freeze))
    raw_path = Path(args.rollout_rows)
    pending_path = raw_path.parent / "active_suite_block.json"
    attempts_path = raw_path.parent / "infrastructure_attempts.jsonl"
    completed = _rollout_completed(raw_path)
    if pending_path.is_file():
        raw = pending_path.read_bytes()
        attempt = {
            "classification": "INCOMPLETE_ROLLOUT_BLOCK_WITHOUT_MATERIALIZED_EPISODES",
            "pending_sha256": hashlib.sha256(raw).hexdigest(),
            "pending_record": json.loads(raw.decode("utf-8")),
            "retry_authorized": True,
        }
        attempt["attempt_sha256"] = _canonical_sha256(attempt)
        existing = {row.get("attempt_sha256") for row in _read_jsonl(attempts_path)}
        if attempt["attempt_sha256"] not in existing:
            _append_jsonl(attempts_path, attempt)
    from lerobot.envs.factory import make_env
    from lerobot.scripts.lerobot_eval import eval_policy_all
    from tca_map.smolvla import official_wsl_libero_rollout as official

    started = time.monotonic()
    new_blocks = 0
    block_audits = []
    checkpoints = sorted(action_freeze["development_checkpoints"], key=lambda row: row["policy_identity"])
    for checkpoint in checkpoints:
        policy_identity = str(checkpoint["policy_identity"])
        required = [f"{policy_identity}|{suite}|{seed}" for suite in SUITE_STEP_CAPS for seed in RESET_SEEDS]
        if all(key in completed for key in required):
            continue
        adapter = _windows_to_wsl(str(checkpoint["path"]))
        adapter_file = Path(adapter) / "adapter_model.safetensors"
        if not adapter_file.is_file() or _sha256_file(adapter_file) != checkpoint["adapter_sha256"]:
            raise Stage0Error("ADAPTER_DRIFT", adapter)
        loader_args = argparse.Namespace(base_path=args.wsl_base_path, lora_root="/")
        spec = official.PolicySpec(policy_identity, adapter)
        print(f"[epoch10b-stage0-rollout] loading {policy_identity}", flush=True)
        loaded: dict[str, Any] | None = None
        try:
            loaded = official._load_policy_and_processors(loader_args, spec)
            for suite in SUITE_STEP_CAPS:
                suite_keys = [f"{policy_identity}|{suite}|{seed}" for seed in RESET_SEEDS]
                if all(key in completed for key in suite_keys):
                    continue
                prior_attempts = sum(
                    1
                    for row in _read_jsonl(attempts_path)
                    if (row.get("pending_record") or {}).get("block_key") == f"{policy_identity}|{suite}"
                )
                if prior_attempts >= 2:
                    raise Stage0Error("PERSISTENT_ROLLOUT_FAILURE", f"{policy_identity}|{suite}")
                _write_json(
                    pending_path,
                    {
                        "block_key": f"{policy_identity}|{suite}",
                        "policy_identity": policy_identity,
                        "suite": suite,
                        "reset_seeds": list(RESET_SEEDS),
                        "started_at_unix": time.time(),
                    },
                )
                env_cfg = official._make_env_cfg(suite, [0])
                envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
                block_started = time.monotonic()
                try:
                    metrics = eval_policy_all(
                        envs=envs,
                        policy=loaded["policy"],
                        env_preprocessor=loaded["env_preprocessor"],
                        env_postprocessor=loaded["env_postprocessor"],
                        preprocessor=loaded["preprocessor"],
                        postprocessor=loaded["postprocessor"],
                        n_episodes=len(RESET_SEEDS),
                        start_seed=RESET_SEEDS[0],
                        max_parallel_tasks=1,
                        max_episodes_rendered=0,
                    )
                finally:
                    _close_nested_envs(envs)
                successes = _extract_successes(metrics, suite)
                rows = []
                for reset_seed, success in zip(RESET_SEEDS, successes, strict=True):
                    key = f"{policy_identity}|{suite}|{reset_seed}"
                    rows.append(
                        {
                            "schema_version": 1,
                            "campaign": CAMPAIGN,
                            "stage": "development_stage0",
                            "episode_key": key,
                            "policy_identity": policy_identity,
                            "lineage_cluster": checkpoint["lineage_cluster"],
                            "optimizer_step": int(checkpoint["optimizer_step"]),
                            "suite": suite,
                            "task_id": 0,
                            "reset_seed": int(reset_seed),
                            "executed": True,
                            "valid": True,
                            "success": bool(success),
                            "timeout_or_failure": not bool(success),
                            "official_suite_step_cap": int(SUITE_STEP_CAPS[suite]),
                            "success_source": "official LeRobot eval_policy_all per-task successes",
                            "heldout_outcome_opened": False,
                        }
                    )
                _append_jsonl(raw_path, rows)
                for row in rows:
                    completed[row["episode_key"]] = row
                if pending_path.exists():
                    pending_path.unlink()
                new_blocks += 1
                audit = {
                    "block_key": f"{policy_identity}|{suite}",
                    "episodes": len(rows),
                    "successes": sum(int(row["success"]) for row in rows),
                    "elapsed_seconds": round(time.monotonic() - block_started, 3),
                }
                block_audits.append(audit)
                print(json.dumps({**audit, "completed_episode_rows": len(completed)}), flush=True)
                if int(args.max_new_rollout_blocks) > 0 and new_blocks >= int(args.max_new_rollout_blocks):
                    report = {
                        "schema_version": 1,
                        "campaign": CAMPAIGN,
                        "status": "EPOCH10B_STAGE0_ROLLOUT_BATCH_COMPLETE_RESUMABLE",
                        "completed_episode_rows": len(completed),
                        "expected_episode_rows": EXPECTED_ROLLOUT_ROWS,
                        "new_blocks": new_blocks,
                        "raw_path": str(raw_path),
                        "raw_sha256": _sha256_file(raw_path),
                    }
                    _write_json(raw_path.parent / "batch_state.json", report)
                    return report
        finally:
            _unload_policy(loaded)
    if len(completed) != EXPECTED_ROLLOUT_ROWS:
        raise Stage0Error("ROLLOUT_CARDINALITY", f"{len(completed)}/{EXPECTED_ROLLOUT_ROWS}")
    report = {
        "schema_version": 1,
        "campaign": CAMPAIGN,
        "status": "EPOCH10B_STAGE0_DEVELOPMENT_ROLLOUTS_COMPLETE",
        "episode_rows": len(completed),
        "unique_episode_keys": len(completed),
        "checkpoint_count": len({row["policy_identity"] for row in completed.values()}),
        "lineage_count": len({row["lineage_cluster"] for row in completed.values()}),
        "task_count": len({row["suite"] for row in completed.values()}),
        "reset_seed_count": len({int(row["reset_seed"]) for row in completed.values()}),
        "successes": sum(int(row["success"]) for row in completed.values()),
        "invalid_or_unexecuted": sum(not bool(row["valid"] and row["executed"]) for row in completed.values()),
        "raw_path": str(raw_path),
        "raw_sha256": _sha256_file(raw_path),
        "new_block_audits": block_audits,
        "wall_time_seconds_this_invocation": round(time.monotonic() - started, 3),
        "development_success_labels_opened": True,
        "heldout_outcomes_opened": False,
    }
    report["canonical_payload_sha256"] = _canonical_sha256(report)
    _write_json(Path(args.rollout_manifest), report)
    return report


def _rankdata(values: Sequence[float]) -> np.ndarray:
    from scipy.stats import rankdata

    return np.asarray(rankdata(np.asarray(values, dtype=float), method="average"), dtype=float)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or np.std(left) <= 1e-15 or np.std(right) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(_rankdata(left), _rankdata(right))[0, 1])


def _kendall(left: Sequence[float], right: Sequence[float]) -> float:
    from scipy.stats import kendalltau

    if len(left) < 2:
        return float("nan")
    return float(kendalltau(left, right, variant="b").statistic)


def _concordance(
    metric: Mapping[tuple[str, str], float],
    performance: Mapping[tuple[str, str], float],
    checkpoint_lineage: Mapping[str, str],
    tasks: Sequence[str],
    checkpoints: Sequence[str],
) -> tuple[float, dict[str, Any]]:
    task_values = []
    task_reports = {}
    for task in tasks:
        concordant = 0.0
        denominator = 0
        performance_ties = 0
        for left_index, left in enumerate(checkpoints):
            for right in checkpoints[left_index + 1 :]:
                if checkpoint_lineage[left] == checkpoint_lineage[right]:
                    continue
                perf_delta = performance[(left, task)] - performance[(right, task)]
                if abs(perf_delta) <= 1e-15:
                    performance_ties += 1
                    continue
                prediction_delta = metric[(right, task)] - metric[(left, task)]
                denominator += 1
                product = perf_delta * prediction_delta
                concordant += 1.0 if product > 0 else 0.5 if abs(product) <= 1e-15 else 0.0
        value = concordant / denominator if denominator else float("nan")
        if math.isfinite(value):
            task_values.append(value)
        task_reports[task] = {
            "concordance": value,
            "informative_cross_lineage_pairs": denominator,
            "performance_ties_excluded": performance_ties,
        }
    macro = float(np.mean(task_values)) if task_values else float("nan")
    return macro, task_reports


def _centered_ranks(
    metric: Mapping[tuple[str, str], float],
    performance: Mapping[tuple[str, str], float],
    tasks: Sequence[str],
    checkpoints: Sequence[str],
) -> dict[str, float]:
    x = []
    y = []
    for task in tasks:
        task_metric = np.asarray([metric[(checkpoint, task)] for checkpoint in checkpoints], dtype=float)
        task_perf = np.asarray([performance[(checkpoint, task)] for checkpoint in checkpoints], dtype=float)
        x.extend((-(task_metric - np.mean(task_metric))).tolist())
        y.extend((task_perf - np.mean(task_perf)).tolist())
    return {"spearman": _spearman(x, y), "kendall_tau_b": _kendall(x, y)}


def _topk_and_regret(
    metric: Mapping[tuple[str, str], float],
    performance: Mapping[tuple[str, str], float],
    tasks: Sequence[str],
    checkpoints: Sequence[str],
) -> dict[str, Any]:
    rows = []
    for task in tasks:
        predicted = sorted(checkpoints, key=lambda checkpoint: (metric[(checkpoint, task)], checkpoint))
        best_value = max(performance[(checkpoint, task)] for checkpoint in checkpoints)
        empirical_best = {checkpoint for checkpoint in checkpoints if performance[(checkpoint, task)] == best_value}
        selection = predicted[0]
        rows.append(
            {
                "task": task,
                "selected": selection,
                "empirical_best": sorted(empirical_best),
                "top1_correct": selection in empirical_best,
                "top3_hit": bool(set(predicted[:3]) & empirical_best),
                "selection_regret": best_value - performance[(selection, task)],
            }
        )
    return {
        "per_task": rows,
        "top1_accuracy": float(np.mean([row["top1_correct"] for row in rows])),
        "top3_accuracy": float(np.mean([row["top3_hit"] for row in rows])),
        "mean_selection_regret": float(np.mean([row["selection_regret"] for row in rows])),
        "maximum_selection_regret": float(np.max([row["selection_regret"] for row in rows])),
    }


def _build_analysis_tables(
    action_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    rollout_rows: Sequence[Mapping[str, Any]],
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    action_by_key = {_policy_state_key(row): row for row in action_rows}
    paired_by_key = {str(row["pair_key"]): row for row in paired_rows}
    if set(action_by_key) != set(paired_by_key):
        raise Stage0Error("PAIR_ACTION_KEY_MISMATCH", "paired intervention and action ledgers differ")
    shuffled = freeze["branch_panel"]["state_shuffled_score_source_by_target_state"]
    checkpoint_lineage = {
        str(row["policy_identity"]): str(row["lineage_cluster"])
        for row in action_rows
    }
    checkpoint_step = {
        str(row["policy_identity"]): int(row["optimizer_step"])
        for row in action_rows
    }
    tasks = sorted({str(row["suite"]) for row in action_rows})
    checkpoints = sorted(checkpoint_lineage)
    demos_by_task = {
        task: sorted({str(row["demo_cluster"]) for row in paired_rows if row["suite"] == task})
        for task in tasks
    }
    state_metric_rows = []
    for key, action in action_by_key.items():
        paired = paired_by_key[key]
        shuffled_source_key = f"{action['policy_identity']}|{shuffled[action['state_id']]}"
        shuffled_score = float(paired_by_key[shuffled_source_key]["icae"])
        baseline = action["baseline_metrics"]
        state_metric_rows.append(
            {
                "policy_identity": action["policy_identity"],
                "lineage_cluster": action["lineage_cluster"],
                "optimizer_step": int(action["optimizer_step"]),
                "suite": action["suite"],
                "demo_cluster": action["demo_cluster"],
                "state_id": action["state_id"],
                "icae": float(paired["icae"]),
                "raw_mse": float(baseline["raw_mse"]),
                "raw_mae": float(baseline["raw_mae"]),
                "action_dimension_normalized_mse": float(baseline["action_dimension_normalized_mse"]),
                "arm_gripper_equal_weight_mse": float(baseline["arm_gripper_equal_weight_mse"]),
                "phase_state_criticality_weighted_normalized_mse": float(
                    baseline["phase_state_criticality_weighted_normalized_mse"]
                ),
                "response_magnitude_control": float(baseline["candidate_response_magnitude_l2"]),
                "unpaired_icae": float(paired["unpaired_icae"]),
                "state_shuffled_icae": shuffled_score,
            }
        )
    grouped_metric: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_demo_metric: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in state_metric_rows:
        grouped_metric[(row["policy_identity"], row["suite"], "all")].append(row)
        grouped_demo_metric[(row["policy_identity"], row["suite"], row["demo_cluster"])].append(row)
    metric_tables: dict[str, dict[tuple[str, str], float]] = {metric: {} for metric in METRICS}
    demo_tables: dict[str, dict[tuple[str, str, str], float]] = {metric: {} for metric in METRICS}
    for (checkpoint, task, _), rows in grouped_metric.items():
        for metric in METRICS:
            metric_tables[metric][(checkpoint, task)] = float(np.mean([float(row[metric]) for row in rows]))
    for (checkpoint, task, demo), rows in grouped_demo_metric.items():
        for metric in METRICS:
            demo_tables[metric][(checkpoint, task, demo)] = float(np.mean([float(row[metric]) for row in rows]))
    performance_values: dict[tuple[str, str], list[bool]] = defaultdict(list)
    performance_by_reset: dict[tuple[str, str, int], float] = {}
    for row in rollout_rows:
        if not bool(row.get("executed")) or not bool(row.get("valid")):
            raise Stage0Error("INVALID_ROLLOUT_ROW", str(row.get("episode_key")))
        key = (str(row["policy_identity"]), str(row["suite"]))
        performance_values[key].append(bool(row["success"]))
        performance_by_reset[(key[0], key[1], int(row["reset_seed"]))] = float(bool(row["success"]))
    performance = {key: float(np.mean(values)) for key, values in performance_values.items()}
    expected_cells = {(checkpoint, task) for checkpoint in checkpoints for task in tasks}
    if set(performance) != expected_cells or any(len(values) != len(RESET_SEEDS) for values in performance_values.values()):
        raise Stage0Error("PERFORMANCE_PANEL_INCOMPLETE", f"cells {len(performance)}/{len(expected_cells)}")
    return {
        "tasks": tasks,
        "checkpoints": checkpoints,
        "checkpoint_lineage": checkpoint_lineage,
        "checkpoint_step": checkpoint_step,
        "demos_by_task": demos_by_task,
        "metric_tables": metric_tables,
        "demo_tables": demo_tables,
        "performance": performance,
        "performance_by_reset": performance_by_reset,
        "state_metric_rows": state_metric_rows,
    }


def _weighted_concordance(
    scores: np.ndarray,
    performance: np.ndarray,
    occurrence_lineages: Sequence[int],
) -> float:
    # Arrays are [task occurrence, checkpoint occurrence].  Each lineage
    # occurrence contributes its two nested stages; pairs within an occurrence
    # are excluded.
    task_values = []
    for task_index in range(scores.shape[0]):
        concordant = 0.0
        denominator = 0
        for left in range(scores.shape[1]):
            for right in range(left + 1, scores.shape[1]):
                if occurrence_lineages[left] == occurrence_lineages[right]:
                    continue
                perf_delta = performance[task_index, left] - performance[task_index, right]
                if abs(perf_delta) <= 1e-15:
                    continue
                prediction_delta = scores[task_index, right] - scores[task_index, left]
                product = perf_delta * prediction_delta
                concordant += 1.0 if product > 0 else 0.5 if abs(product) <= 1e-15 else 0.0
                denominator += 1
        if denominator:
            task_values.append(concordant / denominator)
    return float(np.mean(task_values)) if task_values else float("nan")


def _hierarchical_bootstrap(tables: Mapping[str, Any], best: str, worst: str) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    tasks = list(tables["tasks"])
    checkpoints = list(tables["checkpoints"])
    checkpoint_lineage = tables["checkpoint_lineage"]
    lineages = sorted(set(checkpoint_lineage.values()))
    checkpoints_by_lineage = {
        lineage: sorted(
            [checkpoint for checkpoint in checkpoints if checkpoint_lineage[checkpoint] == lineage],
            key=lambda checkpoint: tables["checkpoint_step"][checkpoint],
        )
        for lineage in lineages
    }
    distributions = {metric: [] for metric in ("icae", "action_dimension_normalized_mse", "unpaired_icae", "state_shuffled_icae")}
    performance_gaps = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_task_indices = rng.integers(0, len(tasks), size=len(tasks))
        sampled_lineage_indices = rng.integers(0, len(lineages), size=len(lineages))
        checkpoint_occurrences: list[str] = []
        occurrence_lineages: list[int] = []
        for occurrence, lineage_index in enumerate(sampled_lineage_indices):
            for checkpoint in checkpoints_by_lineage[lineages[int(lineage_index)]]:
                checkpoint_occurrences.append(checkpoint)
                occurrence_lineages.append(occurrence)
        perf_matrix = np.empty((len(tasks), len(checkpoint_occurrences)), dtype=float)
        score_matrices = {
            metric: np.empty_like(perf_matrix)
            for metric in distributions
        }
        gap_by_task = []
        for task_occurrence, task_index in enumerate(sampled_task_indices):
            task = tasks[int(task_index)]
            demos = tables["demos_by_task"][task]
            sampled_demos = [demos[int(index)] for index in rng.integers(0, len(demos), size=len(demos))]
            sampled_resets = [RESET_SEEDS[int(index)] for index in rng.integers(0, len(RESET_SEEDS), size=len(RESET_SEEDS))]
            gap_by_task.append(
                float(
                    np.mean([tables["performance_by_reset"][(best, task, seed)] for seed in sampled_resets])
                    - np.mean([tables["performance_by_reset"][(worst, task, seed)] for seed in sampled_resets])
                )
            )
            for checkpoint_occurrence, checkpoint in enumerate(checkpoint_occurrences):
                perf_matrix[task_occurrence, checkpoint_occurrence] = float(
                    np.mean([tables["performance_by_reset"][(checkpoint, task, seed)] for seed in sampled_resets])
                )
                for metric in distributions:
                    score_matrices[metric][task_occurrence, checkpoint_occurrence] = float(
                        np.mean([tables["demo_tables"][metric][(checkpoint, task, demo)] for demo in sampled_demos])
                    )
        performance_gaps.append(float(np.mean(gap_by_task)))
        for metric in distributions:
            distributions[metric].append(
                _weighted_concordance(score_matrices[metric], perf_matrix, occurrence_lineages)
            )
    norm = np.asarray(distributions["action_dimension_normalized_mse"], dtype=float)
    result: dict[str, Any] = {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "best_checkpoint": best,
        "worst_checkpoint": worst,
        "best_minus_worst_success_ci95": [
            float(np.nanquantile(performance_gaps, 0.025)),
            float(np.nanquantile(performance_gaps, 0.975)),
        ],
        "metrics": {},
    }
    for metric, values in distributions.items():
        array = np.asarray(values, dtype=float)
        gain = array - norm
        result["metrics"][metric] = {
            "concordance_ci95": [float(np.nanquantile(array, 0.025)), float(np.nanquantile(array, 0.975))],
            "gain_over_normalized_mse_ci95": [
                float(np.nanquantile(gain, 0.025)),
                float(np.nanquantile(gain, 0.975)),
            ],
            "probability_gain_over_normalized_mse_positive": float(np.nanmean(gain > 0.0)),
            "finite_replicates": int(np.isfinite(array).sum()),
        }
    return result


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    freeze = _load_freeze(args)
    action_rows = _read_jsonl(Path(args.action_rows))
    paired_rows = _read_jsonl(Path(args.paired_rows))
    rollout_rows = _read_jsonl(Path(args.rollout_rows))
    if len(action_rows) != EXPECTED_ACTION_ROWS or len(paired_rows) != EXPECTED_ACTION_ROWS:
        raise Stage0Error("STAGE0_SCORE_PANEL_INCOMPLETE", f"actions={len(action_rows)} pairs={len(paired_rows)}")
    if len(rollout_rows) != EXPECTED_ROLLOUT_ROWS:
        raise Stage0Error("STAGE0_ROLLOUT_PANEL_INCOMPLETE", f"{len(rollout_rows)}/{EXPECTED_ROLLOUT_ROWS}")
    tables = _build_analysis_tables(action_rows, paired_rows, rollout_rows, freeze)
    tasks = tables["tasks"]
    checkpoints = tables["checkpoints"]
    checkpoint_lineage = tables["checkpoint_lineage"]
    competitive = [checkpoint for checkpoint in checkpoints if tables["checkpoint_step"][checkpoint] == 100]
    panels = {"full": checkpoints, "competitive_step100": competitive}
    metric_reports: dict[str, Any] = {}
    for metric in METRICS:
        panel_reports = {}
        for panel_name, panel_checkpoints in panels.items():
            concordance, per_task = _concordance(
                tables["metric_tables"][metric],
                tables["performance"],
                checkpoint_lineage,
                tasks,
                panel_checkpoints,
            )
            panel_reports[panel_name] = {
                "equal_task_cross_lineage_concordance": concordance,
                "per_task": per_task,
                "task_centered_rank": _centered_ranks(
                    tables["metric_tables"][metric], tables["performance"], tasks, panel_checkpoints
                ),
                "selection": _topk_and_regret(
                    tables["metric_tables"][metric], tables["performance"], tasks, panel_checkpoints
                ),
            }
        metric_reports[metric] = panel_reports
    macro_performance = {
        checkpoint: float(np.mean([tables["performance"][(checkpoint, task)] for task in tasks]))
        for checkpoint in checkpoints
    }
    best = min(checkpoints, key=lambda checkpoint: (-macro_performance[checkpoint], checkpoint))
    worst = min(checkpoints, key=lambda checkpoint: (macro_performance[checkpoint], checkpoint))
    bands = {
        "low_[0,1/3)": [checkpoint for checkpoint, value in macro_performance.items() if value < 1.0 / 3.0],
        "middle_[1/3,2/3)": [
            checkpoint for checkpoint, value in macro_performance.items() if 1.0 / 3.0 <= value < 2.0 / 3.0
        ],
        "high_[2/3,1]": [checkpoint for checkpoint, value in macro_performance.items() if value >= 2.0 / 3.0],
    }
    bootstrap = _hierarchical_bootstrap(tables, best, worst)
    band_count = sum(bool(values) for values in bands.values())
    distinguishable = bool(bootstrap["best_minus_worst_success_ci95"][0] > 0.0)
    performance_gate = bool(band_count >= 3 or distinguishable)
    icae = float(metric_reports["icae"]["full"]["equal_task_cross_lineage_concordance"])
    normalized = float(
        metric_reports["action_dimension_normalized_mse"]["full"]["equal_task_cross_lineage_concordance"]
    )
    gain = icae - normalized
    probability = float(
        bootstrap["metrics"]["icae"]["probability_gain_over_normalized_mse_positive"]
    )
    clear_margin = float(freeze["stage0_gate"]["strong_baseline_clear_margin"])
    strong_values = {
        baseline: float(metric_reports[baseline]["full"]["equal_task_cross_lineage_concordance"])
        for baseline in STRONG_BASELINES
    }
    dominated_by_every = all(value >= icae + clear_margin for value in strong_values.values())
    negative_reproduction = {}
    for control in ("unpaired_icae", "state_shuffled_icae"):
        concordance = float(metric_reports[control]["full"]["equal_task_cross_lineage_concordance"])
        control_gain = concordance - normalized
        control_probability = float(
            bootstrap["metrics"][control]["probability_gain_over_normalized_mse_positive"]
        )
        negative_reproduction[control] = {
            "concordance": concordance,
            "gain_over_normalized_mse": control_gain,
            "probability_gain_positive": control_probability,
            "reproduces_gain": bool(concordance >= 0.60 and control_gain >= 0.08 and control_probability >= 0.90),
        }
    negative_controls_pass = not any(row["reproduces_gain"] for row in negative_reproduction.values())
    prefix_steps = sum(int(row["prefix_steps"]) for row in paired_rows)
    scored_steps = sum(int(row["scored_control_steps"]) for row in paired_rows)
    total_icae_steps = prefix_steps + scored_steps
    exhaustive_steps = len(checkpoints) * len(RESET_SEEDS) * sum(SUITE_STEP_CAPS.values())
    step_fraction = total_icae_steps / exhaustive_steps
    gate = {
        "performance_identifiable": performance_gate,
        "icae_concordance_at_least_0_60": bool(icae >= 0.60),
        "gain_over_normalized_mse_at_least_0_08": bool(gain >= 0.08),
        "bootstrap_probability_gain_positive_at_least_0_90": bool(probability >= 0.90),
        "not_clearly_dominated_by_every_strong_baseline": not dominated_by_every,
        "negative_controls_do_not_reproduce_gain": negative_controls_pass,
        "total_icae_steps_at_most_20_percent": bool(step_fraction <= 0.20),
    }
    if not performance_gate:
        terminal = "EPOCH10B_ICAE_DEVELOPMENT_PANEL_NONIDENTIFIABLE"
        status = "TERMINAL"
    elif all(gate.values()):
        terminal = None
        status = "EPOCH10B_STAGE0_GO"
    else:
        terminal = "EPOCH10B_ICAE_NO_PREDICTIVE_HEADROOM"
        status = "TERMINAL"
    report: dict[str, Any] = {
        "schema_version": 1,
        "campaign": CAMPAIGN,
        "status": status,
        "terminal_state": terminal,
        "freeze_sha256": _sha256_file(Path(args.stage0_freeze)),
        "input_hashes": {
            "actions": _sha256_file(Path(args.action_rows)),
            "paired_scores": _sha256_file(Path(args.paired_rows)),
            "rollout_rows": _sha256_file(Path(args.rollout_rows)),
            "intervention_manifest": _sha256_file(Path(args.intervention_manifest)),
            "rollout_manifest": _sha256_file(Path(args.rollout_manifest)),
        },
        "denominators": {
            "development_checkpoints": len(checkpoints),
            "whole_seed_lineages": len(set(checkpoint_lineage.values())),
            "tasks": len(tasks),
            "states_per_checkpoint": len(paired_rows) // len(checkpoints),
            "intervention_pairs": len(paired_rows),
            "fresh_branches": len(paired_rows) * 3,
            "official_episodes": len(rollout_rows),
            "common_resets_per_task_checkpoint": len(RESET_SEEDS),
        },
        "performance": {
            "checkpoint_task_success": {
                checkpoint: {task: tables["performance"][(checkpoint, task)] for task in tasks}
                for checkpoint in checkpoints
            },
            "checkpoint_macro_success": macro_performance,
            "best_checkpoint": best,
            "worst_checkpoint": worst,
            "best_minus_worst_point": macro_performance[best] - macro_performance[worst],
            "quality_bands": bands,
            "occupied_band_count": band_count,
            "distinguishable_by_bootstrap": distinguishable,
        },
        "metrics": metric_reports,
        "primary_comparison": {
            "icae_concordance": icae,
            "normalized_mse_concordance": normalized,
            "gain": gain,
            "bootstrap_probability_gain_positive": probability,
        },
        "strong_baselines": strong_values,
        "clearly_dominated_by_every_strong_baseline": dominated_by_every,
        "negative_controls": negative_reproduction,
        "bootstrap": bootstrap,
        "cost": {
            "prefix_simulator_steps": prefix_steps,
            "scored_simulator_steps_including_absorbing_scores": scored_steps,
            "total_icae_simulator_steps": total_icae_steps,
            "exhaustive_full_rollout_step_budget": exhaustive_steps,
            "fraction": step_fraction,
            "maximum_allowed_fraction": 0.20,
        },
        "repeatability": {
            "action_cache_maximum_abs": float(_read_json(Path(args.action_manifest))["maximum_repeatability_abs"]),
            "fresh_branch_constructor_certified_before_stage0": True,
        },
        "gate": gate,
        "heldout_checkpoint_actions_queried": 0,
        "heldout_outcomes_opened": False,
        "confirmation_outcomes_opened": False,
        "ci_mse": "NOT_IMPLEMENTED_NO_PROXY_UNDER_FROZEN_ACTION_PROTOCOL",
    }
    report["canonical_payload_sha256"] = _canonical_sha256(report)
    _write_json(Path(args.stage0_report), report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preregister", "preflight", "interventions", "rollouts", "analyze"))
    parser.add_argument("--action-freeze", default="reports/epoch10b_action_cache_freeze.json")
    parser.add_argument("--action-manifest", default="reports/epoch10b_action_cache_manifest.json")
    parser.add_argument("--action-rows", default="runs/epoch10b_stage0_action_cache/development_actions.jsonl")
    parser.add_argument("--stage0-freeze", default="reports/epoch10b_stage0_preregistration.json")
    parser.add_argument("--host-guard", default="scripts/run_epoch10b_stage0_host_guard.ps1")
    parser.add_argument("--intervention-run-dir", default="runs/epoch10b_stage0_interventions")
    parser.add_argument("--intervention-manifest", default="reports/epoch10b_stage0_intervention_manifest.json")
    parser.add_argument("--paired-rows", default="runs/epoch10b_stage0_interventions/paired_scores.jsonl")
    parser.add_argument("--rollout-rows", default="runs/epoch10b_stage0_rollouts/development_episodes.jsonl")
    parser.add_argument("--rollout-manifest", default="reports/epoch10b_stage0_rollout_manifest.json")
    parser.add_argument("--stage0-report", default="reports/epoch10b_stage0_result.json")
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--max-new-branches", type=int, default=0)
    parser.add_argument("--max-new-rollout-blocks", type=int, default=0)
    parser.add_argument("--wsl-base-path", default="/home/jiheon/assets/checkpoints/smolvla_libero")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "preregister":
            result = preregister(args)
        elif args.mode == "preflight":
            result = preflight(args)
        elif args.mode == "interventions":
            result = run_interventions(args)
        elif args.mode == "rollouts":
            result = run_rollouts(args)
        else:
            result = analyze(args)
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in ("status", "terminal_state", "completed_branch_count", "completed_episode_rows")
                },
                sort_keys=True,
            )
        )
        return 0
    except Stage0Error as exc:
        error = {"status": "STAGE0_EXECUTION_BLOCKED", "code": exc.code, "detail": exc.detail}
        print(json.dumps(error, sort_keys=True), flush=True)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
