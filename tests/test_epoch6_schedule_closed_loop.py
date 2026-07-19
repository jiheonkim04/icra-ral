from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_epoch6_schedule_closed_loop.py"
SPEC = importlib.util.spec_from_file_location("epoch6_schedule_closed_loop", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
closed_loop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(closed_loop)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_frozen_manifest_and_identity_schedule_are_consistent() -> None:
    manifest = closed_loop.validate_manifest()
    assert len(closed_loop.IDENTITIES) == 20
    assert manifest["identity_manifest"] == closed_loop.IDENTITIES
    assert closed_loop.SERIAL_ASSIGNMENTS == {0: list(range(20))}
    assert sorted(sum(closed_loop.SHARDED_ASSIGNMENTS.values(), [])) == list(range(20))
    assert closed_loop.SHARDED_OFFSETS == {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}
    assert closed_loop.HORIZONS == {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
    }


def test_exact_array_and_policy_input_hashes_bind_shape_and_content() -> None:
    state = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert closed_loop.exact_array_hash(state) != closed_loop.exact_array_hash(state.reshape(2, 6))
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    proprio = np.zeros(20, dtype=np.float32)
    base = closed_loop.policy_input_hash(image, image, proprio, "task")
    changed = image.copy()
    changed[0, 0, 0] = 1
    assert base != closed_loop.policy_input_hash(changed, image, proprio, "task")
    assert base != closed_loop.policy_input_hash(image, image, proprio, "other")


def _schedule_result(run_dir: Path, schedule: str) -> None:
    schedule_dir = run_dir / schedule
    _write(
        schedule_dir / "schedule_result.json",
        {
            "status": "SCHEDULE_COMPLETE",
            "completed_episode_count": 20,
            "resource_monitor": {"maximum_swap_used_bytes": 0, "exceptions": []},
            "errors": [],
        },
    )


def _episode(
    run_dir: Path,
    schedule: str,
    identity_index: int,
    success: bool,
    noise_position: int,
    raw_hash: str,
) -> None:
    identity = closed_loop.IDENTITY_BY_INDEX[identity_index]
    _write(
        run_dir / schedule / "episodes" / f"identity_{identity_index:03d}.json",
        {
            "identity": identity,
            "success": success,
            "timeout": not success,
            "exception": None,
            "initial_state_sha256": f"INITIAL-{identity_index}",
            "post_settle_state_sha256": f"POST-{identity_index}",
            "query_rows": [
                {
                    "episode_policy_call_index": 0,
                    "global_noise_position": noise_position,
                    "pre_query_sim_state_sha256": f"POST-{identity_index}",
                    "raw_chunk_sha256": raw_hash,
                }
            ],
        },
    )


def _transaction(
    run_dir: Path,
    ordinal: int,
    shard_id: int,
    arrival_ns: int,
) -> None:
    schedule_dir = run_dir / closed_loop.SCHEDULES[1]
    request_id = f"identity_{ordinal:03d}_query_0000"
    metadata_path, arrays_path = closed_loop.transaction_paths(schedule_dir, request_id)
    raw = np.full(closed_loop.RAW_CHUNK_SHAPE, ordinal, dtype=np.float32)
    processed = np.full(closed_loop.PROCESSED_CHUNK_SHAPE, ordinal, dtype=np.float32)
    closed_loop.write_npz_atomic(arrays_path, raw_chunk=raw, processed_chunk=processed)
    _write(
        metadata_path,
        {
            "request_id": request_id,
            "global_noise_position": ordinal,
            "shard_id": shard_id,
            "request_arrived_monotonic_ns": arrival_ns,
            "raw_chunk_sha256": closed_loop.stage0.hash_array(raw),
            "processed_chunk_sha256": closed_loop.stage0.hash_array(processed),
            "arrays_sha256": closed_loop.stage0.sha256_file(arrays_path),
        },
    )


def _synthetic_gate(run_dir: Path, discordant_count: int) -> None:
    for schedule in closed_loop.SCHEDULES:
        _schedule_result(run_dir, schedule)
    for index in range(20):
        _episode(run_dir, closed_loop.SCHEDULES[0], index, True, index, f"SERIAL-{index}")
        _episode(
            run_dir,
            closed_loop.SCHEDULES[1],
            index,
            index >= discordant_count,
            100 + index,
            f"SHARDED-{index}",
        )
    for ordinal, shard_id in enumerate([3, 2, 1, 0]):
        _transaction(run_dir, ordinal, shard_id, 1000 + ordinal)


def test_adjudication_authorizes_method_design_only_when_all_gates_pass(tmp_path: Path) -> None:
    _synthetic_gate(tmp_path, discordant_count=4)
    result = closed_loop.adjudicate(tmp_path)
    assert result["final_decision"] == "PROBLEM_VERIFIED_METHOD_DESIGN_AUTHORIZED"
    assert result["paired_disagreement_fraction"] == 0.20
    assert result["success_rate_spread"] == 0.20
    assert result["trace_tied_fraction_among_discordant"] == 1.0
    assert result["actual_sharded_first_arrival_order"] == [3, 2, 1, 0]


def test_adjudication_archives_below_practical_effect(tmp_path: Path) -> None:
    _synthetic_gate(tmp_path, discordant_count=1)
    result = closed_loop.adjudicate(tmp_path)
    assert result["final_decision"] == "NO_REPEATABLE_PROBLEM"
    assert not result["method_design_authorized"]


def test_resource_smoke_validator_requires_four_envs_and_clean_host(tmp_path: Path) -> None:
    internal = {
        "status": "CLOSED_LOOP_ACTUAL_PATH_RESOURCE_SMOKE_PASS",
        "simultaneous_env_instances": 4,
        "simultaneous_env_processes": 4,
        "model_inference_calls": 1,
        "raw_chunk_shape": list(closed_loop.RAW_CHUNK_SHAPE),
        "raw_chunk_finite": True,
        "success_check_calls": 0,
        "simulator_actions_executed": 0,
        "reward_success_done_read": False,
        "resource_monitor": {"maximum_swap_used_bytes": 0, "exceptions": []},
        "runtime": {"parameter_devices": ["cuda:0"], "cpu_or_disk_model_offload": False},
    }
    internal_path = tmp_path / "closed_loop_resource_smoke.json"
    _write(internal_path, internal)
    host = {
        "final_decision": "EPOCH6_CLOSED_LOOP_RESOURCE_SMOKE_PASS_CALIBRATED",
        "idle_control_valid": True,
        "sustained_paging_detected": False,
        "clean_state_restored": True,
        "oom_or_kill_signature_detected": False,
        "internal_report_sha256": closed_loop.stage0.sha256_file(internal_path),
        "protocol_sha256": closed_loop.EXPECTED_PROTOCOL_SHA256,
        "execution_manifest_sha256": closed_loop.EXPECTED_EXECUTION_MANIFEST_SHA256,
        "resource_amendment_sha256": closed_loop.EXPECTED_RESOURCE_AMENDMENT_SHA256,
        "monitor_script_sha256": closed_loop.EXPECTED_CLOSED_LOOP_MONITOR_SHA256,
    }
    _write(tmp_path / "closed_loop_resource_smoke_host.json", host)
    assert closed_loop.valid_resource_smoke(tmp_path)
    host["clean_state_restored"] = False
    _write(tmp_path / "closed_loop_resource_smoke_host.json", host)
    assert not closed_loop.valid_resource_smoke(tmp_path)
