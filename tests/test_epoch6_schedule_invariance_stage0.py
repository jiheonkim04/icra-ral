from __future__ import annotations

import importlib.util
import inspect
import json
import os
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_epoch6_schedule_invariance_stage0.py"
SPEC = importlib.util.spec_from_file_location("epoch6_schedule_stage0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
stage0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage0)


def _metadata(input_hash: str = "FIXED") -> dict[str, dict]:
    return {
        name: {
            "exceptions": [],
            "input_sha256": input_hash,
            "provenance": {"source_manifest_sha256": "SOURCE", "checkpoint_manifest_sha256": "CHECKPOINT"},
            "simulator_actions_executed": 0,
            "reward_success_done_read": False,
            "resource_monitor": {"maximum_swap_used_bytes": 0},
        }
        for name in stage0.SEQUENCES
    }


def _sequence(raw: np.ndarray, name: str) -> dict[str, np.ndarray]:
    processed = np.stack(
        [stage0.raw_to_processed_7d(raw[index]) for index in range(stage0.LOGICAL_KEY_COUNT)]
    )
    order = stage0.SEQUENCES[name][1]
    return {
        "raw_chunks": raw,
        "processed_chunks": processed,
        "draw_positions": np.asarray(
            [order.index(index) for index in range(stage0.LOGICAL_KEY_COUNT)], dtype=np.int64
        ),
    }


def _gates() -> dict[str, float | int]:
    return {
        "same_order_cold_restart_hash_match_fraction_min": 1.0,
        "same_order_cold_restart_normalized_rms_max_if_hash_differs": 1e-6,
        "reversed_order_changed_hash_fraction_min": 0.95,
        "median_order_rms_over_independent_noise_rms_min": 0.10,
        "exception_count_max": 0,
    }


def test_frozen_protocol_matches_runner() -> None:
    protocol = stage0.validate_protocol()
    assert protocol["stage0"]["execution_semantics"]["domain_id"] == 3
    assert protocol["stage0"]["simulator_actions_executed"] is False
    assert protocol["stage0"]["reward_success_done_read"] is False


def test_array_hash_uses_shape_and_canonical_float32_bytes() -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert stage0.hash_array(values) == stage0.hash_array(values.astype(np.float32))
    assert stage0.hash_array(values) != stage0.hash_array(values.reshape(2, 6))
    changed = values.copy()
    changed[0, 0] = 0.25
    assert stage0.hash_array(values) != stage0.hash_array(changed)


def test_normalized_rms_matches_frozen_formula() -> None:
    left = np.array([1.0, 1.0], dtype=np.float32)
    right = np.array([0.0, 0.0], dtype=np.float32)
    assert stage0.normalized_rms(left, right) == 1.0
    assert stage0.normalized_rms(left, left) == 0.0


def test_identity_rot6d_and_gripper_conversion() -> None:
    raw = np.zeros((2, 20), dtype=np.float32)
    raw[:, 3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    raw[0, 9] = 0.4
    raw[1, 9] = 0.6
    processed = stage0.raw_to_processed_7d(raw)
    np.testing.assert_allclose(processed[:, 3:6], 0.0, atol=1e-7)
    np.testing.assert_array_equal(processed[:, 6], np.array([-1.0, 1.0], dtype=np.float32))


def test_logical_keys_and_orders_are_stable() -> None:
    assert stage0.logical_key(0)["episode_index"] == 0
    assert stage0.logical_key(19)["episode_index"] == 19
    assert stage0.SEQUENCES["A"][1] == list(range(20))
    assert stage0.SEQUENCES["B"][1] == list(reversed(range(20)))
    assert stage0.SEQUENCES["A"][0] == stage0.SEQUENCES["B"][0]
    assert stage0.SEQUENCES["C"][0] != stage0.SEQUENCES["A"][0]


def test_adjudication_returns_action_level_go_for_valid_large_effect() -> None:
    rng = np.random.default_rng(17)
    base = rng.normal(size=(20, 30, 20)).astype(np.float32)
    repeat = base.copy()
    reversed_order = (base + 0.5).astype(np.float32)
    reference = (base + 1.0).astype(np.float32)
    arrays = {
        "A": _sequence(base, "A"),
        "A_repeat": _sequence(repeat, "A_repeat"),
        "B": _sequence(reversed_order, "B"),
        "C": _sequence(reference, "C"),
    }
    result = stage0.adjudicate_arrays(arrays, _metadata(), _gates())
    assert result["final_decision"] == "ACTION_LEVEL_SCHEDULE_DEPENDENCE_GO"
    assert result["reversed_order_changed_hash_fraction"] == 1.0
    assert all(result["integrity"].values())


def test_cold_restart_failure_takes_precedence() -> None:
    rng = np.random.default_rng(23)
    base = rng.normal(size=(20, 30, 20)).astype(np.float32)
    arrays = {
        "A": _sequence(base, "A"),
        "A_repeat": _sequence(base + 0.01, "A_repeat"),
        "B": _sequence(base + 0.5, "B"),
        "C": _sequence(base + 1.0, "C"),
    }
    result = stage0.adjudicate_arrays(arrays, _metadata(), _gates())
    assert result["final_decision"] == "EVALUATION_INVALID_CANNOT_ISOLATE_SCHEDULE"


def test_unchanged_order_contrast_is_no_material_dependence() -> None:
    rng = np.random.default_rng(29)
    base = rng.normal(size=(20, 30, 20)).astype(np.float32)
    arrays = {
        "A": _sequence(base, "A"),
        "A_repeat": _sequence(base.copy(), "A_repeat"),
        "B": _sequence(base.copy(), "B"),
        "C": _sequence(base + 1.0, "C"),
    }
    result = stage0.adjudicate_arrays(arrays, _metadata(), _gates())
    assert result["final_decision"] == "NO_MATERIAL_ACTION_LEVEL_SCHEDULE_DEPENDENCE"


def test_fixture_capture_has_no_simulator_step_call() -> None:
    source = inspect.getsource(stage0.capture_fixture)
    assert ".step(" not in source
    assert ".set_init_state(" not in source
    assert "forbidden_success_check" in source
    assert "env.set_state" in source


def test_pinned_libero_import_resolves_exact_checkout_without_environment() -> None:
    if os.name == "nt" or not stage0.LIBERO_ROOT.is_dir():
        return
    _benchmark, offscreen_render_env, executed_wrapper = stage0.import_pinned_libero()
    expected_wrapper = (
        stage0.LIBERO_ROOT / "libero" / "libero" / "envs" / "env_wrapper.py"
    ).resolve(strict=True)
    assert executed_wrapper == expected_wrapper
    assert offscreen_render_env.__module__.startswith("libero.libero.")


def test_fixture_loads_exact_pinned_init_state_with_explicit_torch_compatibility() -> None:
    source = inspect.getsource(stage0.capture_fixture)
    assert 'pinned_asset_root / "init_files"' in source
    assert "weights_only=False" in source
    assert "suite.get_task_init_states" not in source


def test_resource_smoke_releases_runtime_before_stopping_monitor_or_writing_report() -> None:
    source = inspect.getsource(stage0.run_resource_smoke)
    release = source.index("model = processor = model_inputs = action = None", source.index("finally:"))
    empty_cache = source.index("torch_module.cuda.empty_cache()", release)
    stop_monitor = source.index('result["resource_monitor"] = monitor.stop()', release)
    write_report = source.index('write_json(run_dir / "resource_smoke.json"', release)
    assert release < empty_cache < stop_monitor < write_report


def test_schedule_smoke_wrappers_persist_exit_and_support_safe_cache_release() -> None:
    root = SCRIPT.parents[1]
    bash = (root / "scripts" / "run_epoch6_schedule_stage0_smoke_wsl.sh").read_text(
        encoding="utf-8"
    )
    powershell = (root / "scripts" / "monitor_epoch6_schedule_stage0_smoke.ps1").read_text(
        encoding="utf-8"
    )
    assert "resource_smoke_child_exit_code.txt" in bash
    assert "resource_smoke_child_exit_code.txt" in powershell
    assert "AllowWslCacheDropAfterChild" in powershell
    assert "/proc/sys/vm/drop_caches" in powershell
    assert "CalibratedResourceGovernance" in powershell
    assert "IdleControlDurationSeconds = 60" in powershell
    assert "SustainedPagingMinConsecutiveSamples = 3" in powershell
    assert "wsl.exe --shutdown" in powershell
    assert "pagefile_allocation_classification" in powershell


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _qualified_resource_smoke(tmp_path: Path, monkeypatch) -> dict:
    provenance = {"source_manifest_sha256": "SOURCE", "checkpoint_manifest_sha256": "CHECKPOINT"}
    monkeypatch.setattr(stage0, "validate_execution_provenance", lambda _run_dir: provenance)
    internal = {
        "status": "ACTUAL_PATH_RESOURCE_SMOKE_PASS",
        "model_inference_calls": 1,
        "raw_chunk_shape": list(stage0.RAW_CHUNK_SHAPE),
        "raw_chunk_finite": True,
        "resource_monitor": {"samples": 8, "exceptions": [], "maximum_swap_used_bytes": 0},
        "runtime": {
            "parameter_devices": ["cuda:0"],
            "device_map_requested": False,
            "cpu_or_disk_model_offload": False,
        },
        "provenance": provenance,
    }
    internal_path = tmp_path / "resource_smoke.json"
    _write_json(internal_path, internal)
    host = {
        "schema_version": "epoch6.schedule_stage0.host_resource_smoke.v2",
        "resource_governance_mode": "CALIBRATED_OUTCOME_FREE_V1",
        "resource_amendment_sha256": stage0.EXPECTED_RESOURCE_AMENDMENT_SHA256,
        "final_decision": "EPOCH6_STAGE0_RESOURCE_SMOKE_PASS_CALIBRATED",
        "child_exit_code": 0,
        "idle_control_valid": True,
        "sustained_paging_detected": False,
        "oom_or_kill_signature_detected": False,
        "clean_state_restored": True,
        "gpu_release_verified": True,
        "scientific_gate_rows": 0,
        "simulator_actions_executed": 0,
        "reward_success_done_read": False,
        "pagefile_current_growth_mib": 21,
        "pagefile_allocation_classification": "NONFATAL_ALLOCATION_ONLY_ABSENT_PRESSURE",
        "internal_report_sha256": stage0.sha256_file(internal_path),
        "protocol_sha256": stage0.EXPECTED_PROTOCOL_SHA256,
        "monitor_script_sha256": stage0.sha256_file(
            stage0.REPO_ROOT / "scripts" / "monitor_epoch6_schedule_stage0_smoke.ps1"
        ),
    }
    _write_json(tmp_path / "resource_smoke_host.json", host)
    return host


def test_calibrated_resource_smoke_accepts_allocation_only_delta(tmp_path: Path, monkeypatch) -> None:
    _qualified_resource_smoke(tmp_path, monkeypatch)
    assert stage0.valid_resource_smoke(tmp_path)


def test_calibrated_resource_smoke_rejects_sustained_paging(tmp_path: Path, monkeypatch) -> None:
    host = _qualified_resource_smoke(tmp_path, monkeypatch)
    host["sustained_paging_detected"] = True
    _write_json(tmp_path / "resource_smoke_host.json", host)
    assert not stage0.valid_resource_smoke(tmp_path)


def test_calibrated_resource_smoke_rejects_failed_clean_state_restore(
    tmp_path: Path, monkeypatch
) -> None:
    host = _qualified_resource_smoke(tmp_path, monkeypatch)
    host["clean_state_restored"] = False
    _write_json(tmp_path / "resource_smoke_host.json", host)
    assert not stage0.valid_resource_smoke(tmp_path)


def test_partial_sequence_round_trip_preserves_completed_keys(tmp_path: Path) -> None:
    raw = np.full((20, 30, 20), np.nan, dtype=np.float32)
    processed = np.full((20, 30, 7), np.nan, dtype=np.float32)
    latencies = np.full(20, np.nan, dtype=np.float64)
    draw_positions = np.full(20, -1, dtype=np.int64)
    raw[0] = 0.25
    processed[0] = stage0.raw_to_processed_7d(raw[0])
    latencies[0] = 0.5
    draw_positions[0] = 0
    provenance = {"source_manifest_sha256": "SOURCE"}
    rows = [
        {
            "logical_index": 0,
            "logical_key": stage0.logical_key(0),
            "logical_key_string": stage0.canonical_key_string(0),
            "draw_position": 0,
            "input_sha256": "INPUT",
            "raw_chunk_sha256": stage0.hash_array(raw[0]),
            "processed_7d_chunk_sha256": stage0.hash_array(processed[0]),
            "latency_seconds": 0.5,
        }
    ]
    rng_state = {
        "python_random_state": stage0.nested_lists(__import__("random").getstate()),
        "numpy_algorithm": "MT19937",
        "numpy_keys": np.arange(624, dtype=np.uint32),
        "numpy_position": 0,
        "numpy_has_gauss": 0,
        "numpy_cached_gaussian": 0.0,
        "torch_cpu": np.arange(32, dtype=np.uint8),
        "torch_cuda": [np.arange(32, dtype=np.uint8)],
    }
    order = stage0.SEQUENCES["A"][1]
    stage0.write_sequence_partial(
        tmp_path,
        "A",
        stage0.ROOT_SEED,
        order,
        provenance,
        raw,
        processed,
        latencies,
        draw_positions,
        rows,
        rng_state,
    )
    arrays, restored_rows, restored_rng = stage0.load_sequence_partial(
        tmp_path, "A", stage0.ROOT_SEED, order, provenance
    )
    assert [row["logical_index"] for row in restored_rows] == [0]
    np.testing.assert_array_equal(arrays["raw_chunks"][0], raw[0])
    assert np.isnan(arrays["raw_chunks"][1]).all()
    assert restored_rng is not None
    np.testing.assert_array_equal(restored_rng["torch_cuda"][0], rng_state["torch_cuda"][0])
