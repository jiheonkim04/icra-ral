from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_epoch6_contact_topology_stage0a.py"
SPEC = importlib.util.spec_from_file_location("epoch6_contact_stage0a", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
stage0a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage0a)


def _edge(left_name: str, left_type: str, right_name: str, right_type: str):
    return (left_name, left_type, right_name, right_type)


def test_protocol_hash_and_frozen_panel() -> None:
    protocol = stage0a.validate_protocol()
    assert len(stage0a.all_tasks(protocol)) == 10
    assert protocol["dataset"]["selected_demo_ids"] == [0, 1, 2, 3, 4, 5]
    assert protocol["dataset"]["forbidden_hdf5_datasets"] == ["rewards", "dones"]


def test_two_state_debounce_removes_single_frame_chatter() -> None:
    edge = _edge("object", "free", "table", "static")
    raw = [set(), {edge}, set(), set()]
    stable = stage0a.debounced_graphs(raw)
    assert stable == [set(), set(), set(), set()]


def test_two_state_debounce_assigns_transition_to_first_persistent_state() -> None:
    edge = _edge("object", "free", "plate", "static")
    raw = [set(), {edge}, {edge}, {edge}]
    stable = stage0a.debounced_graphs(raw)
    assert stable == [set(), {edge}, {edge}, {edge}]
    matrix = stage0a.transition_matrix(stable)
    assert matrix[1, stage0a.TYPED_BINS.index("free-static:birth")] == 1
    assert int(matrix.sum()) == 1


def test_transition_matrix_separates_birth_and_death() -> None:
    edge = _edge("drawer", "articulated", "cabinet", "static")
    graphs = [set(), {edge}, {edge}, set()]
    matrix = stage0a.transition_matrix(graphs)
    assert matrix[1, stage0a.TYPED_BINS.index("articulated-static:birth")] == 1
    assert matrix[3, stage0a.TYPED_BINS.index("articulated-static:death")] == 1


def test_graph_hash_is_order_invariant_and_type_sensitive() -> None:
    first = _edge("a", "free", "b", "static")
    second = _edge("c", "articulated", "d", "static")
    assert stage0a.graph_hash([first, second]) == stage0a.graph_hash([second, first])
    changed = _edge("a", "articulated", "b", "static")
    assert stage0a.graph_hash([first]) != stage0a.graph_hash([changed])


def test_stage0a_source_has_no_simulator_step_or_outcome_dataset_read() -> None:
    source = inspect.getsource(stage0a.extract_task)
    assert ".step(" not in source
    assert '["rewards"]' not in source
    assert '["dones"]' not in source
    assert "set_state_from_flattened" in source
    assert "sim.forward" in source


def test_stage0b_is_not_implemented_or_reachable() -> None:
    choices = inspect.getsource(stage0a.parse_args)
    assert '"stage0b"' not in choices
    assert not hasattr(stage0a, "train_topology_method")


def test_contact_smoke_wrappers_use_the_actual_workspace_path() -> None:
    root = SCRIPT.parents[1]
    bash = (root / "scripts" / "run_epoch6_contact_stage0a_smoke_wsl.sh").read_text(
        encoding="utf-8"
    )
    powershell = (root / "scripts" / "monitor_epoch6_contact_stage0a_smoke.ps1").read_text(
        encoding="utf-8"
    )
    assert "/mnt/c/Users/jiheo/tca_map" in bash
    assert "/mnt/c/Users/jiheo/tca_map" in powershell
    assert "/mnt/c/Users/jiheon/" not in bash + powershell
    assert "resource_smoke_child_exit_code.txt" in bash
    assert "resource_smoke_child_exit_code.txt" in powershell
    assert "AllowWslShutdownAfterChild" in powershell
    assert "wsl.exe --shutdown" in powershell
    assert "AllowWslCacheDropAfterChild" in powershell
    assert "/proc/sys/vm/drop_caches" in powershell


def test_resource_smoke_revalidates_only_its_frozen_task_file() -> None:
    source = inspect.getsource(stage0a.resource_smoke)
    assert "validate_preflight(run_dir, task)" in source
    valid_source = inspect.getsource(stage0a.valid_host_smoke)
    assert "monitor_script_sha256" in valid_source
