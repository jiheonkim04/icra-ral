"""Run the frozen pre-method LIBERO contact-transition topology label gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "reports"
    / "epoch6_contact_transition_topology"
    / "problem_verification_protocol.json"
)
EXPECTED_PROTOCOL_SHA256 = "7FA28AAEEAC9886F36DD5CCD059CA7AC4CD65B21FABFBBCA4AFFA53B0A256240"
RESOURCE_AMENDMENT_PATH = (
    REPO_ROOT
    / "reports"
    / "epoch7_contact_transition_topology"
    / "resource_rule_amendment.json"
)
EXPECTED_RESOURCE_AMENDMENT_SHA256 = (
    "7CCDCE5D9AA0B24C356AF873D0481AF76312D3C7FCF6871C4CA80FD6621ACFEB"
)
LIBERO_ROOT = Path("/mnt/c/assets/repos/LIBERO")
DATA_ROOT = Path("/mnt/c/assets/data/libero")
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
ROOT_SEED = 620260721
EDGE_TYPES = (
    "free-free",
    "free-articulated",
    "free-static",
    "articulated-articulated",
    "articulated-static",
)
TYPED_BINS = tuple(
    f"{edge_type}:{direction}"
    for edge_type in EDGE_TYPES
    for direction in ("birth", "death")
)
TYPE_ORDER = {"free": 0, "articulated": 1, "static": 2}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise RuntimeError(f"refusing to overwrite preserved artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(32 * 1024 * 1024), b""):
            digest.update(chunk)
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
    return digest.hexdigest().upper()


def hash_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_default).encode(
            "utf-8"
        )
    ).hexdigest().upper()


def validate_protocol() -> dict[str, Any]:
    observed = sha256_file(PROTOCOL_PATH)
    if observed != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            f"contact-topology protocol hash mismatch: expected {EXPECTED_PROTOCOL_SHA256}, got {observed}"
        )
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["root_seed"] != ROOT_SEED:
        raise RuntimeError("contact-topology root seed mismatch")
    if protocol["contact_graph"]["edge_types"] != list(EDGE_TYPES):
        raise RuntimeError("contact-topology edge types mismatch")
    return protocol


def all_tasks(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    split = protocol["task_split"]
    tasks = [
        dict(task)
        for partition in ("development_train", "development_tune", "validation")
        for task in split[partition]
    ]
    tokens = [task_token(task) for task in tasks]
    if len(tasks) != 10 or len(set(tokens)) != len(tokens):
        raise RuntimeError("frozen task panel is not ten unique tasks")
    return tasks


def task_token(task: Mapping[str, Any]) -> str:
    return f"{task['suite']}_task{int(task['task_id'])}"


def dataset_path(task: Mapping[str, Any]) -> Path:
    return DATA_ROOT / str(task["suite"]) / f"{task['task']}_demo.hdf5"


def git_text(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_integrity(root: Path) -> dict[str, Any]:
    worktree = subprocess.run(
        ["git", "-C", str(root), "diff", "--ignore-cr-at-eol", "--quiet"]
    )
    index = subprocess.run(["git", "-C", str(root), "diff", "--cached", "--quiet"])
    untracked = git_text(root, ["ls-files", "--others", "--exclude-standard"]).splitlines()
    return {
        "head": git_text(root, ["rev-parse", "HEAD"]),
        "worktree_clean_ignoring_cr_at_eol": worktree.returncode == 0,
        "index_clean": index.returncode == 0,
        "untracked_files": [item for item in untracked if item],
    }


def package_python_manifest(package: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(f"runtime package is unavailable: {package}")
    root = Path(next(iter(spec.submodule_search_locations))).resolve(strict=True)
    files = [
        {"relative_path": str(path.relative_to(root)), "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*.py"))
    ]
    return {"root": str(root), "files": files, "manifest_sha256": hash_payload(files)}


def source_provenance() -> dict[str, Any]:
    libero = git_integrity(LIBERO_ROOT)
    source = {
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "libero": libero,
        "libero_expected_revision": LIBERO_REVISION,
        "libero_env_wrapper_sha256": sha256_file(
            LIBERO_ROOT / "libero" / "libero" / "envs" / "env_wrapper.py"
        ),
        "robosuite_version": importlib.metadata.version("robosuite"),
        "mujoco_version": importlib.metadata.version("mujoco"),
        "h5py_version": importlib.metadata.version("h5py"),
        "robosuite_python_manifest": package_python_manifest("robosuite"),
        "python_executable": sys.executable,
        "python_version": sys.version,
    }
    source["manifest_sha256"] = hash_payload(source)
    return source


def resource_snapshot() -> dict[str, Any]:
    memory: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, value = line.split(":", 1)
        memory[name] = int(value.strip().split()[0]) * 1024
    disk = shutil.disk_usage(REPO_ROOT)
    return {
        "captured_at": utc_now(),
        "mem_total_bytes": memory["MemTotal"],
        "mem_available_bytes": memory["MemAvailable"],
        "swap_total_bytes": memory["SwapTotal"],
        "swap_used_bytes": memory["SwapTotal"] - memory["SwapFree"],
        "disk_free_bytes": disk.free,
    }


def require_safe_resources(snapshot: Mapping[str, Any]) -> None:
    if int(snapshot["swap_used_bytes"]) != 0:
        raise RuntimeError("nonzero WSL swap use is forbidden")
    if int(snapshot["mem_available_bytes"]) < 3 * 1024**3:
        raise RuntimeError("less than 3 GiB WSL memory is available")
    if int(snapshot["disk_free_bytes"]) < 60_000_000_000:
        raise RuntimeError("free disk is below the frozen 60 GB floor")


def static_preflight(run_dir: Path) -> dict[str, Any]:
    protocol = validate_protocol()
    before = resource_snapshot()
    require_safe_resources(before)
    source = source_provenance()
    if source["libero"]["head"] != LIBERO_REVISION:
        raise RuntimeError("pinned LIBERO revision mismatch")
    if (
        not source["libero"]["worktree_clean_ignoring_cr_at_eol"]
        or not source["libero"]["index_clean"]
        or source["libero"]["untracked_files"]
    ):
        raise RuntimeError(f"pinned LIBERO integrity failure: {source['libero']}")
    if (
        source["robosuite_version"] != "1.4.0"
        or source["mujoco_version"] != "3.8.1"
        or source["h5py_version"] != "3.16.0"
    ):
        raise RuntimeError("runtime package version mismatch")
    files = []
    for task in all_tasks(protocol):
        path = dataset_path(task).resolve(strict=True)
        files.append(
            {
                "task": task,
                "token": task_token(task),
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    dataset_manifest = {"files": files, "manifest_sha256": hash_payload(files)}
    write_json(run_dir / "dataset_manifest.json", dataset_manifest)
    result = {
        "schema_version": "epoch6.contact_topology.preflight.v1",
        "captured_at": utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "source": source,
        "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
        "dataset_manifest_file_sha256": sha256_file(run_dir / "dataset_manifest.json"),
        "resources_before": before,
        "contact_labels_read": False,
        "forbidden_dataset_access_count": 0,
        "simulator_actions_executed": 0,
        "success_check_calls": 0,
        "reward_success_done_read": False,
        "status": "STATIC_PREFLIGHT_PASS",
    }
    write_json(run_dir / "static_preflight.json", result)
    return result


def validate_preflight(run_dir: Path, task: Mapping[str, Any] | None = None) -> dict[str, Any]:
    preflight = json.loads((run_dir / "static_preflight.json").read_text(encoding="utf-8"))
    manifest_path = run_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if preflight["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("preflight protocol mismatch")
    if preflight["source"] != source_provenance():
        raise RuntimeError("runtime source changed after preflight")
    if sha256_file(manifest_path) != preflight["dataset_manifest_file_sha256"]:
        raise RuntimeError("dataset manifest file changed after preflight")
    if hash_payload(manifest["files"]) != preflight["dataset_manifest_sha256"]:
        raise RuntimeError("dataset content manifest identity mismatch")
    entries = manifest["files"]
    if task is not None:
        entries = [entry for entry in entries if entry["token"] == task_token(task)]
        if len(entries) != 1:
            raise RuntimeError("task dataset manifest entry is missing or duplicated")
    for entry in entries:
        path = Path(entry["path"])
        if path.stat().st_size != int(entry["bytes"]) or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"dataset file changed after preflight: {path}")
    return {
        "source_manifest_sha256": preflight["source"]["manifest_sha256"],
        "dataset_manifest_sha256": preflight["dataset_manifest_sha256"],
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
    }


def import_pinned_libero() -> tuple[Any, Any, Path]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/jiheon/.libero")
    for name in [name for name in sys.modules if name == "libero" or name.startswith("libero.")]:
        del sys.modules[name]
    namespace = types.ModuleType("libero")
    namespace.__package__ = "libero"
    namespace.__path__ = [str(LIBERO_ROOT / "libero")]
    sys.modules["libero"] = namespace
    import inspect
    import libero.libero.envs.env_wrapper as env_wrapper
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    executed = Path(inspect.getsourcefile(env_wrapper) or "").resolve(strict=True)
    expected = (LIBERO_ROOT / "libero" / "libero" / "envs" / "env_wrapper.py").resolve(
        strict=True
    )
    if executed != expected:
        raise RuntimeError(f"executed LIBERO wrapper is not pinned: {executed}")
    return benchmark, OffScreenRenderEnv, executed


def name_to_id(model: Any, kind: str, name: str) -> int | None:
    function = getattr(model, f"{kind}_name2id", None)
    if function is not None:
        try:
            value = int(function(name))
            return value if value >= 0 else None
        except Exception:
            return None
    return None


def id_to_name(model: Any, kind: str, index: int) -> str:
    function = getattr(model, f"{kind}_id2name", None)
    if function is not None:
        try:
            value = function(int(index))
            if value:
                return str(value)
        except Exception:
            pass
    return f"{kind}_{int(index)}"


def canonical_body_id(model: Any, body_id: int) -> int:
    current = int(body_id)
    chain: list[int] = []
    while current > 0:
        chain.append(current)
        if int(model.body_jntnum[current]) > 0:
            return current
        current = int(model.body_parentid[current])
    return chain[-1] if chain else int(body_id)


def body_type(model: Any, body_id: int) -> str:
    current = int(body_id)
    while current > 0:
        start = int(model.body_jntadr[current])
        count = int(model.body_jntnum[current])
        joint_types = [int(model.jnt_type[index]) for index in range(start, start + count)]
        if 0 in joint_types:
            return "free"
        if 2 in joint_types or 3 in joint_types:
            return "articulated"
        current = int(model.body_parentid[current])
    return "static"


def edge_type(left: str, right: str) -> str:
    ordered = sorted((left, right), key=TYPE_ORDER.__getitem__)
    return f"{ordered[0]}-{ordered[1]}"


Edge = tuple[str, str, str, str]


def active_contact_graph(sim: Any, robot_geom_ids: set[int]) -> tuple[set[Edge], int]:
    model = sim.model
    edges: set[Edge] = set()
    retained_robot_edges = 0
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        geom_left, geom_right = int(contact.geom1), int(contact.geom2)
        if geom_left in robot_geom_ids or geom_right in robot_geom_ids:
            continue
        body_left = canonical_body_id(model, int(model.geom_bodyid[geom_left]))
        body_right = canonical_body_id(model, int(model.geom_bodyid[geom_right]))
        if body_left == body_right:
            continue
        type_left, type_right = body_type(model, body_left), body_type(model, body_right)
        if type_left == "static" and type_right == "static":
            continue
        left = (id_to_name(model, "body", body_left), type_left)
        right = (id_to_name(model, "body", body_right), type_right)
        ordered = sorted((left, right), key=lambda value: (TYPE_ORDER[value[1]], value[0]))
        edges.add((ordered[0][0], ordered[0][1], ordered[1][0], ordered[1][1]))
    return edges, retained_robot_edges


def graph_hash(graph: Iterable[Edge]) -> str:
    return hash_payload([list(edge) for edge in sorted(graph)])


def debounced_graphs(raw_graphs: Sequence[set[Edge]], persistence: int = 2) -> list[set[Edge]]:
    if persistence != 2:
        raise ValueError("the frozen debounce persistence is exactly two states")
    if not raw_graphs:
        return []
    result = [set() for _ in raw_graphs]
    universe = set().union(*raw_graphs)
    for edge in universe:
        values = [edge in graph for graph in raw_graphs]
        stable = values[0]
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[end] == values[start]:
                end += 1
            if start == 0 or end - start >= persistence:
                stable = values[start]
            if stable:
                for index in range(start, end):
                    result[index].add(edge)
            start = end
    return result


def transition_matrix(graphs: Sequence[set[Edge]]) -> np.ndarray:
    matrix = np.zeros((len(graphs), len(TYPED_BINS)), dtype=np.uint8)
    lookup = {name: index for index, name in enumerate(TYPED_BINS)}
    for index in range(1, len(graphs)):
        for edge in graphs[index] - graphs[index - 1]:
            matrix[index, lookup[f"{edge_type(edge[1], edge[3])}:birth"]] = 1
        for edge in graphs[index - 1] - graphs[index]:
            matrix[index, lookup[f"{edge_type(edge[1], edge[3])}:death"]] = 1
    return matrix


def resolve_robot_geoms(env: Any) -> tuple[set[int], list[str], list[str]]:
    model = env.env.sim.model
    names = sorted(
        {
            str(name)
            for robot in env.env.robots
            for name in robot.robot_model.contact_geoms
        }
    )
    resolved: set[int] = set()
    unresolved: list[str] = []
    for name in names:
        index = name_to_id(model, "geom", name)
        if index is None:
            unresolved.append(name)
        else:
            resolved.add(index)
    return resolved, names, unresolved


def make_task_env(task: Mapping[str, Any]) -> tuple[Any, dict[str, Any], Path]:
    benchmark, OffScreenRenderEnv, executed_wrapper = import_pinned_libero()
    suite = benchmark.get_benchmark_dict()[str(task["suite"])]()
    observed_task = suite.get_task(int(task["task_id"]))
    if observed_task.name != task["task"]:
        raise RuntimeError(f"frozen task name mismatch: {observed_task.name}")
    bddl = (
        LIBERO_ROOT
        / "libero"
        / "libero"
        / "bddl_files"
        / str(task["suite"])
        / observed_task.bddl_file
    ).resolve(strict=True)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
    attestation = {"success_check_calls": 0}

    def forbidden_success(*_args: Any, **_kwargs: Any) -> None:
        attestation["success_check_calls"] += 1
        raise RuntimeError("success checking is forbidden during contact-label extraction")

    attestation["original_wrapper_check"] = env.check_success
    attestation["original_inner_check"] = env.env._check_success
    env.check_success = forbidden_success
    env.env._check_success = forbidden_success
    env.seed(ROOT_SEED)
    env.reset()
    attestation["executed_wrapper"] = str(executed_wrapper)
    return env, attestation, bddl


def close_task_env(env: Any, attestation: Mapping[str, Any]) -> None:
    env.check_success = attestation["original_wrapper_check"]
    env.env._check_success = attestation["original_inner_check"]
    env.close()


def require_host_smoke_lock(run_dir: Path) -> None:
    expected = (run_dir.parent / "host_resource_smoke.global.lock.json").resolve()
    provided = os.environ.get("EPOCH6_CONTACT_HOST_LOCK_PATH")
    if not provided or Path(provided).resolve() != expected or not expected.is_file():
        raise RuntimeError("valid host resource-smoke lock is required")


def resource_smoke(run_dir: Path) -> dict[str, Any]:
    protocol = validate_protocol()
    require_host_smoke_lock(run_dir)
    task = all_tasks(protocol)[0]
    provenance = validate_preflight(run_dir, task)
    before = resource_snapshot()
    require_safe_resources(before)
    env = attestation = None
    try:
        import h5py

        env, attestation, bddl = make_task_env(task)
        robot_ids, robot_names, unresolved = resolve_robot_geoms(env)
        with h5py.File(dataset_path(task), "r") as handle:
            state = np.asarray(handle["data"]["demo_0"]["states"][0], dtype=np.float64)
        env.env.sim.set_state_from_flattened(state)
        env.env.sim.forward()
        restored = np.asarray(env.env.sim.get_state().flatten(), dtype=np.float64)
        graph, retained_robot = active_contact_graph(env.env.sim, robot_ids)
        after = resource_snapshot()
        require_safe_resources(after)
        result = {
            "schema_version": "epoch6.contact_topology.resource_smoke.v1",
            "completed_at": utc_now(),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "provenance": provenance,
            "task": task,
            "demo_id": 0,
            "state_index": 0,
            "bddl": str(bddl),
            "state_shape": list(state.shape),
            "state_finite": bool(np.isfinite(state).all()),
            "state_roundtrip_max_abs_error": float(np.max(np.abs(restored - state))),
            "robot_contact_geom_count": len(robot_names),
            "robot_contact_geom_resolved_count": len(robot_ids),
            "robot_contact_geom_unresolved": unresolved,
            "retained_edges_touching_robot": retained_robot,
            "raw_graph_sha256": graph_hash(graph),
            "raw_graph_edge_count": len(graph),
            "resources_before": before,
            "resources_after": after,
            "resource_only_excluded_from_gate": True,
            "contact_label_gate_rows": 0,
            "forbidden_dataset_access_count": 0,
            "simulator_actions_executed": 0,
            "success_check_calls": int(attestation["success_check_calls"]),
            "reward_success_done_read": False,
            "status": "ACTUAL_PATH_CONTACT_RESOURCE_SMOKE_PASS",
        }
    finally:
        if env is not None and attestation is not None:
            close_task_env(env, attestation)
    write_json(run_dir / "resource_smoke.json", result)
    return result


def valid_host_smoke(run_dir: Path) -> bool:
    try:
        internal_path = run_dir / "resource_smoke.json"
        host_path = run_dir / "resource_smoke_host.json"
        internal = json.loads(internal_path.read_text(encoding="utf-8"))
        host = json.loads(host_path.read_text(encoding="utf-8-sig"))
        internal_valid = bool(
            internal["status"] == "ACTUAL_PATH_CONTACT_RESOURCE_SMOKE_PASS"
            and internal["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256
            and internal["contact_label_gate_rows"] == 0
            and internal["forbidden_dataset_access_count"] == 0
            and internal["simulator_actions_executed"] == 0
            and internal["success_check_calls"] == 0
            and not internal["reward_success_done_read"]
            and internal["resources_after"]["swap_used_bytes"] == 0
            and host["internal_report_sha256"] == sha256_file(internal_path)
        )
        old_valid = bool(
            host.get("schema_version") == "epoch6.contact_topology.host_resource_smoke.v1"
            and host.get("final_decision")
            == "EPOCH6_CONTACT_STAGE0A_RESOURCE_SMOKE_PASS"
            and host["child_exit_code"] == 0
            and host["pagefile_current_growth_mib"] <= 0
            and not host["pagefile_write_activity"]
            and host["monitor_script_sha256"]
            == sha256_file(REPO_ROOT / "scripts" / "monitor_epoch6_contact_stage0a_smoke.ps1")
        )
        epoch7_valid = bool(
            sha256_file(RESOURCE_AMENDMENT_PATH)
            == EXPECTED_RESOURCE_AMENDMENT_SHA256
            and host.get("schema_version")
            == "epoch7.contact_topology.host_resource_smoke.v1"
            and host.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256
            and host.get("resource_amendment_sha256")
            == EXPECTED_RESOURCE_AMENDMENT_SHA256
            and host.get("final_decision")
            == "EPOCH7_CONTACT_STAGE0A_RESOURCE_SMOKE_PASS"
            and host["child_exit_code"] == 0
            and host["thresholds"]["baseline_used_fraction_max"] == 0.70
            and host["thresholds"]["peak_used_fraction_max"] == 0.85
            and host["thresholds"]["pagefile_allocation_growth_mib_max"] == 16.0
            and host["baseline"]["used_fraction"] <= 0.70
            and host["peak"]["used_fraction"] <= 0.85
            and host["pagefile_current_growth_mib"] <= 16.0
            and not host["pagefile_write_activity"]
            and host["memory_release_verified"]
            and host["gpu_release_verified"]
            and host["internal_valid"]
            and host["scientific_gate_rows"] == 0
            and host["simulator_actions_executed"] == 0
            and not host["reward_success_done_read"]
            and not host["wsl_shutdown_after_child_requested"]
            and host["wsl_cache_drop_after_child_requested"]
            and host["monitor_script_sha256"]
            == sha256_file(REPO_ROOT / "scripts" / "monitor_epoch7_contact_stage0a_smoke.ps1")
        )
        return internal_valid and (old_valid or epoch7_valid)
    except Exception:
        return False


def read_demo_arrays(path: Path, demo_id: int) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as handle:
        demo = handle["data"][f"demo_{demo_id}"]
        states = np.asarray(demo["states"], dtype=np.float64)
        actions = np.asarray(demo["actions"], dtype=np.float64)
    if states.ndim != 2 or actions.ndim != 2 or actions.shape[1] < 7:
        raise RuntimeError("unexpected demonstration state/action shape")
    if states.shape[0] != actions.shape[0] or states.shape[0] < 2:
        raise RuntimeError("demonstration state/action lengths are invalid")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise RuntimeError("demonstration state/action arrays are nonfinite")
    return states, actions


def extract_task(run_dir: Path, task: Mapping[str, Any]) -> dict[str, Any]:
    protocol = validate_protocol()
    if not valid_host_smoke(run_dir):
        raise RuntimeError("host-qualified contact resource smoke is required")
    provenance = validate_preflight(run_dir, task)
    before = resource_snapshot()
    require_safe_resources(before)
    env = attestation = None
    try:
        env, attestation, bddl = make_task_env(task)
        robot_ids, robot_names, unresolved = resolve_robot_geoms(env)
        hashes_raw: list[str] = []
        hashes_debounced: list[str] = []
        matrices: list[np.ndarray] = []
        gripper_changes_all: list[np.ndarray] = []
        state_indices: list[np.ndarray] = []
        offsets = [0]
        demo_rows: list[dict[str, Any]] = []
        max_roundtrip = 0.0
        restored_states = 0
        retained_robot_edges = 0
        for demo_id in protocol["dataset"]["selected_demo_ids"]:
            states, actions = read_demo_arrays(dataset_path(task), int(demo_id))
            raw_graphs: list[set[Edge]] = []
            for state in states:
                env.env.sim.set_state_from_flattened(state)
                env.env.sim.forward()
                restored = np.asarray(env.env.sim.get_state().flatten(), dtype=np.float64)
                max_roundtrip = max(max_roundtrip, float(np.max(np.abs(restored - state))))
                graph, retained = active_contact_graph(env.env.sim, robot_ids)
                retained_robot_edges += retained
                raw_graphs.append(graph)
                restored_states += 1
            stable_graphs = debounced_graphs(raw_graphs)
            matrix = transition_matrix(stable_graphs)
            transition_frames = np.flatnonzero(np.any(matrix != 0, axis=1))
            gripper_change = np.zeros(len(actions), dtype=np.uint8)
            gripper_change[1:] = (
                (actions[1:, 6] >= 0) != (actions[:-1, 6] >= 0)
            ).astype(np.uint8)
            change_frames = np.flatnonzero(gripper_change)
            outside = [
                int(change_frames.size == 0 or np.min(np.abs(change_frames - frame)) > 2)
                for frame in transition_frames
            ]
            hashes_raw.extend(graph_hash(graph) for graph in raw_graphs)
            hashes_debounced.extend(graph_hash(graph) for graph in stable_graphs)
            matrices.append(matrix)
            gripper_changes_all.append(gripper_change)
            state_indices.append(np.arange(len(states), dtype=np.int32))
            offsets.append(offsets[-1] + len(states))
            demo_rows.append(
                {
                    "demo_id": int(demo_id),
                    "frames": len(states),
                    "eligible_frame_pairs": len(states) - 1,
                    "debounced_transition_frames": int(len(transition_frames)),
                    "off_gripper_transition_frames": int(sum(outside)),
                    "typed_direction_counts": {
                        name: int(matrix[:, index].sum()) for index, name in enumerate(TYPED_BINS)
                    },
                    "raw_graph_sequence_sha256": hash_payload(
                        [graph_hash(graph) for graph in raw_graphs]
                    ),
                    "debounced_graph_sequence_sha256": hash_payload(
                        [graph_hash(graph) for graph in stable_graphs]
                    ),
                }
            )
        transition_values = np.concatenate(matrices, axis=0)
        npz_path = run_dir / f"task_{task_token(task)}.npz"
        temporary = npz_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            raw_graph_hashes=np.asarray(hashes_raw, dtype="U64"),
            debounced_graph_hashes=np.asarray(hashes_debounced, dtype="U64"),
            typed_transitions=transition_values,
            gripper_changes=np.concatenate(gripper_changes_all),
            state_indices=np.concatenate(state_indices),
            demo_offsets=np.asarray(offsets, dtype=np.int64),
            demo_ids=np.asarray(protocol["dataset"]["selected_demo_ids"], dtype=np.int32),
        )
        temporary.replace(npz_path)
        after = resource_snapshot()
        require_safe_resources(after)
        result = {
            "schema_version": "epoch6.contact_topology.task_extraction.v1",
            "completed_at": utc_now(),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "provenance": provenance,
            "task": dict(task),
            "token": task_token(task),
            "bddl": str(bddl),
            "dataset_file": str(dataset_path(task)),
            "demo_rows": demo_rows,
            "states_planned": int(sum(row["frames"] for row in demo_rows)),
            "states_restored": restored_states,
            "state_roundtrip_max_abs_error": max_roundtrip,
            "robot_contact_geom_count": len(robot_names),
            "robot_contact_geom_resolved_count": len(robot_ids),
            "robot_contact_geom_unresolved": unresolved,
            "retained_edges_touching_robot": retained_robot_edges,
            "typed_direction_counts": {
                name: int(transition_values[:, index].sum())
                for index, name in enumerate(TYPED_BINS)
            },
            "npz": str(npz_path),
            "npz_sha256": sha256_file(npz_path),
            "resources_before": before,
            "resources_after": after,
            "dataset_accesses": ["states", "actions"],
            "forbidden_dataset_access_count": 0,
            "simulator_actions_executed": 0,
            "success_check_calls": int(attestation["success_check_calls"]),
            "reward_success_done_read": False,
            "status": "TASK_CONTACT_EXTRACTION_PASS",
        }
    finally:
        if env is not None and attestation is not None:
            close_task_env(env, attestation)
    write_json(run_dir / f"task_{task_token(task)}.json", result)
    return result


def repeat_task(run_dir: Path, task: Mapping[str, Any]) -> dict[str, Any]:
    validate_protocol()
    validate_preflight(run_dir, task)
    metadata_path = run_dir / f"task_{task_token(task)}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    npz_path = run_dir / f"task_{task_token(task)}.npz"
    if sha256_file(npz_path) != metadata["npz_sha256"]:
        raise RuntimeError("task extraction NPZ changed before repeat")
    with np.load(npz_path, allow_pickle=False) as payload:
        expected_hashes = np.asarray(payload["raw_graph_hashes"]).tolist()
        offsets = np.asarray(payload["demo_offsets"], dtype=np.int64)
    env = attestation = None
    compared = matches = 0
    rows: list[dict[str, Any]] = []
    try:
        env, attestation, _bddl = make_task_env(task)
        robot_ids, _robot_names, unresolved = resolve_robot_geoms(env)
        if unresolved:
            raise RuntimeError("robot geom resolution changed before cold repeat")
        for demo_position, demo_id in enumerate(validate_protocol()["dataset"]["selected_demo_ids"]):
            states, _actions = read_demo_arrays(dataset_path(task), int(demo_id))
            for state_index in range(0, len(states), 10):
                env.env.sim.set_state_from_flattened(states[state_index])
                env.env.sim.forward()
                graph, retained = active_contact_graph(env.env.sim, robot_ids)
                if retained:
                    raise RuntimeError("cold repeat retained a robot edge")
                global_index = int(offsets[demo_position] + state_index)
                observed = graph_hash(graph)
                expected = str(expected_hashes[global_index])
                rows.append(
                    {
                        "demo_id": int(demo_id),
                        "state_index": state_index,
                        "expected_sha256": expected,
                        "observed_sha256": observed,
                        "match": observed == expected,
                    }
                )
                compared += 1
                matches += int(observed == expected)
        result = {
            "schema_version": "epoch6.contact_topology.cold_repeat.v1",
            "completed_at": utc_now(),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "task": dict(task),
            "states_compared": compared,
            "hash_matches": matches,
            "hash_match_fraction": matches / compared,
            "rows": rows,
            "forbidden_dataset_access_count": 0,
            "simulator_actions_executed": 0,
            "success_check_calls": int(attestation["success_check_calls"]),
            "reward_success_done_read": False,
            "status": "TASK_COLD_REPEAT_PASS" if matches == compared else "TASK_COLD_REPEAT_FAIL",
        }
    finally:
        if env is not None and attestation is not None:
            close_task_env(env, attestation)
    write_json(run_dir / f"repeat_{task_token(task)}.json", result)
    return result


def load_task_artifacts(run_dir: Path, task: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = json.loads(
        (run_dir / f"task_{task_token(task)}.json").read_text(encoding="utf-8")
    )
    repeat = json.loads(
        (run_dir / f"repeat_{task_token(task)}.json").read_text(encoding="utf-8")
    )
    if sha256_file(run_dir / f"task_{task_token(task)}.npz") != metadata["npz_sha256"]:
        raise RuntimeError("task NPZ integrity mismatch during adjudication")
    return metadata, repeat


def adjudicate(run_dir: Path) -> dict[str, Any]:
    protocol = validate_protocol()
    gates = protocol["stage0a_label_gate"]["gates"]
    tasks = all_tasks(protocol)
    validation_tokens = {task_token(task) for task in protocol["task_split"]["validation"]}
    records = {task_token(task): load_task_artifacts(run_dir, task) for task in tasks}
    all_metadata = [value[0] for value in records.values()]
    validation = [records[token][0] for token in validation_tokens]
    all_repeats = [value[1] for value in records.values()]
    validation_pairs = sum(
        row["eligible_frame_pairs"] for item in validation for row in item["demo_rows"]
    )
    validation_transitions = sum(
        row["debounced_transition_frames"] for item in validation for row in item["demo_rows"]
    )
    validation_off = sum(
        row["off_gripper_transition_frames"] for item in validation for row in item["demo_rows"]
    )
    validation_task_demo_coverage = {
        item["token"]: sum(row["debounced_transition_frames"] > 0 for row in item["demo_rows"])
        for item in validation
    }
    full_counts = {
        name: sum(item["typed_direction_counts"][name] for item in all_metadata)
        for name in TYPED_BINS
    }
    validation_counts = {
        name: sum(item["typed_direction_counts"][name] for item in validation)
        for name in TYPED_BINS
    }
    support_min = int(gates["support_count_per_typed_direction_bin_min"])
    full_supported = sum(value >= support_min for value in full_counts.values())
    validation_supported = sum(value >= support_min for value in validation_counts.values())
    tasks_with_off = sum(
        any(row["off_gripper_transition_frames"] > 0 for row in item["demo_rows"])
        for item in validation
    )
    exceptions: list[str] = []
    for item in all_metadata:
        if item["status"] != "TASK_CONTACT_EXTRACTION_PASS":
            exceptions.append(f"{item['token']}: extraction status")
        if item.get("resources_after", {}).get("swap_used_bytes") != 0:
            exceptions.append(f"{item['token']}: nonzero swap")
    for item in all_repeats:
        if item["status"] != "TASK_COLD_REPEAT_PASS":
            exceptions.append(f"{task_token(item['task'])}: cold repeat status")
    metrics = {
        "exception_count": len(exceptions),
        "exceptions": exceptions,
        "state_roundtrip_max_abs_error": max(
            item["state_roundtrip_max_abs_error"] for item in all_metadata
        ),
        "selected_state_restore_fraction": sum(item["states_restored"] for item in all_metadata)
        / sum(item["states_planned"] for item in all_metadata),
        "robot_contact_geom_resolution_fraction": sum(
            item["robot_contact_geom_resolved_count"] for item in all_metadata
        )
        / sum(item["robot_contact_geom_count"] for item in all_metadata),
        "retained_edges_touching_robot": sum(
            item["retained_edges_touching_robot"] for item in all_metadata
        ),
        "validation_task_demo_transition_coverage": validation_task_demo_coverage,
        "validation_tasks_with_transition_in_at_least_three_demos": sum(
            value >= 3 for value in validation_task_demo_coverage.values()
        ),
        "validation_debounced_transition_frames": validation_transitions,
        "validation_eligible_frame_pairs": validation_pairs,
        "validation_transition_prevalence": validation_transitions / validation_pairs,
        "full_panel_typed_direction_counts": full_counts,
        "validation_typed_direction_counts": validation_counts,
        "full_panel_supported_typed_direction_bins": full_supported,
        "validation_supported_typed_direction_bins": validation_supported,
        "validation_transition_fraction_outside_plus_minus_two_gripper_change_frames": (
            validation_off / validation_transitions if validation_transitions else 0.0
        ),
        "validation_tasks_with_off_gripper_transitions": tasks_with_off,
        "cold_repeat_graph_hash_match_fraction": sum(
            item["hash_matches"] for item in all_repeats
        )
        / sum(item["states_compared"] for item in all_repeats),
        "forbidden_dataset_access_count": sum(
            item["forbidden_dataset_access_count"] for item in all_metadata + all_repeats
        ),
        "simulator_actions_executed": sum(
            item["simulator_actions_executed"] for item in all_metadata + all_repeats
        ),
        "success_check_calls": sum(
            item["success_check_calls"] for item in all_metadata + all_repeats
        ),
        "reward_success_done_read": any(
            item["reward_success_done_read"] for item in all_metadata + all_repeats
        ),
    }
    passed = {
        "exception_count": metrics["exception_count"] <= gates["exception_count_max"],
        "state_roundtrip": metrics["state_roundtrip_max_abs_error"]
        <= gates["state_roundtrip_max_abs_error_max"],
        "state_restore_fraction": metrics["selected_state_restore_fraction"]
        >= gates["selected_state_restore_fraction_min"],
        "robot_geom_resolution": metrics["robot_contact_geom_resolution_fraction"]
        >= gates["robot_contact_geom_resolution_fraction_min"],
        "zero_retained_robot_edges": metrics["retained_edges_touching_robot"]
        <= gates["retained_edges_touching_robot_max"],
        "validation_demo_coverage": metrics[
            "validation_tasks_with_transition_in_at_least_three_demos"
        ]
        >= gates["validation_tasks_with_transition_in_at_least_three_demos_min"],
        "validation_transition_count": metrics["validation_debounced_transition_frames"]
        >= gates["validation_debounced_transition_frames_min"],
        "validation_prevalence_lower": metrics["validation_transition_prevalence"]
        >= gates["validation_transition_prevalence_min"],
        "validation_prevalence_upper": metrics["validation_transition_prevalence"]
        <= gates["validation_transition_prevalence_max"],
        "full_typed_support": metrics["full_panel_supported_typed_direction_bins"]
        >= gates["full_panel_supported_typed_direction_bins_min"],
        "validation_typed_support": metrics["validation_supported_typed_direction_bins"]
        >= gates["validation_supported_typed_direction_bins_min"],
        "off_gripper_fraction": metrics[
            "validation_transition_fraction_outside_plus_minus_two_gripper_change_frames"
        ]
        >= gates[
            "validation_transition_fraction_outside_plus_minus_two_gripper_change_frames_min"
        ],
        "off_gripper_task_coverage": metrics["validation_tasks_with_off_gripper_transitions"]
        >= gates["validation_tasks_with_off_gripper_transitions_min"],
        "cold_repeat": metrics["cold_repeat_graph_hash_match_fraction"]
        >= gates["cold_repeat_graph_hash_match_fraction_min"],
        "zero_forbidden_access": metrics["forbidden_dataset_access_count"]
        <= gates["forbidden_dataset_access_count_max"],
        "zero_actions": metrics["simulator_actions_executed"]
        <= gates["simulator_actions_executed_max"],
        "zero_success_checks": metrics["success_check_calls"]
        <= gates["success_check_calls_max"],
        "zero_outcome_reads": not metrics["reward_success_done_read"],
    }
    decision = (
        protocol["stage0a_label_gate"]["go_decision"]
        if all(passed.values())
        else protocol["stage0a_label_gate"]["no_go_decision"]
    )
    result = {
        "schema_version": "epoch6.contact_topology.stage0a_result.v1",
        "completed_at": utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "execution_type": protocol["execution_type"],
        "metrics": metrics,
        "gates_passed": passed,
        "final_decision": decision,
        "stage0b_authorized": decision == protocol["stage0a_label_gate"]["go_decision"],
        "method_design_authorized": False,
        "vla_training_happened": False,
        "policy_rollout_happened": False,
        "paper_generation_authorized": False,
    }
    write_json(run_dir / "stage0a_result.json", result)
    return result


def find_task(protocol: Mapping[str, Any], token: str) -> dict[str, Any]:
    matches = [task for task in all_tasks(protocol) if task_token(task) == token]
    if len(matches) != 1:
        raise ValueError(f"unknown frozen task token: {token}")
    return matches[0]


def require_parent_lock(run_dir: Path) -> None:
    expected = (run_dir / "run.lock.json").resolve()
    provided = os.environ.get("EPOCH6_CONTACT_PARENT_LOCK")
    if not provided or Path(provided).resolve() != expected or not expected.is_file():
        raise RuntimeError("valid parent run lock is required")


def launch_child(run_dir: Path, mode: str, token: str) -> None:
    label = f"{mode}_{token}"
    stdout_path = run_dir / f"{label}.stdout.log"
    stderr_path = run_dir / f"{label}.stderr.log"
    if stdout_path.exists() or stderr_path.exists():
        raise RuntimeError(f"refusing to overwrite child logs for {label}")
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(ROOT_SEED)
    environment["EPOCH6_CONTACT_PARENT_LOCK"] = str((run_dir / "run.lock.json").resolve())
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        mode,
        "--run-dir",
        str(run_dir),
        "--task-token",
        token,
        "--child",
    ]
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open(
        "x", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"child {label} failed with exit code {completed.returncode}")


def run_all(run_dir: Path, resume: bool) -> dict[str, Any]:
    protocol = validate_protocol()
    if not valid_host_smoke(run_dir):
        raise RuntimeError("host-qualified contact resource smoke is required")
    lock_path = run_dir / "run.lock.json"
    write_json(
        lock_path,
        {
            "status": "active",
            "pid": os.getpid(),
            "created_at": utc_now(),
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        },
    )
    try:
        for task in all_tasks(protocol):
            token = task_token(task)
            if not (resume and (run_dir / f"task_{token}.json").is_file()):
                launch_child(run_dir, "extract-task", token)
            if not (resume and (run_dir / f"repeat_{token}.json").is_file()):
                launch_child(run_dir, "repeat-task", token)
        return adjudicate(run_dir)
    finally:
        if lock_path.exists():
            lock_path.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("preflight", "resource-smoke", "extract-task", "repeat-task", "adjudicate", "run-all"),
        required=True,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-token")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = validate_protocol()
    if args.mode == "preflight":
        static_preflight(run_dir)
    elif args.mode == "resource-smoke":
        resource_smoke(run_dir)
    elif args.mode in ("extract-task", "repeat-task"):
        if not args.child:
            raise RuntimeError("task extraction/repeat must be launched by the locked parent")
        require_parent_lock(run_dir)
        task = find_task(protocol, str(args.task_token))
        if args.mode == "extract-task":
            extract_task(run_dir, task)
        else:
            repeat_task(run_dir, task)
    elif args.mode == "adjudicate":
        adjudicate(run_dir)
    else:
        run_all(run_dir, args.resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
