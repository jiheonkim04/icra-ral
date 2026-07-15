"""Run the frozen SPARC-VLA Stage 0A math and post-residual hook smoke."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_famr_vla_stage0 import (  # noqa: E402
    HF_HOME,
    VLM_PATH,
    _active_linux_workers,
    _clone_batch,
    _directory_hashes,
    _load_policy_and_processors,
    _preprocess,
    _resource_evidence,
    _set_offline_environment,
)
from scripts.run_pcav_vla_stage0 import _postprocess_chunk, _raw_sample  # noqa: E402
from tca_map.smolvla.sparc_vla import (  # noqa: E402
    HIDDEN_WIDTH,
    PROPOSAL_HASH,
    SparcPostResidualAdapter,
    action_safety,
    compute_conceptor,
    conceptor_and_not,
    equal_episode_covariance,
    tensor_sha256,
)


PROPOSAL_FILE = REPO_ROOT / "reports" / "sparc_vla" / "researcher_proposal.md"
PROPOSAL_HASH_FILE = REPO_ROOT / "reports" / "sparc_vla" / "proposal_hash.txt"
RESOURCE_REGISTRY = REPO_ROOT / "reports" / "resource_contention_intervals.json"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "sparc_vla"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "sparc_vla" / "stage0a"
TARGET_FILES = (
    (
        "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf",
        "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf_demo.hdf5",
        0,
        0,
    ),
    (
        "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray_demo.hdf5",
        1,
        1,
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    report = Path(args.report_root)
    run = Path(args.run_root)
    return {
        "report": report,
        "run": run,
        "checkpoint": Path(args.checkpoint),
        "data_root": Path(args.libero_data_root),
        "pid": run / "worker.pid",
        "heartbeat": run / "heartbeat.json",
        "status": run / "status.json",
        "partial": run / "partial_result.json",
        "exit_code": run / "exit_code.txt",
        "adapter": run / "synthetic_adapter.npz",
        "preflight": report / "stage_0a_preflight.json",
        "result_json": report / "stage_0a_result.json",
        "result_md": report / "stage_0a_result.md",
        "validation": report / "stage_0a_validation.json",
        "blocker": report / "stage_0a_implementation_blocker.json",
    }


def _heartbeat(path: Path, state: dict[str, Any], stop: threading.Event) -> None:
    while not stop.wait(10.0):
        _write_json(path, {**state, "updated_at": _utc_now()})


def _observation_rows(data_root: Path) -> list[dict[str, Any]]:
    rows = []
    for task, filename, task_index, episode in TARGET_FILES:
        source = data_root / "libero_90" / filename
        rows.append(
            {
                "task_identity": task,
                "task_language": task.replace("_", " "),
                "source_path": str(source),
                "task_index": task_index,
                "episode": episode,
                "frame": 0,
            }
        )
    return rows


def _preflight(args: argparse.Namespace, paths: Mapping[str, Path]) -> dict[str, Any]:
    import torch

    proposal_hash_file = PROPOSAL_HASH_FILE.read_text(encoding="utf-8").strip()
    rows = _observation_rows(paths["data_root"])
    required = [paths["checkpoint"], VLM_PATH, PROPOSAL_FILE, PROPOSAL_HASH_FILE]
    required.extend(Path(row["source_path"]) for row in rows)
    missing = [str(path) for path in required if not Path(path).exists()]
    workers = _active_linux_workers()
    registry = _read_json(RESOURCE_REGISTRY) if RESOURCE_REGISTRY.is_file() else {"intervals": []}
    result_absent = not paths["result_json"].exists()
    partial_parse_error = None
    partial = None
    if paths["partial"].is_file():
        try:
            partial = _read_json(paths["partial"])
        except Exception as exc:  # pragma: no cover - exercised by runtime preflight
            partial_parse_error = f"{type(exc).__name__}: {exc}"
    proposal_ok = proposal_hash_file == PROPOSAL_HASH and _sha256(PROPOSAL_FILE) == PROPOSAL_HASH
    passed = bool(
        not missing
        and not workers
        and proposal_ok
        and torch.cuda.is_available()
        and result_absent
        and partial_parse_error is None
        and args.mode in {"audit", "hook-smoke"}
    )
    return {
        "passed": passed,
        "mode": args.mode,
        "missing_paths": missing,
        "proposal_hash_expected": PROPOSAL_HASH,
        "proposal_hash_file": proposal_hash_file,
        "proposal_hash_observed": _sha256(PROPOSAL_FILE),
        "proposal_hash_ok": proposal_ok,
        "active_linux_workers": workers,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "result_absent": result_absent,
        "partial_parsed": partial is not None,
        "partial_parse_error": partial_parse_error,
        "planned_observation_count": len(rows),
        "resource_evidence": _resource_evidence(registry, time.time()),
    }


def _noise(seed: int) -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn((1, 50, 32), generator=generator, dtype=torch.float32).to("cuda")


def _predict(policy: Any, batch: Mapping[str, Any], postprocessor: Any, noise: Any) -> np.ndarray:
    import torch

    if hasattr(policy, "reset"):
        policy.reset()
    policy.eval()
    with torch.no_grad():
        native = policy.predict_action_chunk(_clone_batch(batch), noise=noise.clone())
    return _postprocess_chunk(native, postprocessor)


def _synthetic_operator(captured: np.ndarray) -> np.ndarray:
    token_means = captured.mean(axis=2).reshape(-1, HIDDEN_WIDTH)
    first = token_means[::2]
    second = token_means[1::2]
    if first.shape[0] < 2 or second.shape[0] < 2:
        raise RuntimeError("insufficient captures for synthetic conceptor")
    _, success_covariance = equal_episode_covariance([first])
    _, failure_covariance = equal_episode_covariance([second])
    ridge = np.eye(HIDDEN_WIDTH, dtype=np.float64) * 1e-3
    success = compute_conceptor(success_covariance + ridge, 1.0)
    failure = compute_conceptor(failure_covariance + ridge * 0.5, 1.0)
    operator = conceptor_and_not(success, failure)
    if np.array_equal(operator, np.eye(HIDDEN_WIDTH)):
        raise RuntimeError("synthetic operator collapsed to identity")
    return operator


def _run_hook_smoke(args: argparse.Namespace, paths: Mapping[str, Path], preflight: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    paths["run"].mkdir(parents=True, exist_ok=True)
    paths["report"].mkdir(parents=True, exist_ok=True)
    _write_text(paths["pid"], f"{os.getpid()}\n")
    heartbeat_state: dict[str, Any] = {
        "pid": os.getpid(),
        "status": "running",
        "planned_observation_count": len(TARGET_FILES),
        "completed_observation_count": 0,
        "exception_count": 0,
    }
    _write_json(paths["status"], {**heartbeat_state, "started_at": _utc_now()})
    _write_json(paths["heartbeat"], {**heartbeat_state, "updated_at": _utc_now()})
    stop = threading.Event()
    thread = threading.Thread(target=_heartbeat, args=(paths["heartbeat"], heartbeat_state, stop), daemon=True)
    thread.start()
    checkpoint_before = _directory_hashes(paths["checkpoint"])
    started = time.time()
    try:
        _set_offline_environment()
        policy, _, preprocessor, postprocessor = _load_policy_and_processors(paths["checkpoint"])
        rows = _observation_rows(paths["data_root"])
        batches = [_preprocess(preprocessor, _raw_sample(row)) for row in rows]
        noises = [_noise(191900 + index) for index in range(len(rows))]
        bases = [_predict(policy, batch, postprocessor, noise) for batch, noise in zip(batches, noises, strict=True)]

        adapter = SparcPostResidualAdapter(policy, 11)
        adapter.register()
        adapter.reset_capture()
        capture_only = _predict(policy, batches[0], postprocessor, noises[0])
        adapter.assert_complete()
        capture_identity_error = float(np.max(np.abs(capture_only - bases[0])))
        captures = np.stack(adapter.full_tensors, axis=0)
        capture_shapes = [list(capture.shape) for capture in adapter.captures]
        capture_hashes = [capture.tensor_sha256 for capture in adapter.captures]

        adapter.clear_configuration()
        adapter.reset_capture()
        unconfigured = _predict(policy, batches[0], postprocessor, noises[0])
        adapter.assert_complete()
        unconfigured_error = float(np.max(np.abs(unconfigured - bases[0])))

        identity = np.eye(HIDDEN_WIDTH, dtype=np.float64)
        adapter.configure(identity, beta=0.1)
        adapter.reset_capture()
        identity_action = _predict(policy, batches[0], postprocessor, noises[0])
        adapter.assert_complete()
        identity_operator_error = float(np.max(np.abs(identity_action - bases[0])))

        operator = _synthetic_operator(captures)
        adapter.configure(operator, beta=0.1)
        acting_rows = []
        for index, (batch, noise, base) in enumerate(zip(batches, noises, bases, strict=True)):
            adapter.reset_capture()
            ours = _predict(policy, batch, postprocessor, noise)
            adapter.assert_complete()
            safety = action_safety(base, ours)
            acting_rows.append(
                {
                    "observation_index": index,
                    "task_identity": rows[index]["task_identity"],
                    "base_action_sha256": tensor_sha256(base),
                    "ours_action_sha256": tensor_sha256(ours),
                    "action_safety": safety,
                    "activation_delta_norms": [capture.delta_norm for capture in adapter.captures],
                    "maximum_token_delta_norms": [capture.max_token_delta_norm for capture in adapter.captures],
                }
            )
            heartbeat_state["completed_observation_count"] = index + 1
            _write_json(paths["partial"], {**heartbeat_state, "rows": acting_rows, "updated_at": _utc_now()})
            _write_json(paths["heartbeat"], {**heartbeat_state, "updated_at": _utc_now()})

        adapter.save(paths["adapter"])
        adapter.reset_capture()
        synthetic_reference = _predict(policy, batches[0], postprocessor, noises[0])
        adapter.assert_complete()
        adapter.remove()
        loaded = SparcPostResidualAdapter.load(policy, paths["adapter"])
        loaded.register()
        loaded.reset_capture()
        synthetic_reload = _predict(policy, batches[0], postprocessor, noises[0])
        loaded.assert_complete()
        reload_error = float(np.max(np.abs(synthetic_reload - synthetic_reference)))
        loaded.remove()

        removed = _predict(policy, batches[0], postprocessor, noises[0])
        removed_error = float(np.max(np.abs(removed - bases[0])))
        checkpoint_after = _directory_hashes(paths["checkpoint"])
        base_checkpoint_unchanged = checkpoint_before == checkpoint_after
        acting_action_rows = sum(
            float(row["action_safety"]["mean_full_chunk_7d_delta_l2"]) > 1e-6 for row in acting_rows
        )
        all_activation_rows_act = all(
            any(float(value) > 0.0 for value in row["activation_delta_norms"]) for row in acting_rows
        )
        action_safe = all(bool(row["action_safety"]["passed"]) for row in acting_rows)
        implementation_pass = bool(
            capture_identity_error == 0.0
            and unconfigured_error == 0.0
            and identity_operator_error == 0.0
            and removed_error == 0.0
            and reload_error <= 1e-6
            and capture_shapes == [[1, 50, 720]] * 10
            and len(set(capture_hashes)) > 1
            and all_activation_rows_act
            and action_safe
            and base_checkpoint_unchanged
        )
        if implementation_pass and acting_action_rows > 0:
            decision = "SPARC_STAGE_0A_PASS_DISCOVERY_COLLECTION_ALLOWED"
        elif implementation_pass:
            decision = "SPARC_STAGE_0A_DESIGN_FAILURE_NONACTING"
        else:
            decision = "SPARC_STAGE_0A_IMPLEMENTATION_FAILURE"
        result = {
            "method": "SPARC-VLA",
            "proposal_hash": PROPOSAL_HASH,
            "final_decision": decision,
            "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
            "completed_at": _utc_now(),
            "planned_observation_count": len(rows),
            "completed_observation_count": len(acting_rows),
            "exception_count": 0,
            "duplicate_key_count": 0,
            "capture_identity_max_abs_error": capture_identity_error,
            "unconfigured_identity_max_abs_error": unconfigured_error,
            "identity_operator_max_abs_error": identity_operator_error,
            "removed_hook_identity_max_abs_error": removed_error,
            "configured_reload_max_abs_error": reload_error,
            "capture_count": len(capture_shapes),
            "capture_shapes": capture_shapes,
            "unique_capture_hash_count": len(set(capture_hashes)),
            "operator_sha256": tensor_sha256(operator),
            "operator_audit": {
                "eigenvalue_min": float(np.linalg.eigvalsh(operator)[0]),
                "eigenvalue_max": float(np.linalg.eigvalsh(operator)[-1]),
                "finite_fraction": float(np.mean(np.isfinite(operator))),
            },
            "acting_action_row_count": acting_action_rows,
            "all_activation_rows_act": all_activation_rows_act,
            "all_action_rows_safe": action_safe,
            "base_checkpoint_unchanged": base_checkpoint_unchanged,
            "rows": acting_rows,
            "confirmatory_records_read": 0,
            "stage_0b_allowed": decision == "SPARC_STAGE_0A_PASS_DISCOVERY_COLLECTION_ALLOWED",
            "resource_evidence": preflight["resource_evidence"],
        }
        validation = {
            "final_decision": decision,
            "result_json_parsed": True,
            "proposal_hash_matches": result["proposal_hash"] == PROPOSAL_HASH,
            "completed_matches_planned": len(acting_rows) == len(rows),
            "exception_count": 0,
            "duplicate_key_count": 0,
            "identity_checks_pass": all(
                value == 0.0
                for value in (
                    capture_identity_error,
                    unconfigured_error,
                    identity_operator_error,
                    removed_error,
                )
            ),
            "reload_pass": reload_error <= 1e-6,
            "capture_shape_pass": capture_shapes == [[1, 50, 720]] * 10,
            "activation_acting_pass": all_activation_rows_act,
            "action_safety_pass": action_safe,
            "base_checkpoint_unchanged": base_checkpoint_unchanged,
            "confirmatory_records_read": 0,
            "passed": decision == "SPARC_STAGE_0A_PASS_DISCOVERY_COLLECTION_ALLOWED",
        }
        _write_json(paths["result_json"], result)
        _write_json(paths["validation"], validation)
        lines = [
            "# SPARC-VLA Stage 0A Result",
            "",
            f"Decision: `{decision}`",
            "",
            f"- observations: `{len(acting_rows)} / {len(rows)}`",
            "- exceptions / duplicates: `0 / 0`",
            f"- post-residual captures: `{len(capture_shapes)}` at `[1, 50, 720]`",
            f"- capture-only Base error: `{capture_identity_error}`",
            f"- unconfigured Base error: `{unconfigured_error}`",
            f"- identity-operator Base error: `{identity_operator_error}`",
            f"- removed-hook Base error: `{removed_error}`",
            f"- configured reload error: `{reload_error}`",
            f"- acting action rows: `{acting_action_rows} / {len(acting_rows)}`",
            f"- all action rows safe: `{str(action_safe).lower()}`",
            f"- Base checkpoint unchanged: `{str(base_checkpoint_unchanged).lower()}`",
            "- confirmatory records read: `0`",
            f"- Stage 0B allowed: `{str(result['stage_0b_allowed']).lower()}`",
            "",
            "Timing and resource evidence remain subject to the contention registry.",
        ]
        _write_text(paths["result_md"], "\n".join(lines) + "\n")
        heartbeat_state.update(
            {
                "status": "completed",
                "completed_observation_count": len(acting_rows),
                "final_decision": decision,
            }
        )
        _write_json(paths["status"], {**heartbeat_state, "completed_at": _utc_now()})
        _write_json(paths["heartbeat"], {**heartbeat_state, "updated_at": _utc_now()})
        _write_text(paths["exit_code"], "0\n")
        return result
    finally:
        stop.set()
        thread.join(timeout=2.0)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "hook-smoke"), default="audit")
    parser.add_argument("--checkpoint", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--libero-data-root", default="/mnt/c/assets/data/libero")
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    paths = _paths(args)
    paths["report"].mkdir(parents=True, exist_ok=True)
    preflight = _preflight(args, paths)
    _write_json(paths["preflight"], preflight)
    if args.mode == "audit":
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0 if preflight["passed"] else 2
    if not preflight["passed"]:
        print(json.dumps(preflight, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    try:
        result = _run_hook_smoke(args, paths, preflight)
        print(json.dumps({key: result[key] for key in ("final_decision", "stage_0b_allowed")}, indent=2))
        return 0
    except Exception as exc:
        blocker = {
            "method": "SPARC-VLA",
            "proposal_hash": PROPOSAL_HASH,
            "final_decision": "SPARC_STAGE_0A_IMPLEMENTATION_FAILURE",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc().splitlines(),
            "failed_at": _utc_now(),
        }
        _write_json(paths["blocker"], blocker)
        if paths["partial"].is_file():
            try:
                partial = _read_json(paths["partial"])
                partial["exception_count"] = int(partial.get("exception_count", 0)) + 1
                partial["failed_at"] = blocker["failed_at"]
                partial["exception_type"] = blocker["exception_type"]
                _write_json(paths["partial"], partial)
            except Exception:
                pass
        _write_json(paths["status"], {"pid": os.getpid(), "status": "failed", **blocker})
        _write_json(paths["heartbeat"], {"pid": os.getpid(), "status": "failed", "updated_at": _utc_now()})
        _write_text(paths["exit_code"], "1\n")
        print(json.dumps(blocker, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
