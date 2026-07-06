"""Online action-generation bridge inventory and bounded diagnostic.

This module separates valid online/model-head actions from future-HDF5
candidate replay. It can run a tiny matched-init native SmolVLA online rollout
only under a task-local gate. ActionMap/TCA method variants are not rolled out
unless they expose a non-leaking action that maps explicitly to LIBERO 7D.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.datasets.libero_fixed_prior_rollout_diagnostic import _action_stats
from tca_map.datasets.libero_zero_reward_rollout_diagnosis import (
    _best_object_key,
    _distance,
    _extract_eef,
    _extract_pos,
)
from tca_map.heads.actionmap_head import ActionMapHead
from tca_map.heads.tca_map_head import TCAMapHead
from tca_map.smolvla.interface_adapters import (
    ACTION_STRATEGY_GRIPPER_ZERO_HOLD,
    adapt_policy_action_to_env_action,
)
from tca_map.smolvla.libero_learned_policy_rollout import (
    CAMERA_ALIAS_STRATEGY_CURRENT,
    STATE_ADAPTER_STRATEGY_EEF_POS_QUAT_FIRST3,
    _build_batch,
    _ensure_paths,
)
from tca_map.smolvla.load_only_smoke import (
    _external_tokenizer_files,
    _find_files,
    _nvidia_smi,
    _read_tokenizer_dependency,
    _rss_mb,
    _runtime_dependencies,
)
from tca_map.smolvla.single_sample_interface_smoke import _load_policy

SCHEMA_VERSION = "2026-07-06.online_action_generation_bridge.v1"
TASK_GATE = "ALLOW_ONLINE_ACTION_BRIDGE_ROLLOUT"
FORBIDDEN_GATES = (
    "ALLOW_DOWNLOADS",
    "ALLOW_GPU_TRAINING",
    "ALLOW_TINY_TRAINING",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_ROLLOUT",
    "ALLOW_ROLLOUTS",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_FIXED_PRIOR_ROLLOUT_DIAGNOSTIC",
    "ALLOW_ACTION_SOURCE_AUDIT_ROLLOUT",
)
MAX_STEPS = 25
MAX_RUNTIME_SECONDS = 1800


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _as_path(value: str | Path) -> Path:
    text = str(value)
    if os.name != "nt":
        match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
        if match:
            drive = match.group(1).lower()
            rest = match.group(2).replace("\\", "/")
            return Path(f"/mnt/{drive}/{rest}")
    return Path(text)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _safe_l2(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return None
    width = min(a.size, b.size)
    return round(float(np.linalg.norm(a[:width] - b[:width])), 9)


def _match_stats(actions: np.ndarray, reference: np.ndarray, *, near_tol: float = 1e-6) -> dict[str, Any]:
    steps = min(actions.shape[0], reference.shape[0])
    width = min(actions.shape[1], reference.shape[1])
    if steps == 0 or width == 0:
        return {
            "steps_compared": 0,
            "exact_match_rate": 0.0,
            "near_match_rate": 0.0,
            "mean_l2": None,
            "max_l2": None,
            "near_tol": near_tol,
        }
    diff = actions[:steps, :width] - reference[:steps, :width]
    row_l2 = np.linalg.norm(diff, axis=1)
    exact = np.all(actions[:steps, :width] == reference[:steps, :width], axis=1)
    near = row_l2 <= near_tol
    return {
        "steps_compared": int(steps),
        "exact_match_rate": round(float(np.mean(exact)), 9),
        "near_match_rate": round(float(np.mean(near)), 9),
        "mean_l2": round(float(np.mean(row_l2)), 9),
        "max_l2": round(float(np.max(row_l2)), 9),
        "near_tol": near_tol,
    }


def _adapter_probe(action: list[float] | np.ndarray, env_dim: int = 7) -> dict[str, Any]:
    try:
        result = adapt_policy_action_to_env_action(action, env_dim, strategy=ACTION_STRATEGY_GRIPPER_ZERO_HOLD)
        return {"ok": True, "metadata": result.metadata, "error": None, "adapted_dim": len(result.values)}
    except Exception as exc:  # noqa: BLE001 - exact diagnostic.
        return {"ok": False, "metadata": None, "error": str(exc), "adapted_dim": None}


def _source(
    name: str,
    *,
    online: bool,
    future_hdf5: bool,
    closed_loop_valid: bool,
    action_dim: str,
    maps_to_7d: bool,
    heavy_import: bool,
    gpu: bool,
    notes: str,
    adapter_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source_name": name,
        "online_generated_from_current_observation": online,
        "depends_on_future_hdf5_action": future_hdf5,
        "valid_for_closed_loop_rollout_claim": closed_loop_valid,
        "action_dimension_produced": action_dim,
        "can_be_mapped_to_libero_7d_without_silent_padding": maps_to_7d,
        "requires_heavy_vla_import": heavy_import,
        "requires_gpu": gpu,
        "adapter_probe": adapter_probe,
        "notes": notes,
    }


def build_online_action_source_inventory() -> dict[str, Any]:
    hidden = [0.1, -0.2, 0.3, 0.4]
    actionmap = ActionMapHead(grid_size=8).expected_action(ActionMapHead(grid_size=8).predict_heatmap(hidden))
    tca = TCAMapHead(grid_size=8).predict(hidden, ["moka pot", "black bowl"])["action"]
    actionmap_probe = _adapter_probe(actionmap)
    tca_probe = _adapter_probe(tca)
    native_probe = _adapter_probe([0.0] * 6)

    sources = [
        _source(
            "native_smolvla_policy_output",
            online=True,
            future_hdf5=False,
            closed_loop_valid=bool(native_probe["ok"]),
            action_dim="6D_policy_delta_pose",
            maps_to_7d=bool(native_probe["ok"]),
            heavy_import=True,
            gpu=False,
            adapter_probe=native_probe,
            notes="Native SmolVLA can produce an online 6D policy action that maps to 7D only through the explicit gripper strategy; this is a native baseline, not ActionMap/TCA.",
        ),
        _source(
            "existing_actionmap_head_output",
            online=True,
            future_hdf5=False,
            closed_loop_valid=False,
            action_dim=f"{len(actionmap)}D",
            maps_to_7d=bool(actionmap_probe["ok"]),
            heavy_import=False,
            gpu=False,
            adapter_probe=actionmap_probe,
            notes="Current smoke ActionMapHead returns xyz plus one scalar only. The explicit adapter refuses 4D->7D, so this cannot support a rollout claim.",
        ),
        _source(
            "existing_tca_map_head_output",
            online=True,
            future_hdf5=False,
            closed_loop_valid=False,
            action_dim=f"{len(tca)}D",
            maps_to_7d=bool(tca_probe["ok"]),
            heavy_import=False,
            gpu=False,
            adapter_probe=tca_probe,
            notes="Current TCAMapHead wraps the same 4D ActionMap expected_action path, so no valid 7D online TCA rollout source exists yet.",
        ),
        _source(
            "trained_numpy_offline_diagnostic_head",
            online=False,
            future_hdf5=False,
            closed_loop_valid=False,
            action_dim="4D_offline_proxy_prefix",
            maps_to_7d=False,
            heavy_import=False,
            gpu=False,
            notes="Bounded offline NumPy heads train/evaluate on cached HDF5-derived proxy records with ACTION_PREFIX_DIM=4; they are not current-observation rollout policies.",
        ),
        _source(
            "cached_hidden_tokens_or_features",
            online=False,
            future_hdf5=False,
            closed_loop_valid=False,
            action_dim="features_only_no_action",
            maps_to_7d=False,
            heavy_import=False,
            gpu=False,
            notes="Cached features can drive offline proxy heads, but there is no implemented current-simulator-observation feature extractor to produce deployable ActionMap/TCA actions.",
        ),
        _source(
            "hdf5_action_derived_candidates",
            online=False,
            future_hdf5=True,
            closed_loop_valid=False,
            action_dim="7D_hdf5_action_sequence",
            maps_to_7d=True,
            heavy_import=False,
            gpu=False,
            notes="HDF5 actions are valid as expert replay, labels, and action-distribution references only; they are invalid as method rollout actions.",
        ),
    ]
    valid_method_sources = [
        item["source_name"]
        for item in sources
        if item["valid_for_closed_loop_rollout_claim"] and item["source_name"] != "native_smolvla_policy_output"
    ]
    return {
        "sources": sources,
        "valid_native_online_source_found": any(
            item["source_name"] == "native_smolvla_policy_output" and item["valid_for_closed_loop_rollout_claim"]
            for item in sources
        ),
        "valid_actionmap_tca_online_source_found": bool(valid_method_sources),
        "valid_actionmap_tca_sources": valid_method_sources,
        "method_rollout_blocker": None
        if valid_method_sources
        else "current project supports offline candidate selection and expert replay bridge, but not closed-loop online ActionMap/TCA 7D action generation yet",
    }


def _load_first_case(manifest_path: Path, max_steps: int) -> dict[str, Any]:
    import h5py  # type: ignore

    manifest = _load_json(manifest_path)
    pairs = manifest.get("counterfactual_pairs") or []
    if not pairs:
        raise ValueError("counterfactual split manifest has no pairs")
    pair = pairs[0]
    positive_path = _as_path(pair["positive_demo_file"])
    with h5py.File(positive_path, "r") as handle:
        demo_name = sorted(handle["data"].keys())[0]
        demo = handle["data"][demo_name]
        actions = np.asarray(demo["actions"][:max_steps], dtype=np.float64)
        init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"expected HDF5 expert actions [T, 7], got {list(actions.shape)}")
    if actions.shape[0] < max_steps:
        raise ValueError(f"demo has only {actions.shape[0]} actions, requested {max_steps}")
    return {
        "pair_id": pair.get("pair_id"),
        "suite": pair.get("suite") or "libero_10",
        "task_id": pair["positive_task_id"],
        "instruction": pair["positive_instruction"],
        "counterfactual_instruction": pair.get("counterfactual_instruction"),
        "positive_demo_path": str(positive_path),
        "demo_name": demo_name,
        "init_state": init_state,
        "expert_actions": actions[:max_steps],
        "bddl_file": None,
    }


def _bddl_path(libero_root: Path, suite: str, task_id: str) -> Path:
    direct = libero_root / "libero" / "libero" / "bddl_files" / suite / f"{task_id}.bddl"
    if direct.exists():
        return direct
    matches = sorted((libero_root / "libero" / "libero" / "bddl_files" / suite).glob("*.bddl"))
    for path in matches:
        if path.stem == task_id:
            return path
    raise FileNotFoundError(f"could not find BDDL for {suite}/{task_id}")


def _target_metrics(start_obs: dict[str, Any], final_obs: dict[str, Any], instruction: str, counterfactual_instruction: str | None) -> dict[str, Any]:
    target_audit = _best_object_key(start_obs, instruction)
    target_key = target_audit.get("best_key")
    start_eef = _extract_eef(start_obs)
    final_eef = _extract_eef(final_obs)
    target_start = _extract_pos(start_obs, target_key) if target_key else None
    target_final = _extract_pos(final_obs, target_key) if target_key else None
    target_score = None
    if start_eef is not None and final_eef is not None and target_start is not None and target_final is not None:
        start_dist = _distance(start_eef, target_start)
        final_dist = _distance(final_eef, target_final)
        target_score = None if start_dist is None or final_dist is None else round(float(start_dist - final_dist), 6)
    wrong = {"available": False, "distance_change": None, "target_key_audit": None}
    if counterfactual_instruction:
        wrong_audit = _best_object_key(start_obs, counterfactual_instruction)
        wrong_key = wrong_audit.get("best_key")
        wrong_start = _extract_pos(start_obs, wrong_key) if wrong_key else None
        wrong_final = _extract_pos(final_obs, wrong_key) if wrong_key else None
        if start_eef is not None and final_eef is not None and wrong_start is not None and wrong_final is not None:
            start_wrong = _distance(start_eef, wrong_start)
            final_wrong = _distance(final_eef, wrong_final)
            wrong = {
                "available": start_wrong is not None and final_wrong is not None,
                "distance_change": None
                if start_wrong is None or final_wrong is None
                else round(float(start_wrong - final_wrong), 6),
                "target_key_audit": wrong_audit,
            }
        else:
            wrong = {"available": False, "distance_change": None, "target_key_audit": wrong_audit}
    return {
        "target_key_audit": target_audit,
        "target_directed_movement_score": target_score,
        "wrong_target_movement": wrong,
        "eef_displacement_l2": _safe_l2(start_eef, final_eef),
        "target_object_displacement_l2": _safe_l2(target_start, target_final),
    }


def _compact_error(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _run_variant(
    *,
    env_cls: Any,
    bddl_file: Path,
    camera_size: int,
    init_state: np.ndarray,
    expert_actions: np.ndarray,
    variant: str,
    instruction: str,
    counterfactual_instruction: str | None,
    policy: Any | None = None,
    config: Any | None = None,
    tokenizer_root: Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "variant": variant,
        "evidence_type": "valid_closed_loop_online_rollout" if variant == "native_smolvla_online_policy" else "control_only",
        "action_provenance": [],
        "steps_performed": 0,
        "reward_sum": 0.0,
        "final_success": False,
        "done_seen": False,
        "first_done_index": None,
        "first_positive_reward_index": None,
        "first_success_index": None,
        "env_created": False,
        "reset_ok": False,
        "set_init_state_ok": False,
        "error": None,
    }
    actions_used: list[list[float]] = []
    env = None
    final_obs: dict[str, Any] | None = None
    start_obs: dict[str, Any] | None = None
    try:
        env = env_cls(bddl_file_name=str(bddl_file), camera_heights=camera_size, camera_widths=camera_size)
        summary["env_created"] = True
        env.seed(0)
        obs = env.reset()
        summary["reset_ok"] = True
        obs = env.set_init_state(init_state)
        summary["set_init_state_ok"] = True
        start_obs = obs
        action_dim = int(getattr(env, "action_dim", 7) or 7)
        if policy is not None:
            policy.reset()
        import torch

        for step in range(int(expert_actions.shape[0])):
            if variant == "zero_action_exact_init":
                env_action = [0.0] * action_dim
                provenance = "programmatic_zero_action"
                policy_shape = None
            elif variant == "hdf5_expert_replay_exact_init":
                env_action = [float(value) for value in expert_actions[step].tolist()]
                provenance = "hdf5_expert_action_upper_bound_not_method"
                policy_shape = None
            elif variant == "native_smolvla_online_policy":
                if policy is None or config is None or tokenizer_root is None:
                    raise ValueError("native policy variant requires loaded policy/config/tokenizer_root")
                batch, batch_metadata = _build_batch(
                    config,
                    tokenizer_root,
                    obs,
                    instruction,
                    device,
                    CAMERA_ALIAS_STRATEGY_CURRENT,
                    STATE_ADAPTER_STRATEGY_EEF_POS_QUAT_FIRST3,
                )
                noise = torch.zeros((1, config.chunk_size, config.max_action_dim), dtype=torch.float32, device=device)
                with torch.inference_mode():
                    policy_action = policy.select_action(batch, noise=noise)
                adapter = adapt_policy_action_to_env_action(
                    policy_action,
                    action_dim,
                    strategy=ACTION_STRATEGY_GRIPPER_ZERO_HOLD,
                    action_scale=1.0,
                )
                env_action = [float(value) for value in adapter.values]
                provenance = "online_smolvla_model_head_decoded_from_current_observation_instruction"
                policy_shape = list(policy_action.detach().cpu().shape)
                summary["last_batch_metadata"] = batch_metadata
                summary["last_action_adapter_metadata"] = adapter.metadata
            else:
                raise ValueError(f"unsupported variant: {variant}")

            obs, reward, done, _info = env.step(env_action)
            actions_used.append(env_action)
            expert_l2 = _safe_l2(env_action, expert_actions[step])
            summary["action_provenance"].append(
                {
                    "step": step,
                    "source": provenance,
                    "uses_future_hdf5_action": variant == "hdf5_expert_replay_exact_init",
                    "online_generated_from_current_observation": variant == "native_smolvla_online_policy",
                    "model_head_decoded_action": variant == "native_smolvla_online_policy",
                    "policy_action_shape": policy_shape,
                    "env_action_dim": len(env_action),
                    "l2_to_hdf5_expert_same_timestep": expert_l2,
                }
            )
            summary["steps_performed"] += 1
            summary["reward_sum"] += float(reward)
            if reward and summary["first_positive_reward_index"] is None:
                summary["first_positive_reward_index"] = step
            if done and summary["first_done_index"] is None:
                summary["first_done_index"] = step
            summary["done_seen"] = bool(summary["done_seen"] or done)
            try:
                success = bool(env.check_success())
                if success and summary["first_success_index"] is None:
                    summary["first_success_index"] = step
                summary["final_success"] = success
            except Exception:
                pass
            final_obs = obs
        if final_obs is not None and start_obs is not None:
            summary.update(_target_metrics(start_obs, final_obs, instruction, counterfactual_instruction))
    except Exception as exc:  # noqa: BLE001 - diagnostic report.
        summary["error"] = _compact_error(exc)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    action_array = np.asarray(actions_used, dtype=np.float64) if actions_used else np.zeros((0, 7), dtype=np.float64)
    summary["action_shape"] = list(action_array.shape)
    summary["action_stats"] = _action_stats(action_array) if action_array.size else None
    summary["expert_match"] = _match_stats(action_array, expert_actions[: action_array.shape[0]])
    return summary


def run_online_bridge(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    inventory = build_online_action_source_inventory()
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    task_gate_set = _env_flag(TASK_GATE)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "online_action_generation_bridge_passed": False,
        "decision": "stop",
        "policy": {
            "task_local_gate_required": f"{TASK_GATE}=1",
            "task_local_gate_set": task_gate_set,
            "downloads_performed": False,
            "gpu_jobs_performed": False,
            "training_performed": False,
            "lora_training_performed": False,
            "loss_computed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "model_inference_performed": False,
            "rollout_happened": False,
            "diagnostic_rollouts_performed": False,
            "benchmark_rollouts_performed": False,
            "openvla_oft_executed": False,
            "paper_grade_claims_made": False,
            "forbidden_gates_set": forbidden,
        },
        "inventory": inventory,
        "risk_assessment": {
            "task": "online action-generation bridge for matched-init diagnostic",
            "expected_runtime_minutes": "<=30",
            "expected_steps": args.max_steps,
            "device": args.device,
            "simulator_will_run": bool(task_gate_set and inventory["valid_native_online_source_found"]),
            "benchmark_rollout_will_run": False,
            "decision": "proceed" if task_gate_set and not forbidden else "stop",
        },
        "case": None,
        "rollout_results": [],
        "result": {"passed": False, "blocked": True, "blocked_reason": None, "elapsed_sec": None},
        "recommended_next_step": None,
    }

    def stop(reason: str) -> dict[str, Any]:
        report["result"].update({"passed": False, "blocked": True, "blocked_reason": reason, "elapsed_sec": round(time.monotonic() - started, 3)})
        report["recommended_next_step"] = reason
        return report

    if args.max_steps < 1 or args.max_steps > MAX_STEPS:
        return stop(f"max_steps must be between 1 and {MAX_STEPS}")
    if forbidden:
        return stop("Forbidden gate(s) set: " + ", ".join(forbidden))
    if not task_gate_set:
        report["decision"] = "inventory_only_blocked_before_rollout"
        return stop(
            "online source inventory completed; rollout not run because task-local gate is absent. "
            + inventory["method_rollout_blocker"]
        )
    if not inventory["valid_native_online_source_found"]:
        return stop("no valid online action source exists")

    smolvla_ckpt = Path(args.smolvla_ckpt)
    checkpoint_root = Path(args.checkpoint_root)
    hf_home = Path(args.hf_home)
    libero_root = Path(args.libero_root)
    robosuite_root = Path(args.robosuite_root)
    deps = _runtime_dependencies()
    config_files = _find_files(smolvla_ckpt, ["config.json"])
    weight_files = _find_files(smolvla_ckpt, ["model.safetensors", "pytorch_model.bin"], ["*.safetensors", "*.bin"])
    external = _external_tokenizer_files(_read_tokenizer_dependency(smolvla_ckpt), [hf_home, checkpoint_root])
    if not (config_files and weight_files and external.get("found")):
        return stop("SmolVLA local checkpoint/tokenizer/weights are incomplete for native online source")
    if not all(deps.values()):
        missing = [name for name, present in deps.items() if not present]
        return stop("Missing runtime dependencies: " + ", ".join(missing))

    try:
        _ensure_paths(libero_root, robosuite_root)
        from libero.libero.envs import OffScreenRenderEnv

        case = _load_first_case(Path(args.manifest), args.max_steps)
        bddl_file = _bddl_path(libero_root, case["suite"], case["task_id"])
        case["bddl_file"] = str(bddl_file)
        report["case"] = {
            key: value
            for key, value in case.items()
            if key not in {"init_state", "expert_actions"}
        }
        report["case"]["expert_action_shape"] = list(case["expert_actions"].shape)
        report["case"]["max_steps"] = int(args.max_steps)

        report["policy"]["heavy_model_imports_performed"] = True
        policy, config = _load_policy(smolvla_ckpt, hf_home, external, args.device)
        report["policy"]["model_load_performed"] = True
        tokenizer_root = Path(external["root"])

        variants = ["zero_action_exact_init", "hdf5_expert_replay_exact_init", "native_smolvla_online_policy"]
        results = []
        for variant in variants:
            results.append(
                _run_variant(
                    env_cls=OffScreenRenderEnv,
                    bddl_file=bddl_file,
                    camera_size=args.camera_size,
                    init_state=case["init_state"],
                    expert_actions=case["expert_actions"],
                    variant=variant,
                    instruction=case["instruction"],
                    counterfactual_instruction=case.get("counterfactual_instruction"),
                    policy=policy if variant == "native_smolvla_online_policy" else None,
                    config=config if variant == "native_smolvla_online_policy" else None,
                    tokenizer_root=tokenizer_root if variant == "native_smolvla_online_policy" else None,
                    device=args.device,
                )
            )
        report["rollout_results"] = results
        report["policy"]["rollout_happened"] = True
        report["policy"]["diagnostic_rollouts_performed"] = True
        report["policy"]["model_inference_performed"] = any(
            any(step.get("model_head_decoded_action") for step in item.get("action_provenance", [])) for item in results
        )
        native = next((item for item in results if item["variant"] == "native_smolvla_online_policy"), {})
        report["decision"] = "native_online_rollout_completed_actionmap_tca_blocked"
        report["online_action_generation_bridge_passed"] = bool(native and native.get("error") is None)
        report["result"] = {
            "passed": bool(native and native.get("error") is None),
            "blocked": False,
            "blocked_reason": None,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "valid_closed_loop_online_rollout_variants": ["native_smolvla_online_policy"] if native and native.get("error") is None else [],
            "valid_actionmap_tca_rollout_variants": inventory["valid_actionmap_tca_sources"],
            "fixed_prior_tca_valid_rollout_support": False,
            "blocker_classification": "no_nonleaking_online_actionmap_tca_7d_head",
        }
        report["recommended_next_step"] = (
            "Implement/train a minimal non-leaking 7D online diagnostic head for ActionMap/TCA, or package current evidence with an honest offline + bridge caveat."
        )
    except Exception as exc:  # noqa: BLE001 - exact diagnostic.
        report["result"] = {
            "passed": False,
            "blocked": True,
            "blocked_reason": f"{type(exc).__name__}: {exc}",
            "error": _compact_error(exc),
            "elapsed_sec": round(time.monotonic() - started, 3),
        }
        report["recommended_next_step"] = "Fix the online bridge blocker before any rollout-level method claim."

    if float(report["result"].get("elapsed_sec") or 0.0) > MAX_RUNTIME_SECONDS:
        report["result"]["passed"] = False
        report["result"]["blocked"] = True
        report["result"]["blocked_reason"] = "runtime exceeded online bridge budget"
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    inventory = report.get("inventory") or {}
    result = report.get("result") or {}
    lines = [
        "# Online Action-Generation Bridge Report",
        "",
        "This is bounded diagnostic evidence only. It is not benchmark success, SOTA evidence, or paper-grade evidence.",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- passed: `{result.get('passed')}`",
        f"- rollout happened: `{(report.get('policy') or {}).get('rollout_happened')}`",
        f"- valid native online source found: `{inventory.get('valid_native_online_source_found')}`",
        f"- valid ActionMap/TCA online source found: `{inventory.get('valid_actionmap_tca_online_source_found')}`",
        f"- fixed-prior TCA valid rollout support: `{result.get('fixed_prior_tca_valid_rollout_support')}`",
        f"- blocker: `{result.get('blocker_classification') or result.get('blocked_reason')}`",
        "",
        "## Source Inventory",
        "",
    ]
    for source in inventory.get("sources", []):
        lines.append(
            f"- `{source['source_name']}`: dim `{source['action_dimension_produced']}`, "
            f"online `{source['online_generated_from_current_observation']}`, "
            f"future-HDF5 `{source['depends_on_future_hdf5_action']}`, "
            f"valid `{source['valid_for_closed_loop_rollout_claim']}`"
        )
    lines.extend(["", f"Recommended next step: {report.get('recommended_next_step')}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_scaled_report.json")
    parser.add_argument("--report-json", default="reports/online_action_generation_bridge_report.json")
    parser.add_argument("--report-md", default="reports/online_action_generation_bridge_report.md")
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--checkpoint-root", default="C:/assets/checkpoints")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--libero-root", default="C:/assets/repos/LIBERO")
    parser.add_argument("--robosuite-root", default="C:/assets/repos/robosuite")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    args = parser.parse_args(argv)

    report = run_online_bridge(args)
    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, report_md)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"]["passed"] or report["decision"] == "inventory_only_blocked_before_rollout" else 8


if __name__ == "__main__":
    sys.exit(main())
