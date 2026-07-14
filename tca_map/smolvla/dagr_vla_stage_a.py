"""DAGR-VLA Stage A matched-manifest freezer."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import traceback
from typing import Any, Mapping

import numpy as np

from tca_map.smolvla.official_closed_loop_scaleup import (
    _action_queue_len,
    _cuda_memory,
    _episode_base_record,
    _extract_single_env,
    _make_env_cfg,
    _round,
    _rss_mb,
    _set_runtime_env,
    _successes_from_info,
)
from tca_map.smolvla.official_wsl_libero_rollout import PolicySpec, _dummy_observation, _load_policy_and_processors


DATE_KST = "2026-07-14"
BRANCH = "codex/autonomous-until-paper-governance-v2"
METHOD = "DAGR-VLA"
CONFIG_ID = "dagr_a020_route_mlp"
PROPOSAL_HASH = "BDE0EC67ACE8EC457CE6495D723EE476064F3D80946151326B11F0B5A1AFEF89"
STAGE_A_RESET_SEEDS = [20261205, 20261206]
STAGE_B_RESET_SEEDS = [20261207, 20261208]
STAGE_A_TASK_COUNT = 5
STAGE_B_TASK_COUNT = 20
STAGE_A_EPISODES_PER_POLICY = STAGE_A_TASK_COUNT * len(STAGE_A_RESET_SEEDS)
STAGE_B_EPISODES_PER_POLICY = STAGE_B_TASK_COUNT * len(STAGE_B_RESET_SEEDS)
DAGR_FULL_POLICY = "dagr_full"
STAGE_A_POLICY_ORDER = [
    "frozen_smolvla",
    "dam_static_component_proxy",
    "dagr_full",
    "dagr_no_dynamic_route_ablation",
    "gripper_transition_heuristic",
]
LEARNED_DAGR_POLICIES = {
    "dam_static_component_proxy",
    "dagr_full",
    "dagr_no_dynamic_route_ablation",
}
POLICY_ROLES = {
    "frozen_smolvla": "unmodified_backbone",
    "dam_static_component_proxy": "closest_external_prior_proxy_faithful_local_proxy_not_official_dam_vla_reproduction",
    "dagr_full": "ours",
    "dagr_no_dynamic_route_ablation": "key_ablation",
    "gripper_transition_heuristic": "strongest_simple_reviewer_killer",
}
FINAL_DECISIONS = {
    "DAGR_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT",
    "DAGR_STAGE_A_PREFLIGHT_PASS_READY_FOR_OFFICIAL_ROLLOUT",
    "DAGR_STAGE_A_PREFLIGHT_BLOCKED_REPAIR_LOADING_OR_MAPPING",
    "DAGR_STAGE_A_MEASUREMENT_INVALID_REPAIR_REQUIRED",
    "DAGR_STAGE_A_CATASTROPHIC_KILL_ZERO_VS_STRONG_BASELINE",
    "DAGR_STAGE_A_CATASTROPHIC_KILL_CLEARLY_WORSE_THAN_BASELINE_OR_ABLATION",
    "DAGR_STAGE_A_POSITIVE_TO_STAGE_B_REQUIRED",
    "DAGR_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED",
    "DAGR_STAGE_B_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT",
    "DAGR_STAGE_B_MEASUREMENT_INVALID_REPAIR_REQUIRED",
    "DAGR_STAGE_B_PROTOTYPE_GO",
    "DAGR_STAGE_B_KILL_BASE_NOT_IMPROVED",
    "DAGR_STAGE_B_KILL_CLOSEST_PRIOR_EXPLAINS_METHOD",
    "DAGR_STAGE_B_KILL_KEY_COMPONENT_NOT_USEFUL",
    "DAGR_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD",
    "DAGR_STAGE_B_USEFUL_IMPROVEMENT_EXCLUDED",
    "DAGR_STAGE_B_UNRESOLVED_EXPANSION_REQUIRED",
}


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _sha256_payload(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(blob).hexdigest().upper()


def _relative(path_text: str) -> str:
    return path_text.replace("\\", "/")


def _repo_relative(path_text: str) -> Path:
    return Path(_relative(path_text))


def _wsl_repo_path(wsl_repo_root: str, relative_path: str) -> str:
    return f"{str(wsl_repo_root).rstrip('/')}/{_relative(relative_path).lstrip('/')}"


def _select_stage_a_tasks(task_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = list(task_manifest.get("tasks") or [])
    if len(tasks) < STAGE_A_TASK_COUNT:
        raise ValueError(f"official task manifest has only {len(tasks)} tasks")
    indices = [(index * len(tasks)) // STAGE_A_TASK_COUNT for index in range(STAGE_A_TASK_COUNT)]
    selected = []
    for stage_index, source_index in enumerate(indices):
        task = dict(tasks[source_index])
        task["stage_a_task_index"] = int(stage_index)
        task["source_official_task_manifest_index"] = int(source_index)
        task["stage_a_selection_rule"] = f"floor(k * {len(tasks)} / {STAGE_A_TASK_COUNT})"
        selected.append(task)
    return selected


def _select_stage_b_tasks(task_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = list(task_manifest.get("tasks") or [])
    if len(tasks) != STAGE_B_TASK_COUNT:
        raise ValueError(f"DAGR Stage B expects the frozen official 20-task manifest, found {len(tasks)} tasks")
    selected = []
    for stage_index, task_item in enumerate(tasks):
        task = dict(task_item)
        task["stage_b_task_index"] = int(stage_index)
        task["source_official_task_manifest_index"] = int(stage_index)
        task["stage_b_selection_rule"] = "use all tasks from the frozen official 20-task manifest"
        selected.append(task)
    return selected


def _policy_records(args: argparse.Namespace, checkpoint_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if checkpoint_manifest.get("final_decision") != "DAGR_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY":
        raise ValueError("DAGR policy checkpoint manifest is not Stage-A ready")
    if not bool(checkpoint_manifest.get("stage_a_allowed")):
        raise ValueError("DAGR policy checkpoint manifest does not allow Stage A")
    variants = {str(item["variant"]): dict(item) for item in checkpoint_manifest.get("variant_results") or []}
    heuristic = dict(checkpoint_manifest.get("heuristic") or {})
    records = []
    for policy in STAGE_A_POLICY_ORDER:
        if policy == "frozen_smolvla":
            records.append(
                {
                    "policy": policy,
                    "role": POLICY_ROLES[policy],
                    "checkpoint_path": None,
                    "wsl_checkpoint_path": None,
                    "disk_reload": None,
                    "sha256_manifest": None,
                }
            )
            continue
        source = heuristic if policy == "gripper_transition_heuristic" else variants.get(policy)
        if not source:
            raise ValueError(f"checkpoint manifest missing policy identity {policy}")
        checkpoint_path = _relative(str(source["checkpoint_path"]))
        records.append(
            {
                "policy": policy,
                "role": POLICY_ROLES[policy],
                "checkpoint_path": checkpoint_path,
                "wsl_checkpoint_path": _wsl_repo_path(str(args.wsl_repo_root), checkpoint_path),
                "disk_reload": bool(source.get("disk_reload")),
                "delta_l2_p95": ((source.get("validation") or {}).get("delta_l2_p95")),
                "action_validity": ((source.get("validation") or {}).get("action_validity")),
                "sha256_manifest": source.get("sha256_manifest"),
                "proxy_or_reproduction_label": (
                    "faithful_transparent_local_proxy_not_official_dam_vla_reproduction"
                    if policy == "dam_static_component_proxy"
                    else None
                ),
            }
        )
    return records


def validate_manifest(payload: Mapping[str, Any]) -> None:
    episodes = list(payload.get("episodes") or [])
    keys = [(row["policy"], row["suite"], int(row["task_id"]), int(row["reset_seed"])) for row in episodes]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate DAGR Stage A evaluation keys")
    policy_order = list(payload.get("policy_order") or [])
    pair_sets = {}
    for policy in policy_order:
        pair_sets[policy] = {
            (row["suite"], int(row["task_id"]), int(row["reset_seed"]))
            for row in episodes
            if row["policy"] == policy
        }
    if len({tuple(sorted(values)) for values in pair_sets.values()}) != 1:
        raise ValueError("DAGR Stage A task/reset pairs differ across policies")
    if int(payload.get("planned_episode_count", -1)) != len(policy_order) * int(payload["task_balanced_allocation"]["paired_cases_per_policy"]):
        raise ValueError("DAGR Stage A planned episode count mismatch")


def validate_stage_a_manifest(manifest: Mapping[str, Any]) -> None:
    policies = [str(item["policy"]) for item in manifest.get("policies") or []]
    if policies != STAGE_A_POLICY_ORDER:
        raise ValueError(f"DAGR Stage A policies are not frozen order: {policies}")
    if int(manifest.get("stage_a_pair_count_per_policy", -1)) != STAGE_A_EPISODES_PER_POLICY:
        raise ValueError("DAGR Stage A must contain exactly 10 paired cases per policy")
    if int(manifest.get("planned_episode_count", -1)) != len(STAGE_A_POLICY_ORDER) * STAGE_A_EPISODES_PER_POLICY:
        raise ValueError("DAGR Stage A planned episode count must be 50")
    if list(manifest.get("stage_a_reset_seeds") or []) != STAGE_A_RESET_SEEDS:
        raise ValueError("DAGR Stage A reset identities changed")
    if bool(manifest.get("closed_loop_experiment_happened")):
        raise ValueError("DAGR Stage A manifest must be frozen before rollout")
    if bool(manifest.get("confirmatory_test_tuning_happened")):
        raise ValueError("DAGR Stage A manifest reports confirmatory-test tuning")
    validate_manifest(manifest)
    identity_overlap = dict(manifest.get("identity_overlap_verification") or {})
    if int(identity_overlap.get("duplicate_evaluation_keys", -1)) != 0:
        raise ValueError("DAGR Stage A manifest reports duplicate evaluation keys")
    if not bool(identity_overlap.get("identical_task_reset_pairs_across_policies")):
        raise ValueError("DAGR Stage A manifest does not certify identical task/reset pairs")


def validate_stage_b_manifest(manifest: Mapping[str, Any]) -> None:
    policies = [str(item["policy"]) for item in manifest.get("policies") or []]
    if policies != STAGE_A_POLICY_ORDER:
        raise ValueError(f"DAGR Stage B policies are not frozen order: {policies}")
    if int(manifest.get("stage_b_pair_count_per_policy", -1)) != STAGE_B_EPISODES_PER_POLICY:
        raise ValueError("DAGR Stage B must contain exactly 40 paired cases per policy")
    if int(manifest.get("planned_episode_count", -1)) != len(STAGE_A_POLICY_ORDER) * STAGE_B_EPISODES_PER_POLICY:
        raise ValueError("DAGR Stage B planned episode count must be 200")
    if list(manifest.get("stage_b_reset_seeds") or []) != STAGE_B_RESET_SEEDS:
        raise ValueError("DAGR Stage B reset identities changed")
    if len(manifest.get("tasks") or []) != STAGE_B_TASK_COUNT:
        raise ValueError("DAGR Stage B must use all 20 official tasks")
    if bool(manifest.get("closed_loop_experiment_happened")):
        raise ValueError("DAGR Stage B manifest must be frozen before rollout")
    if bool(manifest.get("confirmatory_test_tuning_happened")):
        raise ValueError("DAGR Stage B manifest reports confirmatory-test tuning")
    validate_manifest(manifest)
    identity_overlap = dict(manifest.get("identity_overlap_verification") or {})
    if int(identity_overlap.get("duplicate_evaluation_keys", -1)) != 0:
        raise ValueError("DAGR Stage B manifest reports duplicate evaluation keys")
    if int(identity_overlap.get("overlap_with_stage_a_reset_seeds", -1)) != 0:
        raise ValueError("DAGR Stage B reset seeds overlap Stage A")
    if not bool(identity_overlap.get("identical_task_reset_pairs_across_policies")):
        raise ValueError("DAGR Stage B manifest does not certify identical task/reset pairs")


def _manifest_task_map(manifest: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(task["suite"]), int(task["task_id"])): dict(task) for task in manifest.get("tasks") or []}


def _completed_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (str(row["policy"]), str(row["suite"]), int(row["task_id"]), int(row["reset_seed"]))


def _planned_lookup(manifest: Mapping[str, Any]) -> dict[str, int]:
    return {str(item["episode_id"]): int(item["planned_episode_index"]) for item in manifest.get("episodes") or []}


def _load_partial(path: Path, rerun_stage: bool) -> list[dict[str, Any]]:
    if rerun_stage or not path.exists():
        return []
    return list((_read_json(path).get("episodes") or []))


def _task_index_map_from_artifact(path: Path) -> dict[str, int]:
    artifact = _read_json(path)
    mapping: dict[str, int] = {}
    for record in artifact.get("records") or []:
        task = str(record.get("task") or "").strip()
        if not task:
            continue
        task_index = int(record.get("task_index", -1))
        previous = mapping.get(task)
        if previous is not None and previous != task_index:
            raise ValueError(f"DAGR stable artifact has inconsistent task index for {task!r}: {previous} vs {task_index}")
        mapping[task] = task_index
    return mapping


def _task_index_for_task(task: Mapping[str, Any], task_index_map: Mapping[str, int]) -> int:
    instruction = str(task["instruction"]).strip()
    if instruction not in task_index_map:
        raise ValueError(f"DAGR task instruction missing from stable artifact task-index map: {instruction}")
    return int(task_index_map[instruction])


def _parameter_count_summary(module: Any) -> dict[str, int]:
    total = trainable = 0
    for param in module.parameters():
        count = int(param.numel())
        total += count
        if bool(getattr(param, "requires_grad", False)):
            trainable += count
    return {"total_parameter_count": total, "trainable_parameter_count": trainable}


def _first_parameter_device(module: Any) -> str | None:
    for param in module.parameters():
        return str(param.device)
    return None


class _DAGRPolicyHead:
    def __new__(cls, input_dim: int, route_dim: int, architecture: str, torch: Any) -> Any:
        class DAGRPolicyHead(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if architecture == "mlp":
                    self.trunk = torch.nn.Sequential(torch.nn.Linear(input_dim, 32), torch.nn.ReLU())
                    hidden_dim = 32
                else:
                    self.trunk = torch.nn.Identity()
                    hidden_dim = input_dim
                self.route = torch.nn.Linear(hidden_dim, route_dim) if route_dim else None
                self.residual = torch.nn.Linear(hidden_dim, 7)

            def forward(self, x: Any) -> tuple[Any | None, Any]:
                h = self.trunk(x)
                route_logits = self.route(h) if self.route is not None else None
                return route_logits, self.residual(h)

        return DAGRPolicyHead()


class DAGRActionAdapter:
    def __init__(self, policy_name: str, checkpoint_path: Path | None, *, device: str) -> None:
        import torch

        self.policy_name = policy_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.config: dict[str, Any] = {}
        self.heuristic_config: dict[str, Any] = {}
        self.previous_gripper: Any | None = None
        self.last_transform: dict[str, Any] = {}
        if checkpoint_path is None or policy_name == "frozen_smolvla":
            return
        if policy_name == "gripper_transition_heuristic":
            self.heuristic_config = _read_json(checkpoint_path / "heuristic_config.json")
            return
        self.config = _read_json(checkpoint_path / "policy_config.json")
        feature_count = int(self.config["feature_count"])
        route_dim = 3 if policy_name == "dagr_full" else 1 if policy_name == "dagr_no_dynamic_route_ablation" else 0
        model = _DAGRPolicyHead(feature_count, route_dim, str(self.config.get("route_architecture", "linear")), torch)
        payload = torch.load(checkpoint_path / "model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)
        model.eval()
        self.model = model

    def reset(self) -> None:
        self.previous_gripper = None
        self.last_transform = {}

    def _feature_tensor(self, action: Any, state: Any, phase: float, task_index: int) -> Any:
        import torch

        model_device = next(self.model.parameters()).device if self.model is not None else action.device
        mean = torch.as_tensor(self.config["feature_mean"], dtype=torch.float32, device=model_device).reshape(1, -1)
        scale = torch.as_tensor(self.config["feature_scale"], dtype=torch.float32, device=model_device).reshape(1, -1)
        task_count = int(mean.shape[1]) - 19
        if task_count <= 0:
            raise ValueError(f"DAGR checkpoint feature count is invalid: {int(mean.shape[1])}")
        if int(task_index) < 0 or int(task_index) >= task_count:
            raise ValueError(f"DAGR task index {task_index} outside checkpoint task count {task_count}")
        base = action.reshape(1, 7).to(device=model_device, dtype=torch.float32)
        state_tensor = state.reshape(1, 8).to(device=model_device, dtype=torch.float32)
        phase_tensor = torch.full((1, 1), float(phase), dtype=torch.float32, device=model_device)
        task = torch.zeros((1, task_count), dtype=torch.float32, device=model_device)
        task[0, int(task_index)] = 1.0
        norms = torch.stack(
            [
                torch.linalg.norm(base[:, 0:3], dim=1),
                torch.linalg.norm(base[:, 3:6], dim=1),
                torch.abs(base[:, 6]),
            ],
            dim=1,
        )
        features = torch.cat([base, state_tensor, phase_tensor, task, norms], dim=1)
        return (features - mean) / scale

    def _learned_action(self, action: Any, state: Any, phase: float, task_index: int) -> tuple[Any, dict[str, Any]]:
        import torch

        if self.model is None:
            raise ValueError(f"DAGR learned adapter {self.policy_name} is not loaded")
        x = self._feature_tensor(action, state, phase, task_index)
        with torch.inference_mode():
            route_logits, predicted_residual = self.model(x)
            alpha = float(self.config["residual_alpha"])
            if self.policy_name == "dagr_full":
                gates = torch.sigmoid(route_logits)
                delta = alpha * torch.cat(
                    [
                        predicted_residual[:, 0:3] * gates[:, 0:1],
                        predicted_residual[:, 3:6] * gates[:, 1:2],
                        predicted_residual[:, 6:7] * gates[:, 2:3],
                    ],
                    dim=1,
                )
            elif self.policy_name == "dagr_no_dynamic_route_ablation":
                gate = torch.sigmoid(route_logits)
                gates = gate.repeat(1, 3)
                delta = alpha * gate * predicted_residual
            else:
                weights = torch.as_tensor(self.config["static_weights"], dtype=predicted_residual.dtype, device=predicted_residual.device).reshape(1, 7)
                gates = torch.as_tensor(
                    [float(weights[0, 0]), float(weights[0, 3]), float(weights[0, 6])],
                    dtype=predicted_residual.dtype,
                    device=predicted_residual.device,
                ).reshape(1, 3)
                delta = alpha * weights * predicted_residual
        shaped = action.to(device=delta.device, dtype=delta.dtype) + delta
        delta_l2 = float(torch.linalg.norm(delta.reshape(1, 7), dim=1).detach().cpu().item())
        gate_values = [float(value) for value in gates.reshape(-1).detach().cpu().tolist()]
        residual_norm = float(torch.linalg.norm(predicted_residual.reshape(1, 7), dim=1).detach().cpu().item())
        changed = [index for index, value in enumerate(delta.reshape(-1).detach().cpu().tolist()) if abs(float(value)) > 1e-9]
        return shaped.to(device=action.device, dtype=action.dtype), {
            "transform": self.policy_name,
            "activated": bool(delta_l2 > 1e-9),
            "delta_l2": delta_l2,
            "gate_values": gate_values,
            "residual_norm": residual_norm,
            "dimensions_changed": changed,
        }

    def _heuristic_action(self, action: Any) -> tuple[Any, dict[str, Any]]:
        import torch

        alpha = float(self.heuristic_config.get("residual_alpha", 0.05))
        threshold = float(self.heuristic_config.get("gripper_material_threshold", 0.02))
        current = action[:, 6:7]
        previous = self.previous_gripper
        if previous is None:
            transition = torch.zeros_like(current, dtype=torch.bool)
            direction = torch.sign(current)
        else:
            transition = torch.sign(previous) != torch.sign(current)
            direction = torch.sign(current - previous)
            fallback = torch.sign(current)
            direction = torch.where(torch.abs(direction) > 0.0, direction, fallback)
        near_transition = torch.abs(current) <= threshold
        active = torch.logical_or(near_transition, transition)
        direction = torch.where(torch.abs(direction) > 0.0, direction, torch.ones_like(direction))
        delta = torch.zeros_like(action)
        delta[:, 6:7] = torch.where(active, alpha * direction.to(dtype=action.dtype), torch.zeros_like(current))
        self.previous_gripper = current.detach().clone()
        shaped = action + delta
        delta_l2 = float(torch.linalg.norm(delta.reshape(1, 7), dim=1).detach().cpu().item())
        return shaped, {
            "transform": self.policy_name,
            "activated": bool(delta_l2 > 1e-9),
            "delta_l2": delta_l2,
            "gate_values": None,
            "residual_norm": None,
            "dimensions_changed": [6] if delta_l2 > 1e-9 else [],
            "near_gripper_transition": bool(torch.any(active).detach().cpu().item()),
        }

    def transform(self, action: Any, state: Any, phase: float, task_index: int) -> Any:
        if self.policy_name == "frozen_smolvla":
            self.last_transform = {
                "transform": "identity",
                "activated": False,
                "delta_l2": 0.0,
                "gate_values": None,
                "residual_norm": None,
                "dimensions_changed": [],
            }
            return action
        if self.policy_name == "gripper_transition_heuristic":
            shaped, metrics = self._heuristic_action(action)
        else:
            shaped, metrics = self._learned_action(action, state, phase, task_index)
        self.last_transform = metrics
        return shaped


def _checkpoint_expected_hashes(checkpoint_path: Path) -> dict[str, str]:
    manifest = _read_json(checkpoint_path / "sha256_manifest.json")
    return {str(key): str(value).upper() for key, value in manifest.items()}


def _checkpoint_hash_report(checkpoint_path: Path | None) -> dict[str, Any]:
    if checkpoint_path is None:
        return {"checkpoint_path": None, "all_match": None, "files": {}}
    expected = _checkpoint_expected_hashes(checkpoint_path)
    files = {}
    all_match = True
    for name, expected_sha in expected.items():
        path = checkpoint_path / name
        actual_sha = _sha256_file(path) if path.exists() else None
        match = str(actual_sha).upper() == str(expected_sha).upper() if actual_sha else False
        all_match = all_match and match
        files[name] = {
            "path": str(path),
            "sha256": actual_sha,
            "expected_sha256": expected_sha,
            "match": match,
        }
    return {"checkpoint_path": str(checkpoint_path), "all_match": bool(all_match), "files": files}


def _adapter_for_manifest_policy(manifest_policy: Mapping[str, Any], *, device: str) -> DAGRActionAdapter:
    checkpoint = manifest_policy.get("checkpoint_path")
    checkpoint_path = _repo_relative(str(checkpoint)) if checkpoint else None
    return DAGRActionAdapter(str(manifest_policy["policy"]), checkpoint_path, device=device)


def _dummy_postprocessed_action(loaded: Mapping[str, Any], torch: Any) -> tuple[Any, Any]:
    dummy = loaded["env_preprocessor"](_dummy_observation(torch))
    batch = loaded["preprocessor"](dummy)
    with torch.inference_mode():
        action_chunk = loaded["policy"].predict_action_chunk(batch)
        selected = loaded["policy"].select_action(batch)
    processed = loaded["postprocessor"](selected)
    return processed, batch["observation.state"], action_chunk


def _preflight_policy_record(
    args: argparse.Namespace,
    manifest_policy: Mapping[str, Any],
    loaded: Mapping[str, Any],
    task_index: int,
) -> dict[str, Any]:
    import torch

    checkpoint = manifest_policy.get("checkpoint_path")
    checkpoint_path = _repo_relative(str(checkpoint)) if checkpoint else None
    adapter = _adapter_for_manifest_policy(manifest_policy, device="cuda")
    adapter.reset()
    action, state, action_chunk = _dummy_postprocessed_action(loaded, torch)
    shaped = adapter.transform(action, state, phase=0.0, task_index=int(task_index))
    audit = dict(loaded.get("audit") or {})
    checksum = _checkpoint_hash_report(checkpoint_path)
    delta = shaped - action
    learned_counts = _parameter_count_summary(adapter.model) if adapter.model is not None else None
    return {
        "policy": manifest_policy["policy"],
        "role": manifest_policy.get("role"),
        "proxy_or_reproduction_label": manifest_policy.get("proxy_or_reproduction_label"),
        "checkpoint_path": checkpoint,
        "wsl_checkpoint_path": manifest_policy.get("wsl_checkpoint_path"),
        "checkpoint_hashes": checksum,
        "base_model_path": str(args.base_path),
        "policy_class": audit.get("policy_class"),
        "policy_output_shape": audit.get("action_chunk_shape"),
        "policy_output_device": audit.get("action_chunk_device"),
        "policy_output_finite": audit.get("action_chunk_finite"),
        "postprocessed_action_shape": [int(dim) for dim in action.shape],
        "postprocessed_action_device": str(action.device),
        "dagr_action_shape": [int(dim) for dim in shaped.shape],
        "dagr_action_device": str(shaped.device),
        "dagr_action_finite": bool(torch.isfinite(shaped).all().item()),
        "dagr_action_max_abs": _round(float(torch.max(torch.abs(shaped)).detach().cpu().item()), 6),
        "dagr_delta_l2": _round(float(torch.linalg.norm(delta.reshape(1, 7), dim=1).detach().cpu().item()), 9),
        "dagr_transform": adapter.last_transform,
        "dagr_model_parameter_device": _first_parameter_device(adapter.model) if adapter.model is not None else None,
        "dagr_model_parameter_counts": learned_counts,
        "base_action_chunk_shape": [int(dim) for dim in action_chunk.shape],
        "model_parameter_device": (audit.get("parameter") or {}).get("device"),
        "input_tensor_devices": audit.get("input_tensor_devices"),
        "cuda_memory": audit.get("cuda_memory"),
        "old_custom_libero_7d_route_used": audit.get("old_custom_libero_7d_route_used"),
        "cpu_fallback_detected": bool((audit.get("parameter") or {}).get("device") != "cuda:0"),
    }


def _summarize_stage_a(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_policy_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_policy_task: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy_rows[str(row["policy"])].append(row)
        by_policy_task[(str(row["policy"]), str(row["suite"]), int(row["task_id"]))].append(row)

    by_policy = {}
    for policy in STAGE_A_POLICY_ORDER:
        policy_rows = by_policy_rows[policy]
        valid = [row for row in policy_rows if row.get("failure_status") != "exception"]
        successes = int(sum(1 for row in valid if bool(row.get("success"))))
        per_task = {}
        task_rates = []
        for task_key, task_rows in sorted(
            ((key, value) for key, value in by_policy_task.items() if key[0] == policy),
            key=lambda item: (item[0][1], item[0][2]),
        ):
            valid_task_rows = [row for row in task_rows if row.get("failure_status") != "exception"]
            task_successes = int(sum(1 for row in valid_task_rows if bool(row.get("success"))))
            task_total = len(valid_task_rows)
            rate = task_successes / task_total if task_total else 0.0
            task_rates.append(rate)
            per_task[f"{task_key[1]}/task_{task_key[2]}"] = {
                "successes": task_successes,
                "total": task_total,
                "success_rate": _round(rate, 6),
            }
        transform_rows = [row.get("dagr_transform_summary") or {} for row in valid]
        by_policy[policy] = {
            "successes": successes,
            "total": len(valid),
            "success_rate": _round(successes / len(valid), 6) if valid else 0.0,
            "task_balanced_success_rate": _round(float(np.mean(task_rates)), 6) if task_rates else 0.0,
            "exception_count": int(sum(1 for row in policy_rows if row.get("failure_status") == "exception")),
            "per_task": per_task,
            "action_validity_all_finite": all(bool((row.get("action_validity") or {}).get("finite", False)) for row in valid) if valid else False,
            "action_validity_all_shape_ok": all(bool((row.get("action_validity") or {}).get("shape_ok", False)) for row in valid) if valid else False,
            "policy_latency_mean_s": _round(
                float(np.mean([float(row["policy_latency_mean_s"]) for row in valid if row.get("policy_latency_mean_s") is not None])),
                6,
            )
            if any(row.get("policy_latency_mean_s") is not None for row in valid)
            else None,
            "peak_vram_max_allocated_mb": _round(
                max([float(((row.get("peak_vram") or {}).get("max_allocated_mb")) or 0.0) for row in valid] or [0.0]),
                3,
            ),
            "mean_dagr_delta_l2": _round(
                float(np.mean([float(item.get("mean_delta_l2", 0.0)) for item in transform_rows if item.get("mean_delta_l2") is not None])),
                9,
            )
            if transform_rows
            else None,
            "mean_activation_fraction": _round(
                float(np.mean([float(item.get("activation_fraction", 0.0)) for item in transform_rows if item.get("activation_fraction") is not None])),
                6,
            )
            if transform_rows
            else None,
        }
    return {
        "by_policy": by_policy,
        "exception_count": int(sum(1 for row in rows if row.get("failure_status") == "exception")),
        "completed_episode_count": len(rows),
    }


def _paired_bootstrap_ci(deltas: list[float], *, seed: int = 20261218, samples: int = 5000) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas, dtype=np.float64)
    means = np.mean(rng.choice(arr, size=(samples, arr.size), replace=True), axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [_round(float(low), 6), _round(float(high), 6)]


def _paired_vs_dagr_full(rows: list[Mapping[str, Any]], *, include_bootstrap: bool = False) -> dict[str, Any]:
    by_key = {
        (str(row["policy"]), str(row["suite"]), int(row["task_id"]), int(row["reset_seed"])): bool(row.get("success"))
        for row in rows
        if row.get("failure_status") != "exception"
    }
    out = {}
    for policy in STAGE_A_POLICY_ORDER:
        if policy == DAGR_FULL_POLICY:
            continue
        deltas = []
        wins = losses = ties = 0
        for row in rows:
            if str(row.get("policy")) != DAGR_FULL_POLICY or row.get("failure_status") == "exception":
                continue
            key = (policy, str(row["suite"]), int(row["task_id"]), int(row["reset_seed"]))
            if key not in by_key:
                continue
            delta = float(bool(row.get("success"))) - float(by_key[key])
            deltas.append(delta)
            if delta > 0:
                wins += 1
            elif delta < 0:
                losses += 1
            else:
                ties += 1
        out[policy] = {
            "paired_count": len(deltas),
            "paired_win_count": wins,
            "paired_loss_count": losses,
            "paired_tie_count": ties,
            "paired_success_delta": _round(float(np.mean(deltas)), 6) if deltas else 0.0,
        }
        if include_bootstrap:
            out[policy]["paired_bootstrap_ci"] = _paired_bootstrap_ci(deltas)
    return out


def _stage_a_decision(summary: Mapping[str, Any]) -> str:
    if int(summary.get("exception_count") or 0) > 0:
        return "DAGR_STAGE_A_MEASUREMENT_INVALID_REPAIR_REQUIRED"
    by_policy = summary["by_policy"]
    full = by_policy[DAGR_FULL_POLICY]
    full_successes = int(full["successes"])
    for policy in STAGE_A_POLICY_ORDER:
        if policy == DAGR_FULL_POLICY:
            continue
        baseline = by_policy[policy]
        if full_successes == 0 and int(baseline["successes"]) >= 4:
            return "DAGR_STAGE_A_CATASTROPHIC_KILL_ZERO_VS_STRONG_BASELINE"
        if float(baseline["task_balanced_success_rate"]) - float(full["task_balanced_success_rate"]) >= 0.30:
            return "DAGR_STAGE_A_CATASTROPHIC_KILL_CLEARLY_WORSE_THAN_BASELINE_OR_ABLATION"
    if all(float(full["task_balanced_success_rate"]) > float(by_policy[policy]["task_balanced_success_rate"]) for policy in STAGE_A_POLICY_ORDER if policy != DAGR_FULL_POLICY):
        return "DAGR_STAGE_A_POSITIVE_TO_STAGE_B_REQUIRED"
    return "DAGR_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED"


def _stage_b_decision(summary: Mapping[str, Any], paired: Mapping[str, Any]) -> str:
    if int(summary.get("exception_count") or 0) > 0:
        return "DAGR_STAGE_B_MEASUREMENT_INVALID_REPAIR_REQUIRED"
    by_policy = summary["by_policy"]
    full_rate = float(by_policy[DAGR_FULL_POLICY]["task_balanced_success_rate"])
    base_rate = float(by_policy["frozen_smolvla"]["task_balanced_success_rate"])
    prior_rate = float(by_policy["dam_static_component_proxy"]["task_balanced_success_rate"])
    ablation_rate = float(by_policy["dagr_no_dynamic_route_ablation"]["task_balanced_success_rate"])
    simple_rate = float(by_policy["gripper_transition_heuristic"]["task_balanced_success_rate"])
    strongest_name = max(
        (policy for policy in STAGE_A_POLICY_ORDER if policy != DAGR_FULL_POLICY),
        key=lambda policy: float(by_policy[policy]["task_balanced_success_rate"]),
    )
    strongest_rate = float(by_policy[strongest_name]["task_balanced_success_rate"])
    if full_rate > max(base_rate, prior_rate, ablation_rate, simple_rate) and full_rate - strongest_rate >= 0.10:
        return "DAGR_STAGE_B_PROTOTYPE_GO"
    if simple_rate >= full_rate:
        return "DAGR_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD"
    if ablation_rate >= full_rate:
        return "DAGR_STAGE_B_KILL_KEY_COMPONENT_NOT_USEFUL"
    if prior_rate >= full_rate:
        return "DAGR_STAGE_B_KILL_CLOSEST_PRIOR_EXPLAINS_METHOD"
    if base_rate >= full_rate:
        return "DAGR_STAGE_B_KILL_BASE_NOT_IMPROVED"
    strongest_pair = paired.get(strongest_name) or {}
    ci = strongest_pair.get("paired_bootstrap_ci") or [0.0, 0.0]
    if full_rate <= strongest_rate and float(ci[1]) <= 0.10:
        return "DAGR_STAGE_B_USEFUL_IMPROVEMENT_EXCLUDED"
    return "DAGR_STAGE_B_UNRESOLVED_EXPANSION_REQUIRED"


def _transform_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"mean_delta_l2": 0.0, "max_delta_l2": 0.0, "activation_fraction": 0.0, "mean_gate_values": None}
    deltas = [float(item.get("delta_l2") or 0.0) for item in records]
    gates = [item.get("gate_values") for item in records if item.get("gate_values") is not None]
    gate_mean = None
    if gates:
        gate_mean = [float(value) for value in np.mean(np.asarray(gates, dtype=np.float64), axis=0).tolist()]
    return {
        "mean_delta_l2": _round(float(np.mean(deltas)), 9),
        "max_delta_l2": _round(float(np.max(deltas)), 9),
        "activation_fraction": _round(float(np.mean([value > 1e-9 for value in deltas])), 6),
        "mean_gate_values": gate_mean,
        "changed_dimensions": sorted({int(dim) for item in records for dim in (item.get("dimensions_changed") or [])}),
    }


def trace_one_dagr_episode(
    *,
    env: Any,
    policy: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    adapter: DAGRActionAdapter,
    task_index: int,
    seed: int,
    video_path: Path | None,
) -> dict[str, Any]:
    import torch
    from lerobot.scripts.lerobot_eval import (
        ACTION,
        add_envs_task,
        check_env_attributes_and_types,
        preprocess_observation,
        write_video,
    )

    if env.num_envs != 1:
        raise ValueError("DAGR Stage A trace expects batch size 1")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    policy.reset()
    adapter.reset()
    observation, _ = env.reset(seed=[int(seed)])
    max_steps = int(env.call("_max_episode_steps")[0])
    done = np.array([False])
    rewards: list[float] = []
    successes: list[bool] = []
    action_finite = True
    action_shape_ok = True
    action_max_abs = 0.0
    policy_latencies: list[float] = []
    env_latencies: list[float] = []
    chunks_generated = 0
    queue_observable = True
    terminated_last = False
    truncated_last = False
    frames = []
    transform_records: list[dict[str, Any]] = []

    capture_video = video_path is not None
    if capture_video:
        frames.append(env.envs[0].render())

    check_env_attributes_and_types(env)
    step = 0
    while not np.all(done) and step < max_steps:
        lerobot_observation = preprocess_observation(observation)
        lerobot_observation = add_envs_task(env, lerobot_observation)
        lerobot_observation = env_preprocessor(lerobot_observation)
        batch = preprocessor(lerobot_observation)

        queue_len_before = _action_queue_len(policy, ACTION)
        if queue_len_before is None:
            queue_observable = False
        elif queue_len_before == 0:
            chunks_generated += 1

        start_policy = time.perf_counter()
        with torch.inference_mode():
            selected = policy.select_action(batch)
            action = postprocessor(selected)
            phase = float(step) / max(1.0, float(max_steps - 1))
            action = adapter.transform(action, batch["observation.state"], phase=phase, task_index=int(task_index))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        policy_latencies.append(time.perf_counter() - start_policy)
        transform_records.append(dict(adapter.last_transform))

        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition[ACTION]
        action_numpy = action.to("cpu").numpy()
        action_finite = action_finite and bool(np.isfinite(action_numpy).all())
        action_shape_ok = action_shape_ok and action_numpy.shape == (1, 7)
        action_max_abs = max(action_max_abs, float(np.max(np.abs(action_numpy))))

        start_env = time.perf_counter()
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        env_latencies.append(time.perf_counter() - start_env)
        if capture_video:
            frames.append(env.envs[0].render())

        step_successes = _successes_from_info(info, env.num_envs)
        successes.append(bool(step_successes[0]))
        rewards.append(float(np.asarray(reward).reshape(-1)[0]))
        terminated_last = bool(np.asarray(terminated).reshape(-1)[0])
        truncated_last = bool(np.asarray(truncated).reshape(-1)[0])

        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)
        step += 1

    success = any(successes)
    sum_reward = float(np.sum(rewards)) if rewards else 0.0
    max_reward = float(np.max(rewards)) if rewards else 0.0
    if success:
        termination_reason = "success"
    elif terminated_last:
        termination_reason = "terminated_without_success"
    elif truncated_last or step >= max_steps:
        termination_reason = "max_steps_or_truncated_without_success"
    else:
        termination_reason = "done_without_success"

    saved_video_path = None
    if capture_video and frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        write_video(str(video_path), np.stack(frames), env.unwrapped.metadata["render_fps"])
        saved_video_path = str(video_path)

    return {
        "success": bool(success),
        "sum_reward": _round(sum_reward, 6),
        "max_reward": _round(max_reward, 6),
        "episode_length": int(step),
        "termination_reason": termination_reason,
        "failure_status": "success" if success else "unsuccessful",
        "exception": None,
        "action_validity": {
            "finite": bool(action_finite),
            "shape_ok": bool(action_shape_ok),
            "max_abs": _round(action_max_abs, 6),
        },
        "action_chunks_generated": int(chunks_generated) if queue_observable else None,
        "env_steps": int(step),
        "policy_latency_mean_s": _round(float(np.mean(policy_latencies)), 6) if policy_latencies else None,
        "policy_latency_max_s": _round(float(np.max(policy_latencies)), 6) if policy_latencies else None,
        "env_step_latency_mean_s": _round(float(np.mean(env_latencies)), 6) if env_latencies else None,
        "env_step_latency_max_s": _round(float(np.max(env_latencies)), 6) if env_latencies else None,
        "dagr_transform_summary": _transform_summary(transform_records),
        "peak_vram": _cuda_memory(torch),
        "rss_mb": _rss_mb(),
        "video_path": saved_video_path,
    }


def build_stage_a_manifest(args: argparse.Namespace) -> dict[str, Any]:
    task_manifest = _read_json(Path(args.official_task_manifest))
    checkpoint_manifest = _read_json(Path(args.checkpoint_manifest))
    tasks = _select_stage_a_tasks(task_manifest)
    policies = _policy_records(args, checkpoint_manifest)
    pairs = []
    for task in tasks:
        for seed in STAGE_A_RESET_SEEDS:
            pairs.append(
                {
                    "pair_id": f"{task['suite']}|task_{task['task_id']}|seed_{seed}",
                    "suite": str(task["suite"]),
                    "task_id": int(task["task_id"]),
                    "instruction": str(task["instruction"]),
                    "reset_seed": int(seed),
                    "stage_a_task_index": int(task["stage_a_task_index"]),
                    "source_official_task_manifest_index": int(task["source_official_task_manifest_index"]),
                }
            )
    episodes = []
    planned_index = 0
    for policy in STAGE_A_POLICY_ORDER:
        for pair in pairs:
            episodes.append(
                {
                    "planned_episode_index": int(planned_index),
                    "episode_id": f"{policy}|{pair['suite']}|task_{pair['task_id']}|seed_{pair['reset_seed']}",
                    "policy": policy,
                    "pair_id": pair["pair_id"],
                    "suite": pair["suite"],
                    "task_id": int(pair["task_id"]),
                    "instruction": pair["instruction"],
                    "reset_seed": int(pair["reset_seed"]),
                }
            )
            planned_index += 1
    payload = {
        "schema_version": 1,
        "method": METHOD,
        "config_id": CONFIG_ID,
        "proposal_hash": PROPOSAL_HASH,
        "branch": BRANCH,
        "date": f"{args.date} KST",
        "mode": "stage_a_manifest",
        "final_decision": "DAGR_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT",
        "closed_loop_experiment_happened": False,
        "confirmatory_test_tuning_happened": False,
        "confirmatory_test_identities_used_for_training_or_validation": False,
        "policy_order": list(STAGE_A_POLICY_ORDER),
        "policies": policies,
        "stage_a_reset_seeds": list(STAGE_A_RESET_SEEDS),
        "stage_a_pair_count_per_policy": len(pairs),
        "planned_episode_count": len(episodes),
        "tasks": tasks,
        "pairs": pairs,
        "episodes": episodes,
        "identity_overlap_verification": {
            "stage_a_rollout_reset_seeds": list(STAGE_A_RESET_SEEDS),
            "overlap_with_development_training_identities": 0,
            "overlap_with_development_validation_identities": 0,
            "overlap_with_previous_known_allocated_rollout_identities": 0,
            "duplicate_evaluation_keys": 0,
            "identical_task_reset_pairs_across_policies": True,
            "note": "DAGR training and validation used offline dataset frame splits; Stage A reset seeds were selected after policy identity freeze.",
        },
        "task_balanced_allocation": {
            "task_count": len(tasks),
            "reset_count_per_task": len(STAGE_A_RESET_SEEDS),
            "episodes_per_task_per_policy": len(STAGE_A_RESET_SEEDS),
            "paired_cases_per_policy": len(pairs),
            "fixed_before_rollout": True,
        },
        "task_selection": {
            "source_manifest": str(args.official_task_manifest),
            "source_manifest_sha256": _sha256_file(Path(args.official_task_manifest)),
            "rule": "select 5 global evenly spaced tasks from the frozen official 20-task manifest: floor(k * n / 5)",
            "outcome_dependent": False,
        },
        "reset_identity_selection": {
            "rule": "fresh unused DAGR Stage A block after MTF Stage B reset seeds",
            "reset_seeds": list(STAGE_A_RESET_SEEDS),
            "previous_known_allocations_avoided": [
                "official baseline scale-up reset seeds 20260711..20260715",
                "CBFD/SCVC/PSE reset identities 20260716..20260760",
                "CAVM/FANG/RAC/EvoState/MTF reset identity blocks through 20261204",
            ],
        },
        "partition_separation": {
            "offline_training_splits": ["train"],
            "offline_validation_splits": ["val"],
            "offline_reserved_confirmatory_splits": ["test"],
            "stage_a_rollout_resets_are_frozen_after_checkpoint_selection": True,
            "stage_a_rollout_resets_used_for_policy_training": False,
            "stage_a_rollout_resets_used_for_validation_search": False,
        },
        "frozen_stage_a_rules": {
            "permanent_kill_zero_vs_baseline": "dagr_full has 0/10 while any paired baseline has at least 4/10",
            "permanent_kill_clear_degradation": "dagr_full is at least 30 absolute points below a baseline, prior proxy, simple baseline, or ablation",
            "small_difference_rule": "small differences, ties, and one- or two-episode gaps advance to Stage B",
            "next_stage_count": "Stage B requires at least 40 paired episodes per key policy",
        },
        "execution": {
            "official_path": "LeRobot SmolVLA/LIBERO policy, processors, action queue, relative 7D control, official LIBERO success condition, plus DAGR residual wrapper",
            "policy_order_affects_environment_initialization": False,
            "environment_initialization_rule": "each episode calls env.reset(seed=[reset_seed]) after constructing the task env; the same task/reset pairs are executed for every policy",
            "base_path_default": str(args.base_path),
            "checkpoint_root_default": str(args.checkpoint_root),
            "libero_config_dir_default": str(args.libero_config_dir),
            "partial_result_path": str(args.stage_a_partial_output),
            "result_path": str(args.stage_a_output),
            "preflight_result_path": str(args.stage_a_preflight_output),
            "resume_rule": "resume only missing (policy, suite, task_id, reset_seed) episode keys",
        },
        "checkpoint_manifest": {
            "path": str(args.checkpoint_manifest),
            "sha256": _sha256_file(Path(args.checkpoint_manifest)),
            "checkpoint_root": _relative(str(checkpoint_manifest.get("checkpoint_root"))),
            "policy_identity_count": len(checkpoint_manifest.get("policy_identities") or []),
        },
    }
    payload["canonical_payload_sha256"] = _sha256_payload({key: value for key, value in payload.items() if key != "canonical_payload_sha256"})
    validate_manifest(payload)
    return payload


def build_stage_b_manifest(args: argparse.Namespace) -> dict[str, Any]:
    task_manifest = _read_json(Path(args.official_task_manifest))
    checkpoint_manifest = _read_json(Path(args.checkpoint_manifest))
    stage_a_result = _read_json(Path(args.stage_a_output))
    if stage_a_result.get("final_decision") not in {
        "DAGR_STAGE_A_POSITIVE_TO_STAGE_B_REQUIRED",
        "DAGR_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED",
    }:
        raise ValueError("DAGR Stage B requires a completed Stage A decision requiring Stage B")
    if int(stage_a_result.get("completed_episode_count", -1)) != len(STAGE_A_POLICY_ORDER) * STAGE_A_EPISODES_PER_POLICY:
        raise ValueError("DAGR Stage A result is incomplete")
    if int((stage_a_result.get("summary") or {}).get("exception_count") or 0) != 0:
        raise ValueError("DAGR Stage A result has exceptions; repair/adjudication required before Stage B")

    tasks = _select_stage_b_tasks(task_manifest)
    policies = _policy_records(args, checkpoint_manifest)
    pairs = []
    for task in tasks:
        for seed in STAGE_B_RESET_SEEDS:
            pairs.append(
                {
                    "pair_id": f"{task['suite']}|task_{task['task_id']}|seed_{seed}",
                    "suite": str(task["suite"]),
                    "task_id": int(task["task_id"]),
                    "instruction": str(task["instruction"]),
                    "reset_seed": int(seed),
                    "stage_b_task_index": int(task["stage_b_task_index"]),
                    "source_official_task_manifest_index": int(task["source_official_task_manifest_index"]),
                }
            )

    episodes = []
    planned_index = 0
    for policy in STAGE_A_POLICY_ORDER:
        for pair in pairs:
            episodes.append(
                {
                    "planned_episode_index": int(planned_index),
                    "episode_id": f"{policy}|{pair['suite']}|task_{pair['task_id']}|seed_{pair['reset_seed']}",
                    "policy": policy,
                    "pair_id": pair["pair_id"],
                    "suite": pair["suite"],
                    "task_id": int(pair["task_id"]),
                    "instruction": pair["instruction"],
                    "reset_seed": int(pair["reset_seed"]),
                }
            )
            planned_index += 1

    payload = {
        "schema_version": 1,
        "method": METHOD,
        "config_id": CONFIG_ID,
        "proposal_hash": PROPOSAL_HASH,
        "branch": BRANCH,
        "date": f"{args.date} KST",
        "mode": "stage_b_manifest",
        "final_decision": "DAGR_STAGE_B_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT",
        "closed_loop_experiment_happened": False,
        "confirmatory_test_tuning_happened": False,
        "stage_a_outcome_used_only_for_preregistered_escalation": True,
        "policy_order": list(STAGE_A_POLICY_ORDER),
        "policies": policies,
        "stage_b_reset_seeds": list(STAGE_B_RESET_SEEDS),
        "stage_b_pair_count_per_policy": len(pairs),
        "planned_episode_count": len(episodes),
        "tasks": tasks,
        "pairs": pairs,
        "episodes": episodes,
        "identity_overlap_verification": {
            "stage_a_rollout_reset_seeds": list(STAGE_A_RESET_SEEDS),
            "stage_b_rollout_reset_seeds": list(STAGE_B_RESET_SEEDS),
            "overlap_with_stage_a_reset_seeds": len(set(STAGE_A_RESET_SEEDS) & set(STAGE_B_RESET_SEEDS)),
            "overlap_with_development_training_identities": 0,
            "overlap_with_development_validation_identities": 0,
            "overlap_with_previous_known_allocated_rollout_identities": 0,
            "duplicate_evaluation_keys": 0,
            "identical_task_reset_pairs_across_policies": True,
            "note": "Stage B uses fresh reset seeds and all official tasks after the frozen Stage A decision required Stage B.",
        },
        "task_balanced_allocation": {
            "task_count": len(tasks),
            "reset_count_per_task": len(STAGE_B_RESET_SEEDS),
            "episodes_per_task_per_policy": len(STAGE_B_RESET_SEEDS),
            "paired_cases_per_policy": len(pairs),
            "fixed_before_rollout": True,
        },
        "task_selection": {
            "source_manifest": str(args.official_task_manifest),
            "source_manifest_sha256": _sha256_file(Path(args.official_task_manifest)),
            "rule": "use all tasks from the frozen official 20-task manifest",
            "outcome_dependent": False,
        },
        "reset_identity_selection": {
            "rule": "fresh unused DAGR Stage B block immediately after the DAGR Stage A block",
            "reset_seeds": list(STAGE_B_RESET_SEEDS),
            "previous_known_allocations_avoided": [
                "official baseline scale-up reset seeds 20260711..20260715",
                "CBFD/SCVC/PSE reset identities 20260716..20260760",
                "CAVM/FANG/RAC/EvoState/MTF reset identity blocks through 20261204",
                "DAGR Stage A reset seeds 20261205..20261206",
            ],
        },
        "partition_separation": {
            "offline_training_splits": ["train"],
            "offline_validation_splits": ["val"],
            "offline_reserved_confirmatory_splits": ["test"],
            "stage_b_rollout_resets_are_frozen_after_stage_a_adjudication": True,
            "stage_b_rollout_resets_used_for_policy_training": False,
            "stage_b_rollout_resets_used_for_validation_search": False,
            "stage_b_outcomes_used_for_retuning": False,
        },
        "frozen_stage_b_rules": {
            "prototype_go": "dagr_full must beat Base, closest-prior proxy, key ablation, and simple killer by at least 10 task-balanced points or equivalent positive paired evidence under governance",
            "simple_baseline_kill": "gripper_transition_heuristic matches or beats dagr_full",
            "key_component_kill": "dagr_no_dynamic_route_ablation matches or beats dagr_full",
            "closest_prior_kill": "dam_static_component_proxy matches or beats dagr_full",
            "base_not_improved_kill": "frozen_smolvla matches or beats dagr_full",
            "expansion_rule": "one expansion to 80 paired episodes is allowed only if Stage B is genuinely unresolved",
        },
        "execution": {
            "official_path": "LeRobot SmolVLA/LIBERO policy, processors, action queue, relative 7D control, official LIBERO success condition, plus DAGR residual wrapper",
            "policy_order_affects_environment_initialization": False,
            "environment_initialization_rule": "each episode calls env.reset(seed=[reset_seed]) after constructing the task env; the same task/reset pairs are executed for every policy",
            "base_path_default": str(args.base_path),
            "checkpoint_root_default": str(args.checkpoint_root),
            "libero_config_dir_default": str(args.libero_config_dir),
            "partial_result_path": str(args.stage_b_partial_output),
            "result_path": str(args.stage_b_output),
            "resume_rule": "resume only missing (policy, suite, task_id, reset_seed) episode keys",
        },
        "checkpoint_manifest": {
            "path": str(args.checkpoint_manifest),
            "sha256": _sha256_file(Path(args.checkpoint_manifest)),
            "checkpoint_root": _relative(str(checkpoint_manifest.get("checkpoint_root"))),
            "policy_identity_count": len(checkpoint_manifest.get("policy_identities") or []),
        },
        "stage_a_result": {
            "path": str(args.stage_a_output),
            "sha256": _sha256_file(Path(args.stage_a_output)),
            "final_decision": str(stage_a_result.get("final_decision")),
            "completed_episode_count": int(stage_a_result.get("completed_episode_count", 0)),
            "exception_count": int((stage_a_result.get("summary") or {}).get("exception_count") or 0),
        },
    }
    payload["canonical_payload_sha256"] = _sha256_payload({key: value for key, value in payload.items() if key != "canonical_payload_sha256"})
    validate_stage_b_manifest(payload)
    return payload


def write_stage_a_manifest_md(path: Path, manifest: Mapping[str, Any]) -> None:
    lines = [
        "# DAGR-VLA Stage A Manifest",
        "",
        f"Date: `{manifest['date']}`",
        "",
        f"Final decision: `{manifest['final_decision']}`",
        "",
        f"- method: `{manifest['method']}`",
        f"- config: `{manifest['config_id']}`",
        f"- proposal hash: `{manifest['proposal_hash']}`",
        f"- policies: `{', '.join(manifest['policy_order'])}`",
        f"- reset seeds: `{manifest['stage_a_reset_seeds']}`",
        f"- paired cases per policy: `{manifest['stage_a_pair_count_per_policy']}`",
        f"- planned episodes: `{manifest['planned_episode_count']}`",
        f"- canonical payload sha256: `{manifest['canonical_payload_sha256']}`",
        "",
        "## Tasks",
        "",
    ]
    for task in manifest["tasks"]:
        lines.append(f"- `{task['suite']}/task_{task['task_id']}`: {task['instruction']}")
    lines.extend(
        [
            "",
            "## Frozen Rules",
            "",
            "- five policies only: frozen SmolVLA, DAM-style static component proxy, DAGR full, no-dynamic-route ablation, and gripper-transition heuristic",
            "- `dam_static_component_proxy` is a faithful transparent local proxy, not an official DAM-VLA reproduction",
            "- task/reset pairs are identical across policies and duplicate evaluation keys are zero",
            "- policy order does not choose or perturb reset identities",
            "- official LIBERO success condition is the primary closed-loop outcome",
            "- no confirmatory-test tuning or checkpoint selection from Stage A outcomes",
            "- small differences, ties, and one- or two-episode gaps advance to Stage B",
            "- permanent Stage A kill only under the preregistered catastrophic criteria",
            "",
            "## Execution",
            "",
            f"- partial result path: `{manifest['execution']['partial_result_path']}`",
            f"- final result path: `{manifest['execution']['result_path']}`",
            "- resume only missing `(policy, suite, task_id, reset_seed)` keys",
        ]
    )
    _write_md(path, lines)


def write_stage_b_manifest_reports(args: argparse.Namespace, manifest: Mapping[str, Any]) -> None:
    _write_json(Path(args.stage_b_manifest), manifest)
    lines = [
        "# DAGR-VLA Stage B Manifest",
        "",
        f"Date: `{manifest['date']}`",
        "",
        f"Final decision: `{manifest['final_decision']}`",
        "",
        f"- method: `{manifest['method']}`",
        f"- config: `{manifest['config_id']}`",
        f"- proposal hash: `{manifest['proposal_hash']}`",
        f"- policies: `{', '.join(manifest['policy_order'])}`",
        f"- reset seeds: `{manifest['stage_b_reset_seeds']}`",
        f"- paired cases per policy: `{manifest['stage_b_pair_count_per_policy']}`",
        f"- planned episodes: `{manifest['planned_episode_count']}`",
        f"- canonical payload sha256: `{manifest['canonical_payload_sha256']}`",
        f"- Stage A decision: `{manifest['stage_a_result']['final_decision']}`",
        "",
        "## Tasks",
        "",
    ]
    for task in manifest["tasks"]:
        lines.append(f"- `{task['suite']}/task_{task['task_id']}`: {task['instruction']}")
    lines.extend(
        [
            "",
            "## Frozen Rules",
            "",
            "- all 20 official tasks are used",
            "- reset seeds are fresh relative to DAGR Stage A",
            "- task/reset pairs are identical across policies and duplicate evaluation keys are zero",
            "- `dam_static_component_proxy` remains a faithful transparent local proxy, not an official DAM-VLA reproduction",
            "- no confirmatory-test tuning or checkpoint selection from Stage A or Stage B outcomes",
            "- one expansion to 80 paired episodes is allowed only if Stage B is genuinely unresolved",
            "",
            "## Execution",
            "",
            f"- partial result path: `{manifest['execution']['partial_result_path']}`",
            f"- final result path: `{manifest['execution']['result_path']}`",
            "- resume only missing `(policy, suite, task_id, reset_seed)` keys",
        ]
    )
    _write_md(Path(args.stage_b_manifest_md), lines)


def run_plan(args: argparse.Namespace) -> dict[str, Any]:
    manifest = build_stage_a_manifest(args)
    _write_json(Path(args.stage_a_manifest), manifest)
    write_stage_a_manifest_md(Path(args.stage_a_manifest_md), manifest)
    return manifest


def run_stage_b_plan(args: argparse.Namespace) -> dict[str, Any]:
    manifest = build_stage_b_manifest(args)
    write_stage_b_manifest_reports(args, manifest)
    return manifest


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    _set_runtime_env(args)
    manifest_path = Path(args.stage_a_manifest)
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
    else:
        manifest = build_stage_a_manifest(args)
        _write_json(manifest_path, manifest)
        write_stage_a_manifest_md(Path(args.stage_a_manifest_md), manifest)
    validate_stage_a_manifest(manifest)
    task_index_map = _task_index_map_from_artifact(Path(args.stable_artifact))
    first_task_index = _task_index_for_task(manifest["tasks"][0], task_index_map)
    records = []
    errors = []
    for manifest_policy in manifest["policies"]:
        policy_name = str(manifest_policy["policy"])
        try:
            loaded = _load_policy_and_processors(args, PolicySpec(policy_name))
            records.append(_preflight_policy_record(args, manifest_policy, loaded, first_task_index))
            del loaded
            torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - WSL/CUDA boundary
            errors.append(
                {
                    "policy": policy_name,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc().splitlines()[-24:],
                }
            )
    checkpoint_records = [record for record in records if record.get("checkpoint_path")]
    unique_paths = {str(record["checkpoint_path"]) for record in checkpoint_records}
    identity_hashes = set()
    for record in checkpoint_records:
        files = (record.get("checkpoint_hashes") or {}).get("files") or {}
        if "model.pt" in files:
            identity_hashes.add(str(files["model.pt"].get("sha256")).upper())
        elif "heuristic_config.json" in files:
            identity_hashes.add(str(files["heuristic_config.json"].get("sha256")).upper())
    checkpoint_matches = all((record.get("checkpoint_hashes") or {}).get("all_match") is True for record in checkpoint_records)
    cuda_ok = all(
        not bool(record.get("cpu_fallback_detected"))
        and record.get("policy_output_device") == "cuda:0"
        and (record.get("dagr_model_parameter_device") in {None, "cuda:0"})
        for record in records
    )
    output_ok = all(
        record.get("policy_output_shape") == [1, 50, 7]
        and record.get("base_action_chunk_shape") == [1, 50, 7]
        and record.get("postprocessed_action_shape") == [1, 7]
        and record.get("dagr_action_shape") == [1, 7]
        and bool(record.get("policy_output_finite"))
        and bool(record.get("dagr_action_finite"))
        and float(record.get("dagr_action_max_abs") or 999.0) <= 5.0
        and float(record.get("dagr_delta_l2") or 0.0) <= 1.0
        for record in records
    )
    no_reuse = len(unique_paths) == 4 and len(identity_hashes) == 4
    task_indices = {
        f"{task['suite']}/task_{task['task_id']}": _task_index_for_task(task, task_index_map)
        for task in manifest.get("tasks") or []
    }
    final_decision = "DAGR_STAGE_A_PREFLIGHT_PASS_READY_FOR_OFFICIAL_ROLLOUT"
    if errors or not (checkpoint_matches and cuda_ok and output_ok and no_reuse):
        final_decision = "DAGR_STAGE_A_PREFLIGHT_BLOCKED_REPAIR_LOADING_OR_MAPPING"
    report = {
        "schema_version": 1,
        "method": METHOD,
        "config_id": CONFIG_ID,
        "proposal_hash": PROPOSAL_HASH,
        "date": f"{args.date} KST",
        "mode": "preflight",
        "stage_a_manifest": str(args.stage_a_manifest),
        "stage_a_manifest_sha256": _sha256_file(Path(args.stage_a_manifest)),
        "stable_artifact": str(args.stable_artifact),
        "closed_loop_experiment_happened": False,
        "training_happened": False,
        "policy_count": len(records),
        "checkpoint_policy_count": len(checkpoint_records),
        "checkpoint_checksum_matches": bool(checkpoint_matches),
        "cuda_ok": bool(cuda_ok),
        "policy_output_shape_ok": bool(output_ok),
        "no_accidental_checkpoint_reuse": bool(no_reuse),
        "task_indices": task_indices,
        "records": records,
        "errors": errors,
        "final_decision": final_decision,
        "elapsed_seconds": _round(time.monotonic() - started, 3),
    }
    _write_json(Path(args.stage_a_preflight_output), report)
    return report


def run_stage_a(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from lerobot.envs.factory import make_env

    started = time.monotonic()
    _set_runtime_env(args)
    manifest_path = Path(args.stage_a_manifest)
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
    else:
        manifest = build_stage_a_manifest(args)
        _write_json(manifest_path, manifest)
        write_stage_a_manifest_md(Path(args.stage_a_manifest_md), manifest)
    validate_stage_a_manifest(manifest)
    if Path(args.stage_a_preflight_output).exists():
        preflight = _read_json(Path(args.stage_a_preflight_output))
        if preflight.get("final_decision") != "DAGR_STAGE_A_PREFLIGHT_PASS_READY_FOR_OFFICIAL_ROLLOUT":
            raise ValueError("DAGR Stage A requires a passing preflight before rollout")
    planned_lookup = _planned_lookup(manifest)
    task_map = _manifest_task_map(manifest)
    task_index_map = _task_index_map_from_artifact(Path(args.stable_artifact))
    partial_path = Path(args.stage_a_partial_output)
    rows = _load_partial(partial_path, bool(args.rerun_stage))
    completed = {_completed_key(row) for row in rows}
    policy_audits = {}
    errors = []

    for manifest_policy in manifest["policies"]:
        policy_name = str(manifest_policy["policy"])
        print(f"[dagr-stage-a] policy {policy_name}", flush=True)
        loaded = _load_policy_and_processors(args, PolicySpec(policy_name))
        adapter = _adapter_for_manifest_policy(manifest_policy, device="cuda")
        policy_audits[policy_name] = loaded["audit"]
        for task in manifest["tasks"]:
            env = None
            suite = str(task["suite"])
            task_id = int(task["task_id"])
            task_index = _task_index_for_task(task, task_index_map)
            print(f"[dagr-stage-a] {policy_name} {suite} task_{task_id}", flush=True)
            try:
                env_cfg = _make_env_cfg(suite, [task_id])
                env = _extract_single_env(make_env(env_cfg, n_envs=1, use_async_envs=False), suite, task_id)
                task_record = task_map[(suite, task_id)]
                for seed in STAGE_A_RESET_SEEDS:
                    key = (policy_name, suite, task_id, int(seed))
                    if key in completed:
                        continue
                    episode_id = f"{policy_name}|{suite}|task_{task_id}|seed_{seed}"
                    row = _episode_base_record(policy_name, task_record, int(seed), planned_lookup[episode_id])
                    row["dagr_task_index"] = int(task_index)
                    try:
                        trace = trace_one_dagr_episode(
                            env=env,
                            policy=loaded["policy"],
                            env_preprocessor=loaded["env_preprocessor"],
                            env_postprocessor=loaded["env_postprocessor"],
                            preprocessor=loaded["preprocessor"],
                            postprocessor=loaded["postprocessor"],
                            adapter=adapter,
                            task_index=int(task_index),
                            seed=int(seed),
                            video_path=None,
                        )
                        row.update(trace)
                    except Exception as exc:  # pragma: no cover - simulator boundary
                        row.update(
                            {
                                "success": False,
                                "sum_reward": None,
                                "max_reward": None,
                                "episode_length": None,
                                "termination_reason": "exception",
                                "failure_status": "exception",
                                "exception": {
                                    "type": type(exc).__name__,
                                    "message": str(exc),
                                    "traceback": traceback.format_exc().splitlines()[-24:],
                                },
                                "action_validity": {"finite": False, "shape_ok": False, "max_abs": None},
                                "action_chunks_generated": None,
                                "env_steps": None,
                                "policy_latency_mean_s": None,
                                "policy_latency_max_s": None,
                                "env_step_latency_mean_s": None,
                                "env_step_latency_max_s": None,
                                "dagr_transform_summary": None,
                                "peak_vram": _cuda_memory(torch),
                                "rss_mb": _rss_mb(),
                                "video_path": None,
                            }
                        )
                        errors.append({"episode_id": episode_id, **row["exception"]})
                    rows.append(row)
                    completed.add(key)
                    _write_json(partial_path, {"episodes": rows, "planned_episode_count": manifest["planned_episode_count"]})
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
        del loaded
        del adapter
        torch.cuda.empty_cache()

    summary = _summarize_stage_a(rows)
    paired = _paired_vs_dagr_full(rows)
    final_decision = _stage_a_decision(summary)
    report = {
        "schema_version": 1,
        "method": METHOD,
        "config_id": CONFIG_ID,
        "proposal_hash": PROPOSAL_HASH,
        "branch": BRANCH,
        "date": f"{args.date} KST",
        "mode": "stage-a",
        "closed_loop_experiment_happened": True,
        "confirmatory_test_tuning_happened": False,
        "stage_a_manifest": str(args.stage_a_manifest),
        "stage_a_manifest_sha256": _sha256_file(Path(args.stage_a_manifest)),
        "stage_a_preflight": str(args.stage_a_preflight_output),
        "planned_episode_count": int(manifest["planned_episode_count"]),
        "completed_episode_count": len(rows),
        "policy_load_audits": policy_audits,
        "episodes": rows,
        "summary": summary,
        "paired_vs_dagr_full": paired,
        "errors": errors,
        "final_decision": final_decision,
        "next_step": "Run Stage B on a frozen expansion manifest." if final_decision.endswith("STAGE_B_REQUIRED") else "Adjudicate repair or catastrophic kill under the preregistered governance.",
        "elapsed_seconds": _round(time.monotonic() - started, 3),
    }
    _write_json(Path(args.stage_a_output), report)
    write_stage_a_result_md(Path(args.stage_a_md), report)
    return report


def run_stage_b(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from lerobot.envs.factory import make_env

    started = time.monotonic()
    _set_runtime_env(args)
    manifest_path = Path(args.stage_b_manifest)
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
    else:
        manifest = build_stage_b_manifest(args)
        write_stage_b_manifest_reports(args, manifest)
    validate_stage_b_manifest(manifest)
    planned_lookup = _planned_lookup(manifest)
    task_map = _manifest_task_map(manifest)
    task_index_map = _task_index_map_from_artifact(Path(args.stable_artifact))
    partial_path = Path(args.stage_b_partial_output)
    rows = _load_partial(partial_path, bool(args.rerun_stage))
    completed = {_completed_key(row) for row in rows}
    policy_audits = {}
    errors = []

    for manifest_policy in manifest["policies"]:
        policy_name = str(manifest_policy["policy"])
        print(f"[dagr-stage-b] policy {policy_name}", flush=True)
        loaded = _load_policy_and_processors(args, PolicySpec(policy_name))
        adapter = _adapter_for_manifest_policy(manifest_policy, device="cuda")
        policy_audits[policy_name] = loaded["audit"]
        for task in manifest["tasks"]:
            env = None
            suite = str(task["suite"])
            task_id = int(task["task_id"])
            task_index = _task_index_for_task(task, task_index_map)
            print(f"[dagr-stage-b] {policy_name} {suite} task_{task_id}", flush=True)
            try:
                env_cfg = _make_env_cfg(suite, [task_id])
                env = _extract_single_env(make_env(env_cfg, n_envs=1, use_async_envs=False), suite, task_id)
                task_record = task_map[(suite, task_id)]
                for seed in STAGE_B_RESET_SEEDS:
                    key = (policy_name, suite, task_id, int(seed))
                    if key in completed:
                        continue
                    episode_id = f"{policy_name}|{suite}|task_{task_id}|seed_{seed}"
                    row = _episode_base_record(policy_name, task_record, int(seed), planned_lookup[episode_id])
                    row["dagr_task_index"] = int(task_index)
                    try:
                        trace = trace_one_dagr_episode(
                            env=env,
                            policy=loaded["policy"],
                            env_preprocessor=loaded["env_preprocessor"],
                            env_postprocessor=loaded["env_postprocessor"],
                            preprocessor=loaded["preprocessor"],
                            postprocessor=loaded["postprocessor"],
                            adapter=adapter,
                            task_index=int(task_index),
                            seed=int(seed),
                            video_path=None,
                        )
                        row.update(trace)
                    except Exception as exc:  # pragma: no cover - simulator boundary
                        row.update(
                            {
                                "success": False,
                                "sum_reward": None,
                                "max_reward": None,
                                "episode_length": None,
                                "termination_reason": "exception",
                                "failure_status": "exception",
                                "exception": {
                                    "type": type(exc).__name__,
                                    "message": str(exc),
                                    "traceback": traceback.format_exc().splitlines()[-24:],
                                },
                                "action_validity": {"finite": False, "shape_ok": False, "max_abs": None},
                                "action_chunks_generated": None,
                                "env_steps": None,
                                "policy_latency_mean_s": None,
                                "policy_latency_max_s": None,
                                "env_step_latency_mean_s": None,
                                "env_step_latency_max_s": None,
                                "dagr_transform_summary": None,
                                "peak_vram": _cuda_memory(torch),
                                "rss_mb": _rss_mb(),
                                "video_path": None,
                            }
                        )
                        errors.append({"episode_id": episode_id, **row["exception"]})
                    rows.append(row)
                    completed.add(key)
                    _write_json(partial_path, {"episodes": rows, "planned_episode_count": manifest["planned_episode_count"]})
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
        del loaded
        del adapter
        torch.cuda.empty_cache()

    summary = _summarize_stage_a(rows)
    paired = _paired_vs_dagr_full(rows, include_bootstrap=True)
    final_decision = _stage_b_decision(summary, paired)
    if final_decision == "DAGR_STAGE_B_PROTOTYPE_GO":
        next_step = "Verify Quantized OpenVLA-OFT INT4 transfer and add one second condition."
    elif final_decision == "DAGR_STAGE_B_UNRESOLVED_EXPANSION_REQUIRED":
        next_step = "Freeze and run the one allowed Stage B expansion to 80 paired episodes per key policy."
    else:
        next_step = "Archive or pivot under the preregistered governance; do not retune DAGR from Stage B outcomes."
    report = {
        "schema_version": 1,
        "method": METHOD,
        "config_id": CONFIG_ID,
        "proposal_hash": PROPOSAL_HASH,
        "branch": BRANCH,
        "date": f"{args.date} KST",
        "mode": "stage-b",
        "closed_loop_experiment_happened": True,
        "confirmatory_test_tuning_happened": False,
        "stage_b_manifest": str(args.stage_b_manifest),
        "stage_b_manifest_sha256": _sha256_file(Path(args.stage_b_manifest)),
        "stage_a_result": str(args.stage_a_output),
        "stage_a_result_sha256": _sha256_file(Path(args.stage_a_output)),
        "planned_episode_count": int(manifest["planned_episode_count"]),
        "completed_episode_count": len(rows),
        "policy_load_audits": policy_audits,
        "episodes": rows,
        "summary": summary,
        "paired_vs_dagr_full": paired,
        "errors": errors,
        "final_decision": final_decision,
        "next_step": next_step,
        "elapsed_seconds": _round(time.monotonic() - started, 3),
    }
    _write_json(Path(args.stage_b_output), report)
    write_stage_b_result_md(Path(args.stage_b_md), report)
    return report


def write_stage_a_result_md(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# DAGR-VLA Stage A Result",
        "",
        f"Date: `{report['date']}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- planned episodes: `{report['planned_episode_count']}`",
        f"- completed episodes: `{report['completed_episode_count']}`",
        f"- closed-loop experiment happened: `{report['closed_loop_experiment_happened']}`",
        f"- confirmatory-test tuning happened: `{report['confirmatory_test_tuning_happened']}`",
        f"- elapsed seconds: `{report['elapsed_seconds']}`",
        "",
        "## Policy Summary",
        "",
        "| policy | successes | total | task-balanced success | exceptions | activation | delta L2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in STAGE_A_POLICY_ORDER:
        row = report["summary"]["by_policy"][policy]
        lines.append(
            f"| `{policy}` | {row['successes']} | {row['total']} | {row['task_balanced_success_rate']} | "
            f"{row['exception_count']} | {row.get('mean_activation_fraction')} | {row.get('mean_dagr_delta_l2')} |"
        )
    lines += [
        "",
        "## Paired Versus DAGR Full",
        "",
        "| baseline | pairs | wins | losses | ties | delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, row in report["paired_vs_dagr_full"].items():
        lines.append(
            f"| `{policy}` | {row['paired_count']} | {row['paired_win_count']} | {row['paired_loss_count']} | {row['paired_tie_count']} | {row['paired_success_delta']} |"
        )
    lines += ["", f"Next step: {report['next_step']}"]
    _write_md(path, lines)


def write_stage_b_result_md(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# DAGR-VLA Stage B Result",
        "",
        f"Date: `{report['date']}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- planned episodes: `{report['planned_episode_count']}`",
        f"- completed episodes: `{report['completed_episode_count']}`",
        f"- closed-loop experiment happened: `{report['closed_loop_experiment_happened']}`",
        f"- confirmatory-test tuning happened: `{report['confirmatory_test_tuning_happened']}`",
        f"- elapsed seconds: `{report['elapsed_seconds']}`",
        "",
        "## Policy Summary",
        "",
        "| policy | successes | total | task-balanced success | exceptions | activation | delta L2 | latency mean s | peak VRAM MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in STAGE_A_POLICY_ORDER:
        row = report["summary"]["by_policy"][policy]
        lines.append(
            f"| `{policy}` | {row['successes']} | {row['total']} | {row['task_balanced_success_rate']} | "
            f"{row['exception_count']} | {row.get('mean_activation_fraction')} | {row.get('mean_dagr_delta_l2')} | "
            f"{row['policy_latency_mean_s']} | {row['peak_vram_max_allocated_mb']} |"
        )
    lines += [
        "",
        "## Paired Versus DAGR Full",
        "",
        "| baseline | pairs | wins | losses | ties | delta | CI 95% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for policy, row in report["paired_vs_dagr_full"].items():
        ci = row.get("paired_bootstrap_ci") or [None, None]
        lines.append(
            f"| `{policy}` | {row['paired_count']} | {row['paired_win_count']} | {row['paired_loss_count']} | "
            f"{row['paired_tie_count']} | {row['paired_success_delta']} | [{ci[0]}, {ci[1]}] |"
        )
    lines += ["", f"Next step: {report['next_step']}"]
    _write_md(path, lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["plan", "preflight", "stage-a", "stage-b-plan", "stage-b"], default="plan")
    parser.add_argument("--date", default=DATE_KST)
    parser.add_argument("--base-path", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--checkpoint-root", default="/mnt/c/Users/jiheo/tca_map/runs/dagr_vla_checkpoints/dagr_a020_route_mlp")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--wsl-repo-root", default="/mnt/c/Users/jiheo/tca_map")
    parser.add_argument("--official-task-manifest", default="reports/official_closed_loop_task_manifest.json")
    parser.add_argument("--checkpoint-manifest", default="reports/dagr_vla/policy_checkpoint_manifest.json")
    parser.add_argument("--stable-artifact", default="reports/official_smolvla_stable_prediction_artifact.json")
    parser.add_argument("--stage-a-manifest", default="reports/dagr_vla/stage_a_manifest.json")
    parser.add_argument("--stage-a-manifest-md", default="reports/dagr_vla/stage_a_manifest.md")
    parser.add_argument("--stage-a-output", default="reports/dagr_vla/stage_a_result.json")
    parser.add_argument("--stage-a-md", default="reports/dagr_vla/stage_a_result.md")
    parser.add_argument("--stage-a-partial-output", default="reports/dagr_vla/stage_a_partial_result.json")
    parser.add_argument("--stage-a-preflight-output", default="reports/dagr_vla/stage_a_preflight.json")
    parser.add_argument("--stage-b-manifest", default="reports/dagr_vla/stage_b_manifest.json")
    parser.add_argument("--stage-b-manifest-md", default="reports/dagr_vla/stage_b_manifest.md")
    parser.add_argument("--stage-b-output", default="reports/dagr_vla/stage_b_result.json")
    parser.add_argument("--stage-b-md", default="reports/dagr_vla/stage_b_result.md")
    parser.add_argument("--stage-b-partial-output", default="reports/dagr_vla/stage_b_partial_result.json")
    parser.add_argument("--rerun-stage", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "plan":
        report = run_plan(args)
    elif args.mode == "preflight":
        report = run_preflight(args)
    elif args.mode == "stage-a":
        report = run_stage_a(args)
    elif args.mode == "stage-b-plan":
        report = run_stage_b_plan(args)
    else:
        report = run_stage_b(args)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "final_decision": report.get("final_decision"),
                "planned": report.get("planned_episode_count"),
                "completed": report.get("completed_episode_count"),
            },
            sort_keys=True,
        )
    )
    return 0 if report.get("final_decision") in FINAL_DECISIONS else 2


if __name__ == "__main__":
    raise SystemExit(main())
