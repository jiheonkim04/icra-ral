"""Run the frozen LIFT-VLA Stage 0 audit."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.datasets.lift_counterfactual_manifest import (  # noqa: E402
    PARTITIONS,
    build_counterfactual_manifest,
    parse_goal_tasks,
)
from tca_map.smolvla.lift_vla import (  # noqa: E402
    BASE,
    CAG,
    LAST_STEP_ABLATION,
    LIFT,
    postprocess_native,
    sample_smolvla_variant,
    unpad_native,
)


PROPOSAL_HASH = "3D263AA6FF73B342523D85AD4854145AF4D79DE2B90C6119F417D37A8B08F55F"
GUIDANCE_SCALES = (1.25, 1.5, 2.0)
IDENTITY_SCALE = 1.0
HEADROOM_DIAGNOSTIC_SCALE = 1.5
EXPECTED_NATIVE_SHAPE = (1, 50, 32)
EXPECTED_POLICY_SHAPE = (1, 50, 7)
EXPECTED_LANGUAGE_SHAPE = (1, 48)
MAX_MEMORY_BYTES = int(15.5 * 1024**3)
MAX_LATENCY_RATIO = 4.0
IDENTITY_TOLERANCE = 1e-5
ALLOWED_DECISIONS = {
    "LIFT_STAGE_0_PASS_TO_BOUNDED_VALIDATION",
    "LIFT_DATA_OR_BENCHMARK_FAILURE",
    "LIFT_NO_HEADROOM",
    "LIFT_IMPLEMENTATION_FAILURE",
    "LIFT_DESIGN_FAILURE_PRACTICAL_EQUIVALENCE",
    "LIFT_COMPUTE_INFEASIBLE",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return value.detach().to("cpu").tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _sha256_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _noise_seed(manifest_key: str, chunk_index: int = 0) -> int:
    digest = hashlib.sha256(f"{manifest_key}|chunk={chunk_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _rms(value: Any) -> float:
    import torch

    tensor = torch.as_tensor(value).float()
    return float(torch.sqrt(torch.mean(tensor.square())).item())


def _max_abs(left: Any, right: Any) -> float:
    import torch

    return float(torch.max(torch.abs(torch.as_tensor(left).float() - torch.as_tensor(right).float())).item())


def _finite_and_range_valid(value: Any) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float32)
    finite = np.isfinite(array)
    range_valid = finite & (array >= -1.0) & (array <= 1.0)
    return float(finite.mean()), float(range_valid.mean())


def _set_runtime_environment(args: argparse.Namespace) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ["LIBERO_CONFIG_PATH"] = str(Path(args.libero_config_dir))


class CounterfactualRuntime:
    """Resolve official task IDs and validate target scorers without policy use."""

    def __init__(self, tasks: list[dict[str, Any]], args: argparse.Namespace):
        from libero.libero import benchmark

        self.args = args
        self.tasks = tasks
        self.suite = benchmark.get_benchmark_dict()["libero_goal"]()
        self.benchmark_id_by_stem = {
            Path(task.bddl_file).stem: index for index, task in enumerate(self.suite.tasks)
        }
        self.cached_target_id: int | None = None
        self.cached_env: Any | None = None
        missing = sorted(task["task_id"] for task in tasks if task["task_id"] not in self.benchmark_id_by_stem)
        if missing:
            raise RuntimeError(f"sorted BDDL tasks absent from benchmark registry: {missing}")

    def benchmark_id(self, task_id: str) -> int:
        return int(self.benchmark_id_by_stem[task_id])

    def initial_state(self, source: Mapping[str, Any], state_index: int) -> Any:
        states = self.suite.get_task_init_states(self.benchmark_id(str(source["task_id"])))
        if state_index >= len(states):
            raise IndexError(f"source state index {state_index} exceeds {len(states)} states")
        return states[int(state_index)]

    def env_for_target(self, target_task: str) -> Any:
        from lerobot.envs.libero import LiberoEnv

        target_id = self.benchmark_id(target_task)
        if self.cached_target_id != target_id:
            if self.cached_env is not None:
                self.cached_env.close()
            self.cached_env = LiberoEnv(
                task_suite=self.suite,
                task_id=target_id,
                task_suite_name="libero_goal",
                episode_index=0,
                n_envs=1,
                camera_name="agentview_image,robot0_eye_in_hand_image",
                obs_type="pixels_agent_pos",
                render_mode="rgb_array",
                observation_width=256,
                observation_height=256,
                init_states=False,
                num_steps_wait=0,
                control_mode="relative",
            )
            self.cached_target_id = target_id
        return self.cached_env

    @staticmethod
    def _goal_grounding(inner: Any) -> bool:
        goal_state = list(inner.parsed_problem["goal_state"])
        interest = set(str(item) for item in inner.obj_of_interest)
        selected = [state for state in goal_state if interest.intersection(str(item) for item in state[1:])]
        if not selected:
            selected = goal_state
        return bool(selected) and all(bool(inner._eval_predicate(state)) for state in selected)

    def validate_row(self, row: dict[str, Any], source_state: Any) -> dict[str, Any]:
        errors: list[str] = []
        env = self.env_for_target(str(row["target_task"]))
        inner = env._env.env
        source_array = np.asarray(source_state)
        target_state = np.asarray(env._env.get_sim_state())
        parsed_goal = [[str(item) for item in state] for state in inner.parsed_problem["goal_state"]]
        object_state_names = set(str(name) for name in inner.object_states_dict)
        required_names = {str(item) for state in parsed_goal for item in state[1:]}
        if parsed_goal != row["target_goal_state"]:
            errors.append("instantiated_target_goal_mismatch")
        if source_array.shape != target_state.shape:
            errors.append("source_target_state_shape_mismatch")
        if not required_names.issubset(object_state_names):
            errors.append("target_goal_entity_missing_from_object_state_registry")
        if not callable(getattr(env._env, "check_success", None)):
            errors.append("target_success_scorer_not_callable")

        audit: dict[str, Any] = {
            "target_benchmark_task_id": self.benchmark_id(str(row["target_task"])),
            "source_benchmark_task_id": self.benchmark_id(str(row["source_task"])),
            "target_environment_problem": str(inner.parsed_problem.get("problem_name", "")),
            "instantiated_target_goal_state": parsed_goal,
            "source_state_shape_compatible": source_array.shape == target_state.shape,
            "target_goal_entities_instantiated": required_names.issubset(object_state_names),
            "target_success_scorer_callable": callable(getattr(env._env, "check_success", None)),
            "confirmatory_state_applied": False,
            "confirmatory_success_or_grounding_read": False,
            "errors": errors,
        }
        if row["evidence_partition"] != "confirmatory" and not errors:
            env._env.set_init_state(source_state)
            audit["source_state_applied"] = True
            audit["initial_target_success"] = bool(env._env.check_success())
            audit["initial_target_grounding"] = self._goal_grounding(inner)
        else:
            audit["source_state_applied"] = False
            audit["initial_target_success"] = None
            audit["initial_target_grounding"] = None
        audit["valid"] = not errors
        return audit

    def fresh_episode(self, row: Mapping[str, Any], source_state: Any) -> tuple[Any, dict[str, Any]]:
        from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action

        env = LiberoEnv(
            task_suite=self.suite,
            task_id=self.benchmark_id(str(row["target_task"])),
            task_suite_name="libero_goal",
            episode_index=0,
            n_envs=1,
            camera_name="agentview_image,robot0_eye_in_hand_image",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            observation_width=256,
            observation_height=256,
            init_states=False,
            num_steps_wait=0,
            control_mode="relative",
        )
        raw = env._env.set_init_state(source_state)
        for _ in range(10):
            raw, _, _, _ = env._env.step(get_libero_dummy_action())
        for robot in env._env.robots:
            robot.controller.use_delta = True
        return env, env._format_raw_obs(raw)

    def close(self) -> None:
        if self.cached_env is not None:
            try:
                self.cached_env.close()
            except Exception:
                pass
        self.cached_env = None
        self.cached_target_id = None


def _load_policy(args: argparse.Namespace) -> dict[str, Any]:
    from tca_map.smolvla.official_wsl_libero_rollout import (
        PolicySpec,
        _load_policy_and_processors,
    )

    loader_args = argparse.Namespace(
        base_path=str(args.checkpoint_path),
        lora_root="",
        libero_config_dir=str(args.libero_config_dir),
    )
    return _load_policy_and_processors(loader_args, PolicySpec("frozen_base"))


def _add_observation_batch(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _add_observation_batch(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value[None, ...]
    return value


def _make_batch(observation: Mapping[str, Any], task: str, loaded: Mapping[str, Any]) -> dict[str, Any]:
    from lerobot.scripts.lerobot_eval import preprocess_observation

    policy_observation = preprocess_observation(_add_observation_batch(dict(observation)))
    policy_observation["task"] = [str(task)]
    policy_observation = loaded["env_preprocessor"](policy_observation)
    return loaded["preprocessor"](policy_observation)


def _make_noise(policy: Any, seed: int) -> Any:
    import torch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed))
    return torch.randn(
        (1, int(policy.config.chunk_size), int(policy.config.max_action_dim)),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )


def _sample_once(
    loaded: Mapping[str, Any],
    conditioned_batch: dict[str, Any],
    empty_batch: dict[str, Any],
    noise: Any,
    variant: str,
    omega: float,
) -> tuple[Any, Any, dict[str, Any], float]:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        sample, prefix = sample_smolvla_variant(
            loaded["policy"],
            conditioned_batch,
            empty_batch,
            noise,
            variant=variant,
            omega=float(omega),
        )
        processed = postprocess_native(loaded["policy"], sample.native, loaded["postprocessor"])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return sample, processed, prefix, time.perf_counter() - started


def _tensor_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().to("cpu").numpy()
    return np.asarray(value, dtype=np.float32)


def _environment_chunk(processed: Any, loaded: Mapping[str, Any]) -> Any:
    from lerobot.utils.constants import ACTION

    return loaded["env_postprocessor"]({ACTION: processed})[ACTION]


def _smoke_audit(
    args: argparse.Namespace,
    runtime: CounterfactualRuntime,
    manifest: Mapping[str, Any],
    loaded: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    row = next(item for item in manifest["rows"] if item["evidence_partition"] == "discovery")
    source_task = next(task for task in runtime.tasks if task["task_id"] == row["source_task"])
    source_state = runtime.initial_state(source_task, int(row["source_initial_state_index"]))
    env, observation = runtime.fresh_episode(row, source_state)
    try:
        conditioned_batch = _make_batch(observation, str(row["target_language"]), loaded)
        empty_batch = _make_batch(observation, "", loaded)
    finally:
        env.close()

    policy = loaded["policy"]
    noise = _make_noise(policy, _noise_seed(str(row["manifest_key"])))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    base_samples = []
    base_processed = []
    base_executed = []
    base_latencies = []
    base_prefix = None
    for _ in range(3):
        sample, processed, prefix, latency = _sample_once(
            loaded, conditioned_batch, empty_batch, noise, BASE, IDENTITY_SCALE
        )
        base_samples.append(sample)
        base_processed.append(processed)
        base_executed.append(_environment_chunk(processed, loaded))
        base_latencies.append(latency)
        base_prefix = prefix

    identity, identity_processed, identity_prefix, identity_latency = _sample_once(
        loaded, conditioned_batch, empty_batch, noise, LIFT, IDENTITY_SCALE
    )
    identity_executed = _environment_chunk(identity_processed, loaded)
    identity_native_error = _max_abs(base_samples[0].native, identity.native)
    identity_exec_error = _max_abs(base_executed[0], identity_executed)

    scale_samples: dict[str, Any] = {}
    lift_latencies: list[float] = []
    for scale in GUIDANCE_SCALES:
        cag, cag_processed, cag_prefix, cag_latency = _sample_once(
            loaded, conditioned_batch, empty_batch, noise, CAG, scale
        )
        lift, lift_processed, lift_prefix, lift_latency = _sample_once(
            loaded, conditioned_batch, empty_batch, noise, LIFT, scale
        )
        ablation, ablation_processed, ablation_prefix, ablation_latency = _sample_once(
            loaded, conditioned_batch, empty_batch, noise, LAST_STEP_ABLATION, scale
        )
        lift_latencies.append(lift_latency)
        scale_samples[f"{scale:.2f}"] = {
            "cag": cag,
            "cag_processed": cag_processed,
            "cag_executed": _environment_chunk(cag_processed, loaded),
            "cag_prefix": cag_prefix,
            "cag_latency": cag_latency,
            "lift": lift,
            "lift_processed": lift_processed,
            "lift_executed": _environment_chunk(lift_processed, loaded),
            "lift_prefix": lift_prefix,
            "lift_latency": lift_latency,
            "ablation": ablation,
            "ablation_processed": ablation_processed,
            "ablation_executed": _environment_chunk(ablation_processed, loaded),
            "ablation_prefix": ablation_prefix,
            "ablation_latency": ablation_latency,
        }

    native_repeat_errors = [
        _rms(unpad_native(policy, sample.native) - unpad_native(policy, base_samples[0].native))
        for sample in base_samples[1:]
    ]
    exec_repeat_errors = [
        _rms(_tensor_array(executed)[0, 0] - _tensor_array(base_executed[0])[0, 0])
        for executed in base_executed[1:]
    ]
    e_native = float(np.quantile(native_repeat_errors or [0.0], 0.99))
    e_exec = float(np.quantile(exec_repeat_errors or [0.0], 0.99))
    s_native = float(statistics.median(_rms(unpad_native(policy, sample.native)) for sample in base_samples))
    s_exec = float(statistics.median(_rms(_tensor_array(executed)[0, 0]) for executed in base_executed))
    thresholds = {
        "source": "discovery_only_repeated_same_noise_base",
        "manifest_key": row["manifest_key"],
        "e_native": e_native,
        "s_native": s_native,
        "tau_native": max(100.0 * e_native, 0.01 * s_native, 1e-5),
        "e_exec": e_exec,
        "s_exec": s_exec,
        "tau_exec": max(100.0 * e_exec, 0.01 * s_exec, 1e-5),
        "formula": {
            "native": "max(100*e_native, 0.01*s_native, 1e-5)",
            "executed": "max(100*e_exec, 0.01*s_exec, 1e-5)",
        },
        "validation_data_used": False,
        "confirmatory_data_used": False,
    }

    separation: dict[str, Any] = {}
    action_arrays = [_tensor_array(item) for item in base_executed]
    for scale, items in scale_samples.items():
        lift_native = unpad_native(policy, items["lift"].native)
        cag_native = unpad_native(policy, items["cag"].native)
        ablation_native = unpad_native(policy, items["ablation"].native)
        lift_exec = _tensor_array(items["lift_executed"])[0, 0]
        cag_exec = _tensor_array(items["cag_executed"])[0, 0]
        ablation_exec = _tensor_array(items["ablation_executed"])[0, 0]
        base_exec = action_arrays[0][0, 0]
        finite_fraction, range_fraction = _finite_and_range_valid(_tensor_array(items["lift_executed"]))
        step_rms = [metric["conditioned_minus_empty_rms"] for metric in items["lift"].step_metrics]
        separation[scale] = {
            "lift_vs_cag_native_rms": _rms(lift_native - cag_native),
            "lift_vs_ablation_native_rms": _rms(lift_native - ablation_native),
            "lift_vs_cag_first_action_rms": _rms(lift_exec - cag_exec),
            "lift_vs_ablation_first_action_rms": _rms(lift_exec - ablation_exec),
            "translation_delta_from_base": (lift_exec[:3] - base_exec[:3]).tolist(),
            "rotation_delta_from_base": (lift_exec[3:6] - base_exec[3:6]).tolist(),
            "gripper_delta_from_base": float(lift_exec[6] - base_exec[6]),
            "conditioned_minus_empty_nonzero_step_fraction": float(
                np.mean(np.asarray(step_rms) > 0.0)
            ),
            "finite_fraction": finite_fraction,
            "range_valid_fraction": range_fraction,
            "field_evaluations": {
                "cag": items["cag"].field_evaluations,
                "lift": items["lift"].field_evaluations,
                "ablation": items["ablation"].field_evaluations,
            },
            "latency_seconds": {
                "cag": items["cag_latency"],
                "lift": items["lift_latency"],
                "ablation": items["ablation_latency"],
            },
            "above_discovery_thresholds": bool(
                _rms(lift_native - cag_native) > thresholds["tau_native"]
                and _rms(lift_native - ablation_native) > thresholds["tau_native"]
                and _rms(lift_exec - cag_exec) > thresholds["tau_exec"]
                and _rms(lift_exec - ablation_exec) > thresholds["tau_exec"]
            ),
        }

    language_shape = tuple(int(dim) for dim in conditioned_batch[OBS_LANGUAGE_TOKENS].shape)
    empty_language_shape = tuple(int(dim) for dim in empty_batch[OBS_LANGUAGE_TOKENS].shape)
    native_shape = tuple(int(dim) for dim in base_samples[0].native.shape)
    policy_shape = tuple(int(dim) for dim in _tensor_array(base_processed[0]).shape)
    field_counts_ok = all(
        details["field_evaluations"][name]["total"] == 20
        for details in separation.values()
        for name in ("cag", "lift", "ablation")
    )
    all_actions = action_arrays + [
        _tensor_array(items[name])
        for items in scale_samples.values()
        for name in ("cag_executed", "lift_executed", "ablation_executed")
    ]
    concatenated_actions = np.concatenate(all_actions, axis=1)
    finite_fraction, range_valid_fraction = _finite_and_range_valid(concatenated_actions)
    flat_actions = concatenated_actions.reshape(-1, concatenated_actions.shape[-1])
    base_median_latency = float(statistics.median(base_latencies))
    lift_median_latency = float(statistics.median(lift_latencies))
    latency_ratio = lift_median_latency / max(base_median_latency, 1e-12)
    memory = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
        "limit_bytes": MAX_MEMORY_BYTES,
    }
    smoke = {
        "manifest_key": row["manifest_key"],
        "noise_seed": _noise_seed(str(row["manifest_key"])),
        "shapes": {
            "native": list(native_shape),
            "policy_chunk": list(policy_shape),
            "conditioned_language_tokens": list(language_shape),
            "empty_language_tokens": list(empty_language_shape),
        },
        "empty_language": {
            "task_text": "",
            "token_ids": conditioned_batch[OBS_LANGUAGE_TOKENS].detach().to("cpu").tolist(),
            "empty_token_ids": empty_batch[OBS_LANGUAGE_TOKENS].detach().to("cpu").tolist(),
            "empty_attention_mask": empty_batch[OBS_LANGUAGE_ATTENTION_MASK].detach().to("cpu").tolist(),
        },
        "base_prefix": base_prefix,
        "identity_prefix": identity_prefix,
        "identity": {
            "omega": IDENTITY_SCALE,
            "native_max_abs_error": identity_native_error,
            "postprocessed_max_abs_error": identity_exec_error,
            "latency_seconds": identity_latency,
        },
        "repeatability": {
            "native_rms_errors": native_repeat_errors,
            "first_action_rms_errors": exec_repeat_errors,
        },
        "separation": separation,
        "action_validity": {
            "finite_fraction": finite_fraction,
            "range_valid_fraction": range_valid_fraction,
            "out_of_range_fraction": 1.0 - range_valid_fraction,
            "per_dimension_min": np.min(flat_actions, axis=0).tolist(),
            "per_dimension_max": np.max(flat_actions, axis=0).tolist(),
            "variant_specific_clipping_applied": False,
        },
        "latency": {
            "base_calls_seconds": base_latencies,
            "lift_calls_seconds": lift_latencies,
            "base_median_seconds": base_median_latency,
            "lift_median_seconds": lift_median_latency,
            "lift_over_base_ratio": latency_ratio,
            "limit_ratio": MAX_LATENCY_RATIO,
        },
        "cuda_memory": memory,
        "field_counts_ok": field_counts_ok,
        "shape_gate_passed": bool(
            native_shape == EXPECTED_NATIVE_SHAPE
            and policy_shape == EXPECTED_POLICY_SHAPE
            and language_shape == EXPECTED_LANGUAGE_SHAPE
            and empty_language_shape == EXPECTED_LANGUAGE_SHAPE
        ),
        "identity_gate_passed": bool(
            identity_native_error <= IDENTITY_TOLERANCE
            and identity_exec_error <= IDENTITY_TOLERANCE
        ),
        "activation_gate_passed": bool(
            all(item["conditioned_minus_empty_nonzero_step_fraction"] >= 0.8 for item in separation.values())
        ),
        "action_gate_passed": finite_fraction == 1.0 and range_valid_fraction == 1.0,
        "compute_gate_passed": bool(
            field_counts_ok
            and memory["peak_allocated_bytes"] < MAX_MEMORY_BYTES
            and latency_ratio <= MAX_LATENCY_RATIO
        ),
        "discovery_practical_separation_seen": any(
            item["above_discovery_thresholds"] for item in separation.values()
        ),
    }
    return smoke, thresholds


def _grounding(env: Any) -> bool:
    return CounterfactualRuntime._goal_grounding(env._env.env)


def _headroom_episode(
    row: Mapping[str, Any],
    source_state: Any,
    runtime: CounterfactualRuntime,
    loaded: Mapping[str, Any],
    variant: str,
    omega: float,
) -> dict[str, Any]:
    import torch
    from lerobot.utils.constants import ACTION

    env, observation = runtime.fresh_episode(row, source_state)
    queue: deque[np.ndarray] = deque()
    success = bool(env._env.check_success())
    grounding = _grounding(env)
    action_valid = True
    chunks = 0
    steps = 0
    error = None
    try:
        while steps < 300 and not success:
            if not queue:
                conditioned = _make_batch(observation, str(row["target_language"]), loaded)
                empty = _make_batch(observation, "", loaded)
                noise = _make_noise(loaded["policy"], _noise_seed(str(row["manifest_key"]), chunks))
                sample, processed, _, _ = _sample_once(
                    loaded, conditioned, empty, noise, variant, omega
                )
                expected_count = 10 if variant == BASE else 20
                if sample.field_evaluations["total"] != expected_count:
                    raise RuntimeError(f"{variant} field count changed during rollout")
                policy_chunk = _tensor_array(processed)[0]
                queue.extend(np.asarray(action, dtype=np.float32) for action in policy_chunk)
                chunks += 1
            policy_action = queue.popleft()
            finite, in_range = _finite_and_range_valid(policy_action)
            action_valid = action_valid and finite == 1.0 and in_range == 1.0
            action_tensor = torch.as_tensor(policy_action.reshape(1, 7), device="cuda", dtype=torch.float32)
            transition = loaded["env_postprocessor"]({ACTION: action_tensor})
            environment_action = transition[ACTION].detach().to("cpu").numpy().reshape(-1)
            observation, _, terminated, truncated, info = env.step(environment_action)
            success = bool(info.get("is_success", False))
            steps += 1
            if terminated or truncated:
                break
        if not success:
            grounding = _grounding(env)
        else:
            grounding = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
    return {
        "manifest_key": row["manifest_key"],
        "evidence_partition": row["evidence_partition"],
        "target_task": row["target_task"],
        "variant": variant,
        "omega": float(omega),
        "success": bool(success),
        "target_grounding": bool(grounding),
        "steps": int(steps),
        "chunks": int(chunks),
        "action_valid": bool(action_valid),
        "error": error,
    }


def _headroom_audit(
    manifest: Mapping[str, Any], runtime: CounterfactualRuntime, loaded: Mapping[str, Any]
) -> dict[str, Any]:
    rows = [
        row for row in manifest["rows"] if row["evidence_partition"] in {"discovery", "validation"}
    ]
    task_by_id = {task["task_id"]: task for task in runtime.tasks}
    results: list[dict[str, Any]] = []
    for row in rows:
        source_state = runtime.initial_state(
            task_by_id[row["source_task"]], int(row["source_initial_state_index"])
        )
        results.append(_headroom_episode(row, source_state, runtime, loaded, BASE, IDENTITY_SCALE))
        results.append(
            _headroom_episode(row, source_state, runtime, loaded, CAG, HEADROOM_DIAGNOSTIC_SCALE)
        )
    base = [item for item in results if item["variant"] == BASE]
    cag = [item for item in results if item["variant"] == CAG]
    errors = [item for item in results if item["error"]]
    base_failure = 1.0 - float(np.mean([item["success"] for item in base])) if base else 1.0
    cag_failure = 1.0 - float(np.mean([item["success"] for item in cag])) if cag else 1.0
    cag_grounding_misses = sum(not item["target_grounding"] for item in cag)
    return {
        "policy_scope": ["frozen_smolvla", "training_free_cag_proxy"],
        "lift_inference_performed": False,
        "cag_diagnostic_scale": HEADROOM_DIAGNOSTIC_SCALE,
        "cag_scale_role": "fixed middle-scale headroom diagnostic, not validation selection",
        "planned_rows": len(rows),
        "base_episode_count": len(base),
        "cag_episode_count": len(cag),
        "base_failure_rate": base_failure,
        "cag_residual_failure_rate": cag_failure,
        "cag_target_grounding_miss_count": int(cag_grounding_misses),
        "errors": errors,
        "results": results,
        "passed": bool(
            not errors
            and base_failure >= 0.20
            and cag_failure >= 0.10
            and cag_grounding_misses >= 1
            and all(item["action_valid"] for item in results)
        ),
    }


def _write_result_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# LIFT-VLA Stage 0 Result",
        "",
        f"- proposal hash: `{PROPOSAL_HASH}`",
        f"- final decision: `{report['final_decision']}`",
        f"- completed stage: `{report.get('completed_stage')}`",
        f"- confirmatory policy observations decoded: `{report['confirmatory_policy_observations_decoded']}`",
        f"- confirmatory policy actions computed: `{report['confirmatory_policy_actions_computed']}`",
        f"- elapsed seconds: `{report['elapsed_seconds']}`",
        "",
        "The decision follows the frozen Stage 0 order. No task, reset, scale, threshold, policy, or scorer was changed from the preregistered protocol.",
        "",
    ]
    _atomic_write_text(path, "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    report_dir = Path(args.report_dir)
    manifest_path = report_dir / "counterfactual_manifest.json"
    threshold_path = report_dir / "discovery_thresholds.json"
    result_path = report_dir / "stage_0_result.json"
    blocker_path = report_dir / "implementation_blocker.json"
    report: dict[str, Any] = {
        "schema_version": "lift-vla-stage0-v1",
        "proposal_hash": PROPOSAL_HASH,
        "mode": args.mode,
        "training_happened": False,
        "validation_search_happened": False,
        "confirmatory_policy_observations_decoded": 0,
        "confirmatory_policy_actions_computed": 0,
        "final_decision": None,
        "completed_stage": "preflight",
        "errors": [],
    }
    runtime = None
    try:
        _set_runtime_environment(args)
        bddl_root = Path(args.libero_root) / "libero" / "libero" / "bddl_files" / "libero_goal"
        tasks = parse_goal_tasks(bddl_root)
        report["static_audit"] = {
            "bddl_root": str(bddl_root),
            "sorted_task_count": len(tasks),
            "sorted_task_ids": [task["task_id"] for task in tasks],
            "scene_hash_count": len({task["scene_sha256"] for task in tasks}),
            "partition_rule": {name: list(indices) for name, indices in PARTITIONS.items()},
            "passed": len(tasks) == 10 and len({task["scene_sha256"] for task in tasks}) == 1,
        }
        if not report["static_audit"]["passed"]:
            raise RuntimeError("static BDDL or same-scene partition audit failed")
        report["completed_stage"] = "static_bddl_partition_audit"

        runtime = CounterfactualRuntime(tasks, args)
        manifest = build_counterfactual_manifest(
            tasks,
            runtime.initial_state,
            dynamic_validator=runtime.validate_row,
        )
        _atomic_write_json(manifest_path, manifest)
        report["manifest"] = {
            "path": str(manifest_path),
            "sha256": manifest["canonical_payload_sha256"],
            "row_count": manifest["row_count"],
            "valid_row_count": manifest["valid_row_count"],
            "ready": manifest["ready_for_stage_0_model_load"],
        }
        report["completed_stage"] = "source_scorer_manifest"
        if not manifest["ready_for_stage_0_model_load"]:
            report["final_decision"] = "LIFT_DATA_OR_BENCHMARK_FAILURE"
            return report

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU fallback is forbidden")
        loaded = _load_policy(args)
        report["policy_load_audit"] = loaded["audit"]
        report["completed_stage"] = "cuda_load_and_shape_check"
        smoke, thresholds = _smoke_audit(args, runtime, manifest, loaded)
        report["smoke"] = smoke
        _atomic_write_json(threshold_path, thresholds)
        report["thresholds"] = {"path": str(threshold_path), **thresholds}
        report["completed_stage"] = "discovery_mechanism_compute_smoke"

        if not smoke["shape_gate_passed"] or not smoke["identity_gate_passed"]:
            report["final_decision"] = "LIFT_IMPLEMENTATION_FAILURE"
            return report
        if not smoke["activation_gate_passed"]:
            report["final_decision"] = "LIFT_IMPLEMENTATION_FAILURE"
            return report
        if not smoke["action_gate_passed"] or not smoke["compute_gate_passed"]:
            report["final_decision"] = "LIFT_COMPUTE_INFEASIBLE"
            return report
        if not smoke["discovery_practical_separation_seen"]:
            report["final_decision"] = "LIFT_DESIGN_FAILURE_PRACTICAL_EQUIVALENCE"
            return report

        report["completed_stage"] = "discovery_thresholds_persisted"
        headroom = _headroom_audit(manifest, runtime, loaded)
        report["headroom"] = headroom
        report["completed_stage"] = "base_cag_development_headroom"
        if headroom["errors"]:
            report["final_decision"] = "LIFT_IMPLEMENTATION_FAILURE"
            return report
        if not headroom["passed"]:
            report["final_decision"] = "LIFT_NO_HEADROOM"
            return report
        report["final_decision"] = "LIFT_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
        report["completed_stage"] = "stage_0_adjudicated"
        return report
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        report["errors"].append(message)
        report["traceback_tail"] = traceback.format_exc().splitlines()[-30:]
        if report["completed_stage"] in {"preflight", "static_bddl_partition_audit", "source_scorer_manifest"}:
            report["final_decision"] = "LIFT_DATA_OR_BENCHMARK_FAILURE"
        elif "out of memory" in message.lower() or "cuda" in message.lower() and "memory" in message.lower():
            report["final_decision"] = "LIFT_COMPUTE_INFEASIBLE"
        else:
            report["final_decision"] = "LIFT_IMPLEMENTATION_FAILURE"
        _atomic_write_json(
            blocker_path,
            {
                "proposal_hash": PROPOSAL_HASH,
                "completed_stage": report["completed_stage"],
                "error": message,
                "traceback_tail": report.get("traceback_tail", []),
                "frozen_protocol_changed": False,
            },
        )
        return report
    finally:
        if runtime is not None:
            runtime.close()
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if report.get("final_decision") not in ALLOWED_DECISIONS:
            report["final_decision"] = "LIFT_IMPLEMENTATION_FAILURE"
        _atomic_write_json(result_path, report)
        _write_result_markdown(report_dir / "stage_0_result.md", report)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit"], default="audit")
    parser.add_argument("--checkpoint-path", default="/mnt/c/assets/checkpoints/smolvla")
    parser.add_argument("--libero-root", default="/mnt/c/assets/repos/LIBERO")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--report-dir", default="reports/lift_vla")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(
        json.dumps(
            {
                "final_decision": report["final_decision"],
                "completed_stage": report["completed_stage"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0 if report["final_decision"] in ALLOWED_DECISIONS else 2


if __name__ == "__main__":
    raise SystemExit(main())
