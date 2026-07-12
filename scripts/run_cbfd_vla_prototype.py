"""CBFD-VLA prototype runner."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_echo_vla_first_prototype import _postprocess_action, _preprocess_batch  # noqa: E402
from scripts.run_phase_barrier_vla_prototype import _make_exact_vector_env, _set_runtime_env, _step_success  # noqa: E402
from tca_map.openvla_oft_int4_gate import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_OPENVLA_REPO,
    _assert_no_offload,
    _autocast_payload,
    _cuda_memory_payload,
    _first_parameter_device,
    _hf_device_map,
    _rss_mib,
)
from tca_map.smolvla.cbfd_vla import (  # noqa: E402
    CBFDConfig,
    CBFDExample,
    file_sha256,
    load_cbfd_checkpoint,
    memory_action,
    predict_cbfd_action,
    save_cbfd_checkpoint,
    stage_a_decision,
    task_key,
    train_cbfd_policy,
)
from tca_map.smolvla.official_closed_loop_scaleup import _json_default  # noqa: E402
from tca_map.smolvla.official_wsl_libero_rollout import POLICIES, _cuda_memory, _load_policy_and_processors  # noqa: E402
from tca_map.smolvla.sacf_vla import instruction_to_demo_filename  # noqa: E402


DATE_KST = "2026-07-12"
BRANCH = "codex/autonomous-until-paper-governance-v2"
RESET_IDENTITY_BASE = 20260711
MAX_OFFICIAL_INITIAL_STATE_COUNT = 50
TASKS = [
    {
        "suite": "libero_spatial",
        "task_id": 4,
        "role": "stable_grasp_contact_transition",
        "instruction": "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    },
    {
        "suite": "libero_10",
        "task_id": 4,
        "role": "long_horizon_contact_and_release",
        "instruction": "put the white mug on the left plate and put the yellow and white mug on the right plate",
    },
]
TRAIN_IDENTITIES = [20260711, 20260712, 20260713, 20260714, 20260715]
EVAL_IDENTITIES = [20260716, 20260717, 20260718, 20260719, 20260720]
STAGE_B_IDENTITIES = list(range(20260721, 20260761))
VARIANTS = [
    "frozen_smolvla",
    "direct_distill_proxy",
    "teacher_trace_memory",
    "cbfd_no_retention",
    "cbfd_full",
]


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_md(path: Path, title: str, report: Mapping[str, Any]) -> None:
    lines = [
        f"# {title}",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Final decision: `{report.get('final_decision')}`",
        "",
        f"- mode: `{report.get('mode')}`",
        f"- training happened: `{report.get('training_happened')}`",
        f"- closed-loop experiment happened: `{report.get('closed_loop_experiment_happened')}`",
        f"- summary: `{report.get('summary')}`",
        f"- elapsed seconds: `{report.get('elapsed_seconds')}`",
        "",
        f"Next step: {report.get('next_step')}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _cuda_memory_report() -> dict[str, Any]:
    import torch

    return _cuda_memory(torch)


def _identity_to_initial_state_index(identity: int) -> int:
    index = int(identity) - RESET_IDENTITY_BASE
    if index < 0 or index >= MAX_OFFICIAL_INITIAL_STATE_COUNT:
        raise ValueError(f"identity {identity} maps to invalid official initial state index {index}")
    return index


def _state_from_prepared_or_obs(prepared: Mapping[str, Any] | None, obs: Mapping[str, Any], config: CBFDConfig) -> np.ndarray:
    if prepared is not None and "state" in prepared:
        return np.asarray(prepared["state"], dtype=np.float32).reshape(-1)[: int(config.state_dim)]
    parts: list[np.ndarray] = []
    for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_gripper_qvel"):
        if key in obs:
            parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
    if not parts:
        return np.zeros(int(config.state_dim), dtype=np.float32)
    return np.concatenate(parts, axis=0)[: int(config.state_dim)].astype(np.float32)


def _state_from_smolvla_observation(env: Any, observation: Any, loaded: Mapping[str, Any], config: CBFDConfig) -> np.ndarray:
    batch = _preprocess_batch(env, observation, dict(loaded))
    state = batch.get("observation.state")
    if state is None:
        raise RuntimeError("preprocessed observation has no observation.state")
    try:
        state_np = state.detach().to("cpu").numpy()
    except AttributeError:
        state_np = np.asarray(state)
    return np.asarray(state_np, dtype=np.float32).reshape(-1)[: int(config.state_dim)]


def _policy_action(policy: Any, env: Any, observation: Any, loaded: Mapping[str, Any]) -> np.ndarray:
    import torch

    batch = _preprocess_batch(env, observation, dict(loaded))
    with torch.inference_mode():
        action = policy.select_action(batch)
    return _postprocess_action(action, dict(loaded)).reshape(1, -1)


def _synthetic_examples(count: int, config: CBFDConfig) -> list[CBFDExample]:
    rows: list[CBFDExample] = []
    for index in range(int(count)):
        frac = (index % 20) / 19.0
        key = "libero_spatial/task_4" if index % 2 == 0 else "libero_10/task_4"
        code = -1.0 if "spatial" in key else 1.0
        state = np.asarray([frac, code, np.sin(frac), np.cos(frac), 0.01 * index, 0.0, 0.0, 1.0], dtype=np.float32)
        teacher = np.asarray([0.28 * code, 0.5 * frac, -0.25 * frac, 0.0, 0.0, 0.1, -0.6], dtype=np.float32)
        rows.append(CBFDExample(state=state.tolist(), action=teacher.tolist(), task_key=key, step_fraction=frac, source="teacher", failure_weight=1.0))
        retention = np.asarray([0.05 * code, 0.02, 0.0, 0.0, 0.0, 0.0, -0.1], dtype=np.float32)
        rows.append(CBFDExample(state=(state + 0.015).tolist(), action=retention.tolist(), task_key=key, step_fraction=frac, source="retention", failure_weight=1.0))
    return rows


def _synthetic_mode(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    config = CBFDConfig(hidden_dim=int(args.hidden_dim))
    rows = _synthetic_examples(int(args.synthetic_rows), config)
    direct, direct_stats = train_cbfd_policy(rows, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=1, include_retention=False, use_failure_weights=False)
    no_retention, no_retention_stats = train_cbfd_policy(rows, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=2, include_retention=False, use_failure_weights=True)
    full, full_stats = train_cbfd_policy(rows, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=3, include_retention=True, use_failure_weights=True)
    probe = rows[0]
    memory, _ = memory_action(rows, state=probe.state, step_fraction=probe.step_fraction, task_key_value=probe.task_key, config=config)
    predictions = {
        "direct_distill_proxy": predict_cbfd_action(direct, state=probe.state, step_fraction=probe.step_fraction, task_key_value=probe.task_key).tolist(),
        "cbfd_no_retention": predict_cbfd_action(no_retention, state=probe.state, step_fraction=probe.step_fraction, task_key_value=probe.task_key).tolist(),
        "cbfd_full": predict_cbfd_action(full, state=probe.state, step_fraction=probe.step_fraction, task_key_value=probe.task_key).tolist(),
        "teacher_trace_memory": memory.tolist(),
    }
    finite = all(np.isfinite(np.asarray(value, dtype=np.float32)).all() for value in predictions.values())
    passed = bool(direct_stats["loss_decreased"] and no_retention_stats["loss_decreased"] and full_stats["loss_decreased"] and finite)
    return {
        "mode": "synthetic",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": True,
        "closed_loop_experiment_happened": False,
        "config": config.to_json(),
        "stats": {
            "direct_distill_proxy": direct_stats,
            "cbfd_no_retention": no_retention_stats,
            "cbfd_full": full_stats,
        },
        "predictions": predictions,
        "summary": {"synthetic_passed": passed, "all_predictions_finite": finite},
        "final_decision": "SYNTHETIC_MECHANISM_PASS" if passed else "SYNTHETIC_MECHANISM_FAIL",
        "next_step": "Run teacher acquisition." if passed else "Repair or kill CBFD implementation.",
        "elapsed_seconds": _round(time.time() - started, 3),
    }


def _planned_teacher_rows(tasks: list[Mapping[str, Any]], identities: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for identity in identities:
            rows.append(
                {
                    "suite": str(task["suite"]),
                    "task_id": int(task["task_id"]),
                    "task_key": task_key(str(task["suite"]), int(task["task_id"])),
                    "role": str(task["role"]),
                    "instruction": str(task["instruction"]),
                    "identity": int(identity),
                    "initial_state_index": _identity_to_initial_state_index(int(identity)),
                }
            )
    return rows


def _teacher_acquisition_mode(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from libero.libero import benchmark

    sys.path.insert(0, str(Path(args.openvla_repo)))
    import experiments.robot.openvla_utils as official_openvla_utils

    official_openvla_utils.update_auto_map = lambda pretrained_checkpoint: None
    official_openvla_utils.check_model_logic_mismatch = lambda pretrained_checkpoint: None

    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
    from experiments.robot.libero.run_libero_eval import TASK_MAX_STEPS, GenerateConfig, check_unnorm_key, initialize_model, prepare_observation, process_action
    from experiments.robot.robot_utils import get_action, get_image_resize_size

    started = time.time()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(args.libero_config_dir))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    config = CBFDConfig(hidden_dim=int(args.hidden_dim))
    planned = _planned_teacher_rows(TASKS[: int(args.max_tasks)], TRAIN_IDENTITIES[: int(args.train_identities)])
    partial_path = Path(args.teacher_partial_output)
    episodes: list[dict[str, Any]] = []
    if partial_path.exists() and not bool(args.rerun_teacher_acquisition):
        prior = json.loads(partial_path.read_text(encoding="utf-8-sig"))
        episodes = list(prior.get("episodes") or [])
    completed_keys = {(row.get("task_key"), int(row.get("identity", -1))) for row in episodes}
    cfg = GenerateConfig(
        pretrained_checkpoint=str(Path(args.openvla_checkpoint_dir)),
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=8,
        load_in_4bit=True,
        load_in_8bit=False,
        task_suite_name="libero_spatial",
        num_trials_per_task=1,
        seed=int(args.seed),
    )
    model = None
    env = None
    errors: list[dict[str, Any]] = []
    model_parameter_summary: dict[str, Any] | None = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CPU_FALLBACK_BUG: CUDA unavailable before OpenVLA teacher load")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
        resize_size = get_image_resize_size(cfg)
        device_map = _hf_device_map(model)
        offload_errors = _assert_no_offload(device_map)
        if offload_errors:
            raise RuntimeError("OFFLOAD_FORBIDDEN: " + "; ".join(offload_errors))
        model_parameter_summary = _first_parameter_device(model)
        benchmark_dict = benchmark.get_benchmark_dict()
        suite_cache: dict[str, Any] = {}
        for planned_row in planned:
            key = (planned_row["task_key"], int(planned_row["identity"]))
            if key in completed_keys:
                continue
            row = {**planned_row, "success": False, "exception": None, "trace_rows": []}
            try:
                cfg.task_suite_name = str(planned_row["suite"])
                check_unnorm_key(cfg, model)
                if cfg.task_suite_name not in suite_cache:
                    suite_cache[cfg.task_suite_name] = benchmark_dict[cfg.task_suite_name]()
                task_suite = suite_cache[cfg.task_suite_name]
                task = task_suite.get_task(int(planned_row["task_id"]))
                initial_state = task_suite.get_task_init_states(int(planned_row["task_id"]))[int(planned_row["initial_state_index"])]
                env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
                env.reset()
                obs = env.set_init_state(initial_state)
                action_queue: deque[Any] = deque(maxlen=cfg.num_open_loop_steps)
                max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
                if int(args.max_eval_steps) > 0:
                    max_steps = min(max_steps, int(args.max_eval_steps))
                policy_latencies: list[float] = []
                final_reward = 0.0
                episode_success = False
                t = 0
                while t < max_steps + cfg.num_steps_wait:
                    if t < cfg.num_steps_wait:
                        obs, reward, done, _info = env.step(get_libero_dummy_action(cfg.model_family))
                        final_reward = float(reward)
                        t += 1
                        continue
                    observation, _img = prepare_observation(obs, resize_size)
                    state = _state_from_prepared_or_obs(observation, obs, config)
                    if len(action_queue) == 0:
                        policy_started = time.perf_counter()
                        actions = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                            action_head=action_head,
                            proprio_projector=proprio_projector,
                            noisy_action_projector=noisy_action_projector,
                            use_film=False,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        policy_latencies.append(time.perf_counter() - policy_started)
                        action_queue.extend(actions)
                    raw_action = action_queue.popleft()
                    env_action = np.asarray(process_action(raw_action, cfg.model_family), dtype=np.float32).reshape(-1)[: int(config.action_dim)]
                    step_fraction = float(t - cfg.num_steps_wait) / max(1.0, float(max_steps - 1))
                    row["trace_rows"].append(
                        {
                            "state": [float(value) for value in state.tolist()],
                            "action": [float(value) for value in env_action.tolist()],
                            "step_fraction": _round(step_fraction, 6),
                            "task_key": planned_row["task_key"],
                            "identity": int(planned_row["identity"]),
                        }
                    )
                    obs, reward, done, info = env.step(env_action.tolist())
                    final_reward = float(reward)
                    episode_success = bool(episode_success or done)
                    if done:
                        break
                    t += 1
                row.update(
                    {
                        "success": bool(episode_success),
                        "task_description": task_description,
                        "trace_row_count": len(row["trace_rows"]),
                        "final_reward": final_reward,
                        "policy_latency_mean_s": _round(float(np.mean(policy_latencies)) if policy_latencies else 0.0, 6),
                        "policy_latency_max_s": _round(float(np.max(policy_latencies)) if policy_latencies else 0.0, 6),
                    }
                )
            except Exception as exc:  # pragma: no cover - runtime boundary
                row["exception"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                errors.append({"task_key": row["task_key"], "identity": row["identity"], "exception": row["exception"]})
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                env = None
                episodes.append(row)
                _write_json(partial_path, {"episodes": episodes, "planned_episode_count": len(planned)})
    except Exception as exc:  # pragma: no cover - runtime boundary
        errors.append({"top_level_exception": "".join(traceback.format_exception_only(type(exc), exc)).strip(), "traceback": traceback.format_exc().splitlines()[-24:]})
    finally:
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass
    successful = [row for row in episodes if row.get("success") and not row.get("exception")]
    trace_count = int(sum(len(row.get("trace_rows") or []) for row in successful))
    passed = bool(len(episodes) == len(planned) and not errors and len(successful) == len(planned) and trace_count > 0)
    report = {
        "mode": "teacher-acquisition",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": False,
        "closed_loop_experiment_happened": True,
        "teacher_policy": "quantized_openvla_oft_int4",
        "episodes": episodes,
        "errors": errors,
        "summary": {
            "episodes_completed": len(episodes),
            "episodes_planned": len(planned),
            "successful_episodes": len(successful),
            "teacher_trace_rows": trace_count,
        },
        "final_decision": "TEACHER_ACQUISITION_PASS" if passed else "TEACHER_ACQUISITION_FAIL",
        "next_step": "Run student training." if passed else "Repair or stop CBFD teacher path.",
        "elapsed_seconds": _round(time.time() - started, 3),
        "cuda_memory": _cuda_memory_payload(torch),
        "rss_mib": _rss_mib(),
        "autocast": _autocast_payload(torch),
        "model_parameter": model_parameter_summary,
    }
    return report


def _examples_from_teacher_report(path: Path, config: CBFDConfig) -> list[CBFDExample]:
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    rows: list[CBFDExample] = []
    for episode in report.get("episodes") or []:
        if not episode.get("success") or episode.get("exception"):
            continue
        for item in episode.get("trace_rows") or []:
            rows.append(
                CBFDExample(
                    state=[float(x) for x in item["state"][: int(config.state_dim)]],
                    action=[float(x) for x in item["action"][: int(config.action_dim)]],
                    task_key=str(item["task_key"]),
                    step_fraction=float(item["step_fraction"]),
                    source="teacher",
                    failure_weight=1.0,
                )
            )
    return rows


def _hdf5_state(obs_group: Any, index: int, config: CBFDConfig) -> np.ndarray:
    ee = np.asarray(obs_group["ee_states"][index], dtype=np.float32).reshape(-1)
    gripper = np.asarray(obs_group["gripper_states"][index], dtype=np.float32).reshape(-1)
    return np.concatenate([ee, gripper], axis=0)[: int(config.state_dim)].astype(np.float32)


def _resolve_demo_path(data_root: Path, suite: str, instruction: str) -> Path:
    filename = instruction_to_demo_filename(instruction)
    direct = data_root / suite / filename
    if direct.exists():
        return direct
    matches = sorted((data_root / suite).glob(f"*_{filename}"))
    if matches:
        return matches[0]
    return direct


def _collect_retention_examples(args: argparse.Namespace, config: CBFDConfig) -> tuple[list[CBFDExample], dict[str, Any]]:
    import h5py

    data_root = Path(args.libero_data_root)
    rows: list[CBFDExample] = []
    coverage: dict[str, Any] = {}
    for task in TASKS[: int(args.max_tasks)]:
        suite = str(task["suite"])
        key = task_key(suite, int(task["task_id"]))
        path = _resolve_demo_path(data_root, suite, str(task["instruction"]))
        count = 0
        if not path.exists():
            coverage[key] = {"path": str(path), "exists": False, "rows": 0}
            continue
        with h5py.File(path, "r") as handle:
            for demo_name in sorted(handle["data"].keys()):
                group = handle["data"][demo_name]
                actions = np.asarray(group["actions"], dtype=np.float32)
                obs = group["obs"]
                stride = max(1, int(np.ceil(actions.shape[0] / max(1, int(args.retention_rows_per_task)))))
                for step in range(0, actions.shape[0], stride):
                    if count >= int(args.retention_rows_per_task):
                        break
                    frac = float(step) / max(1.0, float(actions.shape[0] - 1))
                    rows.append(
                        CBFDExample(
                            state=[float(x) for x in _hdf5_state(obs, step, config)],
                            action=[float(x) for x in np.clip(actions[step].reshape(-1)[: int(config.action_dim)], -1.0, 1.0)],
                            task_key=key,
                            step_fraction=frac,
                            source="retention",
                            failure_weight=1.0,
                        )
                    )
                    count += 1
                if count >= int(args.retention_rows_per_task):
                    break
        coverage[key] = {"path": str(path), "exists": True, "rows": count}
    return rows, coverage


def _student_train_mode(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    config = CBFDConfig(hidden_dim=int(args.hidden_dim), failure_weight=float(args.failure_weight), retention_weight=float(args.retention_weight))
    teacher_rows = _examples_from_teacher_report(Path(args.teacher_acquisition_output), config)
    retention_rows, retention_coverage = _collect_retention_examples(args, config)
    if len(teacher_rows) < 10:
        raise RuntimeError(f"not enough teacher rows for CBFD training: {len(teacher_rows)}")
    examples = teacher_rows + retention_rows
    direct, direct_stats = train_cbfd_policy(examples, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=31, include_retention=False, use_failure_weights=False)
    no_retention, no_retention_stats = train_cbfd_policy(examples, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=32, include_retention=False, use_failure_weights=True)
    full, full_stats = train_cbfd_policy(examples, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=33, include_retention=True, use_failure_weights=True)
    paths = {
        "direct_distill_proxy": Path(args.direct_checkpoint),
        "cbfd_no_retention": Path(args.no_retention_checkpoint),
        "cbfd_full": Path(args.full_checkpoint),
    }
    save_cbfd_checkpoint(paths["direct_distill_proxy"], direct, direct_stats)
    save_cbfd_checkpoint(paths["cbfd_no_retention"], no_retention, no_retention_stats)
    save_cbfd_checkpoint(paths["cbfd_full"], full, full_stats)
    passed = bool(direct_stats["loss_decreased"] and no_retention_stats["loss_decreased"] and full_stats["loss_decreased"] and len(retention_rows) > 0)
    return {
        "mode": "student-train",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": True,
        "closed_loop_experiment_happened": False,
        "config": config.to_json(),
        "teacher_rows": len(teacher_rows),
        "retention_rows": len(retention_rows),
        "retention_coverage": retention_coverage,
        "stats": {
            "direct_distill_proxy": direct_stats,
            "cbfd_no_retention": no_retention_stats,
            "cbfd_full": full_stats,
        },
        "checkpoints": {name: str(path) for name, path in paths.items()},
        "checkpoint_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "summary": {"student_train_passed": passed},
        "final_decision": "STUDENT_TRAIN_PASS" if passed else "STUDENT_TRAIN_FAIL",
        "next_step": "Run Stage A." if passed else "Repair or kill student training.",
        "elapsed_seconds": _round(time.time() - started, 3),
    }


def _planned_stage_rows(tasks: list[Mapping[str, Any]], identities: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for task in tasks:
            for identity in identities:
                rows.append(
                    {
                        "variant": variant,
                        "suite": str(task["suite"]),
                        "task_id": int(task["task_id"]),
                        "task_key": task_key(str(task["suite"]), int(task["task_id"])),
                        "role": str(task["role"]),
                        "instruction": str(task["instruction"]),
                        "identity": int(identity),
                    }
                )
    return rows


def _load_train_examples_for_memory(args: argparse.Namespace, config: CBFDConfig) -> list[CBFDExample]:
    teacher_rows = _examples_from_teacher_report(Path(args.teacher_acquisition_output), config)
    if not teacher_rows:
        raise RuntimeError("teacher memory has no rows")
    return teacher_rows


def _run_student_episode(
    *,
    row: Mapping[str, Any],
    loaded: Mapping[str, Any],
    config: CBFDConfig,
    direct_model: Any,
    no_retention_model: Any,
    full_model: Any,
    memory_rows: list[CBFDExample],
    max_eval_steps: int,
) -> dict[str, Any]:
    env = None
    started = time.time()
    action_deltas_full_direct: list[float] = []
    action_deltas_full_memory: list[float] = []
    memory_scores: list[float] = []
    try:
        env = _make_exact_vector_env(str(row["suite"]), int(row["task_id"]), _identity_to_initial_state_index(int(row["identity"])))
        policy = loaded["policy"]
        if hasattr(policy, "reset"):
            policy.reset()
        observation, _ = env.reset(seed=[int(row["identity"])])
        max_steps = int(env.call("_max_episode_steps")[0])
        if int(max_eval_steps) > 0:
            max_steps = min(max_steps, int(max_eval_steps))
        success = False
        rewards: list[float] = []
        policy_latencies: list[float] = []
        steps = 0
        for step in range(max_steps):
            step_fraction = float(step) / max(1.0, float(max_steps - 1))
            state = _state_from_smolvla_observation(env, observation, loaded, config)
            start_policy = time.perf_counter()
            if row["variant"] == "frozen_smolvla":
                action = _policy_action(policy, env, observation, loaded).reshape(-1).astype(np.float32)
            elif row["variant"] == "direct_distill_proxy":
                action = predict_cbfd_action(direct_model, state=state, step_fraction=step_fraction, task_key_value=str(row["task_key"]))
            elif row["variant"] == "teacher_trace_memory":
                action, diag = memory_action(memory_rows, state=state, step_fraction=step_fraction, task_key_value=str(row["task_key"]), config=config)
                memory_scores.append(float(diag["memory_score"]))
            elif row["variant"] == "cbfd_no_retention":
                action = predict_cbfd_action(no_retention_model, state=state, step_fraction=step_fraction, task_key_value=str(row["task_key"]))
            elif row["variant"] == "cbfd_full":
                action = predict_cbfd_action(full_model, state=state, step_fraction=step_fraction, task_key_value=str(row["task_key"]))
            else:
                raise ValueError(f"unknown variant: {row['variant']}")
            if row["variant"] == "cbfd_full":
                direct_action = predict_cbfd_action(direct_model, state=state, step_fraction=step_fraction, task_key_value=str(row["task_key"]))
                memory_action_value, memory_diag = memory_action(memory_rows, state=state, step_fraction=step_fraction, task_key_value=str(row["task_key"]), config=config)
                action_deltas_full_direct.append(float(np.linalg.norm(action - direct_action)))
                action_deltas_full_memory.append(float(np.linalg.norm(action - memory_action_value)))
                memory_scores.append(float(memory_diag["memory_score"]))
            policy_latencies.append(time.perf_counter() - start_policy)
            observation, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32).reshape(1, -1))
            rewards.append(float(np.asarray(reward).reshape(-1)[0]))
            steps = int(step + 1)
            success = bool(success or _step_success(info))
            if success or np.all(terminated | truncated):
                break
        return {
            **dict(row),
            "success": bool(success),
            "exception": None,
            "episode_steps": steps,
            "reward_sum": _round(float(np.sum(rewards)) if rewards else 0.0, 6),
            "policy_latency_mean_s": _round(float(np.mean(policy_latencies)) if policy_latencies else 0.0, 6),
            "policy_latency_max_s": _round(float(np.max(policy_latencies)) if policy_latencies else 0.0, 6),
            "mean_action_delta_full_vs_direct": _round(float(np.mean(action_deltas_full_direct)) if action_deltas_full_direct else 0.0, 6),
            "mean_action_delta_full_vs_memory": _round(float(np.mean(action_deltas_full_memory)) if action_deltas_full_memory else 0.0, 6),
            "mean_memory_score": _round(float(np.mean(memory_scores)) if memory_scores else 0.0, 6),
            "elapsed_seconds": _round(time.time() - started, 3),
            "cuda_memory": _cuda_memory_report(),
        }
    except Exception as exc:  # pragma: no cover - real rollout boundary
        return {
            **dict(row),
            "success": False,
            "exception": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            "episode_steps": 0,
            "elapsed_seconds": _round(time.time() - started, 3),
            "cuda_memory": _cuda_memory_report(),
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _summarize_stage(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_rows = [row for row in rows if row.get("variant") == variant]
        successes = int(sum(1 for row in variant_rows if bool(row.get("success"))))
        total = int(len(variant_rows))
        per_task: dict[str, Any] = {}
        for key in sorted({str(row.get("task_key")) for row in variant_rows}):
            task_rows = [row for row in variant_rows if str(row.get("task_key")) == key]
            task_successes = int(sum(1 for row in task_rows if bool(row.get("success"))))
            per_task[key] = {"successes": task_successes, "total": len(task_rows), "rate": _round(task_successes / max(1, len(task_rows)), 6)}
        task_balanced = float(np.mean([item["rate"] for item in per_task.values()])) if per_task else 0.0
        by_variant[variant] = {
            "successes": successes,
            "total": total,
            "success_rate": _round(successes / max(1, total), 6),
            "task_balanced_success_rate": _round(task_balanced, 6),
            "per_task": per_task,
            "exceptions": int(sum(1 for row in variant_rows if row.get("exception"))),
            "policy_latency_mean_s": _round(float(np.mean([float(row.get("policy_latency_mean_s", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
            "peak_cuda_allocated_mb": _round(
                float(np.max([float((row.get("cuda_memory") or {}).get("max_allocated_mb", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0,
                3,
            ),
            "mean_action_delta_full_vs_direct": _round(float(np.mean([float(row.get("mean_action_delta_full_vs_direct", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
            "mean_action_delta_full_vs_memory": _round(float(np.mean([float(row.get("mean_action_delta_full_vs_memory", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
        }
    strongest_name = max((name for name in VARIANTS if name != "cbfd_full"), key=lambda name: by_variant[name]["task_balanced_success_rate"])
    mechanism_active = bool(
        by_variant["cbfd_full"]["mean_action_delta_full_vs_direct"] > 1e-4
        or by_variant["cbfd_full"]["mean_action_delta_full_vs_memory"] > 1e-4
    )
    return {
        "by_variant": by_variant,
        "strongest_baseline": strongest_name,
        "mechanism_active": mechanism_active,
        "exception_count": int(sum(1 for row in rows if row.get("exception"))),
    }


def _stage_a_mode(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    _set_runtime_env(args)
    args.base_path = str(Path(args.checkpoint))
    args.lora_root = getattr(args, "lora_root", "/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    config = CBFDConfig(hidden_dim=int(args.hidden_dim), failure_weight=float(args.failure_weight), retention_weight=float(args.retention_weight))
    direct_model, direct_stats = load_cbfd_checkpoint(args.direct_checkpoint)
    no_retention_model, no_retention_stats = load_cbfd_checkpoint(args.no_retention_checkpoint)
    full_model, full_stats = load_cbfd_checkpoint(args.full_checkpoint)
    memory_rows = _load_train_examples_for_memory(args, config)
    loaded = _load_policy_and_processors(args, POLICIES[0])
    planned = _planned_stage_rows(TASKS[: int(args.max_tasks)], EVAL_IDENTITIES[: int(args.eval_identities)])
    partial_path = Path(args.stage_a_partial_output)
    episodes: list[dict[str, Any]] = []
    if partial_path.exists() and not bool(args.rerun_stage_a):
        partial = json.loads(partial_path.read_text(encoding="utf-8-sig"))
        episodes = list(partial.get("episodes") or [])
    completed = {(row.get("variant"), row.get("task_key"), int(row.get("identity", -1))) for row in episodes}
    for row in planned:
        key = (row["variant"], row["task_key"], int(row["identity"]))
        if key in completed:
            continue
        result = _run_student_episode(
            row=row,
            loaded=loaded,
            config=config,
            direct_model=direct_model,
            no_retention_model=no_retention_model,
            full_model=full_model,
            memory_rows=memory_rows,
            max_eval_steps=int(args.max_eval_steps),
        )
        episodes.append(result)
        _write_json(partial_path, {"episodes": episodes, "planned_episode_count": len(planned)})
    summary = _summarize_stage(episodes)
    final = stage_a_decision(summary, strongest_baseline=str(summary["strongest_baseline"]))
    return {
        "mode": "stage-a",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": False,
        "closed_loop_experiment_happened": True,
        "config": config.to_json(),
        "planned_episode_count": len(planned),
        "completed_episode_count": len(episodes),
        "episodes": episodes,
        "checkpoint_stats": {
            "direct_distill_proxy": direct_stats,
            "cbfd_no_retention": no_retention_stats,
            "cbfd_full": full_stats,
        },
        "summary": summary,
        "final_decision": final,
        "next_step": "Run Stage B." if "STAGE_B_REQUIRED" in final else "Archive or repair according to governance.",
        "elapsed_seconds": _round(time.time() - started, 3),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "synthetic":
        report = _synthetic_mode(args)
        _write_json(Path(args.synthetic_output), report)
        _write_md(Path(args.synthetic_md), "CBFD-VLA Synthetic Result", report)
        return report
    if args.mode == "teacher-acquisition":
        report = _teacher_acquisition_mode(args)
        _write_json(Path(args.teacher_acquisition_output), report)
        _write_md(Path(args.teacher_acquisition_md), "CBFD-VLA Teacher Acquisition Result", report)
        return report
    if args.mode == "student-train":
        report = _student_train_mode(args)
        _write_json(Path(args.student_train_output), report)
        _write_md(Path(args.student_train_md), "CBFD-VLA Student Training Result", report)
        return report
    if args.mode == "stage-a":
        report = _stage_a_mode(args)
        _write_json(Path(args.stage_a_output), report)
        _write_md(Path(args.stage_a_md), "CBFD-VLA Stage A Result", report)
        return report
    raise ValueError(f"unknown mode: {args.mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "teacher-acquisition", "student-train", "stage-a"], required=True)
    parser.add_argument("--checkpoint", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--libero-data-root", default="/mnt/c/assets/datasets/lerobot_libero")
    parser.add_argument("--openvla-checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--openvla-repo", default=str(DEFAULT_OPENVLA_REPO))
    parser.add_argument("--synthetic-output", default="reports/cbfd_vla/synthetic_result.json")
    parser.add_argument("--synthetic-md", default="reports/cbfd_vla/synthetic_result.md")
    parser.add_argument("--teacher-acquisition-output", default="reports/cbfd_vla/teacher_acquisition_result.json")
    parser.add_argument("--teacher-acquisition-md", default="reports/cbfd_vla/teacher_acquisition_result.md")
    parser.add_argument("--teacher-partial-output", default="reports/cbfd_vla/teacher_acquisition_partial_result.json")
    parser.add_argument("--student-train-output", default="reports/cbfd_vla/student_train_result.json")
    parser.add_argument("--student-train-md", default="reports/cbfd_vla/student_train_result.md")
    parser.add_argument("--stage-a-output", default="reports/cbfd_vla/stage_a_result.json")
    parser.add_argument("--stage-a-md", default="reports/cbfd_vla/stage_a_result.md")
    parser.add_argument("--stage-a-partial-output", default="reports/cbfd_vla/stage_a_partial_result.json")
    parser.add_argument("--direct-checkpoint", default="reports/cbfd_vla/checkpoints/direct_distill_proxy.pt")
    parser.add_argument("--no-retention-checkpoint", default="reports/cbfd_vla/checkpoints/cbfd_no_retention.pt")
    parser.add_argument("--full-checkpoint", default="reports/cbfd_vla/checkpoints/cbfd_full.pt")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--failure-weight", type=float, default=3.0)
    parser.add_argument("--retention-weight", type=float, default=1.0)
    parser.add_argument("--synthetic-rows", type=int, default=96)
    parser.add_argument("--retention-rows-per-task", type=int, default=96)
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--train-identities", type=int, default=5)
    parser.add_argument("--eval-identities", type=int, default=5)
    parser.add_argument("--max-eval-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--rerun-teacher-acquisition", action="store_true")
    parser.add_argument("--rerun-stage-a", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if int(args.max_tasks) < 1 or int(args.max_tasks) > len(TASKS):
        raise SystemExit("--max-tasks must be between 1 and 2")
    if int(args.train_identities) < 1 or int(args.train_identities) > len(TRAIN_IDENTITIES):
        raise SystemExit(f"--train-identities must be between 1 and {len(TRAIN_IDENTITIES)}")
    if int(args.eval_identities) < 1 or int(args.eval_identities) > len(EVAL_IDENTITIES):
        raise SystemExit(f"--eval-identities must be between 1 and {len(EVAL_IDENTITIES)}")
    report = run(args)
    print(json.dumps({"mode": args.mode, "final_decision": report.get("final_decision"), "elapsed_seconds": report.get("elapsed_seconds")}, indent=2, sort_keys=True))
    return 0 if "FAIL" not in str(report.get("final_decision")) and "INVALID" not in str(report.get("final_decision")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
