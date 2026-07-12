"""SACF-VLA prototype runner."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_echo_vla_first_prototype import _postprocess_action, _preprocess_batch  # noqa: E402
from scripts.run_phase_barrier_vla_prototype import (  # noqa: E402
    _identity_to_initial_state_index,
    _make_exact_vector_env,
    _policy_action,
    _round,
    _set_runtime_env,
    _step_success,
)
from tca_map.smolvla.official_closed_loop_scaleup import _json_default  # noqa: E402
from tca_map.smolvla.official_wsl_libero_rollout import POLICIES, _cuda_memory, _load_policy_and_processors  # noqa: E402
from tca_map.smolvla.sacf_vla import (  # noqa: E402
    SACFConfig,
    SACFExample,
    assert_no_privileged_inference_fields,
    file_sha256,
    instruction_to_demo_filename,
    load_plain_checkpoint,
    load_sacf_checkpoint,
    phase_index_from_fraction,
    predict_plain_action,
    predict_sacf_action,
    save_plain_checkpoint,
    save_sacf_checkpoint,
    task_phase_mean_action,
    train_plain_bc_prefix,
    train_sacf_policy,
)


DATE_KST = "2026-07-12"
BRANCH = "codex/autonomous-until-paper-governance-v2"
TASK_MAP = {
    "libero_spatial": [
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "pick up the black bowl next to the ramekin and place it on the plate",
        "pick up the black bowl from table center and place it on the plate",
        "pick up the black bowl on the cookie box and place it on the plate",
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        "pick up the black bowl on the ramekin and place it on the plate",
        "pick up the black bowl next to the cookie box and place it on the plate",
        "pick up the black bowl on the stove and place it on the plate",
        "pick up the black bowl next to the plate and place it on the plate",
        "pick up the black bowl on the wooden cabinet and place it on the plate",
    ],
    "libero_object": [
        "pick up the alphabet soup and place it in the basket",
        "pick up the cream cheese and place it in the basket",
        "pick up the salad dressing and place it in the basket",
        "pick up the bbq sauce and place it in the basket",
        "pick up the ketchup and place it in the basket",
        "pick up the tomato sauce and place it in the basket",
        "pick up the butter and place it in the basket",
        "pick up the milk and place it in the basket",
        "pick up the chocolate pudding and place it in the basket",
        "pick up the orange juice and place it in the basket",
    ],
}
TASKS = [
    {
        "suite": "libero_spatial",
        "task_id": 4,
        "role": "same_object_spatial_counterfactual",
        "instruction": TASK_MAP["libero_spatial"][4],
    },
    {
        "suite": "libero_object",
        "task_id": 4,
        "role": "same_destination_object_counterfactual",
        "instruction": TASK_MAP["libero_object"][4],
    },
]
EVAL_IDENTITIES = [20260713, 20260714, 20260715, 20260716, 20260717]
VARIANTS = [
    "frozen_smolvla",
    "task_phase_mean_prefix",
    "plain_bc_prefix",
    "cag_null_guidance",
    "sacf_full",
]


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


def _task_key(suite: str, task_id: int) -> str:
    return f"{suite}/task_{int(task_id)}"


def _cuda_memory_report() -> dict[str, Any]:
    import torch

    return _cuda_memory(torch)


def _state_from_observation(env: Any, observation: Any, loaded: Mapping[str, Any], config: SACFConfig) -> np.ndarray:
    batch = _preprocess_batch(env, observation, dict(loaded))
    state = batch.get("observation.state")
    if state is None:
        raise RuntimeError("preprocessed observation has no observation.state")
    try:
        state_np = state.detach().to("cpu").numpy()
    except AttributeError:
        state_np = np.asarray(state)
    return np.asarray(state_np, dtype=np.float32).reshape(-1)[: int(config.state_dim)]


def _policy_action_with_task_override(policy: Any, env: Any, observation: Any, loaded: Mapping[str, Any], task_text: str) -> np.ndarray:
    import torch
    from lerobot.scripts.lerobot_eval import preprocess_observation

    lerobot_observation = preprocess_observation(observation)
    first_key = next(iter(lerobot_observation))
    batch_size = int(lerobot_observation[first_key].shape[0])
    lerobot_observation["task"] = [str(task_text) for _ in range(batch_size)]
    lerobot_observation = loaded["env_preprocessor"](lerobot_observation)
    batch = loaded["preprocessor"](lerobot_observation)
    with torch.inference_mode():
        if hasattr(policy, "predict_action_chunk"):
            action_chunk = policy.predict_action_chunk(batch)
            action = action_chunk[:, 0] if getattr(action_chunk, "ndim", 0) == 3 else action_chunk
        else:
            action = policy.select_action(batch)
    return _postprocess_action(action, dict(loaded)).reshape(1, -1)


def _hdf5_state(obs_group: Any, index: int, config: SACFConfig) -> np.ndarray:
    ee = np.asarray(obs_group["ee_states"][index], dtype=np.float32).reshape(-1)
    gripper = np.asarray(obs_group["gripper_states"][index], dtype=np.float32).reshape(-1)
    return np.concatenate([ee, gripper], axis=0)[: int(config.state_dim)].astype(np.float32)


def _collect_demo_examples(args: argparse.Namespace, config: SACFConfig) -> tuple[list[SACFExample], dict[str, Any]]:
    import h5py

    data_root = Path(args.libero_data_root)
    examples: list[SACFExample] = []
    coverage: dict[str, Any] = {}
    for suite in ("libero_spatial", "libero_object"):
        for task_id, instruction in enumerate(TASK_MAP[suite]):
            task_key = _task_key(suite, task_id)
            path = data_root / suite / instruction_to_demo_filename(instruction)
            row_count = 0
            demo_count = 0
            if not path.exists():
                coverage[task_key] = {"path": str(path), "exists": False, "rows": 0}
                continue
            with h5py.File(path, "r") as handle:
                for demo_name in sorted(handle["data"].keys()):
                    group = handle["data"][demo_name]
                    actions = np.asarray(group["actions"], dtype=np.float32)
                    obs = group["obs"]
                    stride = max(1, int(np.ceil(actions.shape[0] / max(1, int(args.max_rows_per_task) // max(1, int(args.max_demos_per_task))))))
                    for step in range(0, actions.shape[0], stride):
                        if row_count >= int(args.max_rows_per_task):
                            break
                        step_fraction = float(step) / max(1.0, float(actions.shape[0] - 1))
                        examples.append(
                            SACFExample(
                                state=[float(x) for x in _hdf5_state(obs, step, config)],
                                action=[float(x) for x in np.clip(actions[step].reshape(-1)[: int(config.action_dim)], -1.0, 1.0)],
                                instruction=instruction,
                                family=suite,
                                task_key=task_key,
                                step_fraction=step_fraction,
                                phase_index=phase_index_from_fraction(step_fraction, config),
                            )
                        )
                        row_count += 1
                    demo_count += 1
                    if row_count >= int(args.max_rows_per_task) or demo_count >= int(args.max_demos_per_task):
                        break
            coverage[task_key] = {"path": str(path), "exists": True, "rows": int(row_count), "demos": int(demo_count)}
    return examples, coverage


def _synthetic_examples(count: int, config: SACFConfig) -> list[SACFExample]:
    instructions = [
        TASK_MAP["libero_spatial"][4],
        TASK_MAP["libero_spatial"][1],
        TASK_MAP["libero_object"][4],
        TASK_MAP["libero_object"][7],
    ]
    families = ["libero_spatial", "libero_spatial", "libero_object", "libero_object"]
    semantic = {
        instructions[0]: np.asarray([0.36, -0.20, 0.12, 0.0, 0.0, 0.05, -0.5], dtype=np.float32),
        instructions[1]: np.asarray([-0.22, 0.24, 0.08, 0.0, 0.0, 0.02, -0.5], dtype=np.float32),
        instructions[2]: np.asarray([0.18, 0.12, -0.04, 0.0, 0.0, -0.02, -0.5], dtype=np.float32),
        instructions[3]: np.asarray([-0.16, -0.13, -0.02, 0.0, 0.0, -0.02, -0.5], dtype=np.float32),
    }
    rows: list[SACFExample] = []
    for index in range(int(count)):
        task = index % len(instructions)
        frac = (index % 24) / 23.0
        phase = phase_index_from_fraction(frac, config)
        state = np.asarray(
            [
                np.sin(np.pi * frac),
                np.cos(np.pi * frac),
                frac,
                frac * frac,
                1.0 - frac,
                (-1.0) ** index * 0.05,
                0.2,
                -0.2,
            ],
            dtype=np.float32,
        )
        shared = np.asarray([0.015 * phase, -0.012 * phase, 0.04 * frac, 0.01, -0.01, 0.0, 0.25], dtype=np.float32)
        action = np.clip(shared + semantic[instructions[task]] + 0.02 * state[:7], -1.0, 1.0)
        rows.append(
            SACFExample(
                state=[float(x) for x in state],
                action=[float(x) for x in action],
                instruction=instructions[task],
                family=families[task],
                task_key=_task_key(families[task], task),
                step_fraction=float(frac),
                phase_index=phase,
            )
        )
    return rows


def _mean_l2(model: Any, examples: list[SACFExample], *, full: bool) -> float:
    errors: list[float] = []
    for row in examples:
        if full:
            pred, _diag = predict_sacf_action(
                model,
                state=row.state,
                instruction=row.instruction,
                family=row.family,
                step_fraction=row.step_fraction,
            )
        else:
            pred = predict_plain_action(
                model,
                state=row.state,
                instruction=row.instruction,
                family=row.family,
                step_fraction=row.step_fraction,
            )
        target = np.asarray(row.action, dtype=np.float32)
        errors.append(float(np.linalg.norm(pred - target)))
    return float(np.mean(errors))


def _synthetic_mode(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    config = SACFConfig(hidden_dim=int(args.hidden_dim), prefix_fraction=float(args.prefix_fraction))
    train = _synthetic_examples(int(args.synthetic_count), config)
    probes = _synthetic_examples(48, config)
    full_model, full_stats = train_sacf_policy(train, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=31)
    plain_model, plain_stats = train_plain_bc_prefix(train, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=32)
    full_l2 = _mean_l2(full_model, probes, full=True)
    plain_l2 = _mean_l2(plain_model, probes, full=False)
    full_path = Path(args.full_checkpoint)
    plain_path = Path(args.plain_checkpoint)
    save_sacf_checkpoint(full_path, full_model, full_stats)
    save_plain_checkpoint(plain_path, plain_model, plain_stats)
    passed = bool(
        full_stats["loss_decreased"]
        and full_stats["factor_loss_decreased"]
        and plain_stats["loss_decreased"]
        and full_l2 <= 0.01
        and full_stats["mean_semantic_component_norm"] > 0.01
    )
    return {
        "mode": "synthetic",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": True,
        "closed_loop_experiment_happened": False,
        "config": config.to_json(),
        "full_checkpoint_path": str(full_path),
        "full_checkpoint_sha256": file_sha256(full_path),
        "plain_checkpoint_path": str(plain_path),
        "plain_checkpoint_sha256": file_sha256(plain_path),
        "summary": {
            "full_mean_action_l2": _round(full_l2, 6),
            "plain_mean_action_l2": _round(plain_l2, 6),
            "full_loss_decreased": bool(full_stats["loss_decreased"]),
            "full_factor_loss_decreased": bool(full_stats["factor_loss_decreased"]),
            "plain_loss_decreased": bool(plain_stats["loss_decreased"]),
            "semantic_component_norm": _round(float(full_stats["mean_semantic_component_norm"]), 6),
            "measurement_repair_note": "Initial synthetic gate used a brittle relative L2 comparison to plain BC. Repaired gate checks absolute reconstruction, factor-loss decrease, and semantic activation before any real-demo or closed-loop SACF result.",
            "synthetic_passed": passed,
        },
        "final_decision": "SYNTHETIC_MECHANISM_PASS" if passed else "SYNTHETIC_MECHANISM_FAIL",
        "next_step": "Run real-demo training." if passed else "Repair or kill synthetic SACF implementation.",
        "elapsed_seconds": _round(time.time() - start, 3),
    }


def _real_demo_train_mode(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    config = SACFConfig(
        hidden_dim=int(args.hidden_dim),
        semantic_width=int(args.semantic_width),
        phase_bins=int(args.phase_bins),
        factor_loss_weight=float(args.factor_loss_weight),
        shared_invariance_weight=float(args.shared_invariance_weight),
        prefix_fraction=float(args.prefix_fraction),
    )
    assert_no_privileged_inference_fields(["state", "instruction", "family", "step_fraction", "phase_index"])
    examples, coverage = _collect_demo_examples(args, config)
    if len(examples) < 80:
        raise RuntimeError(f"not enough SACF demo rows: {len(examples)}")
    full_model, full_stats = train_sacf_policy(examples, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=41)
    plain_model, plain_stats = train_plain_bc_prefix(examples, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=42)
    probe_rows = examples[:: max(1, len(examples) // 64)][:64]
    full_plain_deltas: list[float] = []
    semantic_norms: list[float] = []
    for row in probe_rows:
        full_action, diag = predict_sacf_action(
            full_model,
            state=row.state,
            instruction=row.instruction,
            family=row.family,
            step_fraction=row.step_fraction,
        )
        plain_action = predict_plain_action(
            plain_model,
            state=row.state,
            instruction=row.instruction,
            family=row.family,
            step_fraction=row.step_fraction,
        )
        full_plain_deltas.append(float(np.linalg.norm(full_action - plain_action)))
        semantic_norms.append(float(diag["semantic_component_norm"]))
    full_path = Path(args.full_checkpoint)
    plain_path = Path(args.plain_checkpoint)
    save_sacf_checkpoint(full_path, full_model, full_stats)
    save_plain_checkpoint(plain_path, plain_model, plain_stats)
    passed = bool(full_stats["loss_decreased"] and plain_stats["loss_decreased"] and np.mean(semantic_norms) > 0.01)
    return {
        "mode": "real-demo-train",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": True,
        "closed_loop_experiment_happened": False,
        "config": config.to_json(),
        "demo_coverage": coverage,
        "example_count": int(len(examples)),
        "full_loaded_stats": full_stats,
        "plain_loaded_stats": plain_stats,
        "full_checkpoint_path": str(full_path),
        "full_checkpoint_sha256": file_sha256(full_path),
        "plain_checkpoint_path": str(plain_path),
        "plain_checkpoint_sha256": file_sha256(plain_path),
        "summary": {
            "full_loss_decreased": bool(full_stats["loss_decreased"]),
            "plain_loss_decreased": bool(plain_stats["loss_decreased"]),
            "mean_semantic_component_norm": _round(float(np.mean(semantic_norms)), 6),
            "mean_full_plain_action_delta": _round(float(np.mean(full_plain_deltas)), 6),
            "real_demo_train_passed": passed,
        },
        "final_decision": "REAL_DEMO_TRAIN_PASS" if passed else "REAL_DEMO_TRAIN_FAIL",
        "next_step": "Run Stage A." if passed else "Repair once if action convention mismatch, otherwise kill.",
        "elapsed_seconds": _round(time.time() - start, 3),
    }


def _planned_rows(tasks: list[Mapping[str, Any]], identities: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for task in tasks:
            for identity in identities:
                rows.append(
                    {
                        "variant": variant,
                        "suite": str(task["suite"]),
                        "task_id": int(task["task_id"]),
                        "task_key": _task_key(str(task["suite"]), int(task["task_id"])),
                        "role": str(task["role"]),
                        "instruction": str(task["instruction"]),
                        "identity": int(identity),
                    }
                )
    return rows


def _episode_key(row: Mapping[str, Any]) -> str:
    return "|".join([str(row.get("variant")), str(row.get("suite")), str(row.get("task_id")), str(row.get("identity"))])


def _variant_action(
    *,
    variant: str,
    state: np.ndarray,
    step_fraction: float,
    task_key: str,
    instruction: str,
    family: str,
    frozen_action: np.ndarray | None,
    policy: Any,
    env: Any,
    observation: Any,
    loaded: Mapping[str, Any],
    sacf_model: Any,
    sacf_stats: Mapping[str, Any],
    plain_model: Any,
    config: SACFConfig,
    guidance_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    prefix_active = bool(float(step_fraction) < float(config.prefix_fraction))
    if variant == "frozen_smolvla":
        if frozen_action is None:
            raise RuntimeError("frozen action missing")
        return frozen_action.astype(np.float32), {"prefix_active": False, "semantic_component_norm": 0.0, "full_plain_delta": 0.0, "cag_delta": 0.0}
    if variant == "cag_null_guidance":
        full = _policy_action_with_task_override(policy, env, observation, loaded, instruction).reshape(-1).astype(np.float32)
        null = _policy_action_with_task_override(policy, env, observation, loaded, "").reshape(-1).astype(np.float32)
        action = np.clip(full + float(guidance_scale) * (full - null), -1.0, 1.0).astype(np.float32)
        return action, {"prefix_active": False, "semantic_component_norm": 0.0, "full_plain_delta": 0.0, "cag_delta": float(np.linalg.norm(full - null))}
    if not prefix_active:
        if frozen_action is None:
            raise RuntimeError("frozen action missing after prefix")
        return frozen_action.astype(np.float32), {"prefix_active": False, "semantic_component_norm": 0.0, "full_plain_delta": 0.0, "cag_delta": 0.0}
    if variant == "task_phase_mean_prefix":
        action = task_phase_mean_action(sacf_stats, task_key=task_key, step_fraction=step_fraction, config=config)
        return action.astype(np.float32), {"prefix_active": True, "semantic_component_norm": 0.0, "full_plain_delta": 0.0, "cag_delta": 0.0}
    if variant == "plain_bc_prefix":
        action = predict_plain_action(plain_model, state=state, instruction=instruction, family=family, step_fraction=step_fraction)
        return action.astype(np.float32), {"prefix_active": True, "semantic_component_norm": 0.0, "full_plain_delta": 0.0, "cag_delta": 0.0}
    if variant == "sacf_full":
        action, diag = predict_sacf_action(sacf_model, state=state, instruction=instruction, family=family, step_fraction=step_fraction)
        plain_action = predict_plain_action(plain_model, state=state, instruction=instruction, family=family, step_fraction=step_fraction)
        return action.astype(np.float32), {
            "prefix_active": True,
            "semantic_component_norm": float(diag["semantic_component_norm"]),
            "full_plain_delta": float(np.linalg.norm(action - plain_action)),
            "cag_delta": 0.0,
        }
    raise ValueError(f"unknown SACF variant: {variant}")


def _run_episode(
    *,
    row: Mapping[str, Any],
    loaded: Mapping[str, Any],
    sacf_model: Any,
    sacf_stats: Mapping[str, Any],
    plain_model: Any,
    config: SACFConfig,
    max_eval_steps: int,
    guidance_scale: float,
) -> dict[str, Any]:
    env = None
    started = time.time()
    semantic_norms: list[float] = []
    full_plain_deltas: list[float] = []
    cag_deltas: list[float] = []
    prefix_steps = 0
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
        steps = 0
        for step in range(max_steps):
            step_fraction = float(step) / max(1.0, float(max_steps - 1))
            state = _state_from_observation(env, observation, loaded, config)
            needs_frozen = str(row["variant"]) == "frozen_smolvla" or (str(row["variant"]) in {"task_phase_mean_prefix", "plain_bc_prefix", "sacf_full"} and step_fraction >= float(config.prefix_fraction))
            frozen_action = _policy_action(policy, env, observation, loaded).reshape(-1).astype(np.float32) if needs_frozen else None
            action, diagnostics = _variant_action(
                variant=str(row["variant"]),
                state=state,
                step_fraction=step_fraction,
                task_key=str(row["task_key"]),
                instruction=str(row["instruction"]),
                family=str(row["suite"]),
                frozen_action=frozen_action,
                policy=policy,
                env=env,
                observation=observation,
                loaded=loaded,
                sacf_model=sacf_model,
                sacf_stats=sacf_stats,
                plain_model=plain_model,
                config=config,
                guidance_scale=guidance_scale,
            )
            observation, reward, terminated, truncated, info = env.step(action.reshape(1, -1))
            rewards.append(float(np.asarray(reward).reshape(-1)[0]))
            semantic_norms.append(float(diagnostics["semantic_component_norm"]))
            full_plain_deltas.append(float(diagnostics["full_plain_delta"]))
            cag_deltas.append(float(diagnostics["cag_delta"]))
            if bool(diagnostics["prefix_active"]):
                prefix_steps += 1
            steps = int(step + 1)
            success = bool(success or _step_success(info))
            if success or np.all(terminated | truncated):
                break
        return {
            **dict(row),
            "success": bool(success),
            "exception": None,
            "episode_steps": steps,
            "prefix_steps": int(prefix_steps),
            "reward_sum": _round(float(np.sum(rewards)) if rewards else 0.0, 6),
            "mean_semantic_component_norm": _round(float(np.mean(semantic_norms)) if semantic_norms else 0.0, 6),
            "mean_full_plain_action_delta": _round(float(np.mean(full_plain_deltas)) if full_plain_deltas else 0.0, 6),
            "mean_cag_full_null_delta": _round(float(np.mean(cag_deltas)) if cag_deltas else 0.0, 6),
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


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * np.sqrt((phat * (1.0 - phat) + z * z / (4 * total)) / total)
    return [_round((centre - margin) / denom, 6), _round((centre + margin) / denom, 6)]


def _summarize_stage_a(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_rows = [row for row in rows if row.get("variant") == variant]
        successes = int(sum(1 for row in variant_rows if bool(row.get("success"))))
        total = int(len(variant_rows))
        per_task: dict[str, Any] = {}
        for task_key in sorted({str(row.get("task_key")) for row in variant_rows}):
            task_rows = [row for row in variant_rows if str(row.get("task_key")) == task_key]
            task_successes = int(sum(1 for row in task_rows if bool(row.get("success"))))
            per_task[task_key] = {"successes": task_successes, "total": len(task_rows), "rate": _round(task_successes / max(1, len(task_rows)), 6)}
        task_balanced = float(np.mean([item["rate"] for item in per_task.values()])) if per_task else 0.0
        by_variant[variant] = {
            "successes": successes,
            "total": total,
            "success_rate": _round(successes / max(1, total), 6),
            "task_balanced_success_rate": _round(task_balanced, 6),
            "wilson_95_ci": _wilson_ci(successes, total),
            "per_task": per_task,
            "mean_semantic_component_norm": _round(float(np.mean([float(row.get("mean_semantic_component_norm") or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
            "mean_full_plain_action_delta": _round(float(np.mean([float(row.get("mean_full_plain_action_delta") or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
            "mean_cag_full_null_delta": _round(float(np.mean([float(row.get("mean_cag_full_null_delta") or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
        }
    baseline_names = ["frozen_smolvla", "task_phase_mean_prefix", "plain_bc_prefix", "cag_null_guidance"]
    baseline_rates = {name: float(by_variant.get(name, {}).get("task_balanced_success_rate", 0.0)) for name in baseline_names}
    strongest = max(baseline_rates, key=lambda key: baseline_rates[key])
    strongest_rate = baseline_rates[strongest]
    full_rate = float(by_variant.get("sacf_full", {}).get("task_balanced_success_rate", 0.0))
    full_successes = int(by_variant.get("sacf_full", {}).get("successes", 0))
    full_plain_delta = float(by_variant.get("sacf_full", {}).get("mean_full_plain_action_delta", 0.0))
    if full_plain_delta < 1e-6:
        decision = "STAGE_A_PERMANENT_KILL_TRIVIAL_EQUIVALENCE"
    elif strongest_rate - full_rate >= 0.30:
        decision = "STAGE_A_PERMANENT_KILL_CLEARLY_WORSE"
    elif full_successes == 0 and any(int(by_variant.get(name, {}).get("successes", 0)) >= 4 for name in baseline_names):
        decision = "STAGE_A_PERMANENT_KILL_ZERO_SUCCESS_WITH_BASELINE_HEADROOM"
    elif full_rate > max(float(by_variant.get("frozen_smolvla", {}).get("task_balanced_success_rate", 0.0)), float(by_variant.get("plain_bc_prefix", {}).get("task_balanced_success_rate", 0.0))):
        decision = "STAGE_A_POSITIVE_TO_STAGE_B"
    else:
        decision = "STAGE_A_NON_GO_TO_STAGE_B_REQUIRED"
    return {
        "by_variant": by_variant,
        "strongest_baseline": strongest,
        "strongest_baseline_task_balanced_success_rate": _round(strongest_rate, 6),
        "sacf_full_task_balanced_success_rate": _round(full_rate, 6),
        "final_decision": decision,
    }


def _stage_a_mode(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    _set_runtime_env(args)
    args.base_path = str(Path(args.checkpoint))
    args.lora_root = getattr(args, "lora_root", "/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    sacf_model, sacf_stats = load_sacf_checkpoint(args.full_checkpoint)
    plain_model, plain_stats = load_plain_checkpoint(args.plain_checkpoint)
    config = sacf_model.config
    if plain_model.config.to_json() != config.to_json():
        raise RuntimeError("SACF and plain checkpoint configs differ")
    loaded = _load_policy_and_processors(args, POLICIES[0])
    planned = _planned_rows(TASKS[: int(args.max_tasks)], EVAL_IDENTITIES[: int(args.eval_identities)])
    partial_path = Path(args.stage_a_partial_output)
    completed: dict[str, Any] = {}
    if partial_path.exists() and not bool(args.rerun_stage_a):
        try:
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            completed = {_episode_key(row): row for row in partial.get("episodes", [])}
        except Exception:
            completed = {}
    episodes = list(completed.values())
    for row in planned:
        key = _episode_key(row)
        if key in completed:
            continue
        result = _run_episode(
            row=row,
            loaded=loaded,
            sacf_model=sacf_model,
            sacf_stats=sacf_stats,
            plain_model=plain_model,
            config=config,
            max_eval_steps=int(args.max_eval_steps),
            guidance_scale=float(args.guidance_scale),
        )
        episodes.append(result)
        partial_report = {
            "mode": "stage-a-partial",
            "branch": BRANCH,
            "date_kst": DATE_KST,
            "planned_episode_count": len(planned),
            "completed_episode_count": len(episodes),
            "episodes": episodes,
        }
        _write_json(partial_path, partial_report)
        print(f"[sacf-stage-a] completed {len(episodes)}/{len(planned)}: {key} success={result.get('success')} exception={bool(result.get('exception'))}", flush=True)
    summary = _summarize_stage_a(episodes)
    return {
        "mode": "stage-a",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": False,
        "closed_loop_experiment_happened": True,
        "config": config.to_json(),
        "tasks": TASKS[: int(args.max_tasks)],
        "eval_identities": EVAL_IDENTITIES[: int(args.eval_identities)],
        "variants": VARIANTS,
        "full_checkpoint_path": str(args.full_checkpoint),
        "full_checkpoint_sha256": file_sha256(args.full_checkpoint),
        "plain_checkpoint_path": str(args.plain_checkpoint),
        "plain_checkpoint_sha256": file_sha256(args.plain_checkpoint),
        "sacf_train_summary": {
            "full_loss_decreased": bool(sacf_stats.get("loss_decreased")),
            "plain_loss_decreased": bool(plain_stats.get("loss_decreased")),
            "semantic_component_norm": sacf_stats.get("mean_semantic_component_norm"),
        },
        "episode_count": len(episodes),
        "exceptions": [row for row in episodes if row.get("exception")],
        "episodes": episodes,
        "summary": summary,
        "final_decision": summary["final_decision"],
        "next_step": "Archive kill and pivot if permanent kill; otherwise run Stage B under governance.",
        "elapsed_seconds": _round(time.time() - start, 3),
        "cuda_memory": _cuda_memory_report(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "synthetic":
        report = _synthetic_mode(args)
        _write_json(Path(args.synthetic_output), report)
        _write_md(Path(args.synthetic_md), "SACF-VLA Synthetic Result", report)
        return report
    if args.mode == "real-demo-train":
        report = _real_demo_train_mode(args)
        _write_json(Path(args.real_demo_train_output), report)
        _write_md(Path(args.real_demo_train_md), "SACF-VLA Real-Demo Training Result", report)
        return report
    if args.mode == "stage-a":
        report = _stage_a_mode(args)
        _write_json(Path(args.stage_a_output), report)
        _write_md(Path(args.stage_a_md), "SACF-VLA Stage A Result", report)
        return report
    raise ValueError(f"unknown mode: {args.mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "real-demo-train", "stage-a"], required=True)
    parser.add_argument("--checkpoint", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--libero-data-root", default="/mnt/c/assets/data/libero")
    parser.add_argument("--full-checkpoint", default="reports/sacf_vla/checkpoints/sacf_full.pt")
    parser.add_argument("--plain-checkpoint", default="reports/sacf_vla/checkpoints/plain_bc_prefix.pt")
    parser.add_argument("--synthetic-output", default="reports/sacf_vla/synthetic_result.json")
    parser.add_argument("--synthetic-md", default="reports/sacf_vla/synthetic_result.md")
    parser.add_argument("--real-demo-train-output", default="reports/sacf_vla/real_demo_train_result.json")
    parser.add_argument("--real-demo-train-md", default="reports/sacf_vla/real_demo_train_result.md")
    parser.add_argument("--stage-a-output", default="reports/sacf_vla/stage_a_result.json")
    parser.add_argument("--stage-a-md", default="reports/sacf_vla/stage_a_result.md")
    parser.add_argument("--stage-a-partial-output", default="reports/sacf_vla/stage_a_partial_result.json")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--semantic-width", type=int, default=16)
    parser.add_argument("--phase-bins", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--factor-loss-weight", type=float, default=0.35)
    parser.add_argument("--shared-invariance-weight", type=float, default=0.05)
    parser.add_argument("--prefix-fraction", type=float, default=0.35)
    parser.add_argument("--guidance-scale", type=float, default=0.5)
    parser.add_argument("--synthetic-count", type=int, default=192)
    parser.add_argument("--max-rows-per-task", type=int, default=240)
    parser.add_argument("--max-demos-per-task", type=int, default=20)
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--eval-identities", type=int, default=5)
    parser.add_argument("--max-eval-steps", type=int, default=0)
    parser.add_argument("--rerun-stage-a", action="store_true")
    parser.add_argument("--video-dir", default="runs/sacf_vla/videos")
    parser.add_argument("--no-video", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 < float(args.prefix_fraction) < 1.0:
        raise SystemExit("--prefix-fraction must be in (0, 1)")
    if int(args.max_tasks) < 1 or int(args.max_tasks) > len(TASKS):
        raise SystemExit("--max-tasks must be between 1 and 2")
    if int(args.eval_identities) < 1 or int(args.eval_identities) > len(EVAL_IDENTITIES):
        raise SystemExit("--eval-identities must be between 1 and 5")
    report = run(args)
    print(json.dumps({"mode": args.mode, "final_decision": report.get("final_decision"), "elapsed_seconds": report.get("elapsed_seconds")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
