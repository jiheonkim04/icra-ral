"""Bounded visual review reruns for official SmolVLA/LIBERO failures.

This runner captures videos for a fixed, predeclared subset of hard-slice
episodes. It reuses the official closed-loop policy/environment path and does
not train, tune, or run a full benchmark sweep.
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from tca_map.smolvla.official_closed_loop_scaleup import (
    _extract_single_env,
    _json_default,
    _make_env_cfg,
    _set_runtime_env,
    trace_one_episode,
)
from tca_map.smolvla.official_wsl_libero_rollout import POLICIES, _load_policy_and_processors


MAX_REVIEW_EPISODES = 24

SELECTED_REVIEW_EPISODES = [
    # libero_spatial/task_4: three repeated all-policy failure reset seeds.
    *[
        {"suite": "libero_spatial", "task_id": 4, "reset_seed": seed, "policy": policy, "selection_role": "spatial_all_policy_failure"}
        for seed in [20260712, 20260713, 20260714]
        for policy in ["frozen_base", "rank4_lora_seed_11", "rank4_lora_seed_22", "rank4_lora_seed_33"]
    ],
    # Matched successful resets from the same spatial task.
    {"suite": "libero_spatial", "task_id": 4, "reset_seed": 20260711, "policy": "frozen_base", "selection_role": "spatial_matched_success"},
    {"suite": "libero_spatial", "task_id": 4, "reset_seed": 20260715, "policy": "rank4_lora_seed_11", "selection_role": "spatial_matched_success"},
    # libero_10/task_4: two repeated all-policy failure reset seeds.
    *[
        {"suite": "libero_10", "task_id": 4, "reset_seed": seed, "policy": policy, "selection_role": "libero10_all_policy_failure"}
        for seed in [20260713, 20260715]
        for policy in ["frozen_base", "rank4_lora_seed_11", "rank4_lora_seed_22", "rank4_lora_seed_33"]
    ],
    # Matched successful resets from the same libero_10 task.
    {"suite": "libero_10", "task_id": 4, "reset_seed": 20260712, "policy": "frozen_base", "selection_role": "libero10_matched_success"},
    {"suite": "libero_10", "task_id": 4, "reset_seed": 20260714, "policy": "rank4_lora_seed_11", "selection_role": "libero10_matched_success"},
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _episode_id(policy: str, suite: str, task_id: int, seed: int) -> str:
    return f"{policy}|{suite}|task_{task_id}|seed_{seed}"


def _load_scaleup_rows(path: Path) -> dict[str, dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {row["episode_id"]: row for row in report["scaleup"]["episodes"]}


def _selected_with_originals(scaleup_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if len(SELECTED_REVIEW_EPISODES) > MAX_REVIEW_EPISODES:
        raise ValueError(f"review set has {len(SELECTED_REVIEW_EPISODES)} episodes, max is {MAX_REVIEW_EPISODES}")
    selected = []
    for item in SELECTED_REVIEW_EPISODES:
        episode_id = _episode_id(item["policy"], item["suite"], int(item["task_id"]), int(item["reset_seed"]))
        original = scaleup_rows.get(episode_id)
        if original is None:
            raise KeyError(f"selected review episode missing from original scaleup result: {episode_id}")
        selected.append(
            {
                **item,
                "episode_id": episode_id,
                "instruction": original["instruction"],
                "original_success": bool(original["success"]),
                "original_episode_length": original["episode_length"],
                "original_termination_reason": original["termination_reason"],
                "original_action_chunks_generated": original["action_chunks_generated"],
            }
        )
    return selected


def _write_inventory(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    existing_videos_before: list[str],
) -> None:
    report_dir = Path(args.report_dir)
    completed = [row for row in rows if row.get("rerun_status") == "completed"]
    result_by_id = {row["episode_id"]: row for row in rows}
    lines = [
        "# Closed-Loop Failure Video Inventory",
        "",
        f"Date: {args.date} KST",
        "",
        "## Scope",
        "",
        f"- selected review episodes: `{len(selected)}`",
        f"- maximum allowed episodes: `{MAX_REVIEW_EPISODES}`",
        "- full 400-episode sweep rerun: `false`",
        "- policy training: `false`",
        "- policy set: `frozen_base`, `rank4_lora_seed_11`, `rank4_lora_seed_22`, `rank4_lora_seed_33`",
        "- selected hard tasks: `libero_spatial/task_4`, `libero_10/task_4`",
        f"- completed video reruns: `{len(completed)}`",
        f"- rerun errors: `{len(errors)}`",
        "",
        "## Existing Video Search",
        "",
        "No tracked failure videos were present in the prior 400-episode scaleup. This bounded rerun writes videos under the gitignored `runs/closed_loop_failure_visual_review/videos` directory.",
        "",
        "## Selected Episodes",
        "",
        "| Episode | Role | Original success | Rerun success | Video |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for item in selected:
        row = result_by_id.get(item["episode_id"], {})
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                item["episode_id"],
                item["selection_role"],
                item["original_success"],
                row.get("success"),
                row.get("video_path"),
            )
        )
    _write_md(report_dir / "closed_loop_failure_video_inventory.md", lines)
    _write_json(
        report_dir / "closed_loop_failure_video_rerun_result.json",
        {
            "schema_version": 1,
            "date": args.date,
            "selected_episode_count": len(selected),
            "max_review_episodes": MAX_REVIEW_EPISODES,
            "existing_videos_under_review_root_before_rerun": existing_videos_before,
            "rows": rows,
            "errors": errors,
        },
    )


def run_review(args: argparse.Namespace) -> dict[str, Any]:
    from lerobot.envs.factory import make_env

    _set_runtime_env(args)
    video_root = Path(args.video_dir)
    existing_videos_before = [str(path) for path in sorted(video_root.rglob("*.mp4"))] if video_root.exists() else []
    scaleup_rows = _load_scaleup_rows(Path(args.scaleup_result))
    selected = _selected_with_originals(scaleup_rows)
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        by_policy[item["policy"]].append(item)

    rows = []
    errors = []
    policy_specs = {spec.name: spec for spec in POLICIES}
    for policy_name, policy_items in by_policy.items():
        print(f"[visual-review] policy {policy_name}", flush=True)
        loaded = _load_policy_and_processors(args, policy_specs[policy_name])
        by_task: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for item in policy_items:
            by_task[(item["suite"], int(item["task_id"]))].append(item)
        for (suite, task_id), task_items in by_task.items():
            print(f"[visual-review] {policy_name} {suite}/task_{task_id}", flush=True)
            env = None
            try:
                env = _extract_single_env(make_env(_make_env_cfg(suite, [task_id]), n_envs=1, use_async_envs=False), suite, task_id)
                for item in task_items:
                    video_path = Path(args.video_dir) / policy_name / suite / f"task_{task_id}_seed_{item['reset_seed']}.mp4"
                    row = dict(item)
                    try:
                        trace = trace_one_episode(
                            env=env,
                            policy=loaded["policy"],
                            env_preprocessor=loaded["env_preprocessor"],
                            env_postprocessor=loaded["env_postprocessor"],
                            preprocessor=loaded["preprocessor"],
                            postprocessor=loaded["postprocessor"],
                            seed=int(item["reset_seed"]),
                            video_path=video_path,
                        )
                        row.update(trace)
                        row["rerun_status"] = "completed"
                    except Exception as exc:  # pragma: no cover - simulator boundary
                        row.update(
                            {
                                "success": False,
                                "rerun_status": "exception",
                                "video_path": None,
                                "exception": {
                                    "type": type(exc).__name__,
                                    "message": str(exc),
                                    "traceback": traceback.format_exc().splitlines()[-24:],
                                },
                            }
                        )
                        errors.append({"episode_id": item["episode_id"], **row["exception"]})
                    rows.append(row)
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
        del loaded

    _write_inventory(args, selected, rows, errors, existing_videos_before)
    return {
        "selected_episode_count": len(selected),
        "completed": sum(1 for row in rows if row.get("rerun_status") == "completed"),
        "errors": len(errors),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-07-11")
    parser.add_argument("--base-path", default="/home/jiheon/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/home/jiheon/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--video-dir", default="runs/closed_loop_failure_visual_review/videos")
    parser.add_argument("--scaleup-result", default="reports/official_closed_loop_scaleup_result.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_review(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["errors"] == 0 and result["completed"] == result["selected_episode_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
