"""Metadata-only LIBERO subset manifest builder.

This module reads BDDL task files from a local LIBERO source checkout. It does
not import LIBERO, RoboSuite, MuJoCo, VLA models, or dataset files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = "tca-map-libero-metadata-subset-v0"
DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DATASET_EXTENSIONS = {".hdf5", ".h5", ".npz", ".pkl", ".json", ".jsonl"}


def read_asset_paths(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    in_assets = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if re.match(r"^assets\s*:", line):
            in_assets = True
            continue
        if in_assets and re.match(r"^\S", line):
            break
        match = re.match(r"^\s+([A-Za-z0-9_]+)\s*:\s*(.*)$", line)
        if in_assets and match:
            value = match.group(2).strip().strip('"').strip("'")
            if value and value.lower() != "null":
                values[match.group(1)] = value
    return values


def read_subset_config(path: Path) -> dict:
    config: dict[str, object] = {
        "suites": list(DEFAULT_SUITES),
        "max_tasks_per_suite": 3,
        "max_counterfactual_pairs": 8,
    }
    if not path.exists():
        return config

    lines = path.read_text(encoding="utf-8").splitlines()
    active_list_key: str | None = None
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.endswith(":"):
            key = stripped[:-1].strip()
            active_list_key = key
            if key == "suites":
                config["suites"] = []
            continue
        if active_list_key and stripped.startswith("- "):
            value = stripped[2:].strip().strip('"').strip("'")
            if isinstance(config.get(active_list_key), list):
                config[active_list_key].append(value)
            continue
        active_list_key = None
        match = re.match(r"^([A-Za-z0-9_]+)\s*:\s*(.+)$", stripped)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip().strip('"').strip("'")
        if value.isdigit():
            config[key] = int(value)
        elif value.lower() in {"true", "false"}:
            config[key] = value.lower() == "true"
        else:
            config[key] = value
    return config


def find_bddl_root(libero_root: Path) -> Path | None:
    candidates = [
        libero_root / "libero" / "libero" / "bddl_files",
        libero_root / "libero" / "bddl_files",
        libero_root / "bddl_files",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _extract_balanced_section(text: str, marker: str) -> str:
    start = text.find(marker)
    if start < 0:
        return ""
    depth = 0
    seen_open = False
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
            seen_open = True
        elif char == ")":
            depth -= 1
            if seen_open and depth == 0:
                return text[start : index + 1]
    return text[start:]


def _clean_tokens(section: str, marker: str) -> list[str]:
    cleaned = section.replace(marker, " ")
    cleaned = re.sub(r"[()]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return [token for token in cleaned.split(" ") if token]


def _canonical_object_name(name: str) -> str:
    return re.sub(r"_\d+$", "", name).replace("_", " ")


def parse_bddl_task(path: Path, suite: str, libero_root: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    language_match = re.search(r"\(:language\s+(.+?)\)", text)
    language = language_match.group(1).strip() if language_match else path.stem.replace("_", " ")

    object_section = _extract_balanced_section(text, "(:obj_of_interest")
    object_tokens = _clean_tokens(object_section, ":obj_of_interest")
    goal_section = _extract_balanced_section(text, "(:goal")
    goal_tokens = _clean_tokens(goal_section, ":goal")

    objects_of_interest = [token for token in object_tokens if token != "-"]
    canonical_objects = [_canonical_object_name(token) for token in objects_of_interest]
    goal_text = re.sub(r"\s+", " ", goal_section).strip()

    try:
        relative_path = str(path.relative_to(libero_root)).replace("\\", "/")
    except ValueError:
        relative_path = str(path).replace("\\", "/")

    return {
        "suite": suite,
        "task_id": path.stem,
        "bddl_file": relative_path,
        "language": language,
        "objects_of_interest": objects_of_interest,
        "canonical_objects_of_interest": canonical_objects,
        "goal_tokens": goal_tokens,
        "goal_text": goal_text,
        "metadata_only": True,
        "requires_demo_file": False,
        "requires_simulator": False,
    }


def discover_bddl_tasks(libero_root: Path, suites: Iterable[str], max_tasks_per_suite: int | None = None) -> list[dict]:
    bddl_root = find_bddl_root(libero_root)
    if bddl_root is None:
        return []

    tasks: list[dict] = []
    for suite in suites:
        suite_root = bddl_root / suite
        if not suite_root.exists():
            continue
        task_paths: list[Path] = []
        task_info = suite_root / "tasks_info.txt"
        if task_info.exists():
            for line in task_info.read_text(encoding="utf-8", errors="replace").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                task_paths.append(suite_root / Path(stripped).name)
        else:
            task_paths.extend(sorted(suite_root.glob("*.bddl")))

        seen: set[Path] = set()
        suite_count = 0
        for task_path in task_paths:
            if task_path in seen or not task_path.exists() or task_path.suffix != ".bddl":
                continue
            seen.add(task_path)
            tasks.append(parse_bddl_task(task_path, suite=suite, libero_root=libero_root))
            suite_count += 1
            if max_tasks_per_suite is not None and suite_count >= max_tasks_per_suite:
                break
    return tasks


def make_counterfactual_metadata_pairs(tasks: list[dict], max_pairs: int = 8) -> list[dict]:
    pairs: list[dict] = []
    by_suite: dict[str, list[dict]] = {}
    for task in tasks:
        by_suite.setdefault(task["suite"], []).append(task)

    for suite, suite_tasks in by_suite.items():
        for left in suite_tasks:
            for right in suite_tasks:
                if left["task_id"] == right["task_id"]:
                    continue
                shared = sorted(
                    set(left.get("canonical_objects_of_interest", []))
                    & set(right.get("canonical_objects_of_interest", []))
                )
                if not shared and left.get("suite") != right.get("suite"):
                    continue
                pairs.append(
                    {
                        "pair_id": f"{suite}:{left['task_id']}__vs__{right['task_id']}",
                        "suite": suite,
                        "positive_task_id": left["task_id"],
                        "counterfactual_task_id": right["task_id"],
                        "positive_instruction": left["language"],
                        "counterfactual_instruction": right["language"],
                        "shared_canonical_objects": shared,
                        "swap_type": "metadata_target_or_goal_swap",
                        "requires_demo_actions": False,
                        "requires_simulator": False,
                    }
                )
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def find_dataset_files(root: Path | None, max_files: int = 20) -> list[str]:
    if root is None or not root.exists():
        return []
    files: list[str] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in DATASET_EXTENSIONS:
            files.append(str(path))
            if len(files) >= max_files:
                break
    return files


def build_libero_metadata_subset(
    libero_root: Path,
    libero_data_root: Path | None = None,
    suites: Iterable[str] = DEFAULT_SUITES,
    max_tasks_per_suite: int = 3,
    max_counterfactual_pairs: int = 8,
) -> dict:
    tasks = discover_bddl_tasks(
        libero_root=libero_root,
        suites=suites,
        max_tasks_per_suite=max_tasks_per_suite,
    )
    pairs = make_counterfactual_metadata_pairs(tasks, max_pairs=max_counterfactual_pairs)
    dataset_files = find_dataset_files(libero_data_root)
    bddl_root = find_bddl_root(libero_root)

    ready_for_metadata_only_subset = bool(libero_root.exists() and bddl_root and tasks)
    ready_for_real_dataset_interface_smoke = bool(ready_for_metadata_only_subset and dataset_files)

    return {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "metadata_only": True,
            "downloads_performed": False,
            "gpu_jobs_performed": False,
            "training_performed": False,
            "simulator_executed": False,
            "rollouts_performed": False,
            "heavy_model_imports_performed": False,
            "openvla_oft_executed": False,
            "tokens_read_or_written": False,
            "paper_grade_claims_made": False,
        },
        "paths": {
            "libero_root": str(libero_root),
            "libero_root_exists": libero_root.exists(),
            "bddl_root": str(bddl_root) if bddl_root else None,
            "bddl_root_exists": bool(bddl_root and bddl_root.exists()),
            "libero_data_root": str(libero_data_root) if libero_data_root else None,
            "libero_data_root_exists": bool(libero_data_root and libero_data_root.exists()),
        },
        "suites": list(suites),
        "max_tasks_per_suite": max_tasks_per_suite,
        "selected_task_count": len(tasks),
        "counterfactual_pair_count": len(pairs),
        "tasks": tasks,
        "counterfactual_pairs": pairs,
        "dataset_probe": {
            "data_files_detected": bool(dataset_files),
            "sample_files": dataset_files,
            "note": "Metadata-only manifest can be built without demos; real offline dataset smoke still requires demo files.",
        },
        "ready_for_metadata_only_subset": ready_for_metadata_only_subset,
        "ready_for_real_dataset_interface_smoke": ready_for_real_dataset_interface_smoke,
        "ready_for_rollout": False,
        "recommended_next_step": (
            "Use this manifest to validate target/counterfactual split plumbing only. "
            "Place a documented tiny LIBERO demo subset before real offline dataset interface smoke."
            if ready_for_metadata_only_subset and not ready_for_real_dataset_interface_smoke
            else "Run a bounded offline dataset interface smoke only if demo files are present and risk checks stay green."
        ),
    }


def write_reports(report: dict, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# LIBERO Metadata Subset Report",
        "",
        f"- schema version: `{report['schema_version']}`",
        f"- metadata-only subset ready: `{report['ready_for_metadata_only_subset']}`",
        f"- real dataset interface smoke ready: `{report['ready_for_real_dataset_interface_smoke']}`",
        f"- selected tasks: `{report['selected_task_count']}`",
        f"- counterfactual pairs: `{report['counterfactual_pair_count']}`",
        f"- rollout ready: `{report['ready_for_rollout']}`",
        "",
        "This report is metadata-only. It performs no downloads, GPU jobs, training, rollouts, simulator execution, heavy VLA imports, token access, OpenVLA-OFT execution, or paper-grade claims.",
        "",
        "## Next Step",
        report["recommended_next_step"],
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-file", default="configs/paths.local.yaml")
    parser.add_argument("--config", default="configs/libero_metadata_subset.yaml")
    parser.add_argument("--libero-root", default="")
    parser.add_argument("--libero-data-root", default="")
    parser.add_argument("--report-json", default="reports/libero_metadata_subset_report.json")
    parser.add_argument("--report-md", default="reports/libero_metadata_subset_report.md")
    args = parser.parse_args()

    paths = read_asset_paths(Path(args.paths_file))
    config = read_subset_config(Path(args.config))
    libero_root = Path(args.libero_root or paths.get("libero_root", "C:/assets/repos/LIBERO"))
    data_root_value = args.libero_data_root or paths.get("libero_data_root", "C:/assets/data/libero")
    suites = config.get("suites") or list(DEFAULT_SUITES)

    report = build_libero_metadata_subset(
        libero_root=libero_root,
        libero_data_root=Path(data_root_value) if data_root_value else None,
        suites=[str(suite) for suite in suites],
        max_tasks_per_suite=int(config.get("max_tasks_per_suite", 3)),
        max_counterfactual_pairs=int(config.get("max_counterfactual_pairs", 8)),
    )
    write_reports(report, json_path=Path(args.report_json), markdown_path=Path(args.report_md))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
