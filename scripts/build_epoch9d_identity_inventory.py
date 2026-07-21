#!/usr/bin/env python3
"""Inventory identity and seed references before Epoch 9D allocation.

The inventory is deliberately value-complete but occurrence-compact: every
distinct referenced numeric identity, symbolic identity, and seed is retained,
while source occurrences are counted instead of copied into a very large
ledger.  Epoch 9 development identities are reported separately because they
define the allocation boundary required by the Epoch 9D protocol.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json


OUTPUT = ROOT / "reports/epoch9d_identity_seed_inventory.json"
SCAN_ROOTS = (ROOT / "reports", ROOT / "rollouts")
TEXT_SUFFIXES = {
    ".bib",
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
NUMERIC_IDENTITY_KEY = re.compile(
    r"(?:identity|identities|demo_(?:index|indices)|init_state_(?:id|ids|index|indices)|"
    r"reset_(?:id|ids|index|indices))",
    re.IGNORECASE,
)
SYMBOLIC_IDENTITY_KEY = re.compile(
    r"(?:identity|scene_id|episode_id|reset_id|run_id|pair_id|base_state_id)", re.IGNORECASE
)
SEED_KEY = re.compile(r"seed", re.IGNORECASE)
TEXT_PATTERNS = {
    "demo_index": re.compile(r"\bdemo[_ -]?(\d+)\b", re.IGNORECASE),
    "init_state_index": re.compile(r"\binit(?:ial)?[_ -]?state(?:[_ -]?(?:id|index))?\D{0,8}(\d+)\b", re.IGNORECASE),
    "reset_index": re.compile(r"\breset(?:[_ -]?(?:id|index))?\D{0,8}(\d+)\b", re.IGNORECASE),
    "identity": re.compile(r"\bidentit(?:y|ies)\D{0,12}(\d+)\b", re.IGNORECASE),
    "seed": re.compile(r"\bseed(?:s)?\D{0,8}(\d+)\b", re.IGNORECASE),
}


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def integer_values(value: Any) -> Iterable[int]:
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield int(value)
    elif isinstance(value, list):
        for item in value:
            yield from integer_values(item)


def scalar_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from scalar_strings(item)


def compact_ranges(values: set[int]) -> list[str]:
    if not values:
        return []
    ordered = sorted(values)
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ranges


def partition_hint(path: Path, key_path: tuple[str, ...], enclosing: dict[str, Any] | None) -> str:
    tokens = [relative(path).lower(), *(value.lower() for value in key_path)]
    if enclosing:
        for key in ("partition", "evidence_class", "split", "stage"):
            value = enclosing.get(key)
            if isinstance(value, str):
                tokens.append(value.lower())
    joined = " ".join(tokens)
    if "confirmation" in joined or "confirmatory" in joined:
        return "CONFIRMATION"
    if "validation" in joined:
        return "VALIDATION"
    if "development" in joined or "discovery" in joined or "calibration" in joined:
        return "DEVELOPMENT"
    return "UNSPECIFIED"


def main() -> int:
    numeric: dict[str, set[int]] = defaultdict(set)
    numeric_occurrences: Counter[str] = Counter()
    symbolic: dict[str, set[str]] = defaultdict(set)
    symbolic_occurrences: Counter[str] = Counter()
    seeds: set[int] = set()
    seed_occurrences = 0
    development_numeric: set[int] = set()
    epoch9_development_demo_indices: set[int] = set()
    scanned_files = 0
    parsed_json_files = 0
    text_files = 0
    parse_failures: list[str] = []

    def walk(value: Any, path: Path, key_path: tuple[str, ...] = (), enclosing: dict[str, Any] | None = None) -> None:
        nonlocal seed_occurrences
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = (*key_path, str(key))
                hint = partition_hint(path, child_path, value)
                if SEED_KEY.search(str(key)):
                    found = list(integer_values(child))
                    seeds.update(found)
                    seed_occurrences += len(found)
                if NUMERIC_IDENTITY_KEY.search(str(key)):
                    kind = str(key).lower()
                    found = list(integer_values(child))
                    numeric[kind].update(found)
                    numeric_occurrences[kind] += len(found)
                    if hint == "DEVELOPMENT":
                        development_numeric.update(found)
                    if "epoch9" in relative(path).lower() and hint == "DEVELOPMENT" and "demo" in kind:
                        epoch9_development_demo_indices.update(found)
                if SYMBOLIC_IDENTITY_KEY.search(str(key)):
                    kind = str(key).lower()
                    found_strings = list(scalar_strings(child))
                    symbolic[kind].update(found_strings)
                    symbolic_occurrences[kind] += len(found_strings)
                walk(child, path, child_path, value)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, path, (*key_path, f"[{index}]"), enclosing)

    for scan_root in SCAN_ROOTS:
        for path in sorted(value for value in scan_root.rglob("*") if value.is_file()):
            scanned_files += 1
            path_text = relative(path)
            # Filenames and directory names can carry references even when the
            # payload is binary (for example, NPZ trace names).
            for kind, pattern in TEXT_PATTERNS.items():
                for match in pattern.finditer(path_text):
                    value = int(match.group(1))
                    if kind == "seed":
                        seeds.add(value)
                        seed_occurrences += 1
                    else:
                        numeric[f"path_{kind}"].add(value)
                        numeric_occurrences[f"path_{kind}"] += 1
                        if "development" in path_text.lower():
                            development_numeric.add(value)
                        if "epoch9" in path_text.lower() and "development" in path_text.lower() and kind == "demo_index":
                            epoch9_development_demo_indices.add(value)

            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            text_files += 1
            try:
                text = path.read_text(encoding="utf-8", errors="strict")
            except UnicodeDecodeError:
                parse_failures.append(path_text)
                continue
            if path.suffix.lower() == ".json":
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    parse_failures.append(path_text)
                else:
                    parsed_json_files += 1
                    walk(payload, path)
            for kind, pattern in TEXT_PATTERNS.items():
                for match in pattern.finditer(text):
                    number = int(match.group(1))
                    if kind == "seed":
                        seeds.add(number)
                        seed_occurrences += 1
                    else:
                        numeric[f"text_{kind}"].add(number)
                        numeric_occurrences[f"text_{kind}"] += 1

    # The allocation boundary is task-specific: Epoch 9 uses source-demo/reset
    # indices as identities.  Scene counters and unrelated campaign reset
    # namespaces are not silently conflated with this identity space.
    if not epoch9_development_demo_indices:
        raise RuntimeError("no Epoch 9 development demo identities were discovered")
    epoch9_m = max(epoch9_development_demo_indices)
    result = {
        "schema_version": "epoch9d.identity_seed_inventory.v1",
        "generated_at": timestamp(),
        "scan_scope": ["reports/**", "rollouts/** (including all evidence ledgers under reports)"],
        "scanned_file_count": scanned_files,
        "text_file_count": text_files,
        "parsed_json_file_count": parsed_json_files,
        "text_parse_failures": sorted(set(parse_failures)),
        "numeric_identity_values_by_reference_kind": {
            key: sorted(values) for key, values in sorted(numeric.items())
        },
        "numeric_identity_occurrence_count_by_reference_kind": dict(sorted(numeric_occurrences.items())),
        "symbolic_identity_values_by_reference_kind": {
            key: sorted(values) for key, values in sorted(symbolic.items())
        },
        "symbolic_identity_occurrence_count_by_reference_kind": dict(sorted(symbolic_occurrences.items())),
        "seed_values": sorted(seeds),
        "seed_ranges": compact_ranges(seeds),
        "seed_occurrence_count": seed_occurrences,
        "all_development_numeric_identity_values": sorted(development_numeric),
        "epoch9_development_demo_identity_values": sorted(epoch9_development_demo_indices),
        "epoch9_largest_previously_used_numeric_development_identity_M": epoch9_m,
        "identity_namespace_note": (
            "M is computed in the Epoch 9 source-demo/reset identity namespace used by the frozen "
            "active-probe protocols. Unrelated task IDs, scene counters, and generator seeds remain "
            "enumerated but are not conflated with this namespace."
        ),
        "sealed_epoch9_identity_ranges": {
            "validation": [40, 41, 42, 43, 44],
            "confirmation": [45, 46, 47, 48, 49],
        },
    }
    atomic_write_json(OUTPUT, result)
    print(json.dumps({
        "output": relative(OUTPUT),
        "scanned_files": scanned_files,
        "distinct_seeds": len(seeds),
        "epoch9_M": epoch9_m,
        "epoch9_development_demo_identities": sorted(epoch9_development_demo_indices),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
