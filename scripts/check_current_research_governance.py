from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

ALLOWED_FINAL_STATES = [
    "READY_TO_DRAFT_RAL_PAPER_PACKAGE",
    "AUTONOMOUS_CAMPAIGN_PAUSED_RESUMABLE",
    "HARD_EXTERNAL_BLOCKER",
    "SAFETY_RESOURCE_STOP",
]

ACTIVE_TEXT_FILES = [
    Path("AGENTS.md"),
    Path("reports/current_research_governance.md"),
    Path("reports/codex_delegation_manual.md"),
    Path("reports/autonomous_until_paper_state.md"),
    Path("reports/autonomous_until_paper_final_decision.md"),
    Path("reports/autonomous_ral_campaign_state.md"),
    Path("reports/autonomous_ral_final_decision.md"),
]

ACTIVE_JSON_FILES = [
    Path("reports/autonomous_until_paper_state.json"),
    Path("reports/autonomous_ral_campaign_state.json"),
]


def _repo_path(path: Path, root: Path = REPO_ROOT) -> Path:
    return root / path


def _line_is_deprecated_context(path: Path, lines: list[str], index: int) -> bool:
    line = lines[index]
    if path == Path("reports/current_research_governance.md"):
        before = "\n".join(lines[max(0, index - 8) : index + 1])
        after = "\n".join(lines[index : min(len(lines), index + 8)])
        return (
            "Historical reports, prompts, state files" in before
            and "deprecated historical markers" in after
        )
    if path == Path("AGENTS.md"):
        before = "\n".join(lines[max(0, index - 10) : index + 1])
        after = "\n".join(lines[index : min(len(lines), index + 12)])
        return "## Deprecated Instructions" in before and "## Research Campaign Semantics" in after
    if "not universally prohibited" in line:
        return True
    return False


def _text_violations(path: Path, text: str) -> list[str]:
    lines = text.splitlines()
    violations: list[str] = []
    patterns = [
        ("NO_METHOD_AFTER", re.compile(r"NO_METHOD_AFTER")),
        ("TWO_METHODS_KILLED as a terminal state", re.compile(r"TWO_METHODS_KILLED.*(terminal|final|stop)", re.I)),
        ("one-major-milestone-per-execution", re.compile(r"one-major-milestone-per-execution", re.I)),
        ("TCA-Select required as final method", re.compile(r"TCA-Select.*required.*final method", re.I)),
        ("OpenVLA-OFT INT4 universally prohibited", re.compile(r"OpenVLA-OFT INT4.*universally prohibited", re.I)),
    ]

    for i, line in enumerate(lines):
        if _line_is_deprecated_context(path, lines, i):
            continue
        for label, pattern in patterns:
            if pattern.search(line):
                violations.append(f"{path}:{i + 1}: prohibited active governance text: {label}")
    return violations


def _collect_text_current_decisions(path: Path, text: str) -> list[str]:
    decisions: list[str] = []
    for match in re.finditer(r"Current (?:campaign )?decision:\s*`([^`]+)`", text):
        decisions.append(match.group(1))
    return decisions


def _json_violations(path: Path, data: dict[str, Any]) -> list[str]:
    violations: list[str] = []

    if data.get("maximum_method_cycles") is not None:
        violations.append(f"{path}: maximum_method_cycles must be null in active state")

    if data.get("global_no_method_terminal_allowed") is not False:
        violations.append(f"{path}: global_no_method_terminal_allowed must be false")

    current_decision = str(data.get("current_decision", ""))
    if "NO_METHOD_AFTER" in current_decision:
        violations.append(f"{path}: current_decision contains prohibited no-method terminal")
    if current_decision == "TWO_METHODS_KILLED":
        violations.append(f"{path}: current_decision contains obsolete global terminal")

    final_states = data.get("valid_final_states")
    if final_states != ALLOWED_FINAL_STATES:
        violations.append(f"{path}: valid_final_states must be exactly {ALLOWED_FINAL_STATES}")

    serialized = json.dumps(data, sort_keys=True)
    if "NO_METHOD_AFTER" in serialized:
        violations.append(f"{path}: active state contains prohibited NO_METHOD_AFTER text")
    if "TWO_METHODS_KILLED" in serialized:
        violations.append(f"{path}: active state contains obsolete TWO_METHODS_KILLED text")

    return violations


def validate(root: Path = REPO_ROOT) -> list[str]:
    violations: list[str] = []
    current_decisions: list[tuple[Path, str]] = []

    for rel_path in ACTIVE_TEXT_FILES:
        path = _repo_path(rel_path, root)
        if not path.exists():
            violations.append(f"{rel_path}: missing active governance file")
            continue
        text = path.read_text(encoding="utf-8")
        violations.extend(_text_violations(rel_path, text))
        current_decisions.extend((rel_path, value) for value in _collect_text_current_decisions(rel_path, text))

    for rel_path in ACTIVE_JSON_FILES:
        path = _repo_path(rel_path, root)
        if not path.exists():
            violations.append(f"{rel_path}: missing active state file")
            continue
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        violations.extend(_json_violations(rel_path, data))
        if "current_decision" in data:
            current_decisions.append((rel_path, str(data["current_decision"])))

    distinct_decisions = {value for _, value in current_decisions}
    if len(distinct_decisions) > 1:
        formatted = ", ".join(f"{path}={decision}" for path, decision in current_decisions)
        violations.append(f"Contradictory current_decision values: {formatted}")

    return violations


def main() -> int:
    violations = validate()
    if violations:
        print("Current research governance check failed:")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print("Current research governance check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
