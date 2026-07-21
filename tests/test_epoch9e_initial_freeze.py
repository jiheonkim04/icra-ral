from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SOURCE = "74dd66c32a8b8595e187b13d3ccafe05cae6753b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def protected_snapshot(path: Path) -> tuple[int, int, str]:
    files = sorted(value for value in path.rglob("*") if value.is_file())
    lines = [
        f"{str(value.relative_to(ROOT)).replace(chr(92), '/')}\t{value.stat().st_size}\t{sha256(value)}"
        for value in files
    ]
    return (
        len(files),
        sum(value.stat().st_size for value in files),
        hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest().upper(),
    )


def test_epoch9e_authority_is_append_only_from_exact_source() -> None:
    scope = json.loads((REPORTS / "epoch9e_scope_and_authority_correction.json").read_text(encoding="utf-8"))
    assert scope["source_commit_full"] == SOURCE
    assert scope["epoch9e_branch"] == "codex/epoch9e-nondrag-disengagement-convergence"
    assert scope["append_only"] is True
    assert scope["epoch9d_files_modified"] == []
    assert scope["epoch9d_scientific_scope_correction"]["exact_interpretation"] == [
        "CAUSAL_MASS_SIGNAL_CONFIRMED",
        "VARIANT1_PILOT_FROZEN_NO_GO",
        "PRE_RESPONSE_DISENGAGEMENT_UNRESOLVED",
        "PAPER_NOT_AUTHORIZED",
    ]
    assert scope["sealed_source_demo_identities"]["accessed_by_epoch9e"] == []


def test_every_source_tracked_epoch9_artifact_is_unchanged() -> None:
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE, "HEAD"], cwd=ROOT, check=False
    )
    assert ancestry.returncode == 0
    scope = json.loads((REPORTS / "epoch9e_scope_and_authority_correction.json").read_text(encoding="utf-8"))
    for record in scope["historical_evidence"].values():
        assert sha256(ROOT / record["path"]) == record["sha256"]


def test_epoch9d_raw_causal_go_and_lane_failures_reproduce() -> None:
    audit = json.loads((REPORTS / "epoch9e_endpoint_construct_audit.json").read_text(encoding="utf-8"))
    causal = audit["phase_b_raw_reproduction"]
    lane = audit["variant1_lane_failure_reconstruction"]
    assert causal["trace_binding_count"] == 80
    assert causal["all_trace_hashes_match_raw_rows"] is True
    assert causal["counts"]["rank_correct"] == 28
    assert causal["counts"]["exact_pair_correct_flips"] == 12
    assert causal["paired_mass_intervention"]["mean_m"] == 0.006593329847616967
    assert causal["paired_mass_intervention"]["paired_student_t_95_interval_m"][0] > 0
    assert causal["paired_mass_intervention"]["one_sided_exact_sign_test_p"] < 0.01
    assert all(causal["gates"].values())
    assert lane["causal_exit_count"] == 2
    assert lane["inherited_failure_count"] == 2
    assert lane["causal_exit_phases"] == ["contact_verify_retract", "contact_verify_retract"]
    assert lane["construct"]["epoch9e_rule"].startswith("retain the original")
    assert audit["joint_certification_endpoint_decision"]["full_trajectory_lane_reachability_gate"] == "48/48 RETAINED"


def test_epoch9e_fresh_identity_and_seed_allocations_are_literal_and_disjoint() -> None:
    manifest = json.loads((REPORTS / "epoch9e_fresh_identity_manifest.json").read_text(encoding="utf-8"))
    prior_ids = set(manifest["epoch9_identity_values"])
    prior_seeds = set(manifest["previous_seed_values"])
    allocations = manifest["allocations"]
    new_ids = [number for group in allocations.values() for number in group["identity_ids"]]
    new_seeds = [number for group in allocations.values() for number in group["generator_seeds"]]
    assert manifest["maximum_used_numeric_development_identity_M"] == 20260916
    assert manifest["maximum_prior_epoch9_identity_reference"] == 20261125
    assert manifest["allocation_floor"] == 20261126
    assert min(new_ids) > manifest["maximum_used_numeric_development_identity_M"]
    assert len(new_ids) == len(set(new_ids))
    assert len(new_seeds) == len(set(new_seeds))
    assert not (set(new_ids) & prior_ids)
    assert not (set(new_seeds) & prior_seeds)
    assert not (set(new_ids) & set(range(40, 50)))
    assert all(manifest["disjointness_audit"].values())
    assert not any(manifest["stage_access"].values())


def test_epoch9e_protected_rollouts_remain_exact() -> None:
    assert protected_snapshot(ROOT / "rollouts/2026_07_17") == (
        27,
        5_143_751,
        "25DE8FF5AA6112D7EFF8BCF38D3A4C3F0F3C8C8EE0458E5FA83D17438719EC54",
    )
    assert protected_snapshot(ROOT / "rollouts/2026_07_18") == (
        10,
        924_633,
        "CF701D6F73D4783F016E48A72C093DC9FD6D940B7081DA8FBEC128DB94C24A00",
    )
