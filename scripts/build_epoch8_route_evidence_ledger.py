#!/usr/bin/env python3
"""Build the time-boxed Epoch 8 route ledger from immutable prior ledgers.

This is deliberately conservative: legacy rows that do not state role-specific
episode counts remain UNVERIFIED instead of being reverse-engineered from a
summary percentage.  The generated ledger is an audit bridge, not new evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "autonomous_research_full_history_audit.md"
OUT_JSON = ROOT / "reports" / "epoch8_route_evidence_ledger.json"
OUT_MD = ROOT / "reports" / "epoch8_route_evidence_audit.md"

CLASSES = [
    "EMPIRICAL_PROBLEM_FALSIFICATION",
    "EMPIRICAL_EXACT_METHOD_FALSIFICATION",
    "UNDERPOWERED_OR_AMBIGUOUS",
    "IMPLEMENTATION_INVALID",
    "RESOURCE_BLOCKED",
    "ARTIFACT_BLOCKED",
    "POSITIONING_OR_OVERLAP_RISK",
    "STATIC_IDEA_REJECTION",
    "POSITIVE_PROBLEM_EVIDENCE",
    "POSITIVE_METHOD_EVIDENCE",
]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_bool(value: str):
    table = {"yes": True, "no": False, "partial": "PARTIAL", "diagnostic": "DIAGNOSTIC", "na": None}
    return table.get(value.strip().lower(), "UNVERIFIED")


def extract_paths(value: str) -> list[str]:
    return re.findall(r"`([^`]+)`", value)


def artifact_status(paths: list[str]) -> dict:
    if not paths:
        return {"status": "UNVERIFIED", "present": 0, "referenced": 0, "missing": []}
    present = 0
    missing: list[str] = []
    for raw in paths:
        candidates = list(ROOT.glob(raw)) if "*" in raw else [ROOT / raw]
        if any(p.exists() for p in candidates):
            present += 1
        else:
            missing.append(raw)
    return {
        "status": "PRESENT" if not missing else "PARTIAL_OR_MISSING",
        "present": present,
        "referenced": len(paths),
        "missing": missing,
    }


PROBLEM_GROUPS = {
    "target_grounding_and_action_mapping": ["tca", "actionmap", "css-shield", "gcap", "br-xvla", "ocb-xvla", "mpr-xvla", "prc-xvla"],
    "executable_policy_specification": ["execspec"],
    "temporal_execution_and_chunking": ["phase/event", "tl-chunkrepair", "phasebarrier", "dicd", "ptc-vla", "eac-vla", "4-step openvla requery"],
    "contact_and_interaction_representation": ["contacttube", "contactset", "s2c-vla", "lcg-vla"],
    "visual_robustness_and_view_correction": ["prism", "patchguard", "ocfn", "scvc", "pse-vla", "covi", "vdr-vla"],
    "safety_and_failure_conditioned_control": ["safetrace", "safelora", "censorcredit", "fedo", "sacf", "isac", "fang"],
    "residual_action_correction": ["fcar", "rar", "marc", "iarc", "famr", "sparc", "cfr", "urf", "brid", "cspr", "post-canonical"],
    "memory_retrieval_and_context": ["rcv", "cavm", "dagr", "rap-vla"],
    "history_state_and_representation": ["evostate", "mtf", "pesa", "cala", "nice", "hest", "haste", "kite", "tsc", "mhs", "afid", "mci"],
    "official_prior_residual_ecosystem": ["openvla", "openpi", "pcd", "lightvla", "ript", "vla-gse", "x-vla", "vla-0", "vla-jepa", "r2r-oft", "cr-lightvla", "atcd"],
}


def problem_id(name: str, class_name: str) -> str:
    n = name.lower()
    for group, needles in PROBLEM_GROUPS.items():
        if any(x in n for x in needles):
            return group
    return slug(class_name or name)


DEDUP_METHOD = {
    "TCA-Map": "target_prior_action_heatmap_campaign",
    "TCA-Select": "target_prior_action_heatmap_campaign",
    "ActionMap mini-anchor": "target_prior_action_heatmap_campaign",
    "Custom SmolVLA 7D adapter": "custom_smolvla_7d_adapter_family",
    "TG-7D": "custom_smolvla_7d_adapter_family",
}


def entry_kind(row: dict) -> str:
    n = row["route_name"].lower()
    c = row["method_class"].lower()
    final = row["final_decision"].lower()
    if "candidate" in row["epoch_cycle"].lower() or "not selected" in final or "unselected" in final:
        return "STATIC_CANDIDATE"
    if c in {"prior diagnostic", "resource check", "residual mining"} or "diagnostic" in n or "identity scan" in n or "residual" in n and "vla" not in n:
        return "PRIOR_OR_PROBLEM_DIAGNOSTIC"
    if c in {"simple control", "infrastructure"}:
        return "CONTROL_OR_INFRASTRUCTURE"
    if row["route_name"] in DEDUP_METHOD:
        return "METHOD_VARIANT"
    return "EXACT_METHOD"


def evidence_class(row: dict) -> str:
    n = row["route_name"].lower()
    final = row["final_decision"].lower()
    result = row["observed_result"].lower()
    legacy = row["legacy_evidence_class"].lower()
    kind = entry_kind(row)

    if any(x in final + " " + result for x in ["resource blocker", "resource-blocked", "compute infeasible"]):
        return "RESOURCE_BLOCKED"
    if any(x in final + " " + result for x in ["dependency/checkpoint", "no trained checkpoint", "assets/task/resources not comparable"]):
        return "ARTIFACT_BLOCKED"
    if "unknown" in legacy or "preimplementation" in final or kind == "STATIC_CANDIDATE":
        return "STATIC_IDEA_REJECTION"
    if "invalid_quarantined" in legacy:
        if "resource" in final or "compute" in result or "giB" in row["observed_result"]:
            return "RESOURCE_BLOCKED"
        return "IMPLEMENTATION_INVALID"
    if n in {"post-canonical residual mining", "echo", "x-vla identity scan"} and ("no headroom" in final + result or "10/10" in result):
        return "EMPIRICAL_PROBLEM_FALSIFICATION"
    if n in {"x-vla task1 residual", "x-vla task6 residual", "x-vla libero-90 task75"}:
        return "POSITIVE_PROBLEM_EVIDENCE"
    if n in {"openvla-oft int4 diagnostic", "4-step openvla requery", "lightvla", "x-vla task8", "x-vla spatial task5"}:
        return "POSITIVE_METHOD_EVIDENCE"
    if n == "x-vla libero-90 tasks81/83":
        return "UNDERPOWERED_OR_AMBIGUOUS"
    if n == "cavm-vla":
        return "POSITIVE_METHOD_EVIDENCE"
    if "inconclusive" in legacy:
        return "UNDERPOWERED_OR_AMBIGUOUS"
    if row["closure"] is True and ("valid_canonical" in legacy or "valid_historical" in legacy):
        if kind in {"PRIOR_OR_PROBLEM_DIAGNOSTIC", "CONTROL_OR_INFRASTRUCTURE"}:
            return "POSITIVE_METHOD_EVIDENCE"
        return "EMPIRICAL_EXACT_METHOD_FALSIFICATION"
    if "valid_canonical" in legacy or "valid_historical" in legacy or "superseded" in legacy:
        return "POSITIVE_METHOD_EVIDENCE"
    return "UNDERPOWERED_OR_AMBIGUOUS"


CLOSED_LOOP_OVERRIDES = {
    16: {"base": 12, "prior": 0, "ours": 0, "controls": 36, "note": "three LoRA seeds, 12 episodes each"},
    17: {"base": 0, "prior": 0, "ours": 0, "controls": 0, "note": "0/6 was expert replay, not official policy closed loop"},
    23: {"base": 16, "prior": 36, "ours": 0, "controls": 0, "note": "20 hard-slice plus 16 residual Prior episodes; 16 matched Base residual episodes"},
    24: {"base": 20, "prior": 0, "ours": 20, "controls": 20, "note": "full/Base/ablation frozen panel"},
    27: {"base": 0, "prior": 0, "ours": 10, "controls": 10, "note": "DICD versus direct chunk-index control"},
    30: {"base": 10, "prior": 0, "ours": 10, "controls": 0, "note": "Stage A"},
    31: {"base": 10, "prior": 0, "ours": 10, "controls": 0, "note": "Stage A"},
    32: {"base": 0, "prior": 0, "ours": 80, "controls": 80, "note": "full versus zero-noise control"},
    33: {"base": 10, "prior": 0, "ours": 10, "controls": 0, "note": "Stage A"},
    34: {"base": 40, "prior": 0, "ours": 40, "controls": 0, "note": "Stage B shifted Base"},
    35: {"base": 0, "prior": 0, "ours": 80, "controls": 80, "note": "full versus bright-single control"},
    36: {"base": 0, "prior": 0, "ours": 40, "controls": "UNVERIFIED", "note": "two controls are summarized together at 24/40"},
    37: {"base": 58, "prior": 58, "ours": 58, "controls": 58, "note": "Base, nearest-success memory, full, no-contrast"},
    38: {"base": 40, "prior": 0, "ours": 40, "controls": 40, "note": "Base/full/ablation"},
    40: {"base": 40, "prior": 40, "ours": 40, "controls": 80, "note": "Base, proxy, full, ablation, inverse"},
    41: {"base": 0, "prior": 0, "ours": 40, "controls": 40, "note": "full versus no-retention"},
    42: {"base": 40, "prior": 0, "ours": 40, "controls": 0, "note": "full versus Base"},
    43: {"base": 10, "prior": 0, "ours": 10, "controls": 0, "note": "Stage A"},
    45: {"base": 40, "prior": 40, "ours": 40, "controls": 40, "note": "Base, AAC, full, ablation"},
    75: {"base": 0, "prior": 8, "ours": 0, "controls": 8, "note": "four-step control versus original OpenVLA"},
    78: {"base": 0, "prior": 16, "ours": 0, "controls": 0, "note": "LightVLA and matched OpenVLA, eight each"},
    79: {"base": 0, "prior": 8, "ours": 8, "controls": 0, "note": "CR-LightVLA and parent LightVLA"},
    83: {"base": 0, "prior": 8, "ours": 0, "controls": 0, "note": "official X-VLA prior diagnostic"},
    84: {"base": 0, "prior": 10, "ours": 0, "controls": 0, "note": "X-VLA identity scan"},
    85: {"base": 8, "prior": 8, "ours": 0, "controls": 0, "note": "matched Base/X-VLA diagnostic"},
    88: {"base": "UNVERIFIED", "prior": 1, "ours": 1, "controls": 1, "note": "one-identity residual screen; Base row inherited from problem diagnostic"},
    90: {"base": 8, "prior": 10, "ours": 0, "controls": 0, "note": "X-VLA eight plus OpenVLA INT4 two"},
    93: {"base": 1, "prior": 2, "ours": 0, "controls": 0, "note": "X-VLA shared-residual row plus one OpenVLA solve; only role-explicit rows counted"},
    94: {"base": 2, "prior": 20, "ours": 0, "controls": 0, "note": "X-VLA scan plus matched two-row Base diagnostic"},
    95: {"base": 1, "prior": 20, "ours": 0, "controls": 0, "note": "X-VLA scan plus matched one-row Base diagnostic"},
}


def closed_loop_counts(row: dict) -> dict:
    if row["number"] in CLOSED_LOOP_OVERRIDES:
        out = dict(CLOSED_LOOP_OVERRIDES[row["number"]])
        vals = [out[k] for k in ("base", "prior", "ours", "controls")]
        out["total_observed"] = sum(v for v in vals if isinstance(v, int)) if all(isinstance(v, int) for v in vals) else "PARTIAL_SUM_ONLY"
        out["verification"] = "ROLE_COUNTS_PARSED_OR_CROSS_CHECKED_FROM_LEDGER_SUMMARY"
        return out
    if row["simulator"] is False:
        return {"base": 0, "prior": 0, "ours": 0, "controls": 0, "total_observed": 0, "verification": "NO_SIMULATOR_ROLLOUT_RECORDED"}
    return {"base": "UNVERIFIED", "prior": "UNVERIFIED", "ours": "UNVERIFIED", "controls": "UNVERIFIED", "total_observed": "UNVERIFIED", "verification": "LEGACY_ROW_LACKS_ROLE_SPECIFIC_COUNTS"}


def reopen_condition(cls: str) -> str:
    return {
        "EMPIRICAL_PROBLEM_FALSIFICATION": "A new problem instance or materially different preregistered condition outside the supported scope.",
        "EMPIRICAL_EXACT_METHOD_FALSIFICATION": "A causally distinct mechanism, not a direct reparameterization, with a fresh frozen falsifier.",
        "UNDERPOWERED_OR_AMBIGUOUS": "A preregistered adequately powered execution or resolution of the stated ambiguity without outcome-driven tuning.",
        "IMPLEMENTATION_INVALID": "A semantically null implementation/data repair with a focused test, then rerun of the unchanged scientific stage.",
        "RESOURCE_BLOCKED": "A measured resource-safe execution path preserving the scientific contract.",
        "ARTIFACT_BLOCKED": "A faithful official artifact/checkpoint and compatible local interface with license provenance.",
        "POSITIONING_OR_OVERLAP_RISK": "A current primary-source audit demonstrating a material causal and claim-level distinction.",
        "STATIC_IDEA_REJECTION": "New empirical problem evidence plus a selected executable mechanism; static rejection is not scientific closure.",
        "POSITIVE_PROBLEM_EVIDENCE": "Reusable only within its recorded scope; confirmation must use untouched identities.",
        "POSITIVE_METHOD_EVIDENCE": "Reusable as scoped positive evidence; paper-level claims still require the missing controls and confirmation.",
    }[cls]


def parse_rows() -> list[dict]:
    rows: list[dict] = []
    in_master_ledger = False
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if line.startswith("## 4. Master Method Ledger"):
            in_master_ledger = True
            continue
        if in_master_ledger and line.startswith("## 5."):
            break
        if not in_master_ledger:
            continue
        if not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        fields = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(fields) != 21:
            raise RuntimeError(f"Unexpected master-ledger field count {len(fields)}: {line[:120]}")
        (num, epoch, name, idea, method_class, prior, commit, impl, train, gpu, sim,
         s0, sa, sb, bb2, result, final, legacy, closure, reopen, evidence) = fields
        paths = extract_paths(evidence)
        row = {
            "number": int(num),
            "epoch_cycle": epoch,
            "route_name": name,
            "core_idea": idea,
            "method_class": method_class,
            "closest_prior": prior,
            "branch_or_commit": commit.replace("`", ""),
            "implemented": parse_bool(impl),
            "trained_or_checkpointed": parse_bool(train),
            "gpu_used": parse_bool(gpu),
            "simulator": parse_bool(sim),
            "stage0": parse_bool(s0),
            "stage_a": parse_bool(sa),
            "stage_b": parse_bool(sb),
            "second_backbone": parse_bool(bb2),
            "observed_result": result.replace("`", ""),
            "final_decision": final,
            "legacy_evidence_class": legacy,
            "closure": parse_bool(closure),
            "legacy_reopen": reopen,
            "evidence_paths": paths,
        }
        row["entry_kind"] = entry_kind(row)
        row["unique_thesis_problem_id"] = problem_id(name, method_class)
        row["exact_method_id"] = slug(name) if row["entry_kind"] in {"EXACT_METHOD", "METHOD_VARIANT"} else None
        row["deduplicated_method_id"] = DEDUP_METHOD.get(name, row["exact_method_id"])
        row["evidence_class"] = evidence_class(row)
        row["official_closed_loop_episode_count"] = closed_loop_counts(row)
        row["offline_model_forward_count"] = "UNVERIFIED"
        row["replay_or_demo_count"] = 6 if row["number"] == 17 else "UNVERIFIED"
        row["supported_scope"] = f"Exact legacy route and frozen result only: {row['observed_result']}"
        row["literature_overlap_checked_through"] = "2026-07-20 inherited audit"
        row["literature_overlap"] = f"Closest prior recorded as {prior}; not independently refreshed by the ledger step."
        row["local_artifact_status"] = artifact_status(paths)
        row["resource_status"] = "RESOURCE_BLOCKED" if row["evidence_class"] == "RESOURCE_BLOCKED" else "HISTORICAL_STATUS_ONLY"
        row["valid_reopen_condition"] = reopen_condition(row["evidence_class"])
        rows.append(row)
    if [r["number"] for r in rows] != list(range(1, 96)):
        raise RuntimeError("The inherited route ledger is not exactly rows 1..95")
    return rows


def addendum_entries() -> list[dict]:
    def entry(route_id, problem, method, kind, cls, counts, offline, replay, observed, scope, artifacts, resource, reopen):
        return {
            "route_id": route_id,
            "unique_thesis_problem_id": problem,
            "exact_method_id": method,
            "deduplicated_method_id": method,
            "entry_kind": kind,
            "evidence_class": cls,
            "official_closed_loop_episode_count": counts,
            "offline_model_forward_count": offline,
            "replay_or_demo_count": replay,
            "observed_result": observed,
            "supported_scope": scope,
            "literature_overlap_checked_through": "2026-07-20 inherited audit",
            "literature_overlap": "Requires the dedicated current Epoch 8 primary-source refresh.",
            "local_artifact_status": artifact_status(artifacts),
            "evidence_paths": artifacts,
            "resource_status": resource,
            "valid_reopen_condition": reopen,
        }

    zero = {"base": 0, "prior": 0, "ours": 0, "controls": 0, "total_observed": 0, "verification": "EXPLICIT"}
    return [
        entry(
            "epoch7_language_problem_verification", "language_paraphrase_target_binding", None, "PROBLEM_DIAGNOSTIC",
            "POSITIVE_PROBLEM_EVIDENCE",
            {"base": 60, "prior": 60, "ours": 0, "controls": 35, "total_observed": 155, "verification": "Base canonical/paraphrase 60; CAG-TF 60; lexical control 30; semantic-control five new rows only"},
            "UNVERIFIED", 0,
            "Base canonical 30/30 versus matched paraphrase 19/30; lexical control 24/30; semantic control 25/30; CAG-TF canonical 14/30 and paraphrase 11/30.",
            "Discovery evidence for retained X-VLA on the original 30 matched cases; it does not authorize reuse as confirmation.",
            ["reports/epoch7_selective_language_grounding/base_problem_headroom_adjudication.json", "reports/epoch7_selective_language_grounding/problem_verification_adjudication.json"],
            "SERIAL_LOCAL_PATH_VERIFIED", "Use new reset IDs and held-out paraphrases for confirmation."
        ),
        entry(
            "epoch7_equivalence_selective_action_energy_ranking_v1", "language_paraphrase_target_binding", "equivalence_selective_action_energy_ranking_v1", "EXACT_METHOD",
            "EMPIRICAL_EXACT_METHOD_FALSIFICATION", zero, 180, 0,
            "Only 2/30 frozen ranking violations versus 6/30 required; task coverage 2 versus 3 and family coverage 1 versus 2.",
            "Closes only scalar X-VLA clean-action energy ranking with frozen token-Jaccard feasible-instruction negatives; it does not close language grounding.",
            ["reports/epoch7_selective_language_grounding/base_action_energy_falsifier.json", "reports/epoch7_selective_language_grounding/base_action_energy_adjudication.json"],
            "EXECUTED_WITHIN_LOCAL_BUDGET", "A causally distinct non-energy binding-to-action mechanism with untouched data."
        ),
        entry(
            "epoch7_latent_dynamics_attribution", "latent_object_environment_dynamics_attribution", None, "PROBLEM_DIAGNOSTIC",
            "UNDERPOWERED_OR_AMBIGUOUS", zero, 0, 8,
            "Altered-condition exact-demonstration headroom held for only two collapsed families; no policy was loaded or queried.",
            "The frozen oracle failed its required breadth; altered-task impossibility and the broader latent-dynamics problem are not established.",
            ["reports/epoch7_latent_dynamics_attribution/closure_adjudication.json"],
            "SERIAL_SIMULATOR_PATH_VERIFIED", "A new preregistered legal controller/expert valid under altered dynamics, without outcome-selecting demonstrations."
        ),
        entry(
            "epoch7_policy_rng_reliability", "stochastic_policy_schedule_reliability", None, "POSITIONING_REJECTION",
            "POSITIONING_OR_OVERLAP_RISK", zero, 0, 0,
            "Rejected before outcomes as a one-change variant of the unresolved schedule route and overlapping SDN.",
            "No empirical conclusion about policy-RNG outcome variance.",
            ["reports/epoch7_policy_rng_reliability/overlap_adjudication.json"],
            "LOCALLY_EXECUTABLE_BUT_NOT_SELECTED", "A materially distinct hypothesis and current novelty distinction; static overlap is not scientific closure."
        ),
        entry(
            "epoch7_persistent_completion", "stability_qualified_task_completion", None, "PROBLEM_DIAGNOSTIC",
            "EMPIRICAL_PROBLEM_FALSIFICATION", zero, 0, "UNVERIFIED",
            "Three suffix-recoverable transient successes were all explained by one On predicate family, below the frozen cross-mechanism gate.",
            "Closes only the frozen general cross-mechanism persistent-completion problem formulation, not predicate-specific completion or termination research.",
            ["reports/epoch7_persistent_completion/closure_adjudication.json"],
            "EXPERT_REPLAY_ONLY", "A new predicate-diverse preregistered problem instance outside the frozen ten-task gate."
        ),
        entry(
            "epoch7_contact_transition_topology", "typed_non_gripper_contact_topology", "typed_contact_transition_topology_v1", "EXACT_METHOD",
            "EMPIRICAL_EXACT_METHOD_FALSIFICATION", zero, "UNVERIFIED", 7888,
            "On 7,888 aligned rows and three seeds, visual probes lost to the causal nonvisual control and typed labels did not improve oracle arm prediction.",
            "Closes the exact typed non-gripper edge birth/death visual/action-headroom formulation; not all contact or tactile supervision.",
            ["reports/epoch7_contact_transition_topology/stage0a_adjudication.json", "reports/epoch7_contact_transition_topology/stage0b_adjudication.json"],
            "OFFLINE_DEMONSTRATION_PROBES_ONLY", "A causally distinct contact mechanism with new independent visual/action headroom."
        ),
        entry(
            "epoch6_schedule_invariant_stage0", "stochastic_policy_schedule_reliability", None, "PROBLEM_DIAGNOSTIC",
            "POSITIVE_PROBLEM_EVIDENCE", zero, 80, 0,
            "Same-order cold restart matched 20/20 hashes and reversed order changed 20/20; action-level dependence GO, with outcomes suppressed.",
            "Action sequences only; no official task-success evidence.",
            ["reports/epoch6_schedule_invariant_evaluation/stage0_result.json"],
            "EXECUTED_WITHIN_LOCAL_BUDGET", "Run the frozen official closed-loop panel on an eligible host, or preregister a scientifically distinct protocol."
        ),
        entry(
            "epoch6_schedule_invariant_four_shard_closed_loop", "stochastic_policy_schedule_reliability", None, "RESOURCE_BLOCKED_SUBSTUDY",
            "RESOURCE_BLOCKED", zero, 0, 0,
            "0/40 scientific episodes executed; four environments alone reached 85.16% host RAM before model load.",
            "The exact four-shard official closed-loop effect is unobserved, not a 0% result and not a program-wide blocker.",
            ["reports/epoch6_schedule_invariant_evaluation/closed_loop_resource_blocker.json"],
            "HARD_EXTERNAL_BLOCKER_FOR_EXACT_SUBSTUDY", "Scientifically equivalent clean host with at least 48 GB RAM, zero WSL swap/offload, and unchanged 82% ceiling."
        ),
        entry(
            "epoch8_pcat_action_transport", "language_paraphrase_target_binding", "paired_counterfactual_action_transport_v1", "EXACT_METHOD",
            "EMPIRICAL_EXACT_METHOD_FALSIFICATION", zero, 974, 0,
            "The valid frozen CUDA Stage 0 completed 1,200 optimizer steps but failed canonical retention, equivalence improvement, transport-cosine gain, and transport-NRMSE gain.",
            "Closes only the exact PCAT vector-differential residual-adapter formulation and direct reparameterizations; the language problem remains open.",
            ["reports/epoch8_pcat_stage0/result.json", "reports/epoch8_pcat_stage0_adjudication.json"],
            "EXECUTED_WITHIN_LOCAL_BUDGET", "A causally distinct non-transport language-to-action mechanism with fresh untouched evidence."
        ),
        entry(
            "epoch8_latent_dynamics_feedback_expert", "latent_object_environment_dynamics_attribution", "zero_rotation_cartesian_sweep_expert_v1", "EXACT_FEASIBILITY_ORACLE",
            "EMPIRICAL_EXACT_METHOD_FALSIFICATION", zero, 0, 16,
            "All eight frozen controller configurations completed paired standard/intervened development trials but produced 0/16 target contacts and 0/16 successes.",
            "Closes only the exact zero-rotation Cartesian sweep feasibility expert; it is not evidence against the latent-dynamics attribution problem.",
            ["reports/epoch8_latent_dynamics_feedback_development.json", "reports/epoch8_latent_dynamics_feedback_adjudication.md"],
            "SERIAL_SIMULATOR_PATH_VERIFIED", "A materially different controller/expert frozen before new outcomes."
        ),
        entry(
            "epoch8_two_shard_actual_arrival", "stochastic_policy_schedule_reliability", None, "RESOURCE_BLOCKED_SUBSTUDY",
            "RESOURCE_BLOCKED", zero, 0, 0,
            "Two live environment processes plus the start of one real model load reached 84.66% host RAM, crossing the frozen 82% ceiling before a forward or scientific episode.",
            "The independent two-shard closed-loop effect is unobserved; this does not execute, repair, or relabel the archived four-shard protocol.",
            ["reports/epoch8_two_shard_actual_arrival/run_20260720_2039_kst_resource_repair1/two_shard_resource_smoke_host.json"],
            "RESOURCE_BLOCKED_CURRENT_HOST", "A host-qualified two-process path below the 82% ceiling with zero swap/offload, or a separately novel protocol rather than a repair."
        ),
        entry(
            "epoch8_active_hidden_mass_base_screen", "active_hidden_property_grounding", None, "PROBLEM_DIAGNOSTIC",
            "UNDERPOWERED_OR_AMBIGUOUS",
            {"base": 12, "prior": 0, "ours": 0, "controls": 0, "total_observed": 12, "verification": "EXPLICIT_FROZEN_DISCOVERY_PANEL"},
            0, 0,
            "Valid 12-episode screen: canonical 4/6 and hidden-property 2/6. Front/back canonical was 4/4 while hidden was 0/4 and always contacted the middle distractor, but the frozen 5/6 competence gate failed.",
            "Positive narrow discovery pattern for front/back plus an overall formal Base/headroom non-GO; not a three-position problem verification or confirmation result.",
            ["reports/epoch8_active_latent_property/stage_minus1_result.json"],
            "SERIAL_LOCAL_PATH_VERIFIED", "A new versioned competent target subset with untouched resets and a legal active-probe method, without relabeling the discovery rows."
        ),
        entry(
            "epoch8_probe_response_belief_lda_v1", "active_hidden_property_grounding", "probe_response_belief_lda_v1", "EXACT_METHOD",
            "EMPIRICAL_EXACT_METHOD_FALSIFICATION", zero, 0, 36,
            "The valid repaired 18-pair expert-probe Stage 0 had 6/6 paired heavy-score wins but only 50% held-out absolute classification accuracy on every target, tying the one-feature control and missing the frozen 83.3% gate.",
            "Closes the exact seven-aggregate-feature absolute LDA belief mechanism and direct outcome-tuned ranking reparameterizations; it does not close active property grounding.",
            ["reports/epoch8_active_property_probe_belief_stage0_repair1.json"],
            "EXPERT_REPLAY_ONLY", "A causally distinct temporal or learned response representation with a fresh held-out split and a complete legal probe-return-select closed-loop path."
        ),
        entry(
            "epoch8_scripted_probe_return_inverse_delta_v1", "active_hidden_property_grounding", "scripted_probe_return_inverse_delta_v1", "EXACT_FEASIBILITY_ORACLE",
            "EMPIRICAL_EXACT_METHOD_FALSIFICATION", zero, 0, 6,
            "All six target/mass probes contacted correctly, but only 4/6 returned the bowl within 3 cm and 4/6 returned the end effector within 5 cm; maxima were 3.44 cm and 5.49 cm.",
            "Closes only the exact inverse-delta demonstration-prefix probe-return expert; it does not close visual-feedback controllers or active property grounding.",
            ["reports/epoch8_active_property_probe_return_result.json"],
            "EXPERT_REPLAY_ONLY", "A materially distinct legal visual-feedback controller frozen on new identities before outcomes, plus a complete probe-return-select task path."
        ),
    ]


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    rows = parse_rows()
    addenda = addendum_entries()
    all_entries = rows + addenda
    empirical_problem_classes = {"EMPIRICAL_PROBLEM_FALSIFICATION", "POSITIVE_PROBLEM_EVIDENCE"}
    empirical_method_classes = {"EMPIRICAL_EXACT_METHOD_FALSIFICATION", "POSITIVE_METHOD_EVIDENCE"}
    problem_ids = sorted({e["unique_thesis_problem_id"] for e in all_entries if e["evidence_class"] in empirical_problem_classes})
    method_ids = sorted({e["deduplicated_method_id"] for e in all_entries if e["evidence_class"] in empirical_method_classes and e.get("deduplicated_method_id")})
    class_counts = {c: sum(e["evidence_class"] == c for e in all_entries) for c in CLASSES}
    known_role_counts = {role: 0 for role in ("base", "prior", "ours", "controls")}
    rows_with_unverified_closed_loop = 0
    for e in all_entries:
        counts = e["official_closed_loop_episode_count"]
        if any(counts.get(role) == "UNVERIFIED" for role in known_role_counts):
            rows_with_unverified_closed_loop += 1
        for role in known_role_counts:
            value = counts.get(role)
            if isinstance(value, int):
                known_role_counts[role] += value

    payload = {
        "schema_version": "epoch8.route_evidence_ledger.v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "branch": "codex/epoch8-language-grounding-convergence",
        "source": {
            "path": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": source_hash(SOURCE),
            "claimed_inherited_route_count": 95,
            "parsed_inherited_route_count": len(rows),
        },
        "taxonomy": CLASSES,
        "audit_policy": {
            "classes_1_and_2_only_are_scientific_closures": True,
            "exact_method_closure_is_not_problem_axis_closure": True,
            "resource_artifact_positioning_and_static_rejections_are_not_empirical_closures": True,
            "missing_role_specific_counts_remain_unverified": True,
            "legacy_discovery_is_not_relabelled_confirmation": True,
        },
        "summary": {
            "raw_inherited_route_entry_count": len(rows),
            "corrective_and_epoch8_addendum_count": len(addenda),
            "total_audited_entries": len(all_entries),
            "unique_empirically_adjudicated_problem_count": len(problem_ids),
            "unique_empirically_adjudicated_problem_ids": problem_ids,
            "unique_empirically_adjudicated_method_count": len(method_ids),
            "unique_empirically_adjudicated_method_ids": method_ids,
            "evidence_class_counts": class_counts,
            "official_closed_loop_episode_counts_known_lower_bound": known_role_counts,
            "entries_with_unverified_role_specific_closed_loop_counts": rows_with_unverified_closed_loop,
            "counting_caveat": "Known counts are a conservative lower bound assembled from role-explicit legacy summaries; they are not a deduplicated global episode total because some diagnostics reuse rows.",
        },
        "candidate_a_r_raw_artifact_audit": {
            "epoch7_index_records": {"checked": 15, "matched": 15, "missing": 0, "mismatched": 0},
            "candidate_a_demo_files": {"checked_unique": 10, "matched": 10},
            "candidate_a_and_r_explicit_path_hash_references": {"checked_unique": 64, "matched": 63, "missing": 0, "mismatched": 1},
            "candidate_a_preserved_invalid_attempt_caveat": "base_action_energy_falsifier_attempt2_invalid_gripper.json records protocol SHA BB595F... but its reused path now contains the repaired protocol SHA CB3B9E...; the valid repaired falsifier and its current protocol match. The pre-repair protocol bytes are not independently present at that path.",
            "candidate_r_stage0_run_artifacts": {"checked": 16, "matched": 16},
            "candidate_r_closed_loop_scientific_rows": 0,
        },
        "inherited_routes": rows,
        "corrective_addenda": addenda,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = [
        "# Epoch 8 Route Evidence Audit",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Adjudication",
        "",
        "The inherited statement that 95 routes were 95 completed closed-loop experiments is false. The source contains a mixture of exact methods, variants, controls, diagnostics, static candidates, implementation-invalid attempts, and resource/artifact blocks. Only evidence classes 1 and 2 below are scientific closures, and an exact-method closure never closes its whole problem axis.",
        "",
        "- Whole program: `INDEPENDENT_ROUTE_AUDIT_REQUIRED`",
        "- Exact frozen four-shard sub-study: `HARD_EXTERNAL_BLOCKER`",
        "- Current paper: `PAPER_NOT_AUTHORIZED`",
        "",
        "## Counts",
        "",
        f"- Raw inherited entries: **{len(rows)}**",
        f"- Corrective and Epoch 8 addenda: **{len(addenda)}**",
        f"- Unique empirically adjudicated problem groups: **{len(problem_ids)}**",
        f"- Unique empirically adjudicated method formulations after explicit deduplication: **{len(method_ids)}**",
        f"- Known lower-bound official closed-loop role counts: Base **{known_role_counts['base']}**, Prior **{known_role_counts['prior']}**, Ours **{known_role_counts['ours']}**, controls **{known_role_counts['controls']}**",
        f"- Entries whose legacy summaries do not support role-specific closed-loop counts: **{rows_with_unverified_closed_loop}**",
        "",
        "The role totals are conservative lower bounds, not a global unique-episode total: old diagnostics sometimes reuse rows and many summaries omit role-specific denominators. Those cases remain `UNVERIFIED` in the JSON.",
        "",
        "## Evidence classes",
        "",
        "| Class | Entries | Scientific closure? |",
        "|---|---:|---|",
    ]
    for i, c in enumerate(CLASSES, 1):
        md.append(f"| {i}. `{c}` | {class_counts[c]} | {'Yes, scoped' if i in (1, 2) else 'No'} |")
    md += [
        "",
        "## Candidate A and Candidate R raw-evidence audit",
        "",
        "All 15 records in `epoch7_evidence_index.json` match their supplied SHA-256 hashes. Candidate A's ten unique X-VLA-format HDF5 demonstrations match the hashes embedded in the repaired falsifier. Candidate R's 16 Stage-0 run artifacts and every explicit path/hash pair used by the closed-loop resource blocker are present and match.",
        "",
        "One historical caveat is preserved: the invalid-gripper Candidate A attempt references a protocol path with pre-repair SHA `BB595F...`, while that same path now contains the documented repaired protocol with SHA `CB3B9E...`. Therefore the invalid attempt's original protocol bytes are absent at the referenced path. This does not invalidate the repaired final falsifier, whose result, protocol, manifest, and demonstrations match; it limits independent reconstruction of the discarded invalid attempt.",
        "",
        "Candidate A supports `EMPIRICAL_EXACT_METHOD_FALSIFICATION` only for the frozen scalar action-energy ranking formulation. Candidate R supports `POSITIVE_PROBLEM_EVIDENCE` at the action-sequence level and `RESOURCE_BLOCKED` for the exact four-shard closed loop, with **0/40 episodes executed**.",
        "",
        "## Deduplication policy",
        "",
        "- Repeated reports and repairs are evidence records, not new routes.",
        "- Controls, official-prior diagnostics, and residual scans do not count as Ours methods.",
        "- TCA-Map, TCA-Select, and the ActionMap mini-anchor are retained as three raw entries but one explicitly deduplicated target-prior/action-map campaign family.",
        "- The custom SmolVLA 7-D adapter and TG-7D are retained as two raw entries but one deduplicated adapter family.",
        "- Static idea rejection, overlap, missing artifacts, and resource blocks do not count as scientific falsification.",
        "",
        "## Immediate consequence",
        "",
        "The verified language problem remains open after two scoped exact-method failures: scalar action-energy ranking and PCAT action transport. The independent two-shard study is also resource-blocked with zero scientific episodes. The active hidden-mass screen produced a narrow front/back discovery gap but failed its overall competence gate, and its first legal-response belief mechanism failed the valid frozen Stage 0. None of these scoped results is a problem-axis closure.",
        "",
        "Machine-readable per-route details, evidence paths, counts, supported scope, and reopen conditions are in `reports/epoch8_route_evidence_ledger.json`.",
    ]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
