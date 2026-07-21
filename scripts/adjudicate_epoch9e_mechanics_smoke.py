#!/usr/bin/env python3
"""Adjudicate the outcome-suppressed Epoch 9E mechanics smoke."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tca_map.epoch7_latent_dynamics import atomic_write_json

REPORTS=ROOT/"reports"; PROTOCOL=REPORTS/"epoch9e_mechanics_smoke_protocol.json"; SEAL=REPORTS/"epoch9e_mechanics_execution_seal.json"; RESULT=REPORTS/"epoch9e_mechanics_smoke/result.json"; HOST=REPORTS/"epoch9e_mechanics_smoke/host_resource_monitor.json"; OUTPUT=REPORTS/"epoch9e_mechanics_smoke_adjudication.json"

def sha256(path: Path)->str:
    d=hashlib.sha256()
    with path.open("rb") as h:
        for b in iter(lambda:h.read(1024*1024),b""): d.update(b)
    return d.hexdigest().upper()
def load(path: Path)->dict[str,Any]: return json.loads(path.read_text(encoding="utf-8-sig"))
def smoke_gates(c:dict[str,Any])->dict[str,bool]:
    return {"complete_scenes_8_of_8":c["complete_scenes"]==8,"finite_bounded_actions_16_of_16":c["finite_bounded_actions"]==16,"intended_contact_or_excitation_at_least_15_of_16":c["intended_contact_or_excitation"]>=15,"both_candidates_excited_at_least_7_of_8":c["both_candidates_excited_scenes"]>=7,"full_trajectory_lane_reachable_16_of_16":c["full_trajectory_lane_reachable"]==16,"zero_safety_or_track_events":c["safety_or_track_events"]==0,"liftoff_planar_commands_exact_zero_16_of_16":c["liftoff_zero_planar_probes"]==16,"separation_verified_16_of_16":c["separation_verified_probes"]==16}
def main()->int:
    if OUTPUT.exists(): raise FileExistsError("refusing to overwrite mechanics adjudication")
    protocol=load(PROTOCOL); seal=load(SEAL); result=load(RESULT); host=load(HOST)
    bindings={"protocol":sha256(PROTOCOL)==seal["smoke_protocol_sha256"],"runner":sha256(ROOT/seal["runner_path"])==seal["runner_sha256"],"adjudicator":sha256(Path(__file__))==seal["adjudicator_sha256"],"host":sha256(ROOT/seal["host_wrapper_path"])==seal["host_wrapper_sha256"],"controller":sha256(ROOT/seal["controller_path"])==seal["controller_sha256"],"result":sha256(RESULT)==host["scientific_result_sha256_after_runner"]}
    forbidden_clear=all(result.get(key) is False for key in ("mass_rank_computed","mass_conditioned_response_computed_or_revealed","oracle_task_success_accessed","reward_done_success_accessed")) and all(all(row.get(key) is False for key in ("mass_rank_computed","mass_conditioned_response_computed_or_revealed","oracle_task_success_accessed","reward_done_success_accessed")) for row in result["rows"])
    trace_clear=True
    for row in result["rows"]:
        for audit in row.get("probe_audits",[]):
            path=ROOT/audit["trace_path"]; trace_clear=trace_clear and path.exists() and sha256(path)==audit["trace_sha256"]
    protocol_clean=bool(all(bindings.values()) and forbidden_clear and trace_clear and result["validation_accessed"] is False and result["confirmation_accessed"] is False and host["runner_exit_code"]==0 and host["host_ram_ceiling_breached"] is False and host["peak_host_ram_percent"]<82.0 and result["resource"]["wsl_swap_used_peak_bytes"]==0 and len(result["rows"])==8)
    gates=smoke_gates(result["summary"]); passed=protocol_clean and all(gates.values()); decision="MECHANICS_SMOKE_PASS_FREEZE_CONTROLLER" if passed else "MECHANICS_SMOKE_NO_GO_NO_SCIENTIFIC_REPAIR"
    out={"schema_version":"epoch9e.mechanics_smoke_adjudication.v1","adjudicated_at":datetime.now().astimezone().isoformat(timespec="seconds"),"decision":decision,"mechanics_smoke_pass":passed,"execution_bindings":bindings,"information_boundary_pass":forbidden_clear,"trace_hashes_pass":trace_clear,"protocol_clean":protocol_clean,"counts":result["summary"],"gates":gates,"implementation_repair_used":False,"scientific_outcomes_accessed":False,"validation_accessed":False,"confirmation_accessed":False,"source_hashes":{"protocol":sha256(PROTOCOL),"seal":sha256(SEAL),"result":sha256(RESULT),"host":sha256(HOST)}}
    atomic_write_json(OUTPUT,out); print(json.dumps({"decision":decision,"counts":out["counts"],"gates":gates},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
