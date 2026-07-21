#!/usr/bin/env python3
"""Bind mechanics smoke code and protocol before any smoke result."""
from __future__ import annotations
import hashlib,json,subprocess,sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tca_map.epoch7_latent_dynamics import atomic_write_json
R=ROOT/"reports";OUTPUT=R/"epoch9e_mechanics_execution_seal.json";PROTOCOL=R/"epoch9e_mechanics_smoke_protocol.json";JOINT=R/"epoch9e_joint_certification_protocol.json";RUNNER=ROOT/"scripts/run_epoch9e_mechanics_smoke.py";ADJ=ROOT/"scripts/adjudicate_epoch9e_mechanics_smoke.py";HOST=ROOT/"scripts/run_epoch9e_mechanics_smoke_host.ps1";CONTROLLER=ROOT/"scripts/epoch9e_nondrag_controller.py";ORIGINAL=ROOT/"scripts/run_epoch9b_dynamic_nudge.py"
def sha(p:Path)->str:
 d=hashlib.sha256()
 with p.open("rb") as h:
  for b in iter(lambda:h.read(1024*1024),b""):d.update(b)
 return d.hexdigest().upper()
def rel(p:Path)->str:return str(p.relative_to(ROOT)).replace("\\","/")
def main()->int:
 if OUTPUT.exists():raise FileExistsError("refusing overwrite mechanics seal")
 if (R/"epoch9e_mechanics_smoke/result.json").exists():raise RuntimeError("cannot seal after smoke outcome")
 for p in (PROTOCOL,JOINT,RUNNER,ADJ,HOST,CONTROLLER,ORIGINAL):
  if not p.exists():raise FileNotFoundError(p)
 protocol=json.loads(PROTOCOL.read_text(encoding="utf-8"));seal={"schema_version":"epoch9e.mechanics_execution_seal.v1","sealed_at":datetime.now().astimezone().isoformat(timespec="seconds"),"branch":subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),"source_checkpoint":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"outcomes_accessed_before_seal":False,"smoke_protocol_path":rel(PROTOCOL),"smoke_protocol_sha256":sha(PROTOCOL),"joint_protocol_path":rel(JOINT),"joint_protocol_sha256":sha(JOINT),"runner_path":rel(RUNNER),"runner_sha256":sha(RUNNER),"adjudicator_path":rel(ADJ),"adjudicator_sha256":sha(ADJ),"host_wrapper_path":rel(HOST),"host_wrapper_sha256":sha(HOST),"controller_path":rel(CONTROLLER),"controller_sha256":sha(CONTROLLER),"original_runner_path":rel(ORIGINAL),"original_runner_sha256":sha(ORIGINAL),"frozen_mechanics_gate":protocol["mechanics_pass"],"forbidden_outputs":protocol["must_not_compute_or_reveal"],"runtime":{"serial":True,"environments_at_once":1,"models":0,"host_ram_ceiling_percent":82.0,"wsl_swap_used_peak_bytes":0},"validation_accessed":False,"confirmation_accessed":False};atomic_write_json(OUTPUT,seal);print(json.dumps({"output":rel(OUTPUT),"runner":seal["runner_sha256"],"controller":seal["controller_sha256"]},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
