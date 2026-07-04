import argparse
import json
import os
from pathlib import Path

from tca_map.smolvla import vlm_required_files_acquisition as acquisition


def _write_risk(path: Path, *, gated: bool = False, include_size: bool = True) -> None:
    size = 16 if include_size else None
    payload = {
        "decision": "proceed",
        "ready_for_vlm_weight_acquisition_plan": True,
        "source": {
            "repo_id": acquisition.SOURCE_REPO,
            "official_source": True,
            "private": False,
            "gated": gated,
            "token_login_license_payment_required": gated,
        },
        "files": {
            "root_safetensors": [{"rfilename": "model.safetensors", "size": size}],
            "config_tokenizer_processor_files": [
                {"rfilename": "config.json", "size": size},
                {"rfilename": "tokenizer.json", "size": size},
                {"rfilename": "processor_config.json", "size": size},
            ],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        source_repo=acquisition.SOURCE_REPO,
        risk_report=str(tmp_path / "risk.json"),
        target_root=str(tmp_path / "hf_home" / "HuggingFaceTB" / "SmolVLM2-500M-Video-Instruct"),
        hf_home=str(tmp_path / "hf_home"),
    )


def _fake_downloader(*, source_repo, files, target_root, hf_home):
    assert source_repo == acquisition.SOURCE_REPO
    assert Path(hf_home).name == "hf_home"
    acquired = []
    for item in files:
        path = Path(target_root) / item["rfilename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 16)
        acquired.append({"rfilename": item["rfilename"], "path": str(path), "size": path.stat().st_size})
    return acquired


def test_vlm_required_file_acquisition_requires_download_gate(tmp_path, monkeypatch):
    _write_risk(tmp_path / "risk.json")
    monkeypatch.delenv("ALLOW_DOWNLOADS", raising=False)

    report, code = acquisition.build_report(_args(tmp_path), downloader=_fake_downloader)

    assert code != 0
    assert report["decision"] == "stop"
    assert report["policy"]["downloads_performed"] is False
    assert "ALLOW_DOWNLOADS" in report["recommended_next_step"]


def test_vlm_required_file_acquisition_refuses_forbidden_gate(tmp_path, monkeypatch):
    _write_risk(tmp_path / "risk.json")
    monkeypatch.setenv("ALLOW_DOWNLOADS", "1")
    monkeypatch.setenv("ALLOW_HEAVY_IMPORT", "1")

    report, code = acquisition.build_report(_args(tmp_path), downloader=_fake_downloader)

    assert code != 0
    assert report["decision"] == "stop"
    assert "ALLOW_HEAVY_IMPORT" in report["recommended_next_step"]
    assert report["policy"]["downloads_performed"] is False
    monkeypatch.delenv("ALLOW_HEAVY_IMPORT", raising=False)


def test_vlm_required_file_acquisition_uses_bounded_file_list(tmp_path, monkeypatch):
    _write_risk(tmp_path / "risk.json")
    monkeypatch.setenv("ALLOW_DOWNLOADS", "1")

    report, code = acquisition.build_report(_args(tmp_path), downloader=_fake_downloader)

    assert code == 0
    assert report["decision"] == "acquisition_complete"
    assert report["vlm_required_files_acquisition_passed"] is True
    assert report["ready_for_bounded_vlm_enabled_load_smoke_plan"] is True
    assert report["policy"]["downloads_performed"] is True
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert sorted(item["rfilename"] for item in report["files"]["present_after"]) == [
        "config.json",
        "model.safetensors",
        "processor_config.json",
        "tokenizer.json",
    ]


def test_vlm_required_file_acquisition_stops_for_gated_or_unknown_size(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_DOWNLOADS", "1")
    gated_dir = tmp_path / "gated"
    gated_dir.mkdir()
    _write_risk(gated_dir / "risk.json", gated=True)
    size_dir = tmp_path / "nosize"
    size_dir.mkdir()
    _write_risk(size_dir / "risk.json", include_size=False)

    gated_report, gated_code = acquisition.build_report(_args(gated_dir), downloader=_fake_downloader)
    size_report, size_code = acquisition.build_report(_args(size_dir), downloader=_fake_downloader)

    assert gated_code != 0
    assert gated_report["decision"] == "stop"
    assert "token/login" in gated_report["recommended_next_step"]
    assert size_code != 0
    assert size_report["decision"] == "stop"
    assert "size metadata" in size_report["recommended_next_step"]
