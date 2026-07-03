import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tracked_cloud_handoff_manifest_targets_main():
    manifest = json.loads((REPO_ROOT / "reports" / "cloud_handoff_manifest.json").read_text(encoding="utf-8-sig"))

    assert manifest["git"]["remote_target_branch"] == "main"
    assert manifest["git"]["regenerate_before_remote_execution"] is True
    assert "git checkout main" in manifest["remote_commands"]
    assert "codex/replace-hard-stops-with-risk-assessed-autonomy" not in json.dumps(manifest)


def test_bash_cloud_handoff_markdown_template_does_not_execute_backticks():
    script = (REPO_ROOT / "scripts" / "23_cloud_handoff_manifest.sh").read_text(encoding="utf-8")
    marker = "cat > reports/cloud_handoff_manifest.md <<MD"
    start = script.index(marker)
    body = script[start + len(marker) :]
    body = body[: body.index("\nMD")]

    assert "`" not in body
