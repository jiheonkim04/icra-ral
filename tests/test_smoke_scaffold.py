from pathlib import Path

from tca_map.datasets import make_dummy_samples
from tca_map.heads import TCAMapHead
from tca_map.launch.smoke_test import REPO_ROOT, run_smoke
from tca_map.models import DummyAdapter


def test_dummy_dataset_adapter_and_tca_head_forward():
    samples = make_dummy_samples(count=2)
    assert len(samples) == 2

    adapter = DummyAdapter()
    hidden = adapter.get_hidden_tokens(samples[0]["observation"], samples[0]["instruction"])
    assert hidden

    prediction = TCAMapHead(grid_size=8).predict(
        hidden,
        samples[0]["observation"]["candidate_objects"],
    )
    assert "target" in prediction
    assert "action_heatmap" in prediction
    assert "action" in prediction


def test_smoke_report_schema_contains_pilot_gate():
    run_smoke("train")
    smoke_report_path = Path(REPO_ROOT) / "reports" / "smoke_report.json"
    assert smoke_report_path.exists()

    content = smoke_report_path.read_text(encoding="utf-8")
    assert '"safe_to_run_pilot_gpu"' in content
    assert '"train_smoke_passed"' in content
    assert '"downloads_performed"' in content
