from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_epoch9b_observability_diagnostic.py"
    spec = importlib.util.spec_from_file_location("epoch9b_observability", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys_modules = __import__("sys").modules
    sys_modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_development_examples_are_balanced_and_sealed_safe() -> None:
    module = _module()
    examples, payload = module.load_examples()
    assert len(examples) == 80
    assert sorted({value.demo_index for value in examples}) == list(range(30, 40))
    assert sum(value.label_front_heavier for value in examples) == 40
    assert payload["summary"]["bounded_action_fraction"] == 1.0
    assert payload["summary"]["contact_fraction"] == 1.0


def test_group_folds_have_no_identity_leakage_and_are_balanced() -> None:
    module = _module()
    examples, _ = module.load_examples()
    audit = module.split_audit(examples)
    assert not audit["same_trajectory_or_identity_crosses_fold"]
    assert audit["every_test_fold_position_and_order_balanced"]


def test_trace_sequence_uses_only_deployable_fields() -> None:
    module = _module()
    examples, _ = module.load_examples()
    assert examples[0].sequence.shape == (2, module.SEQUENCE_LENGTH, 60)
    assert np.isfinite(examples[0].sequence).all()
    assert examples[0].position_nuisance.shape == (5,)
    assert examples[0].displacement_nuisance.shape == (3,)
