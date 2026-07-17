from __future__ import annotations

import sys

import numpy as np

from tca_map.xvla_task1 import gradient_smoke
from tca_map.xvla_task1.gradient_smoke import (
    install_optional_server_import_shims,
    select_one_target_clip_start,
    target_count_in_basket,
)


def test_target_count_in_basket_counts_two_targets_by_xy_distance() -> None:
    states = np.zeros((3, 123), dtype=np.float64)
    states[:, 59:62] = [0.0, 0.0, 0.4]
    states[:, 17:20] = [[0.2, 0.0, 0.4], [0.01, 0.0, 0.4], [0.01, 0.0, 0.4]]
    states[:, 52:55] = [[0.2, 0.0, 0.4], [0.2, 0.0, 0.4], [0.01, 0.0, 0.4]]

    counts = target_count_in_basket(states, 0.08)

    assert counts.tolist() == [0, 1, 2]


def test_select_one_target_clip_start_prefers_viable_one_target_index() -> None:
    states = np.zeros((20, 123), dtype=np.float64)
    states[:, 59:62] = [0.0, 0.0, 0.4]
    states[:, 17:20] = [0.2, 0.0, 0.4]
    states[:, 52:55] = [0.2, 0.0, 0.4]
    states[5:, 17:20] = [0.01, 0.0, 0.4]
    states[15:, 52:55] = [0.01, 0.0, 0.4]

    assert select_one_target_clip_start(states, 0.08, clip_steps=10) == 5


def test_optional_server_import_shims_have_module_specs(monkeypatch) -> None:
    optional_names = {"fastapi", "fastapi.responses", "uvicorn", "json_numpy"}
    for name in optional_names:
        monkeypatch.delitem(sys.modules, name, raising=False)

    original_find_spec = gradient_smoke.importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name in {"fastapi", "uvicorn", "json_numpy"}:
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(gradient_smoke.importlib.util, "find_spec", fake_find_spec)

    used = install_optional_server_import_shims()

    assert used == ["fastapi", "uvicorn", "json_numpy"]
    for name in optional_names:
        assert sys.modules[name].__spec__ is not None
        assert sys.modules[name].__spec__.name == name
    assert hasattr(sys.modules["fastapi"], "__path__")
