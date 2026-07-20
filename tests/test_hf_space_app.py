"""Deployability guard for the HF Space (``hf_space/app.py``).

The June audit flagged a missing ``hf_space/requirements.txt``. These tests
pin the fix: the app must import cleanly with no heavy deps, its data helpers
must stay defensive, and every non-stdlib import in ``app.py`` must be covered
by ``hf_space/requirements.txt``.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

HF_SPACE = Path(__file__).resolve().parents[1] / "hf_space"
APP = HF_SPACE / "app.py"
REQUIREMENTS = HF_SPACE / "requirements.txt"


def _load_app():
    spec = importlib.util.spec_from_file_location("hf_space_app", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_app_imports_cleanly_without_ui_deps():
    """Module import must not require streamlit/plotly/pandas (all lazy)."""
    mod = _load_app()
    assert callable(mod.main)


def test_load_json_is_defensive(tmp_path: Path):
    mod = _load_app()
    assert mod._load_json(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert mod._load_json(bad) is None
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"a": 1}))
    assert mod._load_json(good) == {"a": 1}


def test_real_pareto_rows_filters_mock_backends():
    mod = _load_app()
    payload = {
        "configs": [
            {"label": "mock-fp32", "backend": "mock"},
            {"label": "trt-fp16", "backend": "tensorrt"},
        ]
    }
    assert [c["label"] for c in mod._real_pareto_rows(payload)] == ["trt-fp16"]
    assert mod._real_pareto_rows(None) == []
    assert mod._real_pareto_rows({}) == []


def test_requirements_cover_all_app_imports():
    """Every non-stdlib module imported anywhere in app.py must be pinned."""
    tree = ast.parse(APP.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])

    third_party = {
        m for m in imported
        if m not in sys.stdlib_module_names and m != "__future__"
    }
    assert third_party == {"streamlit", "plotly", "pandas"}, (
        f"app.py imports changed ({sorted(third_party)}) — update "
        "hf_space/requirements.txt and this test together"
    )

    req_names = {
        line.split(">=")[0].split("==")[0].strip()
        for line in REQUIREMENTS.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert third_party <= req_names, (
        f"requirements.txt missing: {sorted(third_party - req_names)}"
    )


def test_results_dir_resolves_to_repo_layout():
    """In the repo checkout, the app must find docs/results (not the fallback)."""
    mod = _load_app()
    assert mod._RESULTS_DIR.is_dir()
    assert mod._CPU_INT8_JSON.exists(), "phase3_cpu_int8.json missing from docs/results"
