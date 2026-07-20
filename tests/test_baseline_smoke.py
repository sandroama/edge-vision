"""Tests for the Phase-1 baseline smoke script.

Runs the smoke entrypoint with the mock backend and asserts that:
    1. The CLI returns exit code 0.
    2. The summary banner is present in stdout.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from scripts.run_baseline_smoke import main as smoke_main


def test_smoke_main_with_mock_backend_succeeds():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = smoke_main(
            [
                "--backend",
                "mock",
                "--max-images",
                "4",
                "--mock-recall",
                "1.0",
                "--mock-fp-rate",
                "0.0",
                "--eval-backend",
                "simple",  # don't depend on pycocotools availability in CI
                "--seed",
                "0",
            ]
        )
    assert rc == 0
    out = buf.getvalue()
    assert "[edge-vision]" in out
    assert "DONE" in out
    assert "Precision" in out  # simple backend reports precision/recall/f1


def test_smoke_main_rtdetr_backend_requires_data():
    """``--backend rtdetr`` without --coco-annotations / --coco-images must
    raise SystemExit with a non-None exit code."""
    with pytest.raises(SystemExit) as exc_info:
        smoke_main(["--backend", "rtdetr"])
    assert exc_info.value.code not in (None, 0)
