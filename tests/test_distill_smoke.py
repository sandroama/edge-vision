"""Tests for the Phase-4 distillation smoke script.

Runs the CPU smoke path (``--backend tiny``) and asserts basic invariants.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # noqa: F401 — needed for the smoke

from scripts.run_distill_smoke import main as distill_main  # noqa: E402


@pytest.mark.slow
def test_distill_smoke_tiny_converges(tmp_path: Path):
    out_json = tmp_path / "phase4.json"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = distill_main(
            [
                "--backend",
                "tiny",
                "--epochs",
                "3",
                "--batches",
                "4",
                "--temperature",
                "2.0",
                "--lr",
                "1e-2",
                "--out-json",
                str(out_json),
                "--seed",
                "0",
            ]
        )
    assert rc == 0
    out = buf.getvalue()
    assert "DONE" in out
    assert "Converged" in out

    payload = json.loads(out_json.read_text())
    assert payload["converged"] is True
    assert len(payload["per_epoch"]) == 3
    # First loss > last loss (converging).
    assert payload["initial_loss"] > payload["final_loss"]
