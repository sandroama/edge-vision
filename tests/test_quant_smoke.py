"""Tests for the Phase-3 quantization smoke script.

Runs the script with the synthetic-dataset path so no network or COCO
download is required.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from scripts.run_quant_smoke import main as quant_main


def test_quant_smoke_synthetic_simple_backend(tmp_path: Path):
    out_json = tmp_path / "phase3.json"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = quant_main(
            [
                "--max-images",
                "4",
                "--candidate-recall",
                "0.6",
                "--candidate-fp-rate",
                "0.2",
                "--eval-backend",
                "simple",
                "--out-json",
                str(out_json),
                "--seed",
                "0",
            ]
        )
    assert rc == 0
    out = buf.getvalue()
    assert "RQ-E1" in out
    assert "Reference" in out
    assert "Candidate" in out
    assert out_json.exists()

    payload = json.loads(out_json.read_text())
    assert "reference" in payload
    assert "candidate" in payload
    assert "delta" in payload
    # Candidate should be strictly worse than the perfect-recall reference.
    assert payload["candidate"]["recall"] is not None
    assert payload["reference"]["recall"] >= payload["candidate"]["recall"]
