from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

import scripts.model_parity_smoke as parity


def _browser_evidence() -> dict[str, object]:
    return {
        "runtime": {"name": "onnxruntime-web", "version": "1.20.1"},
        "source_input": {"name": "images", "type": "float32", "shape": [1, 3, 1024, 1024]},
        "derived_input": {"name": "images", "type": "float32", "shape": [1, 3, 1024, 1024]},
        "source_output": {"name": "output0", "type": "float32", "shape": [1, 300, 7]},
        "derived_output": {"name": "output0", "type": "float32", "shape": [1, 300, 7]},
        "declared_contracts_equal": True,
        "output_bytes_equal": True,
        "detections_equal": True,
        "accepted_ship": True,
    }


def test_parity_report_schema_never_contains_paths_metadata_or_tensor_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review_root = tmp_path / "private-review-root"
    review_root.mkdir()
    report = tmp_path / "parity.json"
    monkeypatch.setattr(parity, "validate_admitted_assets", lambda _root: object())
    monkeypatch.setattr(parity, "_browser_parity", lambda _root: _browser_evidence())

    parity.run_parity(review_root, report)

    text = report.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert set(payload) == {
        "accepted_ship",
        "detections_equal",
        "input",
        "output",
        "output_bytes_equal",
        "runtime",
        "verdict",
    }
    assert payload == {
        "runtime": {"name": "onnxruntime-web", "version": "1.20.1"},
        "input": {"name": "images", "type": "float32", "shape": [1, 3, 1024, 1024]},
        "output": {"name": "output0", "type": "float32", "shape": [1, 300, 7]},
        "output_bytes_equal": True,
        "detections_equal": True,
        "accepted_ship": True,
        "verdict": "PASS",
    }
    assert text == json.dumps(payload, sort_keys=True, indent=2) + "\n"
    lowered = text.casefold()
    assert "private-review-root" not in text
    assert "metadata" not in lowered
    assert "tensor" not in lowered
    assert "detections" in lowered and "cx" not in lowered and "confidence" not in lowered


@pytest.mark.parametrize(
    ("field", "mutation"),
    (
        ("derived_input", {"name": "other", "type": "float32", "shape": [1, 3, 1024, 1024]}),
        ("derived_input", {"name": "images", "type": "float64", "shape": [1, 3, 1024, 1024]}),
        ("derived_input", {"name": "images", "type": "float32", "shape": [1, 3, 640, 640]}),
        ("derived_output", {"name": "other", "type": "float32", "shape": [1, 300, 7]}),
        ("derived_output", {"name": "output0", "type": "float64", "shape": [1, 300, 7]}),
        ("derived_output", {"name": "output0", "type": "float32", "shape": [1, 300, 8]}),
        ("declared_contracts_equal", False),
        ("output_bytes_equal", False),
        ("detections_equal", False),
        ("accepted_ship", False),
    ),
)
def test_parity_rejects_shape_name_type_byte_and_ship_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    mutation: object,
) -> None:
    review_root = tmp_path / "review"
    review_root.mkdir()
    evidence = _browser_evidence()
    evidence[field] = mutation
    monkeypatch.setattr(parity, "validate_admitted_assets", lambda _root: object())
    monkeypatch.setattr(parity, "_browser_parity", lambda _root: evidence)

    with pytest.raises(parity.ParityError, match="DEMO_MODEL_PARITY_MISMATCH"):
        parity.run_parity(review_root, tmp_path / "report.json")


def test_parity_cli_uses_fixed_diagnostics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "C:/" + "Users/alice/private?token=secret"
    assert parity.main(["--review-root", secret]) == 1
    assert capsys.readouterr().out == "[FAIL] DEMO_MODEL_PARITY_SCOPE\n"

    monkeypatch.setattr(
        parity,
        "run_parity",
        lambda *_args: (_ for _ in ()).throw(parity.ParityError("runtime")),
    )
    assert parity.main(
        ["--review-root", secret, "--report", str(tmp_path / "report.json")]
    ) == 1
    assert capsys.readouterr().out == "[FAIL] DEMO_MODEL_PARITY_RUNTIME\n"

    completed = subprocess.run(
        [sys.executable, "scripts/model_parity_smoke.py", "--review-root", secret],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 1
    assert completed.stdout == "[FAIL] DEMO_MODEL_PARITY_SCOPE\n"
    assert completed.stderr == ""
