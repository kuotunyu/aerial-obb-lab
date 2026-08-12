from __future__ import annotations

import pytest

from demo.model_source import require_model_path


def test_model_path_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_PATH", raising=False)

    with pytest.raises(RuntimeError, match="MODEL_PATH is required"):
        require_model_path()


def test_model_path_must_be_an_existing_file(tmp_path) -> None:
    missing = tmp_path / "missing.onnx"

    with pytest.raises(RuntimeError, match="does not exist or is not a file"):
        require_model_path(str(missing))


def test_model_path_rejects_unsupported_suffix(tmp_path) -> None:
    model = tmp_path / "owner-model.bin"
    model.write_bytes(b"fixture")

    with pytest.raises(RuntimeError, match="must end with .pt or .onnx"):
        require_model_path(str(model))


def test_existing_supported_model_is_resolved(tmp_path) -> None:
    model = tmp_path / "owner-model.ONNX"
    model.write_bytes(b"fixture")

    assert require_model_path(str(model), allowed_suffixes=(".onnx",)) == model.resolve()
