from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct

import onnx
from onnx import TensorProto, helper
import pytest

from scripts import sanitize_demo_model as sanitizer


def _private_path() -> str:
    return "C:" + chr(92) + "Users" + chr(92) + "owner" + chr(92) + "model.onnx"


def _model(*, metadata: list[tuple[str, str]] | None = None, doc_string: str = "") -> bytes:
    initializer = helper.make_tensor("weights", TensorProto.FLOAT, [1], struct.pack("<f", 3.25), raw=True)
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output0"])],
        "private-free-graph",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 1])],
        [initializer],
    )
    model = helper.make_model(graph, producer_name="test-suite", opset_imports=[helper.make_opsetid("", 13)])
    model.doc_string = doc_string
    for key, value in metadata if metadata is not None else [("converted_from", _private_path()), ("purpose", "public demo")]:
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    return model.SerializeToString(deterministic=True)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load(value: bytes) -> onnx.ModelProto:
    return onnx.load_model_from_string(value)


def _without_metadata(model: onnx.ModelProto) -> bytes:
    cloned = onnx.ModelProto()
    cloned.CopyFrom(model)
    del cloned.metadata_props[:]
    return cloned.SerializeToString(deterministic=True)


def _patch_source_digest(monkeypatch: pytest.MonkeyPatch, source: bytes) -> None:
    monkeypatch.setattr(sanitizer, "SOURCE_SHA256", _digest(source))


def test_sanitize_requires_exact_digest_and_one_admitted_metadata_field() -> None:
    source = _model()

    with pytest.raises(sanitizer.SanitizationError, match="DEMO_MODEL_DIGEST"):
        sanitizer.sanitize_model_bytes(source, expected_source_sha256="0" * 64)

    output, receipt = sanitizer.sanitize_model_bytes(source, expected_source_sha256=_digest(source))

    assert _digest(source) == receipt.source_sha256
    assert _digest(output) == receipt.output_sha256
    assert receipt.source_bytes == len(source)
    assert receipt.output_bytes == len(output)
    assert receipt.modified_field == "ModelProto.metadata_props[0].value"
    assert receipt.modification_date == "2026-08-31"
    assert receipt.removed_metadata_entries == 1
    assert receipt.structural_equivalent and receipt.checker_passed and receipt.privacy_passed and receipt.deterministic


def test_sanitize_removes_only_metadata_entry_zero_and_preserves_all_other_fields() -> None:
    source = _model(metadata=[("converted_from", _private_path()), ("purpose", "public demo"), ("team", "aerial")])
    original = _load(source)

    output, _ = sanitizer.sanitize_model_bytes(source, expected_source_sha256=_digest(source))
    sanitized = _load(output)

    assert [(item.key, item.value) for item in sanitized.metadata_props] == [("purpose", "public demo"), ("team", "aerial")]
    assert _without_metadata(original) == _without_metadata(sanitized)
    assert hashlib.sha256(original.graph.initializer[0].raw_data).hexdigest() == hashlib.sha256(sanitized.graph.initializer[0].raw_data).hexdigest()
    assert [(item.name, item.type.tensor_type.shape.dim[0].dim_value) for item in original.graph.input] == [(item.name, item.type.tensor_type.shape.dim[0].dim_value) for item in sanitized.graph.input]
    assert [(item.name, item.type.tensor_type.shape.dim[0].dim_value) for item in original.graph.output] == [(item.name, item.type.tensor_type.shape.dim[0].dim_value) for item in sanitized.graph.output]
    assert [(item.domain, item.version) for item in original.opset_import] == [(item.domain, item.version) for item in sanitized.opset_import]


def test_sanitize_rejects_zero_multiple_wrong_index_and_nonmetadata_matches() -> None:
    cases = [
        _model(metadata=[]),
        _model(metadata=[("converted_from", _private_path()), ("other", _private_path())]),
        _model(metadata=[("purpose", "public demo"), ("converted_from", _private_path())]),
        _model(metadata=[("purpose", "public demo")], doc_string=_private_path()),
    ]

    for source in cases:
        with pytest.raises(sanitizer.SanitizationError, match="DEMO_MODEL_PRIVACY"):
            sanitizer.sanitize_model_bytes(source, expected_source_sha256=_digest(source))


def test_sanitize_is_deterministic_checker_valid_private_and_transactional(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_bytes = _model()
    _patch_source_digest(monkeypatch, source_bytes)
    source, output, receipt = tmp_path / "source.onnx", tmp_path / "output.onnx", tmp_path / "receipt.json"
    source.write_bytes(source_bytes)

    first = sanitizer.sanitize_official_model(source, output, receipt)
    first_output, first_receipt = output.read_bytes(), receipt.read_bytes()
    second = sanitizer.sanitize_official_model(source, output, receipt)

    assert first == second
    assert output.read_bytes() == first_output
    assert receipt.read_bytes() == first_receipt
    onnx.checker.check_model(_load(first_output))
    assert _private_path().encode("utf-8") not in first_output
    assert not any(item.external_data for item in _load(first_output).graph.initializer)
    assert json.loads(first_receipt) == asdict(first)

    output.write_bytes(b"previous-output")
    receipt.write_bytes(b"previous-receipt")
    source.write_bytes(b"not an onnx model")
    with pytest.raises(sanitizer.SanitizationError, match="DEMO_MODEL_DIGEST"):
        sanitizer.sanitize_official_model(source, output, receipt)
    assert output.read_bytes() == b"previous-output"
    assert receipt.read_bytes() == b"previous-receipt"

    source.write_bytes(source_bytes)
    sanitizer.sanitize_official_model(source, output, receipt)
    previous_output, previous_receipt = output.read_bytes(), receipt.read_bytes()
    real_replace, calls = sanitizer.os.replace, 0

    def fail_second_replace(source_path: object, destination_path: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated")
        real_replace(source_path, destination_path)  # type: ignore[arg-type]

    monkeypatch.setattr(sanitizer.os, "replace", fail_second_replace)
    with pytest.raises(sanitizer.SanitizationError, match="DEMO_MODEL_IO"):
        sanitizer.sanitize_official_model(source, output, receipt)
    assert output.read_bytes() == previous_output
    assert receipt.read_bytes() == previous_receipt


def test_validate_sanitized_model_rejects_graph_tensor_opset_receipt_and_privacy_mutations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_bytes = _model()
    _patch_source_digest(monkeypatch, source_bytes)
    source, output, receipt = tmp_path / "source.onnx", tmp_path / "output.onnx", tmp_path / "receipt.json"
    source.write_bytes(source_bytes)
    sanitizer.sanitize_official_model(source, output, receipt)
    assert sanitizer.validate_sanitized_model(source, output, receipt).output_sha256 == _digest(output.read_bytes())

    def assert_rejected(mutated: onnx.ModelProto, code: str) -> None:
        output.write_bytes(mutated.SerializeToString(deterministic=True))
        with pytest.raises(sanitizer.SanitizationError, match=code):
            sanitizer.validate_sanitized_model(source, output, receipt)
        sanitizer.sanitize_official_model(source, output, receipt)

    graph = _load(output.read_bytes())
    graph.graph.node[0].op_type = "Relu"
    assert_rejected(graph, "DEMO_MODEL_VERIFY")
    tensor = _load(output.read_bytes())
    tensor.graph.initializer[0].raw_data = b"\\x00\\x00\\x80?"
    assert_rejected(tensor, "DEMO_MODEL_VERIFY")
    opset = _load(output.read_bytes())
    opset.opset_import[0].version = 14
    assert_rejected(opset, "DEMO_MODEL_VERIFY")
    privacy = _load(output.read_bytes())
    entry = privacy.metadata_props.add()
    entry.key, entry.value = "restored", _private_path()
    assert_rejected(privacy, "DEMO_MODEL_PRIVACY")

    external = _load(source_bytes)
    external.graph.initializer[0].data_location = TensorProto.EXTERNAL
    location = external.graph.initializer[0].external_data.add()
    location.key, location.value = "location", "weights.bin"
    with pytest.raises(sanitizer.SanitizationError, match="DEMO_MODEL_VERIFY"):
        sanitizer.sanitize_model_bytes(
            external.SerializeToString(deterministic=True),
            expected_source_sha256=_digest(external.SerializeToString(deterministic=True)),
        )

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(sanitizer.SanitizationError, match="DEMO_MODEL_RECEIPT"):
        sanitizer.validate_sanitized_model(source, output, receipt)

    sanitizer.sanitize_official_model(source, output, receipt)
    duplicated = receipt.read_text(encoding="utf-8")
    source_bytes_field = f'  "source_bytes": {len(source_bytes)},\n'
    receipt.write_text(duplicated.replace(source_bytes_field, source_bytes_field * 2, 1), encoding="utf-8")
    with pytest.raises(sanitizer.SanitizationError, match="DEMO_MODEL_RECEIPT"):
        sanitizer.validate_sanitized_model(source, output, receipt)


def test_sanitizer_cli_diagnostics_are_fixed_and_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    secret_argument = "C:" + chr(92) + "Users" + chr(92) + "alice" + chr(92) + "private?token=secret"

    assert sanitizer.main(["invalid-command", "--source", secret_argument]) == 1

    captured = capsys.readouterr()
    assert captured.out == "[FAIL] DEMO_MODEL_SCOPE\n"
    assert captured.err == ""
    assert secret_argument not in captured.out + captured.err
