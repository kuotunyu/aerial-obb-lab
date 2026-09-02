"""Deterministically remove the reviewed private ONNX metadata value."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Iterable

import google.protobuf
import onnx
from onnx import TensorProto

from scripts.prepare_demo_assets import checked_child, is_reparse_point


SOURCE_SHA256 = "02f7c539600296d7389341280beb82da810b15dc09c54cf2bc70f7f610331b38"
MODIFIED_FIELD = "ModelProto.metadata_props[0].value"
MODIFICATION_DATE = "2026-08-31"
ERROR_CODES = {
    "digest": "DEMO_MODEL_DIGEST",
    "privacy": "DEMO_MODEL_PRIVACY",
    "verify": "DEMO_MODEL_VERIFY",
    "receipt": "DEMO_MODEL_RECEIPT",
    "scope": "DEMO_MODEL_SCOPE",
    "io": "DEMO_MODEL_IO",
}
PRIVATE_PATH = re.compile(rb"(?i)(?:[a-z]:[\\/](?:users|home)[\\/]|/(?:home|users)/)")
PRIVATE_TEXT = re.compile(r"(?i)(?:[a-z]:[\\/](?:users|home)[\\/]|/(?:home|users)/)")


class SanitizationError(Exception):
    def __init__(self, category: str) -> None:
        self.code = ERROR_CODES[category]
        super().__init__(self.code)


@dataclass(frozen=True)
class SanitizationReceipt:
    source_bytes: int
    source_sha256: str
    output_bytes: int
    output_sha256: str
    onnx_version: str
    protobuf_version: str
    removed_metadata_entries: int
    modified_field: str
    modification_date: str
    structural_equivalent: bool
    checker_passed: bool
    privacy_passed: bool
    deterministic: bool


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_source_digest(source: bytes, expected_source_sha256: str) -> None:
    if not hmac.compare_digest(_digest(source), expected_source_sha256):
        raise SanitizationError("digest")


def _field_path(prefix: str, name: str) -> str:
    return f"{prefix}.{name}"


def _sensitive_string(value: str) -> bool:
    return PRIVATE_TEXT.search(value) is not None


def _sensitive_bytes(value: bytes) -> bool:
    if PRIVATE_PATH.search(value) is not None:
        return True
    for encoding in ("utf-16-le", "utf-16-be"):
        for offset in (0, 1):
            if _sensitive_string(value[offset:].decode(encoding, errors="ignore")):
                return True
    return False


def _sensitive_fields(message: object, prefix: str) -> Iterable[str]:
    descriptor = message.DESCRIPTOR  # type: ignore[attr-defined]
    for field in descriptor.fields:
        value = getattr(message, field.name)  # type: ignore[arg-type]
        field_path = _field_path(prefix, field.name)
        if field.is_repeated:
            for index, item in enumerate(value):
                item_path = f"{field_path}[{index}]"
                if field.type == field.TYPE_MESSAGE:
                    yield from _sensitive_fields(item, item_path)
                elif field.type == field.TYPE_STRING and _sensitive_string(item):
                    yield item_path
                elif field.type == field.TYPE_BYTES and _sensitive_bytes(item):
                    yield item_path
        elif field.type == field.TYPE_MESSAGE:
            if message.HasField(field.name):  # type: ignore[attr-defined]
                yield from _sensitive_fields(value, field_path)
        elif field.type == field.TYPE_STRING and value and _sensitive_string(value):
            yield field_path
        elif field.type == field.TYPE_BYTES and value and _sensitive_bytes(value):
            yield field_path


def inspect_sensitive_fields(model: onnx.ModelProto) -> list[str]:
    return list(_sensitive_fields(model, "ModelProto"))


def require_exact_match(matches: list[str], modified_field: str) -> None:
    if matches != [modified_field]:
        raise SanitizationError("privacy")


def clone_without_metadata(model: onnx.ModelProto) -> onnx.ModelProto:
    cloned = onnx.ModelProto()
    cloned.CopyFrom(model)
    del cloned.metadata_props[:]
    return cloned


def require_structural_identity(first: onnx.ModelProto, second: onnx.ModelProto) -> None:
    if first.SerializeToString(deterministic=True) != second.SerializeToString(deterministic=True):
        raise SanitizationError("verify")


def _messages(message: object) -> Iterable[object]:
    yield message
    descriptor = message.DESCRIPTOR  # type: ignore[attr-defined]
    for field in descriptor.fields:
        if field.type != field.TYPE_MESSAGE:
            continue
        value = getattr(message, field.name)  # type: ignore[arg-type]
        if field.is_repeated:
            for item in value:
                yield from _messages(item)
        elif message.HasField(field.name):  # type: ignore[attr-defined]
            yield from _messages(value)


def require_no_external_data(model: onnx.ModelProto) -> None:
    for message in _messages(model):
        if isinstance(message, onnx.TensorProto) and (message.data_location == TensorProto.EXTERNAL or message.external_data):
            raise SanitizationError("verify")


def require_private_bytes(value: bytes) -> None:
    if _sensitive_bytes(value):
        raise SanitizationError("privacy")


def _load_model(value: bytes) -> onnx.ModelProto:
    try:
        return onnx.load_model_from_string(value)
    except Exception:
        raise SanitizationError("verify") from None


def _check_model(model: onnx.ModelProto) -> None:
    try:
        onnx.checker.check_model(model)
    except Exception:
        raise SanitizationError("verify") from None


def make_receipt(source: bytes, output: bytes) -> SanitizationReceipt:
    return SanitizationReceipt(
        source_bytes=len(source),
        source_sha256=_digest(source),
        output_bytes=len(output),
        output_sha256=_digest(output),
        onnx_version=onnx.__version__,
        protobuf_version=google.protobuf.__version__,
        removed_metadata_entries=1,
        modified_field=MODIFIED_FIELD,
        modification_date=MODIFICATION_DATE,
        structural_equivalent=True,
        checker_passed=True,
        privacy_passed=True,
        deterministic=True,
    )


def sanitize_model_bytes(source: bytes, *, expected_source_sha256: str) -> tuple[bytes, SanitizationReceipt]:
    require_source_digest(source, expected_source_sha256)
    model = _load_model(source)
    matches = inspect_sensitive_fields(model)
    require_exact_match(matches, MODIFIED_FIELD)
    require_no_external_data(model)
    original_without_metadata = clone_without_metadata(model)
    del model.metadata_props[0]
    derived = model.SerializeToString(deterministic=True)
    validated = _load_model(derived)
    _check_model(validated)
    require_no_external_data(validated)
    require_structural_identity(original_without_metadata, clone_without_metadata(validated))
    require_private_bytes(derived)
    return derived, make_receipt(source, derived)


def _safe_existing_file(path: Path) -> Path:
    parent = Path(os.path.abspath(path.parent))
    try:
        if not parent.is_dir() or is_reparse_point(parent):
            raise SanitizationError("scope")
        candidate = checked_child(parent, Path(path.name))
        if not candidate.is_file() or is_reparse_point(candidate):
            raise SanitizationError("scope")
        return candidate
    except SanitizationError:
        raise
    except Exception:
        raise SanitizationError("scope") from None


def _safe_destination(path: Path) -> Path:
    parent = Path(os.path.abspath(path.parent))
    try:
        for ancestor in [*reversed(parent.parents), parent]:
            if (ancestor.exists() or ancestor.is_symlink()) and is_reparse_point(ancestor):
                raise SanitizationError("scope")
        parent.mkdir(parents=True, exist_ok=True)
        for ancestor in [*reversed(parent.parents), parent]:
            if (ancestor.exists() or ancestor.is_symlink()) and is_reparse_point(ancestor):
                raise SanitizationError("scope")
        safe_parent = parent.resolve(strict=False)
        if not safe_parent.is_dir() or is_reparse_point(safe_parent):
            raise SanitizationError("scope")
        candidate = checked_child(safe_parent, Path(path.name))
        try:
            candidate.resolve(strict=False).relative_to(safe_parent)
        except ValueError:
            raise SanitizationError("scope") from None
        if candidate.exists() and is_reparse_point(candidate):
            raise SanitizationError("scope")
        return candidate
    except SanitizationError:
        raise
    except Exception:
        raise SanitizationError("scope") from None


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except Exception:
        raise SanitizationError("io") from None


def _receipt_bytes(receipt: SanitizationReceipt) -> bytes:
    return (json.dumps(asdict(receipt), sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write_descriptor(descriptor: int, value: bytes) -> None:
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _new_temporary(parent: Path, prefix: str) -> tuple[int, Path]:
    descriptor, temporary = tempfile.mkstemp(prefix=prefix, dir=parent)
    return descriptor, Path(temporary)


def _cleanup(path: Path) -> bool:
    try:
        path.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _write_transactional(entries: tuple[tuple[Path, bytes], ...]) -> None:
    staged: list[tuple[Path, Path]] = []
    backups: dict[Path, Path | None] = {}
    applied: list[Path] = []
    preserved_backups: set[Path] = set()
    failed = False
    try:
        for destination, value in entries:
            descriptor, temporary = _new_temporary(destination.parent, ".demo-model-stage-")
            staged.append((temporary, destination))
            _write_descriptor(descriptor, value)
            backup: Path | None = None
            if destination.exists():
                descriptor, backup = _new_temporary(destination.parent, ".demo-model-backup-")
                backups[destination] = backup
                _write_descriptor(descriptor, destination.read_bytes())
            else:
                backups[destination] = None
        for temporary, destination in staged:
            os.replace(temporary, destination)
            applied.append(destination)
    except Exception:
        failed = True
        for destination in reversed(applied):
            backup = backups[destination]
            try:
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(backup, destination)
            except Exception:
                if backup is not None:
                    preserved_backups.add(backup)
    finally:
        cleanup_failed = False
        for temporary, _ in staged:
            cleanup_failed = not _cleanup(temporary) or cleanup_failed
        for backup in backups.values():
            if backup is not None and backup not in preserved_backups:
                cleanup_failed = not _cleanup(backup) or cleanup_failed
    if failed or cleanup_failed:
        raise SanitizationError("io")


def sanitize_official_model(source: Path, output: Path, receipt: Path) -> SanitizationReceipt:
    safe_source = _safe_existing_file(source)
    safe_output = _safe_destination(output)
    safe_receipt = _safe_destination(receipt)
    if len({safe_source, safe_output, safe_receipt}) != 3:
        raise SanitizationError("scope")
    derived, result = sanitize_model_bytes(_read(safe_source), expected_source_sha256=SOURCE_SHA256)
    _write_transactional(((safe_output, derived), (safe_receipt, _receipt_bytes(result))))
    return result


def _load_receipt(path: Path) -> dict[str, object]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError
            payload[key] = value
        return payload

    try:
        payload = json.loads(_read(path).decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
        if not isinstance(payload, dict):
            raise ValueError
        return payload
    except SanitizationError:
        raise
    except Exception:
        raise SanitizationError("receipt") from None


def validate_sanitized_model(source: Path, output: Path, receipt: Path) -> SanitizationReceipt:
    safe_source = _safe_existing_file(source)
    safe_output = _safe_existing_file(output)
    safe_receipt = _safe_existing_file(receipt)
    expected_output, expected_receipt = sanitize_model_bytes(_read(safe_source), expected_source_sha256=SOURCE_SHA256)
    current_output = _read(safe_output)
    current_model = _load_model(current_output)
    _check_model(current_model)
    require_no_external_data(current_model)
    require_private_bytes(current_output)
    if not hmac.compare_digest(current_output, expected_output):
        raise SanitizationError("verify")
    received_receipt = _load_receipt(safe_receipt)
    expected_payload = asdict(expected_receipt)
    if set(received_receipt) != set(expected_payload) or any(
        type(received_receipt[key]) is not type(expected_payload[key]) or received_receipt[key] != expected_payload[key]
        for key in expected_payload
    ):
        raise SanitizationError("receipt")
    return expected_receipt


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if len(arguments) == 7 and arguments[0] in {"sanitize", "verify"} and arguments[1::2] == ["--source", "--output", "--receipt"]:
            source, output, receipt = (Path(arguments[index]) for index in (2, 4, 6))
            if arguments[0] == "sanitize":
                sanitize_official_model(source, output, receipt)
                print("[OK] DEMO_MODEL_SANITIZED")
            else:
                validate_sanitized_model(source, output, receipt)
                print("[OK] DEMO_MODEL_VERIFIED")
            return 0
        raise SanitizationError("scope")
    except SanitizationError as error:
        print(f"[FAIL] {error.code}")
        return 1
    except Exception:
        print(f"[FAIL] {ERROR_CODES['io']}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
