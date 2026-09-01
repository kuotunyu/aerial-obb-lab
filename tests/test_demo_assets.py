from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from urllib.error import HTTPError

from PIL import Image
import pytest

import scripts.prepare_demo_assets as demo_assets
from scripts.sanitize_demo_model import SanitizationReceipt
from scripts.prepare_demo_assets import (
    AssetPreparationError,
    AssetReceipt,
    AssetSpec,
    OFFICIAL_ASSETS,
    acquire_assets,
    main,
    publish_assets,
    urlopen_transport,
    validate_receipts,
)


def _jpeg_bytes(expected_bytes: int, color: tuple[int, int, int] = (21, 82, 160)) -> bytes:
    image = Image.new("RGB", (3, 2), color=color)
    stream = io.BytesIO()
    image.save(stream, format="JPEG")
    payload = stream.getvalue()
    return payload + (b"\0" * (expected_bytes - len(payload)))


def _png_bytes(expected_bytes: int) -> bytes:
    image = Image.new("RGB", (3, 2), color=(21, 82, 160))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    payload = stream.getvalue()
    return payload + (b"\0" * (expected_bytes - len(payload)))


def _license_bytes(expected_bytes: int) -> bytes:
    prefix = b"GNU AFFERO GENERAL PUBLIC LICENSE\nVersion 3\n"
    return prefix + (b"L" * (expected_bytes - len(prefix)))


def _body_for(asset_id: str, expected_bytes: int) -> bytes:
    if asset_id == "boats-image":
        return _jpeg_bytes(expected_bytes)
    if asset_id == "ultralytics-license":
        return _license_bytes(expected_bytes)
    return b"\x08ONNX" + (b"M" * (expected_bytes - 5))


def _content_type_for(asset_id: str) -> str:
    return {
        "boats-image": "application/octet-stream",
        "obb-model": "application/octet-stream",
        "ultralytics-license": "text/plain; charset=utf-8",
    }[asset_id]


class FakeTransport:
    def __init__(self, responses: dict[str, tuple[bytes, tuple[str, ...], str]]) -> None:
        self.responses = responses

    def __call__(self, spec: AssetSpec) -> tuple[bytes, tuple[str, ...], str]:
        return self.responses[spec.asset_id]


@pytest.fixture
def fake_transport(monkeypatch: pytest.MonkeyPatch) -> FakeTransport:
    transport = FakeTransport(
        {
            spec.asset_id: (
                _body_for(spec.asset_id, spec.expected_bytes),
                (),
                _content_type_for(spec.asset_id),
            )
            for spec in OFFICIAL_ASSETS
        }
    )
    monkeypatch.setattr(
        demo_assets,
        "OFFICIAL_SHA256",
        {
            spec.asset_id: hashlib.sha256(
                transport.responses[spec.asset_id][0]
            ).hexdigest()
            for spec in OFFICIAL_ASSETS
        },
    )
    return transport


def _acquire_review(review_root: Path, transport: FakeTransport) -> dict[str, AssetReceipt]:
    return acquire_assets(review_root, transport)


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _admitted_review(
    tmp_path: Path,
    fake_transport: FakeTransport,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, SanitizationReceipt]:
    review_root = tmp_path / "external-review"
    monkeypatch.setattr(
        demo_assets,
        "OFFICIAL_SHA256",
        {
            spec.asset_id: hashlib.sha256(
                fake_transport.responses[spec.asset_id][0]
            ).hexdigest()
            for spec in OFFICIAL_ASSETS
        },
    )
    receipts = _acquire_review(review_root, fake_transport)
    derivative = b"privacy-safe-derivative"
    sanitization = SanitizationReceipt(
        source_bytes=receipts["obb-model"].bytes,
        source_sha256=receipts["obb-model"].sha256,
        output_bytes=len(derivative),
        output_sha256=hashlib.sha256(derivative).hexdigest(),
        onnx_version="1.22.0",
        protobuf_version="6.32.0",
        removed_metadata_entries=1,
        modified_field="ModelProto.metadata_props[0].value",
        modification_date="2026-08-31",
        structural_equivalent=True,
        checker_passed=True,
        privacy_passed=True,
        deterministic=True,
    )
    sanitized = review_root / "sanitized"
    sanitized.mkdir()
    (sanitized / "yolo26n-obb-privacy-sanitized.onnx").write_bytes(derivative)
    (sanitized / "sanitization-receipt.json").write_text(
        json.dumps(asdict(sanitization), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        demo_assets,
        "validate_sanitized_model",
        lambda *_args: sanitization,
    )
    monkeypatch.setattr(demo_assets, "require_browser_parity", lambda _root: None)
    return review_root, sanitization


def test_validate_admitted_assets_requires_exact_source_and_sanitized_layout(
    tmp_path: Path,
    fake_transport: FakeTransport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review_root, sanitization = _admitted_review(
        tmp_path, fake_transport, monkeypatch
    )

    admitted = demo_assets.validate_admitted_assets(review_root)

    assert admitted.sanitization == sanitization
    assert set(admitted.receipts) == {
        "boats-image",
        "obb-model",
        "ultralytics-license",
    }
    (review_root / "unexpected.bin").write_bytes(b"not admitted")
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_RECEIPT"):
        demo_assets.validate_admitted_assets(review_root)


def test_publish_never_copies_upstream_model_and_writes_exact_derivative_set(
    tmp_path: Path,
    fake_transport: FakeTransport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    subprocess.run(
        ["git", "init", str(repo_root)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    monkeypatch.setattr(demo_assets, "REPO_ROOT", repo_root)
    review_root, sanitization = _admitted_review(
        tmp_path, fake_transport, monkeypatch
    )
    pages_root = repo_root / "demo" / "web"

    publish_assets(review_root, pages_root)

    assert sorted(_files(pages_root)) == [
        "THIRD_PARTY_NOTICES.md",
        "demo-model.json",
        "models/yolo26n-obb-privacy-sanitized.onnx",
        "samples/boats.jpg",
        "third_party/ULTRALYTICS-AGPL-3.0.txt",
        "third_party/yolo26n-obb-privacy-sanitization.json",
    ]
    assert not (pages_root / "models" / "yolo26n-obb.onnx").exists()
    assert (
        hashlib.sha256(
            (pages_root / "models" / "yolo26n-obb-privacy-sanitized.onnx").read_bytes()
        ).hexdigest()
        == sanitization.output_sha256
    )


def test_manifest_and_sanitization_record_are_closed_and_privacy_safe(
    tmp_path: Path,
    fake_transport: FakeTransport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    subprocess.run(
        ["git", "init", str(repo_root)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    monkeypatch.setattr(demo_assets, "REPO_ROOT", repo_root)
    review_root, sanitization = _admitted_review(
        tmp_path, fake_transport, monkeypatch
    )
    pages_root = repo_root / "demo" / "web"

    publish_assets(review_root, pages_root)

    manifest_text = (pages_root / "demo-model.json").read_text(encoding="utf-8")
    record_text = (
        pages_root / "third_party" / "yolo26n-obb-privacy-sanitization.json"
    ).read_text(encoding="utf-8")
    notice_text = (pages_root / "THIRD_PARTY_NOTICES.md").read_text(
        encoding="utf-8"
    )
    manifest = json.loads(manifest_text)
    record = json.loads(record_text)
    assert set(manifest) == {
        "classes",
        "defaultConfidence",
        "id",
        "image",
        "input",
        "license",
        "model",
        "notice",
        "output",
        "provenance",
        "sanitization",
        "schemaVersion",
    }
    assert set(record) == {
        "derivative",
        "license",
        "provenance",
        "sanitizer",
        "schemaVersion",
        "source",
        "transformation",
        "verification",
    }
    assert manifest["model"]["path"] == "models/yolo26n-obb-privacy-sanitized.onnx"
    assert manifest["model"]["sourceSha256"] == sanitization.source_sha256
    assert manifest["sanitization"]["path"] == "third_party/yolo26n-obb-privacy-sanitization.json"
    assert record["transformation"] == {
        "modificationDate": "2026-08-31",
        "modifiedField": "ModelProto.metadata_props[0].value",
        "modificationStatus": "metadata-only",
        "removedMetadataEntries": 1,
    }
    assert record["verification"]["browserParityPassed"] is True
    public_text = manifest_text + record_text
    assert "C:" + "\\Users\\" not in public_text
    assert "raw_header" not in public_text
    assert "redirect_hosts" not in public_text
    assert "tensor" not in public_text.casefold()
    assert OFFICIAL_ASSETS[1].source_url in notice_text
    assert (
        "https://github.com/kuotunyu/aerial-obb-lab/blob/"
        "924fda756801f906e6cb2ea174978fd4b6c37c2c/"
        "scripts/sanitize_demo_model.py"
    ) in notice_text


def test_official_asset_specs_are_immutable_and_same_origin_publishable() -> None:
    assert OFFICIAL_ASSETS == (
        AssetSpec("boats-image", "https://ultralytics.com/images/boats.jpg", 194_872, ("ultralytics.com", "www.ultralytics.com", "github.com", "release-assets.githubusercontent.com"), "samples/boats.jpg"),
        AssetSpec("obb-model", "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.onnx", 10_207_250, ("github.com", "release-assets.githubusercontent.com"), "models/yolo26n-obb.onnx"),
        AssetSpec("ultralytics-license", "https://raw.githubusercontent.com/ultralytics/assets/v8.4.0/LICENSE", 34_523, ("raw.githubusercontent.com",), "third_party/ULTRALYTICS-AGPL-3.0.txt"),
    )
    assert list(inspect.signature(acquire_assets).parameters) == ["review_root", "transport"]
    assert list(inspect.signature(validate_receipts).parameters) == ["review_root"]
    assert list(inspect.signature(publish_assets).parameters) == ["review_root", "pages_root"]
    assert demo_assets.OFFICIAL_SHA256 == {
        "boats-image": "8c5ada657cf8110a9f8aaac954c1dd96cde0187315b581276c32b0d1863e756f",
        "obb-model": "02f7c539600296d7389341280beb82da810b15dc09c54cf2bc70f7f610331b38",
        "ultralytics-license": "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0",
    }
    with pytest.raises(FrozenInstanceError):
        OFFICIAL_ASSETS[0].asset_id = "different"  # type: ignore[misc]
    with pytest.raises(TypeError):
        demo_assets.OFFICIAL_SHA256["boats-image"] = "different"  # type: ignore[index]


def test_acquire_rejects_status_redirect_host_length_and_content_type_drift(tmp_path: Path, fake_transport: FakeTransport, monkeypatch: pytest.MonkeyPatch) -> None:
    boat = OFFICIAL_ASSETS[0]

    class StatusResponse(io.BytesIO):
        def getcode(self) -> int: return 503
        def geturl(self) -> str: return boat.source_url
        @property
        def headers(self) -> dict[str, str]: return {"Content-Type": "image/jpeg"}
        def __enter__(self) -> "StatusResponse": return self
        def __exit__(self, *_: object) -> None: self.close()

    monkeypatch.setattr(demo_assets.urllib.request, "build_opener", lambda *_: type("Opener", (), {"open": lambda *_args, **_kwargs: StatusResponse(b"x")})())
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_STATUS"):
        urlopen_transport(boat)

    cases = (
        ("redirect", (_body_for(boat.asset_id, boat.expected_bytes), ("untrusted.example",), "image/jpeg"), "DEMO_ASSET_REDIRECT"),
        ("length", (b"short", (), "image/jpeg"), "DEMO_ASSET_LENGTH"),
        ("content-type", (_body_for(boat.asset_id, boat.expected_bytes), (), "text/plain"), "DEMO_ASSET_MEDIA"),
    )
    for name, response, code in cases:
        fake_transport.responses[boat.asset_id] = response
        with pytest.raises(AssetPreparationError, match=code):
            _acquire_review(tmp_path / name, fake_transport)


def test_production_transport_streams_success_and_http_error_bodies_with_a_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    boat = OFFICIAL_ASSETS[0]

    class GuardedStream(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            assert 0 <= size <= 65_536
            return super().read(size)

    class Response(GuardedStream):
        def getcode(self) -> int: return 200
        def geturl(self) -> str: return boat.source_url
        @property
        def headers(self) -> dict[str, str]: return {"Content-Type": "image/jpeg"}
        def __enter__(self) -> "Response": return self
        def __exit__(self, *_: object) -> None: self.close()

    monkeypatch.setattr(demo_assets.urllib.request, "build_opener", lambda *_: type("Opener", (), {"open": lambda *_args, **_kwargs: Response(_jpeg_bytes(boat.expected_bytes))})())
    body, redirects, content_type = urlopen_transport(boat)
    assert (len(body), redirects, content_type) == (boat.expected_bytes, (), "image/jpeg")

    error_stream = GuardedStream(b"too much error body")
    error = HTTPError(boat.source_url, 503, "ignored", {}, error_stream)
    monkeypatch.setattr(demo_assets.urllib.request, "build_opener", lambda *_: type("Opener", (), {"open": lambda *_args, **_kwargs: (_ for _ in ()).throw(error)})())
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_STATUS"):
        urlopen_transport(boat)


def test_acquire_writes_digest_receipt_without_path_query_header_or_raw_error(tmp_path: Path, fake_transport: FakeTransport) -> None:
    review_root = tmp_path / "external-review"
    boat = OFFICIAL_ASSETS[0]
    fake_transport.responses[boat.asset_id] = (_body_for(boat.asset_id, boat.expected_bytes), ("www.ultralytics.com",), "application/octet-stream")
    receipts = _acquire_review(review_root, fake_transport)
    serialized = (review_root / "receipt.json").read_text(encoding="utf-8")
    payload = json.loads(serialized)
    assert serialized == json.dumps(payload, sort_keys=True, indent=2) + "\n"
    assert "?" not in serialized and "Authorization" not in serialized and "external-review" not in serialized
    assert "error" not in serialized.casefold()
    assert receipts["boats-image"] == AssetReceipt("boats-image", boat.source_url, ("www.ultralytics.com",), boat.expected_bytes, hashlib.sha256(_body_for(boat.asset_id, boat.expected_bytes)).hexdigest(), "image/jpeg", 3, 2)
    assert receipts["obb-model"].media_type == "application/onnx"


def test_acquire_rejects_decodable_non_jpeg_boats_bytes(tmp_path: Path, fake_transport: FakeTransport) -> None:
    boat = OFFICIAL_ASSETS[0]
    fake_transport.responses[boat.asset_id] = (_png_bytes(boat.expected_bytes), (), "image/jpeg")

    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_MEDIA"):
        _acquire_review(tmp_path / "external-review", fake_transport)


def test_acquire_rejects_unknown_review_member_without_calling_transport(tmp_path: Path, fake_transport: FakeTransport) -> None:
    review_root = tmp_path / "external-review"
    review_root.mkdir()
    unknown = review_root / "unreviewed.bin"
    unknown.write_bytes(b"keep this user file")
    calls: list[str] = []

    def recording_transport(spec: AssetSpec) -> tuple[bytes, tuple[str, ...], str]:
        calls.append(spec.asset_id)
        return fake_transport(spec)

    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_RECEIPT"):
        acquire_assets(review_root, recording_transport)
    assert calls == []
    assert unknown.read_bytes() == b"keep this user file"


def test_acquire_rejects_stale_sanitized_members_without_network(
    tmp_path: Path,
    fake_transport: FakeTransport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review_root, _sanitization = _admitted_review(
        tmp_path, fake_transport, monkeypatch
    )
    calls: list[str] = []

    def recording_transport(spec: AssetSpec) -> tuple[bytes, tuple[str, ...], str]:
        calls.append(spec.asset_id)
        return fake_transport(spec)

    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_RECEIPT"):
        acquire_assets(review_root, recording_transport)
    assert calls == []


def test_validate_receipts_rejects_missing_extra_or_changed_bytes(tmp_path: Path, fake_transport: FakeTransport) -> None:
    review_root = tmp_path / "external-review"
    _acquire_review(review_root, fake_transport)
    (review_root / OFFICIAL_ASSETS[0].public_relative_path).unlink()
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_RECEIPT"):
        validate_receipts(review_root)
    _acquire_review(review_root, fake_transport)
    (review_root / "unreviewed.bin").write_bytes(b"not approved")
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_RECEIPT"):
        validate_receipts(review_root)
    (review_root / "unreviewed.bin").unlink()
    (review_root / OFFICIAL_ASSETS[1].public_relative_path).write_bytes(b"changed")
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_LENGTH"):
        validate_receipts(review_root)


def test_validate_receipts_rejects_extra_top_level_and_asset_receipt_fields(tmp_path: Path, fake_transport: FakeTransport) -> None:
    review_root = tmp_path / "external-review"
    _acquire_review(review_root, fake_transport)
    receipt_path = review_root / "receipt.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["raw_error"] = "do not accept undeclared fields"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_RECEIPT"):
        validate_receipts(review_root)

    _acquire_review(review_root, fake_transport)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["assets"]["boats-image"]["raw_header"] = "do not accept undeclared fields"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_RECEIPT"):
        validate_receipts(review_root)


def test_validate_receipts_rejects_decodable_non_jpeg_boats_mutation(tmp_path: Path, fake_transport: FakeTransport) -> None:
    review_root = tmp_path / "external-review"
    _acquire_review(review_root, fake_transport)
    boat = OFFICIAL_ASSETS[0]
    mutated = _png_bytes(boat.expected_bytes)
    (review_root / boat.public_relative_path).write_bytes(mutated)
    receipt_path = review_root / "receipt.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["assets"][boat.asset_id]["sha256"] = hashlib.sha256(mutated).hexdigest()
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_MEDIA"):
        validate_receipts(review_root)


def test_containment_helpers_reject_reparse_components_and_escape_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "review"
    root.mkdir()
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_SCOPE"):
        demo_assets.checked_child(root, Path("..") / "outside")
    (root / "models").mkdir()
    monkeypatch.setattr(demo_assets, "is_reparse_point", lambda path: path.name == "models")
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_SCOPE"):
        demo_assets.checked_child(root, Path("models") / "model.onnx")


def test_acquire_keeps_existing_batch_when_a_later_asset_fails(tmp_path: Path, fake_transport: FakeTransport, monkeypatch: pytest.MonkeyPatch) -> None:
    review_root = tmp_path / "external-review"
    _acquire_review(review_root, fake_transport)
    before = _files(review_root)
    fake_transport.responses["boats-image"] = (_jpeg_bytes(OFFICIAL_ASSETS[0].expected_bytes, (200, 20, 20)), (), "image/jpeg")
    monkeypatch.setattr(
        demo_assets,
        "OFFICIAL_SHA256",
        {
            **demo_assets.OFFICIAL_SHA256,
            "boats-image": hashlib.sha256(
                fake_transport.responses["boats-image"][0]
            ).hexdigest(),
        },
    )

    def failing_transport(spec: AssetSpec) -> tuple[bytes, tuple[str, ...], str]:
        if spec.asset_id == "obb-model": raise AssetPreparationError("network")
        return fake_transport(spec)

    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_NETWORK"):
        acquire_assets(review_root, failing_transport)
    assert _files(review_root) == before
    assert not list(review_root.glob(".demo-assets-stage-*"))


def test_publish_rejects_review_root_inside_git_and_wrong_pages_root(tmp_path: Path, fake_transport: FakeTransport, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"; repo_root.mkdir()
    subprocess.run(
        ["git", "init", str(repo_root)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    monkeypatch.setattr(demo_assets, "REPO_ROOT", repo_root)
    pages_root = repo_root / "demo" / "web"
    external_review = tmp_path / "external-review"
    _acquire_review(external_review, fake_transport)
    calls: list[str] = []

    def recording_transport(spec: AssetSpec) -> tuple[bytes, tuple[str, ...], str]:
        calls.append(spec.asset_id)
        return fake_transport(spec)

    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_SCOPE"):
        acquire_assets(repo_root / "review", recording_transport)
    assert calls == []
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_SCOPE"):
        publish_assets(repo_root / "review", pages_root)
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_SCOPE"):
        publish_assets(external_review, repo_root / "wrong-pages")


def test_external_review_scope_works_without_git_and_contains_snapshot_root(
    tmp_path: Path,
    fake_transport: FakeTransport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    monkeypatch.setattr(demo_assets, "REPO_ROOT", snapshot_root)
    external_review = tmp_path / "external-review"

    receipts = acquire_assets(external_review, fake_transport)

    assert set(receipts) == {"boats-image", "obb-model", "ultralytics-license"}
    calls: list[str] = []

    def recording_transport(spec: AssetSpec) -> tuple[bytes, tuple[str, ...], str]:
        calls.append(spec.asset_id)
        return fake_transport(spec)

    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_SCOPE"):
        acquire_assets(snapshot_root / "review", recording_transport)
    assert calls == []


def test_external_review_scope_rejects_every_real_git_worktree(
    tmp_path: Path,
    fake_transport: FakeTransport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    linked_root = tmp_path / "linked"
    repo_root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo_root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Release Test",
            "-c",
            "user.email=release-test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        ["git", "worktree", "add", "-qb", "linked-review", str(linked_root)],
        cwd=repo_root,
        check=True,
    )
    monkeypatch.setattr(demo_assets, "REPO_ROOT", repo_root)
    calls: list[str] = []

    def recording_transport(spec: AssetSpec) -> tuple[bytes, tuple[str, ...], str]:
        calls.append(spec.asset_id)
        return fake_transport(spec)

    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_SCOPE"):
        acquire_assets(linked_root / "review", recording_transport)
    assert calls == []


def test_publish_rejects_stale_managed_page_leaf(tmp_path: Path, fake_transport: FakeTransport, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"; repo_root.mkdir()
    subprocess.run(
        ["git", "init", str(repo_root)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    monkeypatch.setattr(demo_assets, "REPO_ROOT", repo_root)
    review_root = tmp_path / "external-review"
    _acquire_review(review_root, fake_transport)
    pages_root = repo_root / "demo" / "web"
    (pages_root / "models").mkdir(parents=True)
    (pages_root / "models" / "stale.onnx").write_bytes(b"stale")

    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_SCOPE"):
        publish_assets(review_root, pages_root)
    assert _files(pages_root) == {"models/stale.onnx": b"stale"}


def test_publish_validates_complete_batch_and_restores_prior_batch_on_failure(tmp_path: Path, fake_transport: FakeTransport, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"; repo_root.mkdir()
    subprocess.run(["git", "init", str(repo_root)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
    monkeypatch.setattr(demo_assets, "REPO_ROOT", repo_root)
    pages_root = repo_root / "demo" / "web"
    review_root = tmp_path / "external-review"
    review_root, sanitization = _admitted_review(tmp_path, fake_transport, monkeypatch)
    publish_assets(review_root, pages_root)
    assert sorted(_files(pages_root)) == [
        "THIRD_PARTY_NOTICES.md",
        "demo-model.json",
        "models/yolo26n-obb-privacy-sanitized.onnx",
        "samples/boats.jpg",
        "third_party/ULTRALYTICS-AGPL-3.0.txt",
        "third_party/yolo26n-obb-privacy-sanitization.json",
    ]
    manifest = json.loads((pages_root / "demo-model.json").read_text(encoding="utf-8"))
    assert set(manifest) == {
        "classes", "defaultConfidence", "id", "image", "input", "license", "model",
        "notice", "output", "provenance", "sanitization", "schemaVersion",
    }
    assert manifest["model"]["sha256"] == sanitization.output_sha256
    assert manifest["model"]["sourceSha256"] == sanitization.source_sha256
    assert manifest["output"] == {"name": "output0", "dims": [1, "N", 7], "type": "float32", "rowWidth": 7, "layout": ["cx", "cy", "w", "h", "confidence", "class", "angleRadians"]}
    assert not ({"results", "detections", "boxes", "tensor", "runtime", "url"} & set(manifest))

    before = _files(pages_root)
    real_write_text = Path.write_text

    def corrupt_staged_manifest(path: Path, data: str, *args: object, **kwargs: object) -> int:
        if path.name == "demo-model.json" and ".demo-assets-stage-" in path.as_posix():
            data += " "
        return real_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", corrupt_staged_manifest)
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_RECEIPT"):
        publish_assets(review_root, pages_root)
    assert _files(pages_root) == before
    monkeypatch.setattr(Path, "write_text", real_write_text)
    real_replace, calls = os.replace, 0

    def fail_second_replace(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2: raise OSError("simulated")
        real_replace(source, destination)  # type: ignore[arg-type]

    monkeypatch.setattr(demo_assets.os, "replace", fail_second_replace)
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_SCOPE"):
        publish_assets(review_root, pages_root)
    assert _files(pages_root) == before
    assert not list(pages_root.glob(".demo-assets-stage-*"))

    monkeypatch.setattr(demo_assets.os, "replace", real_replace)
    (pages_root / "samples" / "boats.jpg").write_bytes(b"previous-sample")
    before_rollback_failure = _files(pages_root)
    rollback_calls = 0

    def fail_replacement_and_first_rollback(source: object, destination: object) -> None:
        nonlocal rollback_calls
        rollback_calls += 1
        if rollback_calls in {2, 3}:
            raise OSError("simulated")
        real_replace(source, destination)  # type: ignore[arg-type]

    monkeypatch.setattr(demo_assets.os, "replace", fail_replacement_and_first_rollback)
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_SCOPE"):
        publish_assets(review_root, pages_root)
    assert _files(pages_root) == before_rollback_failure


def test_gallery_receipt_admits_only_the_published_three_sample_bytes(tmp_path: Path) -> None:
    """Changing a gallery digest or adding boats must make asset publication refuse it."""
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((root / "release/sample-gallery-sources.json").read_text(encoding="utf-8"))
    pages = tmp_path / "web"
    (pages / "samples").mkdir(parents=True)
    for item in receipt["samples"]:
        (pages / item["path"]).write_bytes((root / "demo/web" / item["path"]).read_bytes())
    receipt_path = tmp_path / "sample-gallery-sources.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    demo_assets.validate_gallery_publication(pages, receipt_path)
    (pages / "samples" / "boats.jpg").write_bytes(b"stale")
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_RECEIPT"):
        demo_assets.validate_gallery_publication(pages, receipt_path)


def test_cli_disables_legacy_boats_acquisition_without_a_gallery_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The public preparation CLI must fail closed instead of acquiring boats."""
    assert main(["acquire", "--review-root", str(tmp_path / "outside-review")]) == 1
    assert capsys.readouterr().out == "[FAIL] DEMO_ASSET_RECEIPT\n"


def test_gallery_receipt_rejects_mutated_canonical_sample_contract(tmp_path: Path) -> None:
    """A consumer must reject source, guardrail, and private-shape drift."""
    source_receipt = Path(__file__).resolve().parents[1] / "release/sample-gallery-sources.json"
    payload = json.loads(source_receipt.read_text(encoding="utf-8"))
    pages = tmp_path / "web" / "samples"; pages.mkdir(parents=True)
    for item in payload["samples"]:
        (pages.parent / item["path"]).write_bytes((Path(__file__).resolve().parents[1] / "demo/web" / item["path"]).read_bytes())
    receipt = tmp_path / "receipt.json"
    for path, value in [
        (("samples", 0, "source", "provider"), "private source"),
        (("samples", 1, "derivation", "jpegQuality"), 91),
        (("samples", 2, "guardrails", "threshold"), 0.3),
        (("samples", 0, "source", "extra"), "x"),
    ]:
        mutated = deepcopy(payload); target = mutated
        for key in path[:-1]: target = target[key]
        target[path[-1]] = value
        receipt.write_text(json.dumps(mutated), encoding="utf-8")
        with pytest.raises(AssetPreparationError, match="DEMO_ASSET_RECEIPT"):
            demo_assets.validate_gallery_publication(pages.parent, receipt)


def test_cli_diagnostics_are_fixed_and_do_not_echo_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    secret_argument = "C:/" + "Users/alice/private?token=secret"
    assert main(["invalid-command", "--review-root", secret_argument]) == 1
    assert capsys.readouterr().out == "[FAIL] DEMO_ASSET_SCOPE\n"


def test_git_worktree_roots_decodes_utf8_bytes_independent_of_host_locale(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    unicode_root = tmp_path / "工作區"
    subprocess.run(
        ["git", "init", "-q"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )

    def fake_run(*_args: object, **kwargs: object) -> SimpleNamespace:
        assert kwargs == {
            "check": True,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": False,
        }
        return SimpleNamespace(stdout=f"worktree {unicode_root}\n".encode("utf-8"))

    monkeypatch.setattr(demo_assets.subprocess, "run", fake_run)

    assert unicode_root.resolve() in demo_assets._git_worktree_roots(tmp_path)
