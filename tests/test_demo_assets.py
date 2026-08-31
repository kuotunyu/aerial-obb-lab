from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
from urllib.error import HTTPError

from PIL import Image
import pytest

import scripts.prepare_demo_assets as demo_assets
from scripts.prepare_demo_assets import (
    AssetPreparationError,
    AssetReceipt,
    AssetSpec,
    OFFICIAL_ASSETS,
    acquire_assets,
    main,
    publish_assets,
    urlopen_transport,
    verify_review_assets,
)


def _jpeg_bytes(expected_bytes: int, color: tuple[int, int, int] = (21, 82, 160)) -> bytes:
    image = Image.new("RGB", (3, 2), color=color)
    stream = io.BytesIO()
    image.save(stream, format="JPEG")
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
def fake_transport() -> FakeTransport:
    return FakeTransport(
        {
            spec.asset_id: (
                _body_for(spec.asset_id, spec.expected_bytes),
                (),
                _content_type_for(spec.asset_id),
            )
            for spec in OFFICIAL_ASSETS
        }
    )


def _acquire_review(review_root: Path, transport: FakeTransport) -> dict[str, AssetReceipt]:
    return acquire_assets(OFFICIAL_ASSETS, review_root, transport)


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_official_asset_specs_are_immutable_and_same_origin_publishable() -> None:
    assert OFFICIAL_ASSETS == (
        AssetSpec("boats-image", "https://ultralytics.com/images/boats.jpg", 194_872, ("ultralytics.com", "www.ultralytics.com", "github.com", "release-assets.githubusercontent.com"), "samples/boats.jpg"),
        AssetSpec("obb-model", "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.onnx", 10_207_250, ("github.com", "release-assets.githubusercontent.com"), "models/yolo26n-obb.onnx"),
        AssetSpec("ultralytics-license", "https://raw.githubusercontent.com/ultralytics/assets/v8.4.0/LICENSE", 34_523, ("raw.githubusercontent.com",), "third_party/ULTRALYTICS-AGPL-3.0.txt"),
    )
    assert list(inspect.signature(acquire_assets).parameters) == ["specs", "review_root", "transport"]
    assert list(inspect.signature(verify_review_assets).parameters) == ["specs", "review_root"]
    assert list(inspect.signature(publish_assets).parameters) == ["review_root", "pages_root"]
    with pytest.raises(FrozenInstanceError):
        OFFICIAL_ASSETS[0].asset_id = "different"  # type: ignore[misc]


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


def test_validate_receipts_rejects_missing_extra_or_changed_bytes(tmp_path: Path, fake_transport: FakeTransport) -> None:
    review_root = tmp_path / "external-review"
    _acquire_review(review_root, fake_transport)
    (review_root / OFFICIAL_ASSETS[0].public_relative_path).unlink()
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_RECEIPT"):
        verify_review_assets(OFFICIAL_ASSETS, review_root)
    _acquire_review(review_root, fake_transport)
    (review_root / "unreviewed.bin").write_bytes(b"not approved")
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_RECEIPT"):
        verify_review_assets(OFFICIAL_ASSETS, review_root)
    (review_root / "unreviewed.bin").unlink()
    (review_root / OFFICIAL_ASSETS[1].public_relative_path).write_bytes(b"changed")
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_LENGTH"):
        verify_review_assets(OFFICIAL_ASSETS, review_root)


def test_containment_helpers_reject_reparse_components_and_escape_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "review"
    root.mkdir()
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_SCOPE"):
        demo_assets.checked_child(root, Path("..") / "outside")
    (root / "models").mkdir()
    monkeypatch.setattr(demo_assets, "is_reparse_point", lambda path: path.name == "models")
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_SCOPE"):
        demo_assets.checked_child(root, Path("models") / "model.onnx")


def test_acquire_keeps_existing_batch_when_a_later_asset_fails(tmp_path: Path, fake_transport: FakeTransport) -> None:
    review_root = tmp_path / "external-review"
    _acquire_review(review_root, fake_transport)
    before = _files(review_root)
    fake_transport.responses["boats-image"] = (_jpeg_bytes(OFFICIAL_ASSETS[0].expected_bytes, (200, 20, 20)), (), "image/jpeg")

    def failing_transport(spec: AssetSpec) -> tuple[bytes, tuple[str, ...], str]:
        if spec.asset_id == "obb-model": raise AssetPreparationError("network")
        return fake_transport(spec)

    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_NETWORK"):
        acquire_assets(OFFICIAL_ASSETS, review_root, failing_transport)
    assert _files(review_root) == before
    assert not list(review_root.glob(".demo-assets-stage-*"))


def test_publish_rejects_review_root_inside_git_and_wrong_pages_root(tmp_path: Path, fake_transport: FakeTransport, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"; repo_root.mkdir(); (repo_root / ".git").mkdir()
    monkeypatch.setattr(demo_assets, "REPO_ROOT", repo_root)
    pages_root = repo_root / "demo" / "web"
    external_review = tmp_path / "external-review"
    _acquire_review(external_review, fake_transport)
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_SCOPE"):
        publish_assets(repo_root / "review", pages_root)
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_SCOPE"):
        publish_assets(external_review, repo_root / "wrong-pages")


def test_publish_rejects_stale_managed_page_leaf(tmp_path: Path, fake_transport: FakeTransport, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"; repo_root.mkdir(); (repo_root / ".git").mkdir()
    monkeypatch.setattr(demo_assets, "REPO_ROOT", repo_root)
    review_root = tmp_path / "external-review"
    _acquire_review(review_root, fake_transport)
    pages_root = repo_root / "demo" / "web"
    (pages_root / "models").mkdir(parents=True)
    (pages_root / "models" / "stale.onnx").write_bytes(b"stale")

    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_SCOPE"):
        publish_assets(review_root, pages_root)
    assert _files(pages_root) == {"models/stale.onnx": b"stale"}


def test_publish_writes_only_three_approved_paths_and_closed_manifest(tmp_path: Path, fake_transport: FakeTransport, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"; repo_root.mkdir(); (repo_root / ".git").mkdir()
    monkeypatch.setattr(demo_assets, "REPO_ROOT", repo_root)
    pages_root = repo_root / "demo" / "web"
    review_root = tmp_path / "external-review"
    receipts = _acquire_review(review_root, fake_transport)
    publish_assets(review_root, pages_root)
    assert sorted(_files(pages_root)) == [
        "THIRD_PARTY_NOTICES.md",
        "demo-model.json",
        "models/yolo26n-obb.onnx",
        "samples/boats.jpg",
        "third_party/ULTRALYTICS-AGPL-3.0.txt",
    ]
    manifest = json.loads((pages_root / "demo-model.json").read_text(encoding="utf-8"))
    assert set(manifest) == {
        "classes", "defaultConfidence", "id", "image", "input", "license", "model",
        "notice", "output", "schemaVersion",
    }
    assert manifest["image"]["sha256"] == receipts["boats-image"].sha256
    assert manifest["model"]["sha256"] == receipts["obb-model"].sha256
    assert manifest["license"]["sha256"] == receipts["ultralytics-license"].sha256
    assert manifest["output"] == {"name": "output0", "rowWidth": 7, "layout": ["cx", "cy", "w", "h", "confidence", "class", "angleRadians"]}
    assert not ({"results", "detections", "boxes", "tensor", "runtime", "url"} & set(manifest))
    before = _files(pages_root)
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


def test_cli_diagnostics_are_fixed_and_do_not_echo_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    secret_argument = "C:/Users/alice/private?token=secret"
    assert main(["invalid-command", "--review-root", secret_argument]) == 1
    assert capsys.readouterr().out == "[FAIL] DEMO_ASSET_SCOPE\n"
