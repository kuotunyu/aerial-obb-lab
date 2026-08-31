from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import io
import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.prepare_demo_assets import (
    AssetPreparationError,
    HTTPResponse,
    OFFICIAL_ASSETS,
    acquire_assets,
    main,
    publish_assets,
    validate_receipts,
)


def _jpeg_bytes(expected_bytes: int) -> bytes:
    image = Image.new("RGB", (3, 2), color=(21, 82, 160))
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
        "boats-image": "image/jpeg",
        "obb-model": "application/octet-stream",
        "ultralytics-license": "text/plain; charset=utf-8",
    }[asset_id]


class FakeTransport:
    def __init__(self, responses: dict[str, HTTPResponse]) -> None:
        self.responses = responses

    def __call__(self, source_url: str) -> HTTPResponse:
        return self.responses[source_url]


@pytest.fixture
def fake_transport() -> FakeTransport:
    responses = {}
    for spec in OFFICIAL_ASSETS:
        responses[spec.source_url] = HTTPResponse(
            status=200,
            final_url=spec.source_url,
            redirect_urls=(),
            headers={"Content-Type": _content_type_for(spec.asset_id)},
            body=_body_for(spec.asset_id, spec.expected_bytes),
        )
    return FakeTransport(responses)


def _acquire_review(review_root: Path, transport: FakeTransport) -> dict:
    acquire_assets(review_root, transport=transport)
    return json.loads((review_root / "receipt.json").read_text(encoding="utf-8"))


def test_official_asset_specs_are_immutable_and_same_origin_publishable() -> None:
    from scripts.prepare_demo_assets import AssetSpec

    assert OFFICIAL_ASSETS == (
        AssetSpec(
            asset_id="boats-image",
            source_url="https://ultralytics.com/images/boats.jpg",
            expected_bytes=194_872,
            allowed_redirect_hosts=(
                "ultralytics.com",
                "www.ultralytics.com",
                "github.com",
                "release-assets.githubusercontent.com",
            ),
            public_relative_path="samples/boats.jpg",
        ),
        AssetSpec(
            asset_id="obb-model",
            source_url=(
                "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
                "yolo26n-obb.onnx"
            ),
            expected_bytes=10_207_250,
            allowed_redirect_hosts=("github.com", "release-assets.githubusercontent.com"),
            public_relative_path="models/yolo26n-obb.onnx",
        ),
        AssetSpec(
            asset_id="ultralytics-license",
            source_url="https://raw.githubusercontent.com/ultralytics/assets/v8.4.0/LICENSE",
            expected_bytes=34_523,
            allowed_redirect_hosts=("raw.githubusercontent.com",),
            public_relative_path="third_party/ULTRALYTICS-AGPL-3.0.txt",
        ),
    )
    with pytest.raises(FrozenInstanceError):
        OFFICIAL_ASSETS[0].asset_id = "different"  # type: ignore[misc]


def test_acquire_rejects_status_redirect_host_length_and_content_type_drift(
    tmp_path: Path, fake_transport: FakeTransport
) -> None:
    boat = OFFICIAL_ASSETS[0]
    cases = (
        ("status", replace(fake_transport.responses[boat.source_url], status=503), "DEMO_ASSET_STATUS"),
        (
            "redirect",
            replace(
                fake_transport.responses[boat.source_url],
                redirect_urls=("https://untrusted.example/boats.jpg",),
            ),
            "DEMO_ASSET_REDIRECT",
        ),
        (
            "length",
            replace(fake_transport.responses[boat.source_url], body=b"short"),
            "DEMO_ASSET_LENGTH",
        ),
        (
            "content-type",
            replace(
                fake_transport.responses[boat.source_url],
                headers={"Content-Type": "application/octet-stream"},
            ),
            "DEMO_ASSET_MEDIA",
        ),
    )
    for name, response, code in cases:
        fake_transport.responses[boat.source_url] = response
        with pytest.raises(AssetPreparationError, match=code):
            acquire_assets(tmp_path / name, transport=fake_transport)
        fake_transport.responses[boat.source_url] = HTTPResponse(
            status=200,
            final_url=boat.source_url,
            redirect_urls=(),
            headers={"Content-Type": "image/jpeg"},
            body=_body_for(boat.asset_id, boat.expected_bytes),
        )


def test_acquire_writes_digest_receipt_without_path_query_header_or_raw_error(
    tmp_path: Path, fake_transport: FakeTransport
) -> None:
    review_root = tmp_path / "external-review"
    boat = OFFICIAL_ASSETS[0]
    fake_transport.responses[boat.source_url] = replace(
        fake_transport.responses[boat.source_url],
        final_url="https://www.ultralytics.com/boats.jpg?private=query",
        redirect_urls=("https://www.ultralytics.com/boats.jpg?private=query",),
        headers={"Content-Type": "image/jpeg", "Authorization": "secret-header"},
    )

    receipt = _acquire_review(review_root, fake_transport)

    serialized = (review_root / "receipt.json").read_text(encoding="utf-8")
    assert serialized == json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    assert "private=query" not in serialized
    assert "secret-header" not in serialized
    assert "external-review" not in serialized
    assert "error" not in serialized.casefold()
    assert receipt["assets"]["boats-image"]["sha256"] == hashlib.sha256(
        _body_for("boats-image", boat.expected_bytes)
    ).hexdigest()
    assert receipt["assets"]["boats-image"]["redirect_hosts"] == ["www.ultralytics.com"]
    assert receipt["assets"]["boats-image"]["media"] == {
        "height": 2,
        "media_type": "image/jpeg",
        "width": 3,
    }


def test_validate_receipts_rejects_missing_extra_or_changed_bytes(
    tmp_path: Path, fake_transport: FakeTransport
) -> None:
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
    model_path = review_root / OFFICIAL_ASSETS[1].public_relative_path
    model_path.write_bytes(b"changed")
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_LENGTH"):
        validate_receipts(review_root)


def test_publish_rejects_review_root_inside_git_and_wrong_pages_root(
    tmp_path: Path, fake_transport: FakeTransport
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    pages_root = repo_root / "demo" / "web"
    external_review = tmp_path / "external-review"
    _acquire_review(external_review, fake_transport)

    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_SCOPE"):
        publish_assets(repo_root / "review", pages_root, repo_root=repo_root)
    with pytest.raises(AssetPreparationError, match="DEMO_ASSET_SCOPE"):
        publish_assets(external_review, repo_root / "wrong-pages", repo_root=repo_root)


def test_publish_writes_only_three_approved_paths_and_closed_manifest(
    tmp_path: Path, fake_transport: FakeTransport
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    pages_root = repo_root / "demo" / "web"
    review_root = tmp_path / "external-review"
    receipt = _acquire_review(review_root, fake_transport)

    publish_assets(review_root, pages_root, repo_root=repo_root)

    files = sorted(
        path.relative_to(pages_root).as_posix()
        for path in pages_root.rglob("*")
        if path.is_file()
    )
    assert files == [
        "THIRD_PARTY_NOTICES.md",
        "demo-model.json",
        "models/yolo26n-obb.onnx",
        "samples/boats.jpg",
        "third_party/ULTRALYTICS-AGPL-3.0.txt",
    ]
    manifest = json.loads((pages_root / "demo-model.json").read_text(encoding="utf-8"))
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
        "schemaVersion",
    }
    assert manifest["image"] == {
        "bytes": 194_872,
        "height": 2,
        "mediaType": "image/jpeg",
        "path": "samples/boats.jpg",
        "sha256": receipt["assets"]["boats-image"]["sha256"],
        "width": 3,
    }
    assert manifest["model"] == {
        "bytes": 10_207_250,
        "license": "AGPL-3.0-only",
        "path": "models/yolo26n-obb.onnx",
        "release": "v8.4.0",
        "sha256": receipt["assets"]["obb-model"]["sha256"],
        "source": OFFICIAL_ASSETS[1].source_url,
    }
    assert manifest["license"] == {
        "bytes": 34_523,
        "path": "third_party/ULTRALYTICS-AGPL-3.0.txt",
        "sha256": receipt["assets"]["ultralytics-license"]["sha256"],
    }
    assert manifest["output"] == {
        "name": "output0",
        "rowWidth": 7,
        "layout": ["cx", "cy", "w", "h", "confidence", "class", "angleRadians"],
    }
    forbidden = {"results", "detections", "boxes", "tensor", "runtime", "url"}
    assert not (set(manifest) & forbidden)
    assert not (set(manifest["output"]) & forbidden)


def test_cli_diagnostics_are_fixed_and_do_not_echo_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    secret_argument = "C:/Users/alice/private?token=secret"

    assert main(["invalid-command", secret_argument]) == 1

    assert capsys.readouterr().out == "[FAIL] DEMO_ASSET_SCOPE\n"
