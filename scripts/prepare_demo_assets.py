"""Prepare reviewed official assets for the local browser demonstration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
import urllib.request

from PIL import Image, UnidentifiedImageError


ERROR_CODES = {
    "network": "DEMO_ASSET_NETWORK",
    "redirect": "DEMO_ASSET_REDIRECT",
    "status": "DEMO_ASSET_STATUS",
    "length": "DEMO_ASSET_LENGTH",
    "digest": "DEMO_ASSET_DIGEST",
    "media": "DEMO_ASSET_MEDIA",
    "scope": "DEMO_ASSET_SCOPE",
    "receipt": "DEMO_ASSET_RECEIPT",
}
MODEL_HARD_CEILING = 15 * 1024 * 1024


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    source_url: str
    expected_bytes: int
    allowed_redirect_hosts: tuple[str, ...]
    public_relative_path: str


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    final_url: str
    redirect_urls: tuple[str, ...]
    headers: Mapping[str, str]
    body: bytes


OFFICIAL_ASSETS = (
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


class AssetPreparationError(Exception):
    def __init__(self, category: str) -> None:
        self.code = ERROR_CODES[category]
        super().__init__(self.code)


class _TrackingRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self) -> None:
        super().__init__()
        self.redirect_urls: list[str] = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        self.redirect_urls.append(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def urllib_transport(source_url: str) -> HTTPResponse:
    """Fetch one URL while retaining redirect targets for allowlist validation."""
    redirects = _TrackingRedirectHandler()
    opener = urllib.request.build_opener(redirects)
    try:
        with opener.open(source_url, timeout=30) as response:
            return HTTPResponse(
                status=response.getcode(),
                final_url=response.geturl(),
                redirect_urls=tuple(redirects.redirect_urls),
                headers=dict(response.headers.items()),
                body=response.read(),
            )
    except HTTPError as error:
        return HTTPResponse(
            status=error.code,
            final_url=error.geturl(),
            redirect_urls=tuple(redirects.redirect_urls),
            headers=dict(error.headers.items()) if error.headers else {},
            body=error.read(),
        )
    except (OSError, URLError) as error:
        raise AssetPreparationError("network") from error


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").casefold()
    except ValueError:
        return ""


def _content_type(headers: Mapping[str, str]) -> str:
    for name, value in headers.items():
        if name.casefold() == "content-type":
            return value.split(";", 1)[0].strip().casefold()
    return ""


def _expected_content_type(spec: AssetSpec) -> str:
    return {
        "boats-image": "image/jpeg",
        "obb-model": "application/octet-stream",
        "ultralytics-license": "text/plain",
    }[spec.asset_id]


def _media_facts(spec: AssetSpec, body: bytes) -> dict[str, object]:
    if spec.asset_id == "boats-image":
        try:
            with Image.open(io.BytesIO(body)) as image:
                image.load()
                if image.width <= 0 or image.height <= 0:
                    raise AssetPreparationError("media")
                return {
                    "media_type": "image/jpeg",
                    "width": image.width,
                    "height": image.height,
                }
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise AssetPreparationError("media") from error
    if spec.asset_id == "ultralytics-license":
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AssetPreparationError("media") from error
        if "GNU AFFERO GENERAL PUBLIC LICENSE" not in text or "Version 3" not in text:
            raise AssetPreparationError("media")
        return {"media_type": "text/plain"}
    return {"media_type": "application/octet-stream"}


def _validated_response(spec: AssetSpec, response: HTTPResponse) -> tuple[dict[str, object], list[str]]:
    if response.status != 200:
        raise AssetPreparationError("status")
    redirect_hosts = [_host(url) for url in response.redirect_urls]
    final_host = _host(response.final_url)
    if not final_host or any(host not in spec.allowed_redirect_hosts for host in [*redirect_hosts, final_host]):
        raise AssetPreparationError("redirect")
    if len(response.body) != spec.expected_bytes or (
        spec.asset_id == "obb-model"
        and (spec.expected_bytes > MODEL_HARD_CEILING or len(response.body) > MODEL_HARD_CEILING)
    ):
        raise AssetPreparationError("length")
    if _content_type(response.headers) != _expected_content_type(spec):
        raise AssetPreparationError("media")
    return _media_facts(spec, response.body), redirect_hosts


def acquire_assets(review_root: Path, transport: Callable[[str], HTTPResponse] = urllib_transport) -> None:
    """Acquire the fixed official asset set into an external review directory."""
    root = Path(review_root).resolve()
    receipt_assets: dict[str, dict[str, object]] = {}
    for spec in OFFICIAL_ASSETS:
        try:
            response = transport(spec.source_url)
        except AssetPreparationError:
            raise
        except Exception as error:  # transport implementations may raise non-urllib errors
            raise AssetPreparationError("network") from error
        media, redirect_hosts = _validated_response(spec, response)
        destination = root / spec.public_relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.body)
        receipt_assets[spec.asset_id] = {
            "bytes": len(response.body),
            "media": media,
            "redirect_hosts": redirect_hosts,
            "sha256": hashlib.sha256(response.body).hexdigest(),
            "source_host": _host(spec.source_url),
        }
    receipt_payload = {"assets": receipt_assets, "schemaVersion": 1}
    (root / "receipt.json").write_text(
        json.dumps(receipt_payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _read_receipt(review_root: Path) -> dict[str, object]:
    try:
        receipt = json.loads((review_root / "receipt.json").read_text(encoding="utf-8"))
        if not isinstance(receipt, dict) or set(receipt) != {"assets", "schemaVersion"}:
            raise ValueError
        if receipt["schemaVersion"] != 1 or not isinstance(receipt["assets"], dict):
            raise ValueError
        if set(receipt["assets"]) != {spec.asset_id for spec in OFFICIAL_ASSETS}:
            raise ValueError
        return receipt
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise AssetPreparationError("receipt") from error


def validate_receipts(review_root: Path) -> dict[str, object]:
    """Revalidate every approved byte stream and every privacy-safe receipt fact."""
    root = Path(review_root).resolve()
    receipt = _read_receipt(root)
    expected_files = {"receipt.json", *(spec.public_relative_path for spec in OFFICIAL_ASSETS)}
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise AssetPreparationError("receipt")
    assets = receipt["assets"]
    if not isinstance(assets, dict):
        raise AssetPreparationError("receipt")
    for spec in OFFICIAL_ASSETS:
        details = assets[spec.asset_id]
        if not isinstance(details, dict) or set(details) != {
            "bytes", "media", "redirect_hosts", "sha256", "source_host"
        }:
            raise AssetPreparationError("receipt")
        if details["bytes"] != spec.expected_bytes or details["source_host"] != _host(spec.source_url):
            raise AssetPreparationError("receipt")
        redirect_hosts = details["redirect_hosts"]
        if (
            not isinstance(redirect_hosts, list)
            or not all(isinstance(host, str) and host in spec.allowed_redirect_hosts for host in redirect_hosts)
        ):
            raise AssetPreparationError("receipt")
        body = (root / spec.public_relative_path).read_bytes()
        if len(body) != spec.expected_bytes or (
            spec.asset_id == "obb-model"
            and (spec.expected_bytes > MODEL_HARD_CEILING or len(body) > MODEL_HARD_CEILING)
        ):
            raise AssetPreparationError("length")
        digest = hashlib.sha256(body).hexdigest()
        if details["sha256"] != digest:
            raise AssetPreparationError("digest")
        if details["media"] != _media_facts(spec, body):
            raise AssetPreparationError("receipt")
    return receipt


def _git_worktree_roots(repo_root: Path) -> set[Path]:
    roots = {repo_root.resolve()}
    try:
        output = subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "list", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout
    except OSError:
        return roots
    for line in output.splitlines():
        if line.startswith("worktree "):
            roots.add(Path(line.removeprefix("worktree ")).resolve())
    return roots


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _manifest(receipt: dict[str, object]) -> dict[str, object]:
    assets = receipt["assets"]
    if not isinstance(assets, dict):
        raise AssetPreparationError("receipt")
    image = assets["boats-image"]
    model = assets["obb-model"]
    license_asset = assets["ultralytics-license"]
    if not isinstance(image, dict) or not isinstance(model, dict) or not isinstance(license_asset, dict):
        raise AssetPreparationError("receipt")
    image_media = image["media"]
    if not isinstance(image_media, dict):
        raise AssetPreparationError("receipt")
    return {
        "schemaVersion": 1,
        "id": "ultralytics-yolo26n-obb-demo",
        "image": {
            "path": "samples/boats.jpg",
            "mediaType": "image/jpeg",
            "bytes": image["bytes"],
            "sha256": image["sha256"],
            "width": image_media["width"],
            "height": image_media["height"],
        },
        "model": {
            "path": "models/yolo26n-obb.onnx",
            "source": OFFICIAL_ASSETS[1].source_url,
            "release": "v8.4.0",
            "license": "AGPL-3.0-only",
            "bytes": model["bytes"],
            "sha256": model["sha256"],
        },
        "license": {
            "path": "third_party/ULTRALYTICS-AGPL-3.0.txt",
            "bytes": license_asset["bytes"],
            "sha256": license_asset["sha256"],
        },
        "input": {
            "name": "images",
            "dims": [1, 3, 1024, 1024],
            "type": "float32",
            "channelOrder": "RGB",
            "normalization": "divide-by-255",
            "letterboxValue": 114,
        },
        "output": {
            "name": "output0",
            "rowWidth": 7,
            "layout": ["cx", "cy", "w", "h", "confidence", "class", "angleRadians"],
        },
        "classes": [
            "plane", "ship", "storage tank", "baseball diamond", "tennis court",
            "basketball court", "ground track field", "harbor", "bridge", "large vehicle",
            "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool",
        ],
        "defaultConfidence": 0.25,
        "notice": "THIRD_PARTY_NOTICES.md",
    }


def _notices(receipt: dict[str, object]) -> str:
    assets = receipt["assets"]
    if not isinstance(assets, dict):
        raise AssetPreparationError("receipt")
    license_asset = assets["ultralytics-license"]
    if not isinstance(license_asset, dict):
        raise AssetPreparationError("receipt")
    return (
        "# Third-party notices\n\n"
        "The bundled Ultralytics YOLO26n-OBB model and sample image are subject to "
        "AGPL-3.0-only. The unmodified license text is included at "
        "`third_party/ULTRALYTICS-AGPL-3.0.txt`.\n\n"
        f"License SHA-256: `{license_asset['sha256']}`\n"
    )


def publish_assets(review_root: Path, pages_root: Path, *, repo_root: Path) -> None:
    """Copy validated review bytes into exactly the approved Pages locations."""
    review = Path(review_root).resolve()
    repo = Path(repo_root).resolve()
    pages = Path(pages_root).resolve()
    if any(_is_within(review, worktree) for worktree in _git_worktree_roots(repo)):
        raise AssetPreparationError("scope")
    if pages != (repo / "demo" / "web").resolve():
        raise AssetPreparationError("scope")
    receipt = validate_receipts(review)
    for spec in OFFICIAL_ASSETS:
        destination = pages / spec.public_relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((review / spec.public_relative_path).read_bytes())
    pages.mkdir(parents=True, exist_ok=True)
    (pages / "demo-model.json").write_text(
        json.dumps(_manifest(receipt), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (pages / "THIRD_PARTY_NOTICES.md").write_text(_notices(receipt), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run a deliberately terse command-line interface with fixed diagnostics."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if len(arguments) == 2 and arguments[0] == "acquire":
            acquire_assets(Path(arguments[1]))
            return 0
        if len(arguments) == 2 and arguments[0] == "validate":
            validate_receipts(Path(arguments[1]))
            return 0
        if len(arguments) == 5 and arguments[0] == "publish" and arguments[3] == "--repo-root":
            publish_assets(Path(arguments[1]), Path(arguments[2]), repo_root=Path(arguments[4]))
            return 0
        raise AssetPreparationError("scope")
    except AssetPreparationError as error:
        print(f"[FAIL] {error.code}")
        return 1
    except Exception:
        print(f"[FAIL] {ERROR_CODES['network']}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
