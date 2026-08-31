"""Fail-closed local preparation for the three approved demo assets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Callable
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
CHUNK_SIZE = 65_536
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    source_url: str
    expected_bytes: int
    allowed_redirect_hosts: tuple[str, ...]
    public_relative_path: str


@dataclass(frozen=True)
class AssetReceipt:
    asset_id: str
    source_url: str
    redirect_hosts: tuple[str, ...]
    bytes: int
    sha256: str
    media_type: str
    width: int | None
    height: int | None


OFFICIAL_ASSETS = (
    AssetSpec("boats-image", "https://ultralytics.com/images/boats.jpg", 194_872,
              ("ultralytics.com", "www.ultralytics.com", "github.com", "release-assets.githubusercontent.com"), "samples/boats.jpg"),
    AssetSpec("obb-model", "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.onnx", 10_207_250,
              ("github.com", "release-assets.githubusercontent.com"), "models/yolo26n-obb.onnx"),
    AssetSpec("ultralytics-license", "https://raw.githubusercontent.com/ultralytics/assets/v8.4.0/LICENSE", 34_523,
              ("raw.githubusercontent.com",), "third_party/ULTRALYTICS-AGPL-3.0.txt"),
)


class AssetPreparationError(Exception):
    def __init__(self, category: str) -> None:
        self.code = ERROR_CODES[category]
        super().__init__(self.code)


def _host(value: str) -> str:
    try:
        return (urlsplit(value).hostname or "").casefold()
    except ValueError:
        return ""


def is_reparse_point(path: Path) -> bool:
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, OSError):
        return path.is_symlink()
    return path.is_symlink() or bool(attributes & 0x400)


def _checked_root(root: Path, *, create: bool = False) -> Path:
    raw = Path(os.path.abspath(root))
    components = [*reversed(raw.parents), raw]
    for component in components:
        if (component.exists() or component.is_symlink()) and is_reparse_point(component):
            raise AssetPreparationError("scope")
    if create:
        raw.mkdir(parents=True, exist_ok=True)
    elif not raw.is_dir():
        raise AssetPreparationError("scope")
    resolved = raw.resolve(strict=False)
    if resolved != raw.resolve(strict=False):
        raise AssetPreparationError("scope")
    return resolved


def checked_child(root: Path, relative: Path) -> Path:
    """Return a non-reparse leaf proven to remain beneath the permitted root."""
    safe_root = _checked_root(root)
    child = Path(relative)
    if child.is_absolute() or not child.parts or any(part in {"", ".", ".."} for part in child.parts):
        raise AssetPreparationError("scope")
    candidate = safe_root.joinpath(*child.parts)
    current = safe_root
    for part in child.parts:
        current = current / part
        if (current.exists() or current.is_symlink()) and is_reparse_point(current):
            raise AssetPreparationError("scope")
    try:
        candidate.resolve(strict=False).relative_to(safe_root)
    except ValueError as error:
        raise AssetPreparationError("scope") from error
    return candidate


def _content_type(headers: object) -> str:
    items = headers.items() if hasattr(headers, "items") else ()
    for name, value in items:
        if str(name).casefold() == "content-type":
            return str(value).split(";", 1)[0].strip().casefold()
    return ""


def _stream_cap(spec: AssetSpec) -> int:
    return min(spec.expected_bytes, MODEL_HARD_CEILING) if spec.asset_id == "obb-model" else spec.expected_bytes


def _read_capped(stream: object, cap: int) -> bytes:
    parts: list[bytes] = []
    total = 0
    while True:
        remaining = cap + 1 - total
        if remaining <= 0:
            raise AssetPreparationError("length")
        chunk = stream.read(min(CHUNK_SIZE, remaining))  # type: ignore[union-attr]
        if not chunk:
            return b"".join(parts)
        total += len(chunk)
        if total > cap:
            raise AssetPreparationError("length")
        parts.append(chunk)


class _Redirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, spec: AssetSpec) -> None:
        super().__init__()
        self.spec = spec
        self.hosts: list[str] = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        host = _host(newurl)
        if not host or host not in self.spec.allowed_redirect_hosts:
            raise AssetPreparationError("redirect")
        self.hosts.append(host)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def urlopen_transport(spec: AssetSpec) -> tuple[bytes, tuple[str, ...], str]:
    """Production-only HTTP transport: bounded, allowlisted and privacy-safe."""
    redirects = _Redirects(spec)
    opener = urllib.request.build_opener(redirects)
    try:
        with opener.open(spec.source_url, timeout=30) as response:
            body = _read_capped(response, _stream_cap(spec))
            if response.getcode() != 200:
                raise AssetPreparationError("status")
            final_host = _host(response.geturl())
            if not final_host or final_host not in spec.allowed_redirect_hosts:
                raise AssetPreparationError("redirect")
            return body, tuple(redirects.hosts), _content_type(response.headers)
    except HTTPError as error:
        _read_capped(error, _stream_cap(spec))
        raise AssetPreparationError("status") from None
    except AssetPreparationError:
        raise
    except (OSError, URLError):
        raise AssetPreparationError("network") from None


def _media_receipt(spec: AssetSpec, body: bytes, content_type: str, redirect_hosts: tuple[str, ...]) -> AssetReceipt:
    if len(body) != spec.expected_bytes or (spec.asset_id == "obb-model" and len(body) > MODEL_HARD_CEILING):
        raise AssetPreparationError("length")
    if not all(host in spec.allowed_redirect_hosts for host in redirect_hosts):
        raise AssetPreparationError("redirect")
    content_type = content_type.split(";", 1)[0].strip().casefold()
    media_type: str
    width: int | None = None
    height: int | None = None
    if spec.asset_id == "boats-image":
        if content_type not in {"image/jpeg", "application/octet-stream"}:
            raise AssetPreparationError("media")
        try:
            with Image.open(io.BytesIO(body)) as image:
                image.load()
                width, height = image.width, image.height
        except (UnidentifiedImageError, OSError, ValueError):
            raise AssetPreparationError("media") from None
        if not width or not height:
            raise AssetPreparationError("media")
        media_type = "image/jpeg"
    elif spec.asset_id == "obb-model":
        if content_type != "application/octet-stream":
            raise AssetPreparationError("media")
        media_type = "application/onnx"
    else:
        if content_type != "text/plain":
            raise AssetPreparationError("media")
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            raise AssetPreparationError("media") from None
        if "GNU AFFERO GENERAL PUBLIC LICENSE" not in text or "Version 3" not in text:
            raise AssetPreparationError("media")
        media_type = "text/plain"
    return AssetReceipt(spec.asset_id, spec.source_url, redirect_hosts, len(body), hashlib.sha256(body).hexdigest(), media_type, width, height)


def _receipt_payload(receipts: dict[str, AssetReceipt]) -> dict[str, object]:
    return {"assets": {asset_id: asdict(receipt) for asset_id, receipt in receipts.items()}, "schemaVersion": 1}


def _walk_files(root: Path) -> set[str]:
    safe_root = _checked_root(root)
    files: set[str] = set()
    for directory, names, file_names in os.walk(safe_root, followlinks=False):
        current = Path(directory)
        for name in [*names, *file_names]:
            candidate = current / name
            if is_reparse_point(candidate):
                raise AssetPreparationError("scope")
        for name in file_names:
            files.add((current / name).relative_to(safe_root).as_posix())
    return files


def _load_receipts(review_root: Path) -> dict[str, AssetReceipt]:
    receipt_path = checked_child(review_root, Path("receipt.json"))
    try:
        raw = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != {"assets", "schemaVersion"} or raw["schemaVersion"] != 1:
            raise ValueError
        assets = raw["assets"]
        if not isinstance(assets, dict) or set(assets) != {spec.asset_id for spec in OFFICIAL_ASSETS}:
            raise ValueError
        receipt_keys = {"asset_id", "source_url", "redirect_hosts", "bytes", "sha256", "media_type", "width", "height"}
        if not all(isinstance(value, dict) and set(value) == receipt_keys for value in assets.values()):
            raise ValueError
        receipts = {
            asset_id: AssetReceipt(
                asset_id=value["asset_id"], source_url=value["source_url"], redirect_hosts=tuple(value["redirect_hosts"]),
                bytes=value["bytes"], sha256=value["sha256"], media_type=value["media_type"], width=value["width"], height=value["height"],
            )
            for asset_id, value in assets.items()
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        raise AssetPreparationError("receipt") from None
    return receipts


def validate_receipts(review_root: Path) -> dict[str, AssetReceipt]:
    root = _checked_root(review_root)
    expected = {"receipt.json", *(spec.public_relative_path for spec in OFFICIAL_ASSETS)}
    if _walk_files(root) != expected:
        raise AssetPreparationError("receipt")
    receipts = _load_receipts(root)
    for spec in OFFICIAL_ASSETS:
        body = checked_child(root, Path(spec.public_relative_path)).read_bytes()
        stored = receipts.get(spec.asset_id)
        current = _media_receipt(spec, body, {"boats-image": "image/jpeg", "obb-model": "application/octet-stream", "ultralytics-license": "text/plain"}[spec.asset_id], stored.redirect_hosts if stored else ())
        if stored != current:
            if stored and stored.bytes != len(body):
                raise AssetPreparationError("length")
            if stored and stored.sha256 != current.sha256:
                raise AssetPreparationError("digest")
            raise AssetPreparationError("receipt")
    return receipts


def _replace_batch(root: Path, stage: Path, paths: tuple[str, ...]) -> None:
    backups = stage / "backups"
    originals: dict[str, bool] = {}
    for relative in paths:
        destination = checked_child(root, Path(relative))
        source = checked_child(stage, Path(relative))
        if not source.is_file():
            raise AssetPreparationError("scope")
        destination.parent.mkdir(parents=True, exist_ok=True)
        checked_child(root, Path(relative))
        originals[relative] = destination.is_file()
        if originals[relative]:
            backup = checked_child(backups, Path(relative)) if backups.exists() else backups / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(destination, backup)
    applied: list[str] = []
    try:
        for relative in paths:
            os.replace(checked_child(stage, Path(relative)), checked_child(root, Path(relative)))
            applied.append(relative)
    except (AssetPreparationError, OSError):
        for relative in reversed(applied):
            destination = checked_child(root, Path(relative))
            backup = backups / relative
            if originals[relative] and backup.is_file():
                os.replace(backup, destination)
            elif destination.exists():
                destination.unlink()
        raise AssetPreparationError("scope") from None


def _new_stage(root: Path) -> Path:
    return Path(tempfile.mkdtemp(prefix=".demo-assets-stage-", dir=_checked_root(root, create=True)))


def _reject_unknown_review_members(root: Path) -> None:
    allowed_files = {"receipt.json", *(spec.public_relative_path for spec in OFFICIAL_ASSETS)}
    allowed_directories = {"samples", "models", "third_party"}
    safe_root = _checked_root(root)
    for directory, names, file_names in os.walk(safe_root, followlinks=False):
        current = Path(directory)
        for name in names:
            candidate = current / name
            relative = candidate.relative_to(safe_root).as_posix()
            if is_reparse_point(candidate) or relative not in allowed_directories:
                raise AssetPreparationError("receipt")
        for name in file_names:
            candidate = current / name
            relative = candidate.relative_to(safe_root).as_posix()
            if is_reparse_point(candidate) or relative not in allowed_files:
                raise AssetPreparationError("receipt")


def acquire_assets(review_root: Path, transport: Callable[[AssetSpec], tuple[bytes, tuple[str, ...], str]] = urlopen_transport) -> dict[str, AssetReceipt]:
    root = _checked_root(review_root, create=True)
    _reject_unknown_review_members(root)
    stage = _new_stage(root)
    try:
        receipts: dict[str, AssetReceipt] = {}
        for spec in OFFICIAL_ASSETS:
            try:
                body, redirects, content_type = transport(spec)
            except AssetPreparationError:
                raise
            except Exception:
                raise AssetPreparationError("network") from None
            receipt = _media_receipt(spec, body, content_type, tuple(redirects))
            destination = checked_child(stage, Path(spec.public_relative_path))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(body)
            receipts[spec.asset_id] = receipt
        checked_child(stage, Path("receipt.json")).write_text(json.dumps(_receipt_payload(receipts), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        validate_receipts(stage)
        _replace_batch(root, stage, tuple([*(spec.public_relative_path for spec in OFFICIAL_ASSETS), "receipt.json"]))
        return receipts
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _git_worktree_roots(repo_root: Path) -> set[Path]:
    roots = {repo_root.resolve()}
    try:
        output = subprocess.run(["git", "-C", str(repo_root), "worktree", "list", "--porcelain"], check=False, capture_output=True, text=True).stdout
    except OSError:
        return roots
    for line in output.splitlines():
        if line.startswith("worktree "):
            roots.add(Path(line.removeprefix("worktree ")).resolve())
    return roots


def _manifest(receipts: dict[str, AssetReceipt]) -> dict[str, object]:
    image, model, license_asset = receipts["boats-image"], receipts["obb-model"], receipts["ultralytics-license"]
    return {
        "schemaVersion": 1, "id": "ultralytics-yolo26n-obb-demo",
        "image": {"path": "samples/boats.jpg", "mediaType": "image/jpeg", "bytes": image.bytes, "sha256": image.sha256, "width": image.width, "height": image.height},
        "model": {"path": "models/yolo26n-obb.onnx", "source": OFFICIAL_ASSETS[1].source_url, "release": "v8.4.0", "license": "AGPL-3.0-only", "bytes": model.bytes, "sha256": model.sha256},
        "license": {"path": "third_party/ULTRALYTICS-AGPL-3.0.txt", "bytes": license_asset.bytes, "sha256": license_asset.sha256},
        "input": {"name": "images", "dims": [1, 3, 1024, 1024], "type": "float32", "channelOrder": "RGB", "normalization": "divide-by-255", "letterboxValue": 114},
        "output": {"name": "output0", "rowWidth": 7, "layout": ["cx", "cy", "w", "h", "confidence", "class", "angleRadians"]},
        "classes": ["plane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court", "ground track field", "harbor", "bridge", "large vehicle", "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool"],
        "defaultConfidence": 0.25, "notice": "THIRD_PARTY_NOTICES.md",
    }


def _notices(receipts: dict[str, AssetReceipt]) -> str:
    return "# Third-party notices\n\nThe bundled Ultralytics assets are subject to AGPL-3.0-only. The unmodified license text is included at `third_party/ULTRALYTICS-AGPL-3.0.txt`.\n\nLicense SHA-256: `" + receipts["ultralytics-license"].sha256 + "`\n"


def _reject_stale_managed_pages(pages: Path) -> None:
    approved = {spec.public_relative_path for spec in OFFICIAL_ASSETS}
    for directory in ("samples", "models", "third_party"):
        checked_child(pages, Path(directory))
    for relative in _walk_files(pages):
        if relative.split("/", 1)[0] in {"samples", "models", "third_party"} and relative not in approved:
            raise AssetPreparationError("scope")


def publish_assets(review_root: Path, pages_root: Path) -> None:
    review = _checked_root(review_root)
    repo = _checked_root(REPO_ROOT)
    expected_pages = (repo / "demo" / "web").resolve(strict=False)
    requested_pages = Path(os.path.abspath(pages_root)).resolve(strict=False)
    if requested_pages != expected_pages or any(review == worktree or review.is_relative_to(worktree) for worktree in _git_worktree_roots(repo)):
        raise AssetPreparationError("scope")
    pages = _checked_root(requested_pages, create=True)
    _reject_stale_managed_pages(pages)
    receipts = validate_receipts(review)
    stage = _new_stage(pages)
    targets = tuple([*(spec.public_relative_path for spec in OFFICIAL_ASSETS), "demo-model.json", "THIRD_PARTY_NOTICES.md"])
    try:
        for spec in OFFICIAL_ASSETS:
            destination = checked_child(stage, Path(spec.public_relative_path))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(checked_child(review, Path(spec.public_relative_path)).read_bytes())
        checked_child(stage, Path("demo-model.json")).write_text(json.dumps(_manifest(receipts), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        checked_child(stage, Path("THIRD_PARTY_NOTICES.md")).write_text(_notices(receipts), encoding="utf-8")
        _replace_batch(pages, stage, targets)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if len(arguments) == 3 and arguments[0] in {"acquire", "verify"} and arguments[1] == "--review-root":
            if arguments[0] == "acquire":
                acquire_assets(Path(arguments[2]))
                print("[OK] DEMO_ASSETS_ACQUIRED")
            else:
                validate_receipts(Path(arguments[2]))
                print("[OK] DEMO_ASSETS_VERIFIED")
            return 0
        if len(arguments) == 5 and arguments[0] == "publish" and arguments[1] == "--review-root" and arguments[3] == "--pages-root":
            publish_assets(Path(arguments[2]), Path(arguments[4]))
            print("[OK] DEMO_ASSETS_PUBLISHED")
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
