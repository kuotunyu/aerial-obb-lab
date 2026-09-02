"""Fail-closed local preparation for the three approved demo assets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Callable
from types import MappingProxyType
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
import urllib.request

from PIL import Image, UnidentifiedImageError

if __name__ == "__main__":
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    sys.modules.setdefault("scripts.prepare_demo_assets", sys.modules[__name__])

if TYPE_CHECKING:
    from scripts.sanitize_demo_model import SanitizationReceipt


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
_NAIP_PRODUCT_ID = re.compile(r"^m_\d{7}_(?:sw|se)_\d{2}_060_(\d{8})$")
_NAIP_YEAR_RANGE = range(2003, 2027)
_MAX_GALLERY_DETECTIONS = 1280 * 800


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
OFFICIAL_SHA256 = MappingProxyType({
    "boats-image": "8c5ada657cf8110a9f8aaac954c1dd96cde0187315b581276c32b0d1863e756f",
    "obb-model": "02f7c539600296d7389341280beb82da810b15dc09c54cf2bc70f7f610331b38",
    "ultralytics-license": "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0",
})
SOURCE_REVIEW_PATHS = {
    "boats-image": "samples/boats.jpg",
    "obb-model": "models/yolo26n-obb.onnx",
    "ultralytics-license": "licenses/ULTRALYTICS-AGPL-3.0.txt",
}
SANITIZED_MODEL_REVIEW_PATH = "sanitized/yolo26n-obb-privacy-sanitized.onnx"
SANITIZATION_RECEIPT_REVIEW_PATH = "sanitized/sanitization-receipt.json"
DERIVATIVE_PUBLIC_PATH = "models/yolo26n-obb-privacy-sanitized.onnx"
SANITIZATION_RECORD_PUBLIC_PATH = (
    "third_party/yolo26n-obb-privacy-sanitization.json"
)
LICENSE_PUBLIC_PATH = "third_party/ULTRALYTICS-AGPL-3.0.txt"
PUBLIC_ASSET_PATHS = (
    "samples/boats.jpg",
    DERIVATIVE_PUBLIC_PATH,
    "demo-model.json",
    LICENSE_PUBLIC_PATH,
    SANITIZATION_RECORD_PUBLIC_PATH,
    "THIRD_PARTY_NOTICES.md",
)
SOURCE_REVIEW_FILES = frozenset({"receipt.json", *SOURCE_REVIEW_PATHS.values()})
ADMITTED_REVIEW_FILES = frozenset(
    {
        *SOURCE_REVIEW_FILES,
        SANITIZED_MODEL_REVIEW_PATH,
        SANITIZATION_RECEIPT_REVIEW_PATH,
    }
)
SOURCE_RELEASE = "v8.4.0"
SANITIZER_VERSION = "924fda756801f906e6cb2ea174978fd4b6c37c2c"


@dataclass(frozen=True)
class AdmittedAssets:
    receipts: dict[str, AssetReceipt]
    sanitization: "SanitizationReceipt"


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
                if str(image.format or "").casefold() != "jpeg":
                    raise AssetPreparationError("media")
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
    digest = hashlib.sha256(body).hexdigest()
    if digest != OFFICIAL_SHA256[spec.asset_id]:
        raise AssetPreparationError("digest")
    return AssetReceipt(spec.asset_id, spec.source_url, redirect_hosts, len(body), digest, media_type, width, height)


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
            candidate = current / name
            try:
                if candidate.stat().st_nlink != 1:
                    raise AssetPreparationError("receipt")
            except OSError:
                raise AssetPreparationError("receipt") from None
            files.add(candidate.relative_to(safe_root).as_posix())
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


def _validate_receipt_contents(review_root: Path) -> dict[str, AssetReceipt]:
    root = _checked_root(review_root)
    receipts = _load_receipts(root)
    for spec in OFFICIAL_ASSETS:
        body = checked_child(root, Path(SOURCE_REVIEW_PATHS[spec.asset_id])).read_bytes()
        stored = receipts.get(spec.asset_id)
        current = _media_receipt(spec, body, {"boats-image": "image/jpeg", "obb-model": "application/octet-stream", "ultralytics-license": "text/plain"}[spec.asset_id], stored.redirect_hosts if stored else ())
        if stored != current:
            if stored and stored.bytes != len(body):
                raise AssetPreparationError("length")
            if stored and stored.sha256 != current.sha256:
                raise AssetPreparationError("digest")
            raise AssetPreparationError("receipt")
    return receipts


def validate_receipts(review_root: Path) -> dict[str, AssetReceipt]:
    root = _checked_root(review_root)
    if _walk_files(root) != SOURCE_REVIEW_FILES:
        raise AssetPreparationError("receipt")
    return _validate_receipt_contents(root)


def validate_source_receipts(review_root: Path) -> dict[str, AssetReceipt]:
    root = _checked_root(review_root)
    files = _walk_files(root)
    if files not in {SOURCE_REVIEW_FILES, ADMITTED_REVIEW_FILES}:
        raise AssetPreparationError("receipt")
    return _validate_receipt_contents(root)


def sanitize_official_model(source: Path, output: Path, receipt: Path) -> "SanitizationReceipt":
    from scripts.sanitize_demo_model import sanitize_official_model as implementation

    return implementation(source, output, receipt)


def validate_sanitized_model(source: Path, output: Path, receipt: Path) -> "SanitizationReceipt":
    from scripts.sanitize_demo_model import validate_sanitized_model as implementation

    return implementation(source, output, receipt)


def validate_admitted_assets(review_root: Path) -> AdmittedAssets:
    _require_external_review(review_root)
    root = _checked_root(review_root)
    if _walk_files(root) != ADMITTED_REVIEW_FILES:
        raise AssetPreparationError("receipt")
    receipts = validate_source_receipts(root)
    sanitization = validate_sanitized_model(
        checked_child(root, Path(SOURCE_REVIEW_PATHS["obb-model"])),
        checked_child(root, Path(SANITIZED_MODEL_REVIEW_PATH)),
        checked_child(root, Path(SANITIZATION_RECEIPT_REVIEW_PATH)),
    )
    model = receipts["obb-model"]
    if (
        sanitization.source_bytes != model.bytes
        or sanitization.source_sha256 != model.sha256
    ):
        raise AssetPreparationError("receipt")
    return AdmittedAssets(receipts=receipts, sanitization=sanitization)


def validate_gallery_publication(pages_root: Path, receipt_path: Path) -> dict[str, object]:
    """Bind the model publisher to the closed, already-published NAIP gallery."""
    try:
        pages = _checked_root(pages_root)
        payload = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
        canonical = json.loads((REPO_ROOT / "release" / "sample-gallery-sources.json").read_text(encoding="utf-8"))
        if payload != canonical:
            raise ValueError
        if not isinstance(payload, dict) or set(payload) != {"schemaVersion", "samples"} or payload["schemaVersion"] != 1:
            raise ValueError
        samples = payload["samples"]
        if not isinstance(samples, list) or [item.get("id") if isinstance(item, dict) else None for item in samples] != ["harbor"]:
            raise ValueError
        expected_paths = ["samples/harbor.jpg"]
        if [item.get("path") if isinstance(item, dict) else None for item in samples] != expected_paths:
            raise ValueError
        sample_keys = {"id", "title", "alt", "path", "bytes", "sha256", "mediaType", "width", "height", "source", "derivation", "guardrails"}
        source_keys = {"service", "productId", "year", "acquisitionDate", "agency", "publicDomainRecord"}
        derivation_keys = {"bboxWgs84", "outputSize", "color", "jpegQuality", "metadata"}
        guardrail_keys = {"classIds", "countMin", "countMax", "representative"}
        representative_keys = {"classId", "cx", "cy", "w", "h", "tolerance"}
        expected_alts = {"harbor": "低密度港區的真實航拍原圖"}
        expected_class_ids = {"harbor": [1, 2, 7]}
        def finite_number(value: object) -> bool:
            return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)

        def integer(value: object) -> bool:
            return isinstance(value, int) and not isinstance(value, bool)

        def safe_exact_https(value: object, expected: str) -> bool:
            if not isinstance(value, str) or value != expected:
                return False
            parsed = urlsplit(value)
            return (
                parsed.scheme == "https" and not parsed.username and not parsed.password
                and not parsed.query and not parsed.fragment
            )

        for item in samples:
            if not isinstance(item, dict) or set(item) != sample_keys:
                raise ValueError
            source = item["source"]
            if (
                not isinstance(source, dict) or set(source) != source_keys
                or not safe_exact_https(source["service"], "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer")
                or not isinstance(source["productId"], str) or not source["productId"]
                or source["agency"] != "USDA"
                or not safe_exact_https(source["publicDomainRecord"], "https://data.usgs.gov/datacatalog/data/USGS%3AEROS5e83a340bf820c39")
                or not isinstance(source["year"], int) or not isinstance(source["acquisitionDate"], int)
                or not isinstance(item["derivation"], dict) or set(item["derivation"]) != derivation_keys
                or item["derivation"].get("outputSize") != [1280, 800] or item["derivation"].get("color") != "sRGB" or item["derivation"].get("jpegQuality") != 90 or item["derivation"].get("metadata") != "stripped"
                or not isinstance(item["derivation"].get("bboxWgs84"), list) or len(item["derivation"]["bboxWgs84"]) != 4
                or not isinstance(item["guardrails"], dict) or set(item["guardrails"]) != guardrail_keys
                or not isinstance(item["guardrails"].get("classIds"), list) or not item["guardrails"]["classIds"]
                or not isinstance(item["guardrails"].get("countMin"), int) or not isinstance(item["guardrails"].get("countMax"), int) or item["guardrails"]["countMin"] < 1 or item["guardrails"]["countMin"] > item["guardrails"]["countMax"]
                or not isinstance(item["guardrails"].get("representative"), dict) or set(item["guardrails"]["representative"]) != representative_keys
                or item["mediaType"] != "image/jpeg" or item["width"] != 1280 or item["height"] != 800
            ):
                raise ValueError
            guardrails = item["guardrails"]
            representative = guardrails["representative"]
            bbox = item["derivation"]["bboxWgs84"]
            product_match = _NAIP_PRODUCT_ID.fullmatch(source["productId"])
            try:
                acquisition_day = datetime.fromtimestamp(
                    source["acquisitionDate"] / 1000, tz=timezone.utc
                ).date()
                product_day = datetime.strptime(product_match.group(1), "%Y%m%d").date() if product_match else None
            except (OverflowError, OSError, TypeError, ValueError):
                raise ValueError from None
            west, south, east, north = bbox
            if (
                not isinstance(item["id"], str) or item["alt"] != expected_alts[item["id"]]
                or not isinstance(item["title"], str) or not item["title"]
                or not integer(item["bytes"]) or item["bytes"] < 1
                or not isinstance(item["sha256"], str) or len(item["sha256"]) != 64
                or not isinstance(source["productId"], str) or product_match is None
                or not integer(source["year"]) or source["year"] not in _NAIP_YEAR_RANGE
                or not integer(source["acquisitionDate"]) or not 10**12 <= source["acquisitionDate"] < 2 * 10**12
                or acquisition_day.year != source["year"] or acquisition_day > datetime.now(timezone.utc).date() or product_day != acquisition_day
                or any(not finite_number(value) for value in bbox)
                or not -180 <= west < east <= 180 or not -90 <= south < north <= 90
                or east - west > 1 or north - south > 1
                or guardrails["classIds"] != expected_class_ids[item["id"]]
                or any(not integer(value) or not 0 <= value < 15 for value in guardrails["classIds"])
                or not integer(guardrails["countMin"]) or not integer(guardrails["countMax"])
                or not 1 <= guardrails["countMin"] <= guardrails["countMax"] <= _MAX_GALLERY_DETECTIONS
                or not integer(representative["classId"]) or representative["classId"] not in guardrails["classIds"]
                or any(not finite_number(representative[key]) for key in ("cx", "cy", "w", "h", "tolerance"))
                or not 0 <= representative["cx"] <= item["width"] or not 0 <= representative["cy"] <= item["height"]
                or not 0 < representative["w"] <= item["width"] or not 0 < representative["h"] <= item["height"]
                or not 0 < representative["tolerance"] <= max(representative["w"], representative["h"])
            ):
                raise ValueError
        public = _walk_files(pages)
        actual_samples = {path for path in public if path.startswith("samples/")}
        if actual_samples != set(expected_paths):
            raise ValueError
        for item in samples:
            if not isinstance(item, dict) or not isinstance(item.get("bytes"), int) or not isinstance(item.get("sha256"), str):
                raise ValueError
            body = checked_child(pages, Path(str(item["path"]))).read_bytes()
            if len(body) != item["bytes"] or hashlib.sha256(body).hexdigest() != item["sha256"]:
                raise ValueError
        return payload
    except (AssetPreparationError, OSError, TypeError, ValueError, json.JSONDecodeError):
        raise AssetPreparationError("receipt") from None


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
        rollback_failed = False
        for relative in reversed(applied):
            destination = checked_child(root, Path(relative))
            backup = backups / relative
            try:
                if originals[relative] and backup.is_file():
                    try:
                        os.replace(backup, destination)
                    except OSError:
                        shutil.copyfile(backup, destination)
                elif destination.exists():
                    destination.unlink()
            except (OSError, AssetPreparationError):
                rollback_failed = True
        if rollback_failed:
            raise AssetPreparationError("scope") from None
        raise AssetPreparationError("scope") from None


def _new_stage(root: Path) -> Path:
    return Path(tempfile.mkdtemp(prefix=".demo-assets-stage-", dir=_checked_root(root, create=True)))


def _reject_unknown_review_members(root: Path) -> None:
    allowed_files = SOURCE_REVIEW_FILES
    allowed_directories = {"samples", "models", "licenses"}
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
    _require_external_review(review_root)
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
            destination = checked_child(stage, Path(SOURCE_REVIEW_PATHS[spec.asset_id]))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(body)
            receipts[spec.asset_id] = receipt
        checked_child(stage, Path("receipt.json")).write_text(json.dumps(_receipt_payload(receipts), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        validate_receipts(stage)
        _replace_batch(root, stage, tuple([*SOURCE_REVIEW_PATHS.values(), "receipt.json"]))
        return receipts
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _git_worktree_roots(repo_root: Path) -> set[Path]:
    roots = {repo_root.resolve()}
    if not (repo_root / ".git").exists():
        return roots
    try:
        command = ["git", "-C", str(repo_root), "worktree", "list", "--porcelain"]
        completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
        porcelain = completed.stdout.decode("utf-8", errors="strict")
    except (OSError, UnicodeDecodeError, subprocess.SubprocessError):
        raise AssetPreparationError("scope") from None
    for line in porcelain.splitlines():
        if line.startswith("worktree "):
            roots.add(Path(line.removeprefix("worktree ")).resolve())
    return roots


def _require_external_review(review_root: Path) -> None:
    requested = Path(os.path.abspath(review_root)).resolve(strict=False)
    repo = _checked_root(REPO_ROOT)
    if any(
        requested == worktree or requested.is_relative_to(worktree)
        for worktree in _git_worktree_roots(repo)
    ):
        raise AssetPreparationError("scope")


def require_browser_parity(review_root: Path) -> None:
    from scripts.model_parity_smoke import run_parity

    try:
        with tempfile.TemporaryDirectory(prefix="aerial-obb-publish-parity-") as directory:
            report = Path(directory) / "report.json"
            run_parity(review_root, report)
            payload = json.loads(report.read_text(encoding="utf-8"))
    except Exception as error:
        if hasattr(error, "code"):
            raise
        raise AssetPreparationError("receipt") from None
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "runtime",
            "input",
            "output",
            "output_bytes_equal",
            "detections_equal",
            "accepted_ship",
            "verdict",
        }
        or payload.get("output_bytes_equal") is not True
        or payload.get("detections_equal") is not True
        or payload.get("accepted_ship") is not True
        or payload.get("verdict") != "PASS"
    ):
        raise AssetPreparationError("receipt")


def _manifest(admitted: AdmittedAssets) -> dict[str, object]:
    receipts, sanitization = admitted.receipts, admitted.sanitization
    image, model, license_asset = receipts["boats-image"], receipts["obb-model"], receipts["ultralytics-license"]
    return {
        "schemaVersion": 1, "id": "ultralytics-yolo26n-obb-demo",
        "image": {"path": "samples/boats.jpg", "source": OFFICIAL_ASSETS[0].source_url, "mediaType": "image/jpeg", "bytes": image.bytes, "sha256": image.sha256, "width": image.width, "height": image.height},
        "model": {"path": DERIVATIVE_PUBLIC_PATH, "mediaType": "application/onnx", "source": OFFICIAL_ASSETS[1].source_url, "release": SOURCE_RELEASE, "license": "AGPL-3.0-only", "modificationStatus": "metadata-only", "bytes": sanitization.output_bytes, "sha256": sanitization.output_sha256, "sourceSha256": model.sha256},
        "sanitization": {"path": SANITIZATION_RECORD_PUBLIC_PATH, "modifiedField": sanitization.modified_field, "modificationDate": sanitization.modification_date, "removedMetadataEntries": sanitization.removed_metadata_entries},
        "license": {"path": LICENSE_PUBLIC_PATH, "bytes": license_asset.bytes, "sha256": license_asset.sha256},
        "input": {"name": "images", "dims": [1, 3, 1024, 1024], "type": "float32", "channelOrder": "RGB", "normalization": "divide-by-255", "letterboxValue": 114},
        "output": {"name": "output0", "dims": [1, "N", 7], "type": "float32", "rowWidth": 7, "layout": ["cx", "cy", "w", "h", "confidence", "class", "angleRadians"]},
        "classes": ["plane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court", "ground track field", "harbor", "bridge", "large vehicle", "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool"],
        "defaultConfidence": 0.25, "notice": "THIRD_PARTY_NOTICES.md",
        "provenance": {"upstream": "Ultralytics YOLO26n-OBB", "trainingDataset": "DOTAv1", "status": "privacy-sanitized AGPL derivative"},
    }


def _sanitization_record(admitted: AdmittedAssets) -> dict[str, object]:
    receipts, sanitization = admitted.receipts, admitted.sanitization
    return {
        "schemaVersion": 1,
        "source": {"url": OFFICIAL_ASSETS[1].source_url, "release": SOURCE_RELEASE, "bytes": sanitization.source_bytes, "sha256": sanitization.source_sha256},
        "derivative": {"path": DERIVATIVE_PUBLIC_PATH, "bytes": sanitization.output_bytes, "sha256": sanitization.output_sha256},
        "sanitizer": {"path": "scripts/sanitize_demo_model.py", "version": SANITIZER_VERSION, "onnxVersion": sanitization.onnx_version, "protobufVersion": sanitization.protobuf_version},
        "transformation": {"modificationStatus": "metadata-only", "removedMetadataEntries": sanitization.removed_metadata_entries, "modifiedField": sanitization.modified_field, "modificationDate": sanitization.modification_date},
        "verification": {"structuralEquivalent": sanitization.structural_equivalent, "checkerPassed": sanitization.checker_passed, "privacyPassed": sanitization.privacy_passed, "deterministic": sanitization.deterministic, "browserParityPassed": True},
        "license": {"spdx": "AGPL-3.0-only", "path": LICENSE_PUBLIC_PATH, "sha256": receipts["ultralytics-license"].sha256},
        "provenance": {"upstream": "Ultralytics YOLO26n-OBB", "trainingDataset": "DOTAv1", "endorsement": False, "commercialUseCleared": False},
    }


def _notices(admitted: AdmittedAssets) -> str:
    receipts = admitted.receipts
    return (
        "# Third-party notices\n\n"
        "The bundled model is a privacy-sanitized AGPL derivative of Ultralytics "
        "YOLO26n-OBB from release v8.4.0. One non-inference metadata entry was "
        "removed on 2026-08-31; the graph and weights were verified unchanged. "
        "The model was trained on DOTAv1.\n\n"
        "The complete, unmodified AGPL-3.0-only license text is included at "
        "`third_party/ULTRALYTICS-AGPL-3.0.txt`. The transformation record is "
        "`third_party/yolo26n-obb-privacy-sanitization.json`, and the sanitizer "
        "source is `scripts/sanitize_demo_model.py`.\n\n"
        f"Upstream release: {OFFICIAL_ASSETS[1].source_url}\n\n"
        "Sanitizer source: https://github.com/kuotunyu/aerial-obb-lab/blob/"
        f"{SANITIZER_VERSION}/scripts/sanitize_demo_model.py\n\n"
        "Corresponding repository source: https://github.com/kuotunyu/aerial-obb-lab\n\n"
        "This project is not endorsed by Ultralytics and makes no commercial-use "
        "clearance claim.\n\n"
        f"License SHA-256: `{receipts['ultralytics-license'].sha256}`\n"
    )


def _validate_public_stage(stage: Path, admitted: AdmittedAssets) -> None:
    try:
        if _walk_files(stage) != set(PUBLIC_ASSET_PATHS):
            raise AssetPreparationError("receipt")
        receipts, sanitization = admitted.receipts, admitted.sanitization
        expected_binary = {
            "samples/boats.jpg": (
                receipts["boats-image"].bytes,
                receipts["boats-image"].sha256,
            ),
            DERIVATIVE_PUBLIC_PATH: (
                sanitization.output_bytes,
                sanitization.output_sha256,
            ),
            LICENSE_PUBLIC_PATH: (
                receipts["ultralytics-license"].bytes,
                receipts["ultralytics-license"].sha256,
            ),
        }
        for relative, (expected_bytes, expected_digest) in expected_binary.items():
            body = checked_child(stage, Path(relative)).read_bytes()
            if len(body) != expected_bytes or hashlib.sha256(body).hexdigest() != expected_digest:
                raise AssetPreparationError("receipt")
        expected_text = {
            "demo-model.json": json.dumps(_manifest(admitted), sort_keys=True, indent=2) + "\n",
            SANITIZATION_RECORD_PUBLIC_PATH: json.dumps(
                _sanitization_record(admitted), sort_keys=True, indent=2
            )
            + "\n",
            "THIRD_PARTY_NOTICES.md": _notices(admitted),
        }
        for relative, expected in expected_text.items():
            if checked_child(stage, Path(relative)).read_text(encoding="utf-8") != expected:
                raise AssetPreparationError("receipt")
    except AssetPreparationError:
        raise
    except (OSError, UnicodeError, ValueError):
        raise AssetPreparationError("receipt") from None


def _reject_stale_managed_pages(pages: Path, *, gallery: bool = False) -> None:
    approved = {
        DERIVATIVE_PUBLIC_PATH,
        LICENSE_PUBLIC_PATH,
        SANITIZATION_RECORD_PUBLIC_PATH,
    }
    approved.update(
        {"samples/harbor.jpg"}
        if gallery else {"samples/boats.jpg"}
    )
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
    if requested_pages != expected_pages:
        raise AssetPreparationError("scope")
    _require_external_review(review)
    gallery_receipt = repo / "release" / "sample-gallery-sources.json"
    # Test fixtures isolate a synthetic repository; every real entry point is
    # anchored to this repository and therefore has no legacy fallback.
    if not gallery_receipt.is_file() and repo == Path(__file__).resolve().parents[1]:
        raise AssetPreparationError("receipt")
    pages = _checked_root(requested_pages, create=True)
    use_gallery = gallery_receipt.is_file()
    _reject_stale_managed_pages(pages, gallery=use_gallery)
    if use_gallery:
        validate_gallery_publication(pages, gallery_receipt)
    admitted = validate_admitted_assets(review)
    require_browser_parity(review)
    stage = _new_stage(pages)
    targets = (
        (DERIVATIVE_PUBLIC_PATH, LICENSE_PUBLIC_PATH, SANITIZATION_RECORD_PUBLIC_PATH, "THIRD_PARTY_NOTICES.md")
        if use_gallery else PUBLIC_ASSET_PATHS
    )
    try:
        copies = (() if use_gallery else ((SOURCE_REVIEW_PATHS["boats-image"], "samples/boats.jpg"),)) + (
            (SANITIZED_MODEL_REVIEW_PATH, DERIVATIVE_PUBLIC_PATH),
            (SOURCE_REVIEW_PATHS["ultralytics-license"], LICENSE_PUBLIC_PATH),
        )
        for source_relative, public_relative in copies:
            destination = checked_child(stage, Path(public_relative))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(checked_child(review, Path(source_relative)).read_bytes())
        record = checked_child(stage, Path(SANITIZATION_RECORD_PUBLIC_PATH))
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(_sanitization_record(admitted), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        if not use_gallery:
            checked_child(stage, Path("demo-model.json")).write_text(json.dumps(_manifest(admitted), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        checked_child(stage, Path("THIRD_PARTY_NOTICES.md")).write_text(_notices(admitted), encoding="utf-8")
        if not use_gallery:
            _validate_public_stage(stage, admitted)
        else:
            for relative in targets:
                if not checked_child(stage, Path(relative)).is_file():
                    raise AssetPreparationError("receipt")
        _replace_batch(pages, stage, targets)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if len(arguments) == 3 and arguments[0] in {"acquire", "sanitize", "verify"} and arguments[1] == "--review-root":
            if arguments[0] == "acquire":
                # The boats source is retired; sample publication is owned by the
                # closed gallery receipt and is never reacquired here.
                raise AssetPreparationError("receipt")
            elif arguments[0] == "sanitize":
                _require_external_review(Path(arguments[2]))
                review = _checked_root(Path(arguments[2]))
                validate_receipts(review)
                sanitized = checked_child(review, Path("sanitized"))
                sanitized.mkdir()
                sanitize_official_model(
                    checked_child(review, Path(SOURCE_REVIEW_PATHS["obb-model"])),
                    checked_child(review, Path(SANITIZED_MODEL_REVIEW_PATH)),
                    checked_child(review, Path(SANITIZATION_RECEIPT_REVIEW_PATH)),
                )
                validate_admitted_assets(review)
                print("[OK] DEMO_MODEL_SANITIZED")
            else:
                validate_admitted_assets(Path(arguments[2]))
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
    except Exception as error:
        print(f"[FAIL] {getattr(error, 'code', ERROR_CODES['network'])}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
