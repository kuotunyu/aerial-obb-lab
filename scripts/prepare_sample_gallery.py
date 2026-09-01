"""Acquire and admit reviewed USGS NAIP candidates outside a Git worktree."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
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
from typing import Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
import urllib.request

from PIL import Image, UnidentifiedImageError


NAIP_SERVICE = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer"
NAIP_PUBLIC_DOMAIN_RECORD = "https://data.usgs.gov/datacatalog/data/USGS%3AEROS5e83a340bf820c39"
OUTPUT_SIZE = (1280, 800)
JPEG_QUALITY = 90
DEFAULT_CONFIDENCE = 0.25
SOURCE_FIELDS = (
    "OBJECTID", "Name", "Year", "raster_name", "download_url", "acquisition_date",
    "agency", "resolution_value", "resolution_units", "band_count", "sensor_type",
)
SERVICE_JSON_CAP = 256 * 1024
RASTER_CAP = 25 * 1024 * 1024
_AGENCY_IDENTITY = re.compile(r"(?<![a-z0-9])(?:usda|fsa)(?![a-z0-9])")
_NAIP_YEAR_RANGE = range(2003, 2027)
CHUNK_SIZE = 65_536
REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DOWNLOAD_HOST = "earthexplorer.usgs.gov"


@dataclass(frozen=True)
class CandidateRecipe:
    candidate_id: str
    category: Literal["airfield", "sports-complex", "harbor"]
    bbox_wgs84: tuple[float, float, float, float]


CANDIDATE_RECIPES = (
    CandidateRecipe("airfield-watsonville", "airfield", (-121.797682, 36.929906, -121.785682, 36.937406)),
    CandidateRecipe("airfield-reid-hillview", "airfield", (-121.825300, 37.330133, -121.813300, 37.337633)),
    CandidateRecipe("airfield-santa-monica", "airfield", (-118.456705, 34.012771, -118.444705, 34.020271)),
    CandidateRecipe("sports-big-league-manteca", "sports-complex", (-121.265210, 37.784685, -121.253210, 37.792185)),
    CandidateRecipe("sports-twin-creeks", "sports-complex", (-122.006126, 37.411869, -121.994126, 37.419369)),
    CandidateRecipe("sports-ken-mercer", "sports-complex", (-121.897930, 37.677402, -121.885930, 37.684902)),
    CandidateRecipe("harbor-port-hueneme", "harbor", (-119.216719, 34.144170, -119.200719, 34.154170)),
    CandidateRecipe("harbor-redwood-city", "harbor", (-122.216577, 37.508270, -122.200577, 37.518270)),
    CandidateRecipe("harbor-stockton", "harbor", (-121.334614, 37.946035, -121.318614, 37.956035)),
)
RECIPE_BY_ID = {recipe.candidate_id: recipe for recipe in CANDIDATE_RECIPES}
Transport = Callable[[dict[str, object]], tuple[bytes, str]]


class GalleryError(Exception):
    """Public-safe failure category; never carries transport or path details."""

    def __init__(self, code: str = "GALLERY_RECORD") -> None:
        self.code = code
        super().__init__(code)


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").casefold()
    except ValueError:
        return ""


def _safe_https(url: str, host: str, path_prefix: str | None = None) -> bool:
    try:
        split = urlsplit(url)
    except ValueError:
        return False
    return (
        split.scheme == "https"
        and split.hostname is not None
        and split.hostname.casefold() == host
        and not split.query
        and not split.fragment
        and (path_prefix is None or split.path.startswith(path_prefix))
    )


def _safe_service_response(url: str, endpoint: str) -> bool:
    """The server normally echoes our query string in its final response URL."""
    try:
        split = urlsplit(url)
    except ValueError:
        return False
    return (
        split.scheme == "https"
        and (split.hostname or "").casefold() == "imagery.nationalmap.gov"
        and split.path == f"/arcgis/rest/services/USGSNAIPPlus/ImageServer/{endpoint}"
        and not split.fragment
    )


def _normalized_official_download_url(value: object) -> str:
    if not isinstance(value, str):
        raise GalleryError("GALLERY_RECORD")
    try:
        split = urlsplit(value)
    except ValueError:
        raise GalleryError("GALLERY_RECORD") from None
    host = (split.hostname or "").casefold()
    raw_segments = split.path.split("/")
    if (
        split.scheme != "https"
        or host != OFFICIAL_DOWNLOAD_HOST
        or split.username is not None
        or split.password is not None
        or split.port not in {None, 443}
        or split.query
        or split.fragment
        or not split.path.startswith("/")
        or any(segment in {"", ".", ".."} for segment in raw_segments[1:])
        or not any(segment.casefold() == "naip" for segment in raw_segments[1:])
    ):
        raise GalleryError("GALLERY_RECORD")
    return urlunsplit(("https", OFFICIAL_DOWNLOAD_HOST, split.path, "", ""))


def is_reparse_point(path: Path) -> bool:
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, OSError):
        return path.is_symlink()
    return path.is_symlink() or bool(attributes & 0x400)


def _checked_root(root: Path, *, create: bool = False) -> Path:
    raw = Path(os.path.abspath(root))
    for component in [*reversed(raw.parents), raw]:
        if (component.exists() or component.is_symlink()) and is_reparse_point(component):
            raise GalleryError("GALLERY_SCOPE")
    if create:
        raw.mkdir(parents=True, exist_ok=True)
    if not raw.is_dir():
        raise GalleryError("GALLERY_SCOPE")
    return raw.resolve(strict=False)


def _checked_child(root: Path, name: str) -> Path:
    if not name or Path(name).is_absolute() or "/" in name or "\\" in name or name in {".", ".."}:
        raise GalleryError("GALLERY_SCOPE")
    safe_root = _checked_root(root)
    child = safe_root / name
    if child.exists() or child.is_symlink():
        if is_reparse_point(child):
            raise GalleryError("GALLERY_SCOPE")
    try:
        child.resolve(strict=False).relative_to(safe_root)
    except ValueError:
        raise GalleryError("GALLERY_SCOPE") from None
    return child


def checked_descendant(root: Path, path: Path) -> Path:
    """Return a reparse-free existing-or-new descendant of an exact review root."""
    safe_root = _checked_root(root)
    requested = Path(path)
    if ".." in requested.parts:
        raise GalleryError("GALLERY_SCOPE")
    child = Path(os.path.abspath(requested))
    try:
        relative = child.relative_to(safe_root)
    except ValueError:
        raise GalleryError("GALLERY_SCOPE") from None
    if not relative.parts:
        raise GalleryError("GALLERY_SCOPE")
    current = safe_root
    for component in relative.parts:
        current = current / component
        if (current.exists() or current.is_symlink()) and is_reparse_point(current):
            raise GalleryError("GALLERY_SCOPE")
    return current


def _git_worktree_roots(repo_root: Path) -> set[Path]:
    try:
        response = subprocess.run(
            ["git", "worktree", "list", "--porcelain"], cwd=repo_root,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False,
        )
    except (OSError, subprocess.CalledProcessError):
        return {repo_root.resolve(strict=False)}
    roots = {repo_root.resolve(strict=False)}
    for line in response.stdout.decode("utf-8", errors="strict").splitlines():
        if line.startswith("worktree "):
            roots.add(Path(line.removeprefix("worktree ")).resolve(strict=False))
    return roots


def _require_external_new_review_root(root: Path) -> Path:
    requested = Path(os.path.abspath(root))
    if requested.exists() or requested.is_symlink():
        raise GalleryError("GALLERY_SCOPE")
    resolved_parent = _checked_root(requested.parent)
    review = resolved_parent / requested.name
    for worktree in _git_worktree_roots(REPO_ROOT):
        if review == worktree or review.is_relative_to(worktree):
            raise GalleryError("GALLERY_SCOPE")
    review.mkdir()
    return _checked_root(review)


def _read_capped(stream: object, cap: int) -> bytes:
    parts: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(min(CHUNK_SIZE, cap + 1 - total))  # type: ignore[union-attr]
        if not chunk:
            return b"".join(parts)
        total += len(chunk)
        if total > cap:
            raise GalleryError("GALLERY_NETWORK")
        parts.append(chunk)


def urlopen_transport(request: dict[str, object]) -> tuple[bytes, str]:
    """Bounded production transport; URLs only live in process memory."""
    url = request.get("url")
    kind = request.get("kind")
    if not isinstance(url, str) or kind not in {"query", "identify", "export"}:
        raise GalleryError("GALLERY_NETWORK")
    cap = SERVICE_JSON_CAP if kind in {"query", "identify"} else RASTER_CAP
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            body = _read_capped(response, cap)
            if response.getcode() != 200:
                raise GalleryError("GALLERY_NETWORK")
            return body, str(response.geturl())
    except HTTPError as error:
        _read_capped(error, cap)
        raise GalleryError("GALLERY_NETWORK") from None
    except GalleryError:
        raise
    except (OSError, URLError):
        raise GalleryError("GALLERY_NETWORK") from None


def _inset_points(bbox: tuple[float, float, float, float]) -> tuple[tuple[float, float], ...]:
    west, south, east, north = bbox
    dx, dy = (east - west) * 0.001, (north - south) * 0.001
    return (
        ((west + east) / 2, (south + north) / 2),
        (west + dx, south + dy), (west + dx, north - dy),
        (east - dx, south + dy), (east - dx, north - dy),
    )


def _query_request(recipe: CandidateRecipe, point: tuple[float, float]) -> dict[str, object]:
    params = {
        "f": "json", "where": "1=1", "returnGeometry": "false", "outFields": ",".join(SOURCE_FIELDS),
        "geometryType": "esriGeometryPoint", "geometry": f"{point[0]:.8f},{point[1]:.8f}",
        "inSR": "4326", "spatialRel": "esriSpatialRelIntersects",
    }
    return {"kind": "query", "url": f"{NAIP_SERVICE}/query?{urlencode(params)}", "point": point, "bbox": recipe.bbox_wgs84}


def _query_attributes(body: bytes, final_url: str) -> dict[int, dict[str, object]]:
    if len(body) > SERVICE_JSON_CAP or not _safe_service_response(final_url, "query"):
        raise GalleryError("GALLERY_RECORD")
    try:
        payload = json.loads(body.decode("utf-8"))
        features = payload["features"]
        if not isinstance(features, list) or not features:
            raise ValueError
        records: dict[int, dict[str, object]] = {}
        for feature in features:
            attributes = feature["attributes"]
            object_id = attributes.get("OBJECTID") if isinstance(attributes, dict) else None
            if isinstance(object_id, bool) or not isinstance(object_id, int) or set(attributes) != set(SOURCE_FIELDS) or object_id in records:
                raise ValueError
            records[object_id] = attributes
    except (UnicodeError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        raise GalleryError("GALLERY_RECORD") from None
    return records


def _identify_request(recipe: CandidateRecipe, point: tuple[float, float]) -> dict[str, object]:
    params = {
        "f": "json", "geometry": json.dumps(
            {"x": point[0], "y": point[1], "spatialReference": {"wkid": 4326}},
            separators=(",", ":"),
        ),
        "geometryType": "esriGeometryPoint", "returnCatalogItems": "true", "returnGeometry": "false",
    }
    return {"kind": "identify", "url": f"{NAIP_SERVICE}/identify?{urlencode(params)}", "point": point, "bbox": recipe.bbox_wgs84}


def _identify_catalog(body: bytes, final_url: str) -> tuple[set[int], int]:
    if len(body) > SERVICE_JSON_CAP or not _safe_service_response(final_url, "identify"):
        raise GalleryError("GALLERY_RECORD")
    try:
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError
        catalog = payload["catalogItems"]
        features = catalog["features"]
        visibility = payload["catalogItemVisibilities"]
        if not isinstance(catalog, dict) or not isinstance(features, list) or not isinstance(visibility, list) or not features or len(features) != len(visibility):
            raise ValueError
        ids: set[int] = set()
        contributors: list[int] = []
        for feature, visible in zip(features, visibility):
            if not isinstance(feature, dict) or not isinstance(feature.get("attributes"), dict):
                raise ValueError
            object_id = feature["attributes"].get("OBJECTID")
            if isinstance(object_id, bool) or not isinstance(object_id, int) or object_id in ids:
                raise ValueError
            if isinstance(visible, bool) or not isinstance(visible, (int, float)) or not math.isfinite(visible) or visible < 0:
                raise ValueError
            ids.add(object_id)
            if visible > 0:
                contributors.append(object_id)
        if len(contributors) != 1:
            raise ValueError
        return ids, contributors[0]
    except (UnicodeError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        raise GalleryError("GALLERY_RECORD") from None


def _mosaic_object_id(body: bytes, final_url: str) -> int:
    """Validate the documented Identify structure and return its visible item for tests."""
    return _identify_catalog(body, final_url)[1]


def _identify_catalog_ids(body: bytes, final_url: str) -> set[int]:
    """Validate Identify response shape while retaining every documented catalog identity."""
    return _identify_catalog(body, final_url)[0]


def _valid_source(attributes: dict[str, object], recipe: CandidateRecipe) -> tuple[int, dict[str, object]]:
    values = {field: attributes.get(field) for field in SOURCE_FIELDS}
    object_id = values["OBJECTID"]
    if isinstance(object_id, bool) or not isinstance(object_id, int):
        raise GalleryError("GALLERY_RECORD")
    name, raster_name = values["Name"], values["raster_name"]
    if not isinstance(name, str) or not name.strip() or not isinstance(raster_name, str) or not raster_name.strip():
        raise GalleryError("GALLERY_RECORD")
    agency = str(values["agency"] or "").casefold()
    product = f"{name} {raster_name}".casefold()
    year, acquisition_date = values["Year"], values["acquisition_date"]
    if not _AGENCY_IDENTITY.search(agency):
        raise GalleryError("GALLERY_RECORD")
    if any(term in product or term in agency for term in ("hro", "commercial")):
        raise GalleryError("GALLERY_RECORD")
    if isinstance(year, bool) or not isinstance(year, int) or year not in _NAIP_YEAR_RANGE:
        raise GalleryError("GALLERY_RECORD")
    if isinstance(acquisition_date, str) and acquisition_date:
        try:
            parsed_date = date.fromisoformat(acquisition_date)
        except ValueError:
            raise GalleryError("GALLERY_RECORD") from None
        valid_date = acquisition_date == parsed_date.isoformat()
    elif isinstance(acquisition_date, int) and not isinstance(acquisition_date, bool):
        try:
            parsed_date = datetime.fromtimestamp(acquisition_date / 1000, tz=timezone.utc).date()
        except (OverflowError, OSError, ValueError):
            raise GalleryError("GALLERY_RECORD") from None
        valid_date = 10**12 <= acquisition_date < 2 * 10**12
    else:
        raise GalleryError("GALLERY_RECORD")
    if not valid_date or parsed_date.year not in _NAIP_YEAR_RANGE:
        raise GalleryError("GALLERY_RECORD")
    values["download_url"] = _normalized_official_download_url(values["download_url"])
    west, south, east, north = recipe.bbox_wgs84
    if not (-124.9 < west < east < -66.8 and 24.3 < south < north < 49.5):
        raise GalleryError("GALLERY_RECORD")
    return object_id, values


def _export_request(recipe: CandidateRecipe, object_id: int) -> dict[str, object]:
    mosaic_rule = json.dumps({"mosaicMethod": "esriMosaicLockRaster", "lockRasterIds": [object_id]}, separators=(",", ":"))
    params = {
        "f": "image", "bbox": ",".join(f"{value:.6f}" for value in recipe.bbox_wgs84),
        "bboxSR": "4326", "imageSR": "4326", "size": "1280,800", "format": "jpg", "interpolation": "RSP_BilinearInterpolation",
        "mosaicRule": mosaic_rule,
    }
    return {"kind": "export", "url": f"{NAIP_SERVICE}/exportImage?{urlencode(params)}", "bbox": recipe.bbox_wgs84, "objectId": object_id, "mosaicRule": mosaic_rule}


def _encode_jpeg(body: bytes) -> bytes:
    if len(body) > RASTER_CAP:
        raise GalleryError("GALLERY_RECORD")
    try:
        with Image.open(io.BytesIO(body)) as source:
            source.load()
            if source.mode in {"RGBA", "LA"} or (source.mode == "P" and "transparency" in source.info):
                rgba = source.convert("RGBA")
                image = Image.new("RGB", rgba.size, (0, 0, 0))
                image.paste(rgba, mask=rgba.getchannel("A"))
            else:
                image = source.convert("RGB")
            if image.size != OUTPUT_SIZE:
                image = image.resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
            result = io.BytesIO()
            image.save(result, format="JPEG", quality=JPEG_QUALITY, subsampling=0, optimize=False, progressive=False, exif=b"", icc_profile=None, comment=b"")
            return result.getvalue()
    except (UnidentifiedImageError, OSError, ValueError):
        raise GalleryError("GALLERY_RECORD") from None


def _common_source_qualified_id(
    captures: list[dict[int, dict[str, object]]], recipe: CandidateRecipe
) -> int:
    if len(captures) != 5:
        raise GalleryError("GALLERY_RECORD")
    qualified: list[set[int]] = []
    for capture in captures:
        source_ids: set[int] = set()
        for object_id, attributes in capture.items():
            try:
                selected_id, _source = _valid_source(dict(attributes), recipe)
            except GalleryError:
                continue
            if selected_id != object_id:
                raise GalleryError("GALLERY_RECORD")
            source_ids.add(object_id)
        qualified.append(source_ids)
    shared = set.intersection(*qualified)
    if len(shared) != 1:
        raise GalleryError("GALLERY_SOURCE_REJECTED")
    return shared.pop()


def acquire_candidate(recipe: CandidateRecipe, review_root: Path, transport: Transport = urlopen_transport) -> dict[str, object]:
    """Query one locked NAIP raster, derive deterministic review bytes and return its safe record."""
    root = Path(review_root)
    if not root.exists():
        root = _require_external_new_review_root(root)
    else:
        root = _checked_root(root)
        for worktree in _git_worktree_roots(REPO_ROOT):
            if root == worktree or root.is_relative_to(worktree):
                raise GalleryError("GALLERY_SCOPE")
        allowed = {f"{item.candidate_id}.jpg" for item in CANDIDATE_RECIPES}
        if any(path.name not in allowed for path in root.iterdir()):
            raise GalleryError("GALLERY_SCOPE")
    target = _checked_child(root, f"{recipe.candidate_id}.jpg")
    if target.exists():
        raise GalleryError("GALLERY_SCOPE")
    captures = [_query_attributes(*transport(_query_request(recipe, point))) for point in _inset_points(recipe.bbox_wgs84)]
    object_id = _common_source_qualified_id(captures, recipe)
    identified_catalogs = [_identify_catalog_ids(*transport(_identify_request(recipe, point))) for point in _inset_points(recipe.bbox_wgs84)]
    if any(object_id not in catalog for catalog in identified_catalogs):
        raise GalleryError("GALLERY_RECORD")
    object_id, source = _valid_source(captures[0][object_id], recipe)
    export_body, export_url = transport(_export_request(recipe, object_id))
    if not _safe_service_response(export_url, "exportImage"):
        raise GalleryError("GALLERY_RECORD")
    jpeg = _encode_jpeg(export_body)
    target.write_bytes(jpeg)
    return {
        "schemaVersion": 1, "candidateId": recipe.candidate_id, "category": recipe.category,
        "source": {
            "service": NAIP_SERVICE, "publicDomainRecord": NAIP_PUBLIC_DOMAIN_RECORD,
            "objectId": object_id, "rasterIds": [object_id], "name": source["Name"], "rasterName": source["raster_name"],
            "year": source["Year"], "acquisitionDate": source["acquisition_date"], "agency": source["agency"],
            "downloadUrl": source["download_url"], "responseSha256": hashlib.sha256(export_body).hexdigest(),
            "bboxWgs84": list(recipe.bbox_wgs84), "resolutionValue": source["resolution_value"],
            "resolutionUnits": source["resolution_units"], "bandCount": source["band_count"], "sensorType": source["sensor_type"],
        },
        "derivation": {"outputSize": list(OUTPUT_SIZE), "color": "sRGB", "jpegQuality": JPEG_QUALITY, "metadata": "stripped", "threshold": DEFAULT_CONFIDENCE, "classFilter": []},
        "image": {"reviewName": target.name, "bytes": len(jpeg), "sha256": hashlib.sha256(jpeg).hexdigest(), "mediaType": "image/jpeg"},
    }


_TOP_LEVEL = {"schemaVersion", "candidateId", "category", "source", "derivation", "image"}
_SOURCE = {"service", "publicDomainRecord", "objectId", "rasterIds", "name", "rasterName", "year", "acquisitionDate", "agency", "downloadUrl", "responseSha256", "bboxWgs84", "resolutionValue", "resolutionUnits", "bandCount", "sensorType"}
_DERIVATION = {"outputSize", "color", "jpegQuality", "metadata", "threshold", "classFilter"}
_IMAGE = {"reviewName", "bytes", "sha256", "mediaType"}
MODEL_SHA256 = "a0a1a2dd357067e8c6c9f5ce7bb33487188423f9722e813be880da4f9badcd97"
_REPORT_KEYS = {"schemaVersion", "threshold", "modelSha256", "candidates"}
_CANDIDATE_KEYS = {"candidateId", "category", "runCompleted", "numericRuntime", "detections", "visualReview"}
_DETECTION_KEYS = {"classId", "confidence", "cx", "cy", "w", "h", "angle"}


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def source_valid_pool(records: object, review_root: Path) -> tuple[dict[str, object], ...]:
    if not isinstance(records, (list, tuple)):
        raise ValueError("GALLERY_OBSERVATION")
    pool: list[dict[str, object]] = []
    seen: set[str] = set()
    counts = {"airfield": 0, "sports-complex": 0, "harbor": 0}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("GALLERY_OBSERVATION")
        try:
            validate_candidate_record(record, review_root)
            candidate_id = record["candidateId"]
            category = record["category"]
        except (GalleryError, KeyError, TypeError):
            raise ValueError("GALLERY_OBSERVATION") from None
        if not isinstance(candidate_id, str) or candidate_id in seen or category not in counts:
            raise ValueError("GALLERY_OBSERVATION")
        seen.add(candidate_id)
        counts[category] += 1
        pool.append(record)
    if any(count < 2 or count > 3 for count in counts.values()):
        raise ValueError("GALLERY_OBSERVATION")
    return tuple(pool)


def validate_observations(
    report: dict[str, object], records: object | None = None, review_root: Path | None = None
) -> None:
    """Validate the closed report against the exact acquired source-valid pool."""
    if not isinstance(report, dict) or set(report) != _REPORT_KEYS:
        raise ValueError("GALLERY_OBSERVATION")
    if report["schemaVersion"] != 1 or report["threshold"] != DEFAULT_CONFIDENCE or report["modelSha256"] != MODEL_SHA256:
        raise ValueError("GALLERY_OBSERVATION")
    candidates = report["candidates"]
    if not isinstance(candidates, list):
        raise ValueError("GALLERY_OBSERVATION")
    try:
        pool = source_valid_pool(records, review_root) if records is not None and review_root is not None else ()
    except (GalleryError, ValueError):
        raise ValueError("GALLERY_OBSERVATION") from None
    if not pool:
        raise ValueError("GALLERY_OBSERVATION")
    expected = {str(record["candidateId"]): str(record["category"]) for record in pool}
    seen: set[str] = set()
    for item in candidates:
        if not isinstance(item, dict) or set(item) != _CANDIDATE_KEYS:
            raise ValueError("GALLERY_OBSERVATION")
        candidate_id, category = item["candidateId"], item["category"]
        if not isinstance(candidate_id, str) or candidate_id in seen or expected.get(candidate_id) != category:
            raise ValueError("GALLERY_OBSERVATION")
        if item["runCompleted"] is not True or item["visualReview"] != "unreviewed":
            raise ValueError("GALLERY_OBSERVATION")
        seen.add(candidate_id)
        runtime = item["numericRuntime"]
        if isinstance(runtime, bool) or not isinstance(runtime, (int, float)) or not math.isfinite(runtime) or runtime < 0:
            raise ValueError("GALLERY_OBSERVATION")
        detections = item["detections"]
        if not isinstance(detections, list):
            raise ValueError("GALLERY_OBSERVATION")
        for detection in detections:
            if not isinstance(detection, dict) or set(detection) != _DETECTION_KEYS:
                raise ValueError("GALLERY_OBSERVATION")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in detection.values()):
                raise ValueError("GALLERY_OBSERVATION")
    if seen != set(expected):
        raise ValueError("GALLERY_OBSERVATION")


def validate_candidate_record(record: dict[str, object], review_root: Path) -> None:
    """Validate a closed, public-safe admission record without exposing its internals."""
    try:
        if not isinstance(record, dict) or set(record) != _TOP_LEVEL or record["schemaVersion"] != 1:
            raise ValueError
        candidate_id, category = record["candidateId"], record["category"]
        recipe = RECIPE_BY_ID.get(candidate_id) if isinstance(candidate_id, str) else None
        if recipe is None or category != recipe.category:
            raise ValueError
        source, derivation, image = record["source"], record["derivation"], record["image"]
        if not all(isinstance(value, dict) for value in (source, derivation, image)):
            raise ValueError
        if set(source) != _SOURCE or set(derivation) != _DERIVATION or set(image) != _IMAGE:
            raise ValueError
        if source["service"] != NAIP_SERVICE or source["publicDomainRecord"] != NAIP_PUBLIC_DOMAIN_RECORD:
            raise ValueError
        if source["bboxWgs84"] != list(recipe.bbox_wgs84) or not isinstance(source["objectId"], int) or source["rasterIds"] != [source["objectId"]]:
            raise ValueError
        source_attributes = {
            "OBJECTID": source["objectId"], "Name": source["name"], "Year": source["year"],
            "raster_name": source["rasterName"], "download_url": source["downloadUrl"],
            "acquisition_date": source["acquisitionDate"], "agency": source["agency"],
            "resolution_value": source["resolutionValue"], "resolution_units": source["resolutionUnits"],
            "band_count": source["bandCount"], "sensor_type": source["sensorType"],
        }
        selected_id, canonical_source = _valid_source(source_attributes, recipe)
        if selected_id != source["objectId"] or source["downloadUrl"] != canonical_source["download_url"]:
            raise ValueError
        if not _digest(source["responseSha256"]):
            raise ValueError
        if derivation != {"outputSize": [1280, 800], "color": "sRGB", "jpegQuality": 90, "metadata": "stripped", "threshold": 0.25, "classFilter": []}:
            raise ValueError
        if image["reviewName"] != f"{candidate_id}.jpg" or image["mediaType"] != "image/jpeg" or not isinstance(image["bytes"], int) or not _digest(image["sha256"]):
            raise ValueError
        serialized = json.dumps(record, sort_keys=True)
        blocked = ("?", "token", "authorization", "header", "rawerror", "stack", "traceback", "c:\\", "file://", "\\\\")
        if any(value in serialized.casefold() for value in blocked):
            raise ValueError
        name = str(image["reviewName"])
        if Path(name).is_absolute() or "/" in name or "\\" in name:
            raise ValueError
        review = _checked_root(review_root)
        candidate = _checked_child(review, name)
        if not candidate.is_file() or candidate.is_symlink():
            raise ValueError
        data = candidate.read_bytes()
        if len(data) != image["bytes"] or hashlib.sha256(data).hexdigest() != image["sha256"] or not data.startswith(b"\xff\xd8\xff") or not data.endswith(b"\xff\xd9"):
            raise ValueError
        with Image.open(io.BytesIO(data)) as decoded:
            decoded.load()
            if decoded.format != "JPEG" or decoded.size != OUTPUT_SIZE or decoded.mode != "RGB":
                raise ValueError
            if decoded.getexif() or decoded.info.get("icc_profile") is not None or decoded.info.get("comment") is not None:
                raise ValueError
    except (KeyError, TypeError, ValueError):
        raise GalleryError("GALLERY_RECORD") from None
    except (OSError, UnidentifiedImageError):
        raise GalleryError("GALLERY_RECORD") from None


def validate_approved_gallery(report: dict[str, object], review_root: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    try:
        if set(report) != {"schemaVersion", "threshold", "records", "visualReview"} or report["schemaVersion"] != 1 or report["threshold"] != DEFAULT_CONFIDENCE:
            raise ValueError
        records, visual = report["records"], report["visualReview"]
        if not isinstance(records, list) or not isinstance(visual, dict) or len(records) != 3:
            raise ValueError
        for record in records:
            validate_candidate_record(record, review_root)
            if visual.get(record["candidateId"]) != "approved":
                raise ValueError
        if set(visual) != {record["candidateId"] for record in records}:
            raise ValueError
        by_category = {record["category"]: record for record in records}
        if set(by_category) != {"airfield", "sports-complex", "harbor"}:
            raise ValueError
        return tuple(by_category[category] for category in ("airfield", "sports-complex", "harbor"))  # type: ignore[return-value]
    except (GalleryError, KeyError, TypeError, ValueError):
        raise GalleryError("GALLERY_RECORD") from None


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise GalleryError("GALLERY_RECORD") from None


def _acquired_pool(review_root: Path) -> tuple[dict[str, object], ...]:
    data = _read_json(_checked_child(review_root, "candidate-records.json"))
    if not isinstance(data, dict) or set(data) != {"schemaVersion", "records"} or data.get("schemaVersion") != 1:
        raise GalleryError("GALLERY_RECORD")
    try:
        return source_valid_pool(data["records"], review_root)
    except (GalleryError, KeyError, ValueError):
        raise GalleryError("GALLERY_RECORD") from None


def acquire_all(
    review_root: Path, transport: Transport = urlopen_transport
) -> list[dict[str, object]]:
    """Admit the source-valid pool atomically while keeping source rejections private."""
    root = _require_external_new_review_root(review_root)
    stage = _checked_child(root, ".gallery-stage")
    records: list[dict[str, object]] = []
    try:
        for recipe in CANDIDATE_RECIPES:
            try:
                records.append(acquire_candidate(recipe, stage, transport))
            except GalleryError as error:
                if error.code == "GALLERY_SOURCE_REJECTED":
                    continue
                raise
        counts = {
            category: sum(record["category"] == category for record in records)
            for category in ("airfield", "sports-complex", "harbor")
        }
        if any(count < 2 or count > 3 for count in counts.values()):
            raise GalleryError("GALLERY_RECORD")
        for record in records:
            name = record["image"]["reviewName"]  # type: ignore[index]
            source = _checked_child(stage, str(name))
            destination = _checked_child(root, str(name))
            if not source.is_file() or destination.exists():
                raise GalleryError("GALLERY_SCOPE")
            os.replace(source, destination)
        _checked_child(root, "candidate-records.json").write_text(
            json.dumps({"schemaVersion": 1, "records": records}, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return records
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def approve(review_root: Path, observations: Path, pointer: Path) -> None:
    root = _checked_root(review_root)
    observed = _read_json(observations)
    if not isinstance(observed, dict):
        raise GalleryError("GALLERY_RECORD")
    try:
        pool = _acquired_pool(root)
        validate_observations(observed, pool, root)
    except (GalleryError, ValueError):
        raise GalleryError("GALLERY_RECORD") from None
    observed_ids = {record["candidateId"] for record in pool}
    choices: list[dict[str, object]] = []
    for category in ("airfield", "sports-complex", "harbor"):
        options = [record for record in pool if record.get("category") == category and record.get("candidateId") in observed_ids]
        print(f"{category}: " + ", ".join(str(option["candidateId"]) for option in options))
        selected = input("Approved candidate ID: ").strip()
        matches = [option for option in options if option.get("candidateId") == selected]
        if len(matches) != 1:
            raise GalleryError("GALLERY_RECORD")
        choices.append(matches[0])
    report = {"schemaVersion": 1, "threshold": DEFAULT_CONFIDENCE, "records": choices, "visualReview": {record["candidateId"]: "approved" for record in choices}}
    validate_approved_gallery(report, root)
    approved = _checked_child(root, "approved-gallery.json")
    if approved.exists() or pointer.exists():
        raise GalleryError("GALLERY_SCOPE")
    approved.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(root) + "\n", encoding="utf-8")


def verify_approved(review_root: Path) -> None:
    root = _checked_root(review_root)
    report = _read_json(_checked_child(root, "approved-gallery.json"))
    if not isinstance(report, dict):
        raise GalleryError("GALLERY_RECORD")
    approved = validate_approved_gallery(report, root)
    pool = _acquired_pool(root)
    by_id = {record["candidateId"]: record for record in pool}
    if any(by_id.get(record["candidateId"]) != record for record in approved):
        raise GalleryError("GALLERY_RECORD")
    observed = _read_json(_checked_child(root, "observations.json"))
    if not isinstance(observed, dict):
        raise GalleryError("GALLERY_RECORD")
    try:
        validate_observations(observed, pool, root)
    except ValueError:
        raise GalleryError("GALLERY_RECORD") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", choices=("acquire", "approve", "verify-approved"))
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--pointer", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "acquire" and args.observations is None and args.pointer is None:
            acquire_all(args.review_root)
        elif args.command == "approve" and args.observations is not None and args.pointer is not None:
            approve(args.review_root, args.observations, args.pointer)
        elif args.command == "verify-approved" and args.observations is None and args.pointer is None:
            verify_approved(args.review_root)
        else:
            raise GalleryError("GALLERY_SCOPE")
    except GalleryError as error:
        print(f"[FAIL] {error.code}")
        return 1
    print("[OK] GALLERY_ADMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
