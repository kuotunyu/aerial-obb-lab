"""Offline release evidence and bounded-claim checks.

This module intentionally uses only the Python standard library. It never imports an ML runtime,
opens a remote connection, or reads ignored private files.
"""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

CLAIM_FILES = {
    "matched-evaluation": {
        "README.en.md": ("-0.05", "-0.13", "near-tie"),
        "README.md": ("-0.05", "-0.13", "持平略降"),
        "docs/training_results.md": ("-0.05", "-0.13", "slight regression"),
        "docs/model_card.md": ("-0.05", "-0.13", "slight regression"),
    },
    "export-smoke": {
        "README.en.md": ("DOTA8", "0.9950", "not full DOTAv1 production certification"),
        "README.md": ("DOTA8", "0.9950", "不是完整 DOTAv1 production certification"),
    },
    "t4-benchmark": {
        "README.en.md": ("Tesla T4", "batch=1", "1024", "20.22", "49.4", "historical"),
        "README.md": ("Tesla T4", "batch=1", "1024", "20.22", "49.4", "歷史"),
    },
    "analysis": {
        "README.en.md": ("28,853", "456", "1.76", "2.43", "100%"),
        "README.md": ("28,853", "456", "1.76", "2.43", "100%"),
        "docs/analysis_results.md": ("28853", "456", "ground-truth geometry"),
    },
    "browser-scope": {
        "README.en.md": ("user-supplied", "yolo26m-obb", "does not represent"),
        "README.md": ("使用者自行提供", "yolo26m-obb", "不代表"),
        "docs/model_card.md": ("user-supplied", "yolo26m-obb", "does not represent"),
        "demo/web/README.md": ("user-supplied", "yolo26m-obb", "does not represent"),
    },
}

PUBLIC_PRESENTATION_FILES = (
    "README.md",
    "README.en.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/training_results.md",
    "docs/analysis_results.md",
    "docs/model_card.md",
    "docs/per_class_metrics.json",
    "release/evidence.json",
    "demo/web/README.md",
    "demo/web/index.html",
    "demo/web/app.js",
    "notebooks/03_recover_per_class_metrics_colab.py",
)
PRIVATE_HF_REPO_IDENTIFIERS = (
    "aerial-obb-lab-" + "model-archive",
    "yolo26m-obb-" + "dota",
    "yolo26-obb-" + "aerial-detection",
    "dotav1-" + "split-cache",
)
PRIVATE_HF_REPO_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9._-])(?:"
    + "|".join(re.escape(identifier) for identifier in PRIVATE_HF_REPO_IDENTIFIERS)
    + r")(?![A-Za-z0-9._-])",
    re.I,
)
OWNER_HF_ARTIFACT_RE = re.compile(
    r"(?:https://huggingface\.co/(?:spaces/)?)?(?:[A-Za-z0-9._-]+/)?(?:"
    + "|".join(re.escape(identifier) for identifier in PRIVATE_HF_REPO_IDENTIFIERS)
    + r")(?![A-Za-z0-9._-])",
    re.I,
)
WORKFLOW_LINK_RE = re.compile(r"/actions/workflows/([^/?#)]+\.ya?ml)", re.I)
RETIRED_DEMO_REFERENCES = ("Gradio", "demo/space/", "demo/space-static/")

UNSUPPORTED_CAUSAL_NMS_PHRASES = (
    "why OBB beats horizontal boxes",
    "NMS sees these as duplicate detections and suppresses true positives",
    "detections an HBB NMS would wrongly suppress",
    "NMS 會把這些視為重複偵測而誤殺",
)
PRIVATE_NAMES = {".env", "notes.private.md"}
PRIVATE_FRAGMENTS = ("interview", "面試")
RUNTIME_PREFIXES = (
    ".pytest_cache/",
    ".venv/",
    "build/",
    "datasets/",
    "dist/",
    "runs/",
    "wandb/",
)
RUNTIME_PARTS = ("/__pycache__/",)
LOCAL_PATH_RE = re.compile(
    r'''(?i)(?:[A-Z]:[\\/](?:Users|Documents and Settings)[\\/]|/(?:Users|home)/[^/\s"']+/)'''
)
LOCAL_PATH_BYTES_RE = re.compile(
    rb'''(?i)(?:[A-Z]:[\\/](?:Users|Documents and Settings)[\\/]|/(?:Users|home)/[^/\x00\s"']+/)'''
)
TOKEN_SHAPED_SECRET_RE = re.compile(
    r"(?i)\b(?:gh[pousr]_[A-Za-z0-9]{24,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})\b"
)
TEXT_SUFFIXES = {".css", ".html", ".js", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
FORBIDDEN_MODEL_SUFFIXES = {".pt", ".onnx", ".engine", ".torchscript", ".tflite", ".mlpackage"}
APPROVED_DEMO_MODEL = "demo/web/models/yolo26n-obb-privacy-sanitized.onnx"
APPROVED_DEMO_MODEL_BYTES = 10207127
APPROVED_DEMO_MODEL_SHA256 = "a0a1a2dd357067e8c6c9f5ce7bb33487188423f9722e813be880da4f9badcd97"
SOURCE_MODEL_SHA256 = "02f7c539600296d7389341280beb82da810b15dc09c54cf2bc70f7f610331b38"
SOURCE_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.onnx"
APPROVED_MODEL_LICENSE = "AGPL-3.0-only"
APPROVED_MODEL_LICENSE_FILE = "demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt"
APPROVED_MODEL_LICENSE_SHA256 = "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"
APPROVED_MODEL_LICENSE_SOURCE_URL = (
    "https://raw.githubusercontent.com/ultralytics/assets/v8.4.0/LICENSE"
)
APPROVED_SANITIZATION_RECORD = (
    "demo/web/third_party/yolo26n-obb-privacy-sanitization.json"
)
APPROVED_MODIFICATION_STATUS = "metadata-only"
APPROVED_MODIFICATION_DATE = "2026-08-31"
APPROVED_MODIFIED_FIELD = "ModelProto.metadata_props[0].value"
CANONICAL_LF_ARTIFACTS = {
    "README.md", "README.en.md", "THIRD_PARTY_NOTICES.md", "RELEASE_CHECKLIST.md", "CHANGELOG.md",
    "demo/web/README.md", "demo/web/THIRD_PARTY_NOTICES.md",
    "demo/web/app.js",
    "demo/web/index.html",
    "demo/web/style.css",
    APPROVED_MODEL_LICENSE_FILE,
    APPROVED_SANITIZATION_RECORD,
}
RAW_BINARY_DIGEST_MODE = "raw-binary"
REQUIRED_BUNDLED_THIRD_PARTY_ARTIFACTS = {
    "demo/web/fonts/IBMPlexSansCondensed-SemiBold.woff2",
    APPROVED_DEMO_MODEL,
    "demo/web/samples/harbor.jpg",
    "demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt",
}
REQUIRED_REVIEWED_PUBLIC_ARTIFACTS = {
    "README.md",
    "README.en.md",
    "THIRD_PARTY_NOTICES.md",
    "RELEASE_CHECKLIST.md",
    "CHANGELOG.md",
    "demo/web/app.js",
    "demo/web/README.md",
    "demo/web/THIRD_PARTY_NOTICES.md",
    "demo/web/fonts/IBMPlexSansCondensed-SemiBold.woff2",
    "demo/web/index.html",
    APPROVED_DEMO_MODEL,
    "demo/web/samples/harbor.jpg",
    "demo/web/style.css",
    "demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt",
    "demo/web/third_party/yolo26n-obb-privacy-sanitization.json",
    "docs/assets/browser-workbench.png",
}
GALLERY_SAMPLE_PATHS = ("demo/web/samples/harbor.jpg",)
DOTA_DERIVED_VISUAL_RE = re.compile(r"^assets/hbb_vs_obb_.*\.(?:jpg|jpeg|png)$", re.I)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_digest_mode_error(entry: dict) -> str | None:
    relative = str(entry.get("path", ""))
    mode = entry.get("digest_mode")
    if relative in CANONICAL_LF_ARTIFACTS:
        if mode != "canonical-lf":
            return f"{relative}: canonical text artifact digest_mode must be canonical-lf"
        return None
    if "digest_mode" in entry and mode != RAW_BINARY_DIGEST_MODE:
        return (
            f"{relative}: binary artifact digest_mode must be absent or raw-binary"
        )
    return None


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def verify_evidence(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    evidence = load_json(root / "release" / "evidence.json")
    metrics = load_json(root / "docs" / "per_class_metrics.json")
    analysis = load_json(root / "docs" / "analysis_results.json")

    if evidence.get("schema_version") != 1:
        errors.append("release/evidence.json: unsupported schema_version")
    if evidence.get("release_candidate") != "unreleased-pages-candidate":
        errors.append("release/evidence.json: current Pages candidate identity is missing")

    matched = evidence["matched_evaluation"]
    for key, metric_key in (("mAP50", "mAP50"), ("mAP50_95", "mAP50-95")):
        if not _close(matched["fine_tuned"][key], metrics["aggregate"][metric_key]):
            errors.append(f"matched_evaluation.fine_tuned.{key} differs from per_class_metrics.json")
        calculated = round((matched["fine_tuned"][key] - matched["baseline"][key]) * 100, 2)
        if calculated != matched["delta_percentage_points"][key]:
            errors.append(f"matched_evaluation.delta_percentage_points.{key} is inconsistent")
        if matched["delta_percentage_points"][key] >= 0:
            errors.append(f"matched_evaluation.{key} must preserve the negative result")
    if matched["interpretation"] != "near-tie/slight regression":
        errors.append("matched_evaluation interpretation overstates the result")

    per_class = metrics["per_class"]
    for key in ("mAP50", "mAP50-95"):
        class_mean = sum(float(row[key]) for row in per_class) / len(per_class)
        if not _close(class_mean, metrics["aggregate"][key]):
            errors.append(f"per-class mean {key} differs from aggregate")

    smoke = evidence["export_smoke"]
    if smoke["dataset"] != "DOTA8 val" or smoke["production_certification"] is not False:
        errors.append("export_smoke must be DOTA8-only and non-certifying")
    if set(smoke["backend_mAP50"].values()) != {0.995}:
        errors.append("export_smoke backend values differ from accepted 0.9950 result")

    benchmark = evidence["t4_benchmark"]
    if (benchmark["device"], benchmark["batch"], benchmark["imgsz"]) != (
        "NVIDIA Tesla T4",
        1,
        1024,
    ):
        errors.append("T4 benchmark environment is not fully scoped")
    if benchmark["universal_performance_claim"] is not False:
        errors.append("T4 benchmark must not be a universal performance claim")
    tensor_rt = next(row for row in benchmark["results"] if row["backend"] == "TensorRT FP16")
    if tensor_rt["mean_ms"] != 20.22 or tensor_rt["fps"] != 49.4:
        errors.append("accepted TensorRT result changed")

    geometry = evidence["geometry_analysis"]
    object_count = sum(int(row["n"]) for row in analysis["inflation"].values())
    if object_count != geometry["objects"]:
        errors.append("geometry object count differs from analysis_results.json")
    weighted_inflation = sum(
        int(row["n"]) * float(row["mean"]) for row in analysis["inflation"].values()
    ) / object_count
    if not _close(geometry.get("overall_inflation_mean", math.nan), weighted_inflation):
        errors.append("geometry overall inflation mean differs from analysis_results.json")
    for name, expected in geometry["inflation"].items():
        actual = analysis["inflation"][name]
        if not _close(expected["mean"], actual["mean"]) or not _close(expected["p90"], actual["p90"]):
            errors.append(f"geometry inflation differs for {name}")
    for name, expected in geometry["phantom_overlap"].items():
        actual = analysis["overlap"][name]
        if actual["hbb_iou>=0.3"] != expected["hbb_iou_ge_0_3"]:
            errors.append(f"geometry HBB pair count differs for {name}")
        if actual["hbb_iou>=0.3_but_obb_iou<0.1"] != expected["obb_iou_lt_0_1"]:
            errors.append(f"geometry phantom pair count differs for {name}")

    browser = evidence["browser_demo"]
    if browser["represents_fine_tuned_medium_accuracy"] or browser["represents_t4_latency"]:
        errors.append("browser demo must not inherit medium accuracy or T4 latency evidence")
    expected_browser = {
        "distribution_mode": "public-agpl-privacy-sanitized-demo-model-plus-byom",
        "showcase_enabled": False,
        "demo_inference_performed": True,
        "model_bundled": True,
        "demo_images": ["demo/web/samples/harbor.jpg"],
        "default_demo_image": "demo/web/samples/harbor.jpg",
        "sample_count": 1,
        "sample_selection": "fixed-no-selector",
        "confidence": 0.25,
        "per_image_tuning": False,
        "precomputed_results": False,
        "represents_accuracy_evaluation": False,
        "demo_model": APPROVED_DEMO_MODEL,
        "runtime_load": "lazy-on-demo-detect-or-byom-selection",
        "layout": "workbench-31-69",
        "responsive_breakpoint_px": 960,
        "primary_action_first_viewport": True,
    }
    for field, expected in expected_browser.items():
        if browser.get(field) != expected:
            errors.append(f"browser_demo.{field} is inconsistent with the reviewed real demo")

    owner_visibility = evidence.get("owner_visibility_follow_up", {})
    model_archive = owner_visibility.get("historical_model_archive", {})
    historical_space = owner_visibility.get("historical_space", {})
    if model_archive.get("anonymous_http_status") != 401:
        errors.append("historical model archive must deny anonymous access")
    if (
        historical_space.get("anonymous_api_http_status") != 401
        or historical_space.get("anonymous_page_http_status") != 401
    ):
        errors.append("historical Space must deny anonymous API and page access")
    if owner_visibility.get("remote_mutation_performed_by_local_workflow") is not False:
        errors.append("owner visibility follow-up must remain read-only")
    return errors


def verify_artifacts(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    manifest = load_json(root / "release" / "artifact-manifest.json")
    if manifest.get("schema_version") != 2:
        errors.append("release/artifact-manifest.json: unsupported schema_version")
        return errors
    if manifest.get("release_candidate") != "unreleased-pages-candidate":
        errors.append("release/artifact-manifest.json: current Pages candidate identity is missing")
    if (
        manifest.get("distribution_mode")
        != "public-agpl-privacy-sanitized-demo-model-plus-byom"
    ):
        errors.append("artifact manifest must declare the privacy-sanitized real-demo distribution")
    if manifest.get("policy", {}).get("commercial_use_cleared") is not False:
        errors.append("artifact manifest must not claim commercial-use clearance")
    entries = manifest.get("bundled_third_party_artifacts", [])
    paths = [entry.get("path") for entry in entries]
    if len(paths) != len(set(paths)):
        errors.append("artifact manifest contains duplicate paths")
    if set(paths) != REQUIRED_BUNDLED_THIRD_PARTY_ARTIFACTS:
        errors.append("artifact manifest bundled third-party inventory is not exact")

    reviewed_entries = manifest.get("reviewed_public_artifacts", [])
    reviewed_paths = [entry.get("path") for entry in reviewed_entries]
    if len(reviewed_paths) != len(set(reviewed_paths)):
        errors.append("artifact manifest contains duplicate reviewed public paths")
    if set(reviewed_paths) != REQUIRED_REVIEWED_PUBLIC_ARTIFACTS:
        errors.append("artifact manifest reviewed public inventory is not exact")

    for entry in entries + reviewed_entries:
        relative = entry.get("path", "")
        path = root / relative
        if entry in entries:
            for field in ("kind", "provenance", "source_url", "license", "restrictions"):
                if not entry.get(field):
                    errors.append(f"{relative}: missing artifact field {field}")
        digest_mode_error = _artifact_digest_mode_error(entry)
        if digest_mode_error:
            errors.append(digest_mode_error)
            continue
        if not path.is_file():
            errors.append(f"{relative}: artifact is missing")
            continue
        payload = path.read_bytes()
        if entry.get("digest_mode") == "canonical-lf":
            payload = (
                payload.decode("utf-8")
                .replace("\r\n", "\n")
                .replace("\r", "\n")
                .encode("utf-8")
            )
        if len(payload) != entry.get("bytes"):
            errors.append(f"{relative}: byte size differs from manifest")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry.get("sha256"):
            errors.append(f"{relative}: SHA-256 differs from manifest")

    if _approved_license_entry(manifest) is None:
        errors.append(
            f"{APPROVED_MODEL_LICENSE_FILE}: canonical-LF license identity is not exact"
        )
    errors.extend(verify_code_only_paths(committed_paths(root), manifest))
    errors.extend(_verify_demo_model_contract(root, manifest))
    errors.extend(_verify_gallery_contract(root, manifest))
    return errors


def _verify_gallery_contract(root: Path, manifest: dict) -> list[str]:
    """Bind the admitted public gallery to its closed receipt and manifest entries."""
    errors: list[str] = []
    receipt = load_json(root / "release" / "sample-gallery-sources.json")
    demo = load_json(root / "demo" / "web" / "demo-model.json")
    samples = receipt.get("samples")
    if receipt.get("schemaVersion") != 1 or not isinstance(samples, list):
        return ["sample-gallery receipt is not a supported closed record"]
    receipt_paths = tuple(f"demo/web/{sample.get('path')}" for sample in samples)
    demo_paths = tuple(f"demo/web/{sample.get('path')}" for sample in demo.get("samples", []))
    if receipt_paths != GALLERY_SAMPLE_PATHS or demo_paths != GALLERY_SAMPLE_PATHS:
        errors.append("public sample gallery inventory is not exact")
        return errors
    if "defaultSampleId" in demo:
        errors.append("single-harbor demo must not declare a selectable default")
    entries = {entry.get("path"): entry for entry in manifest.get("bundled_third_party_artifacts", [])}
    for receipt_sample, demo_sample, path in zip(samples, demo["samples"], GALLERY_SAMPLE_PATHS):
        entry = entries.get(path)
        source = receipt_sample.get("source", {})
        derivation = receipt_sample.get("derivation", {})
        expected_entry = {
            "path": path,
            "sample_id": receipt_sample.get("id"),
            "sample_title": receipt_sample.get("title"),
            "media_type": receipt_sample.get("mediaType"),
            "width": receipt_sample.get("width"),
            "height": receipt_sample.get("height"),
            "bytes": receipt_sample.get("bytes"),
            "sha256": receipt_sample.get("sha256"),
            "kind": "public-domain NAIP aerial sample derivative",
            "provenance": (
                f"USGS/USDA NAIP product {source.get('productId')}, {source.get('year')}; "
                "crop/resample/metadata removal, bbox "
                f"{derivation.get('bboxWgs84')}, {derivation.get('outputSize', [None, None])[0]}x"
                f"{derivation.get('outputSize', [None, None])[1]} {derivation.get('color')} JPEG "
                f"quality {derivation.get('jpegQuality')}."
            ),
            "source_url": source.get("service"),
            "source_product_id": source.get("productId"),
            "source_year": source.get("year"),
            "source_acquisition_date": source.get("acquisitionDate"),
            "source_agency": source.get("agency"),
            "public_domain_record": source.get("publicDomainRecord"),
            "modification_status": "crop/resample/metadata removal",
            "derivation": derivation,
            "alt": receipt_sample.get("alt"),
            "guardrails": receipt_sample.get("guardrails"),
            "license": "Public Domain",
            "restrictions": [
                "Curated integration example only; not accuracy, evaluation, or model-quality evidence.",
                "No USGS or USDA endorsement is implied.",
            ],
        }
        if entry is None or any(entry.get(key) != value for key, value in expected_entry.items()):
            errors.append(f"{path}: manifest NAIP record differs from receipt")
        if demo_sample != receipt_sample:
            errors.append(f"{path}: demo manifest sample differs from receipt")
        file_path = root / path
        if not file_path.is_file() or file_path.stat().st_size != receipt_sample.get("bytes"):
            errors.append(f"{path}: approved JPEG bytes differ from receipt")
        elif hashlib.sha256(file_path.read_bytes()).hexdigest() != receipt_sample.get("sha256"):
            errors.append(f"{path}: approved JPEG digest differs from receipt")
    return errors


def _approved_model_entry(manifest: dict) -> dict | None:
    matches = [
        entry
        for entry in manifest.get("bundled_third_party_artifacts", [])
        if entry.get("path") == APPROVED_DEMO_MODEL
    ]
    if len(matches) != 1:
        return None
    entry = matches[0]
    expected = {
        "path": APPROVED_DEMO_MODEL,
        "bytes": APPROVED_DEMO_MODEL_BYTES,
        "sha256": APPROVED_DEMO_MODEL_SHA256,
        "source_url": SOURCE_MODEL_URL,
        "source_sha256": SOURCE_MODEL_SHA256,
        "modification_status": APPROVED_MODIFICATION_STATUS,
        "modification_date": APPROVED_MODIFICATION_DATE,
        "sanitization_record": APPROVED_SANITIZATION_RECORD,
        "license": APPROVED_MODEL_LICENSE,
        "license_file": APPROVED_MODEL_LICENSE_FILE,
    }
    if (
        any(entry.get(field) != value for field, value in expected.items())
        or (
            "digest_mode" in entry
            and entry.get("digest_mode") != RAW_BINARY_DIGEST_MODE
        )
    ):
        return None
    return entry


def _approved_license_entry(manifest: dict) -> dict | None:
    matches = [
        entry
        for entry in manifest.get("bundled_third_party_artifacts", [])
        if entry.get("path") == APPROVED_MODEL_LICENSE_FILE
    ]
    if len(matches) != 1:
        return None
    entry = matches[0]
    expected = {
        "path": APPROVED_MODEL_LICENSE_FILE,
        "bytes": 34523,
        "sha256": APPROVED_MODEL_LICENSE_SHA256,
        "digest_mode": "canonical-lf",
        "source_url": APPROVED_MODEL_LICENSE_SOURCE_URL,
        "license": APPROVED_MODEL_LICENSE,
    }
    if any(entry.get(field) != value for field, value in expected.items()):
        return None
    return entry


def _verify_demo_model_contract(root: Path, manifest: dict) -> list[str]:
    errors: list[str] = []
    model_entry = _approved_model_entry(manifest)
    if model_entry is None:
        return [f"{APPROVED_DEMO_MODEL}: exact manifest-bound model entry is missing"]
    demo = load_json(root / "demo" / "web" / "demo-model.json")
    receipt = load_json(
        root / "demo" / "web" / "third_party" / "yolo26n-obb-privacy-sanitization.json"
    )
    expected_relative = APPROVED_DEMO_MODEL.removeprefix("demo/web/")
    expected = (expected_relative, APPROVED_DEMO_MODEL_BYTES, APPROVED_DEMO_MODEL_SHA256)
    if (
        demo.get("model", {}).get("path"),
        demo.get("model", {}).get("bytes"),
        demo.get("model", {}).get("sha256"),
    ) != expected:
        errors.append("demo-model.json: derivative identity differs from artifact manifest")
    demo_model = demo.get("model", {})
    if (
        demo_model.get("source"),
        demo_model.get("sourceSha256"),
        demo_model.get("modificationStatus"),
        demo_model.get("license"),
        demo_model.get("release"),
    ) != (
        SOURCE_MODEL_URL,
        SOURCE_MODEL_SHA256,
        APPROVED_MODIFICATION_STATUS,
        APPROVED_MODEL_LICENSE,
        "v8.4.0",
    ):
        errors.append("demo-model.json: model provenance contract differs from artifact manifest")
    if (
        demo.get("license", {}).get("path"),
        demo.get("license", {}).get("sha256"),
        demo.get("license", {}).get("bytes"),
    ) != (
        APPROVED_MODEL_LICENSE_FILE.removeprefix("demo/web/"),
        APPROVED_MODEL_LICENSE_SHA256,
        34523,
    ):
        errors.append("demo-model.json: license contract differs from artifact manifest")
    if (
        demo.get("sanitization", {}).get("path"),
        demo.get("sanitization", {}).get("modificationDate"),
        demo.get("sanitization", {}).get("modifiedField"),
        demo.get("sanitization", {}).get("removedMetadataEntries"),
    ) != (
        APPROVED_SANITIZATION_RECORD.removeprefix("demo/web/"),
        APPROVED_MODIFICATION_DATE,
        APPROVED_MODIFIED_FIELD,
        1,
    ):
        errors.append("demo-model.json: sanitization contract differs from artifact manifest")
    if (
        receipt.get("derivative", {}).get("path"),
        receipt.get("derivative", {}).get("bytes"),
        receipt.get("derivative", {}).get("sha256"),
    ) != expected:
        errors.append("sanitization receipt: derivative identity differs from artifact manifest")
    if (
        receipt.get("source", {}).get("url"),
        receipt.get("source", {}).get("sha256"),
        receipt.get("source", {}).get("release"),
        receipt.get("source", {}).get("bytes"),
    ) != (SOURCE_MODEL_URL, SOURCE_MODEL_SHA256, "v8.4.0", 10207250):
        errors.append("sanitization receipt: source identity differs from artifact manifest")
    if (
        receipt.get("license", {}).get("path"),
        receipt.get("license", {}).get("sha256"),
        receipt.get("license", {}).get("spdx"),
    ) != (
        APPROVED_MODEL_LICENSE_FILE.removeprefix("demo/web/"),
        APPROVED_MODEL_LICENSE_SHA256,
        APPROVED_MODEL_LICENSE,
    ):
        errors.append("sanitization receipt: license contract differs from artifact manifest")
    transformation = receipt.get("transformation", {})
    if (
        transformation.get("modificationStatus"),
        transformation.get("modificationDate"),
        transformation.get("modifiedField"),
        transformation.get("removedMetadataEntries"),
    ) != (
        APPROVED_MODIFICATION_STATUS,
        APPROVED_MODIFICATION_DATE,
        APPROVED_MODIFIED_FIELD,
        1,
    ):
        errors.append("sanitization receipt: modification record is incomplete")
    if receipt.get("provenance", {}).get("commercialUseCleared") is not False:
        errors.append("sanitization receipt must not claim commercial-use clearance")
    return errors


def verify_code_only_paths(relative_paths: list[str], manifest: dict) -> list[str]:
    """Admit one exact derivative and reject every other model/DOTA visual."""
    errors: list[str] = []
    normalized = {path.replace("\\", "/") for path in relative_paths}
    for relative in sorted(normalized):
        if Path(relative).suffix.casefold() in FORBIDDEN_MODEL_SUFFIXES:
            if relative == APPROVED_DEMO_MODEL:
                if _approved_model_entry(manifest) is None:
                    errors.append(f"public release model exception is not exact: {relative}")
            else:
                errors.append(f"public release contains unapproved model binary: {relative}")
        if DOTA_DERIVED_VISUAL_RE.match(relative):
            errors.append(f"code-only release contains DOTA-derived visual: {relative}")
    for entry in manifest.get("excluded_historical_artifacts", []):
        relative = str(entry.get("path", "")).replace("\\", "/")
        if relative in normalized:
            errors.append(f"excluded historical artifact is still distributed: {relative}")
    return errors


def verify_privacy_paths(relative_paths: list[str]) -> list[str]:
    errors: list[str] = []
    for raw in relative_paths:
        relative = raw.replace("\\", "/")
        while relative.startswith("./"):
            relative = relative[2:]
        lowered = relative.casefold()
        basename = relative.rsplit("/", 1)[-1].casefold()
        if basename in PRIVATE_NAMES or any(fragment.casefold() in lowered for fragment in PRIVATE_FRAGMENTS):
            errors.append(f"private release member: {relative}")
        elif lowered.startswith(RUNTIME_PREFIXES) or any(part in f"/{lowered}/" for part in RUNTIME_PARTS):
            errors.append(f"runtime release member: {relative}")
    return errors


def verify_text_privacy(root: Path, paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if LOCAL_PATH_RE.search(text):
            errors.append(f"{path.relative_to(root).as_posix()}: absolute local user path")
        if PRIVATE_HF_REPO_IDENTIFIER_RE.search(text):
            errors.append(
                f"{path.relative_to(root).as_posix()}: private Hugging Face repository identifier"
            )
        if TOKEN_SHAPED_SECRET_RE.search(text):
            errors.append(f"{path.relative_to(root).as_posix()}: token-shaped secret")
    return errors


def verify_binary_privacy(root: Path, paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for relative in paths:
        path = relative if relative.is_absolute() else root / relative
        if LOCAL_PATH_BYTES_RE.search(path.read_bytes()):
            errors.append(f"{path.relative_to(root).as_posix()}: absolute local user path")
    return errors


def committed_paths(root: Path = ROOT) -> list[str]:
    if not (root / ".git").exists():
        excluded = {".pytest_cache", ".venv", "__pycache__", "build", "dist"}
        return sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and not any(part in excluded for part in path.relative_to(root).parts)
        )
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root, text=False)
    return [item.decode("utf-8") for item in output.split(b"\0") if item]


def verify_committed_privacy(root: Path = ROOT) -> list[str]:
    relative_paths = committed_paths(root)
    files = [root / relative for relative in relative_paths if (root / relative).is_file()]
    manifest = load_json(root / "release" / "artifact-manifest.json")
    binary_paths = [
        root / entry["path"]
        for entry in manifest.get("bundled_third_party_artifacts", [])
        if (root / entry["path"]).is_file()
    ]
    return (
        verify_privacy_paths(relative_paths)
        + verify_text_privacy(root, files)
        + verify_binary_privacy(root, binary_paths)
    )


def _claim_block(text: str, claim_id: str) -> str | None:
    start = f"<!-- claim:{claim_id} -->"
    end = f"<!-- /claim:{claim_id} -->"
    if text.count(start) != 1 or text.count(end) != 1:
        return None
    return text.split(start, 1)[1].split(end, 1)[0]


def unsupported_claim_errors(text: str) -> list[str]:
    """Reject causal detector outcomes that the geometry-only evidence cannot establish."""
    normalized = " ".join(text.split())
    if any(phrase in normalized for phrase in UNSUPPORTED_CAUSAL_NMS_PHRASES):
        return ["unsupported causal NMS outcome claim"]
    return []


def verify_claims(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    checked_paths: set[str] = set()
    for claim_id, documents in CLAIM_FILES.items():
        for relative, required_tokens in documents.items():
            path = root / relative
            text = path.read_text(encoding="utf-8")
            block = _claim_block(text, claim_id)
            if block is None:
                errors.append(f"{relative}: missing unique {claim_id} claim block")
                continue
            for token in required_tokens:
                if token not in block:
                    errors.append(f"{relative}: {claim_id} claim lacks {token!r}")
            if relative not in checked_paths:
                errors.extend(f"{relative}: {error}" for error in unsupported_claim_errors(text))
                checked_paths.add(relative)
    return errors


def verify_readme_language_structure(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    canonical = (root / "README.md").read_text(encoding="utf-8")
    english = (root / "README.en.md").read_text(encoding="utf-8")
    if not canonical.startswith("正體中文 | [English](README.en.md)"):
        errors.append("README.md: canonical zh-TW language navigation is missing")
    if not english.startswith("[正體中文](README.md) | English"):
        errors.append("README.en.md: English language navigation is missing")
    return errors


def verify_public_links(root: Path = ROOT) -> list[str]:
    """Keep public presentation links valid and independent of retired/private surfaces."""
    errors: list[str] = []
    for relative in PUBLIC_PRESENTATION_FILES:
        text = (root / relative).read_text(encoding="utf-8")
        if OWNER_HF_ARTIFACT_RE.search(text):
            errors.append(f"{relative}: owner Hugging Face artifact reference")
        for workflow_name in sorted(set(WORKFLOW_LINK_RE.findall(text))):
            workflow_path = Path(".github") / "workflows" / workflow_name
            if not (root / workflow_path).is_file():
                errors.append(
                    f"{relative}: workflow badge target does not exist: "
                    f"{workflow_path.as_posix()}"
                )
        if relative in {"README.md", "README.en.md"}:
            for reference in RETIRED_DEMO_REFERENCES:
                if reference.casefold() in text.casefold():
                    errors.append(f"{relative}: retired demo surface reference: {reference}")
    return errors


def main() -> int:
    errors = (
        verify_evidence(ROOT)
        + verify_claims(ROOT)
        + verify_readme_language_structure(ROOT)
        + verify_public_links(ROOT)
        + verify_artifacts(ROOT)
        + verify_committed_privacy(ROOT)
    )
    if errors:
        for error in errors:
            print(f"[FAIL] {error}", file=sys.stderr)
        return 1
    print("[OK] Release evidence, claims, exact real-demo artifacts, BYOM, and committed privacy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
