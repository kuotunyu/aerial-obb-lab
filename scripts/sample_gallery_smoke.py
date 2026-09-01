"""Run the existing BYOM browser path against external NAIP review candidates."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from threading import Thread
from typing import Iterator

if __name__ == "__main__":
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    sys.modules.setdefault("scripts.sample_gallery_smoke", sys.modules[__name__])

from scripts.prepare_sample_gallery import (
    CANDIDATE_RECIPES,
    DEFAULT_CONFIDENCE,
    GalleryError,
    RECIPE_BY_ID,
    _checked_root,
    _git_worktree_roots,
    is_reparse_point,
    validate_candidate_record,
)


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo" / "web"
MODEL_SHA256 = "a0a1a2dd357067e8c6c9f5ce7bb33487188423f9722e813be880da4f9badcd97"
_REPORT_KEYS = {"schemaVersion", "threshold", "modelSha256", "candidates"}
_CANDIDATE_KEYS = {"candidateId", "category", "runCompleted", "numericRuntime", "detections", "visualReview"}
_DETECTION_KEYS = {"classId", "confidence", "cx", "cy", "w", "h", "angle"}
_DESCRIPTION = re.compile(
    r"class=(?P<class_name>[^;]+); confidence=(?P<confidence>-?[\d.]+); "
    r"center-x=(?P<cx>-?[\d.]+) px; center-y=(?P<cy>-?[\d.]+) px; "
    r"width=(?P<w>-?[\d.]+) px; height=(?P<h>-?[\d.]+) px; angle=(?P<angle>-?[\d.]+)\N{DEGREE SIGN}\."
)
_CLASS_IDS = {
    "plane": 0, "ship": 1, "storage tank": 2, "baseball diamond": 3, "tennis court": 4,
    "basketball court": 5, "ground track field": 6, "harbor": 7, "bridge": 8, "large vehicle": 9,
    "small vehicle": 10, "helicopter": 11, "roundabout": 12, "soccer ball field": 13, "swimming pool": 14,
}
BYOM_MODEL_READY_LABEL = "Local ONNX model ready"


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request: object, client_address: object) -> None:
        pass


@contextmanager
def _static_server() -> Iterator[str]:
    server = _QuietServer(("127.0.0.1", 0), partial(_QuietHandler, directory=str(DEMO)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _external(root: Path) -> Path:
    safe = _checked_root(root)
    if any(safe == worktree or safe.is_relative_to(worktree) for worktree in _git_worktree_roots(ROOT)):
        raise GalleryError("GALLERY_SCOPE")
    return safe


def _safe_child(root: Path, name: str) -> Path:
    if not name or Path(name).is_absolute() or "/" in name or "\\" in name:
        raise GalleryError("GALLERY_SCOPE")
    child = root / name
    if child.exists() or child.is_symlink():
        if is_reparse_point(child):
            raise GalleryError("GALLERY_SCOPE")
    try:
        child.resolve(strict=False).relative_to(root)
    except ValueError:
        raise GalleryError("GALLERY_SCOPE") from None
    return child


def validate_observations(report: dict[str, object]) -> None:
    """Fail closed on unsafe or synthetic observation-report shapes."""
    if not isinstance(report, dict) or set(report) != _REPORT_KEYS:
        raise ValueError("GALLERY_OBSERVATION")
    if report["schemaVersion"] != 1 or report["threshold"] != DEFAULT_CONFIDENCE or report["modelSha256"] != MODEL_SHA256:
        raise ValueError("GALLERY_OBSERVATION")
    candidates = report["candidates"]
    if not isinstance(candidates, list):
        raise ValueError("GALLERY_OBSERVATION")
    seen: set[str] = set()
    for item in candidates:
        if not isinstance(item, dict) or set(item) != _CANDIDATE_KEYS:
            raise ValueError("GALLERY_OBSERVATION")
        candidate_id, category = item["candidateId"], item["category"]
        recipe = RECIPE_BY_ID.get(candidate_id) if isinstance(candidate_id, str) else None
        if recipe is None or candidate_id in seen or category != recipe.category or item["runCompleted"] is not True or item["visualReview"] != "unreviewed":
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
            for key, value in detection.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("GALLERY_OBSERVATION")


def source_valid_pool(records: object, review_root: Path) -> tuple[dict[str, object], ...]:
    if not isinstance(records, list):
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


def byom_model_ready(label: str) -> bool:
    return label.strip() == BYOM_MODEL_READY_LABEL


def _parse_canvas_descriptions(description: str) -> list[dict[str, float | int]]:
    if description.startswith("目前篩選"):
        return []
    matches = list(_DESCRIPTION.finditer(description))
    if not matches or " ".join(match.group(0) for match in matches) != description:
        raise ValueError("GALLERY_OBSERVATION")
    detections: list[dict[str, float | int]] = []
    for match in matches:
        values = match.groupdict()
        class_id = _CLASS_IDS.get(values["class_name"])
        if class_id is None:
            raise ValueError("GALLERY_OBSERVATION")
        detections.append({"classId": class_id, "confidence": float(values["confidence"]),
                           "cx": float(values["cx"]), "cy": float(values["cy"]), "w": float(values["w"]),
                           "h": float(values["h"]), "angle": float(values["angle"])})
    return detections


def _candidate_result(page: object, recipe_id: str, category: str) -> dict[str, object]:
    description = page.locator("#canvasDescription").inner_text()
    detections = _parse_canvas_descriptions(description)
    rows = page.locator("#resultsBody tr:not([data-empty='true'])").count()
    if rows != len(detections) or not page.locator("#canvasFrame").is_visible():
        raise ValueError("GALLERY_OBSERVATION")
    runtime = page.locator("#runtimeValue").inner_text().removesuffix(" ms").strip()
    try:
        elapsed = float(runtime)
    except ValueError:
        raise ValueError("GALLERY_OBSERVATION") from None
    return {"candidateId": recipe_id, "category": category, "runCompleted": True, "numericRuntime": elapsed, "detections": detections, "visualReview": "unreviewed"}


def run_smoke(review_root: Path, model: Path, report: Path, screenshot_dir: Path) -> None:
    review = _external(review_root)
    report = _safe_child(review, report.name) if report.parent.resolve(strict=False) == review else report
    if report.exists() or report.is_symlink():
        raise GalleryError("GALLERY_SCOPE")
    screenshots = _safe_child(review, screenshot_dir.name) if screenshot_dir.parent.resolve(strict=False) == review else screenshot_dir
    if screenshots.exists() or screenshots.is_symlink():
        raise GalleryError("GALLERY_SCOPE")
    screenshots.mkdir(parents=True, exist_ok=False)
    if hashlib.sha256(model.read_bytes()).hexdigest() != MODEL_SHA256:
        raise GalleryError("GALLERY_RECORD")
    try:
        batch = json.loads(_safe_child(review, "candidate-records.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise GalleryError("GALLERY_RECORD") from None
    if not isinstance(batch, dict) or set(batch) != {"schemaVersion", "records"} or batch["schemaVersion"] != 1:
        raise GalleryError("GALLERY_RECORD")
    try:
        records = source_valid_pool(batch["records"], review)
    except ValueError:
        raise GalleryError("GALLERY_RECORD") from None
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise GalleryError("GALLERY_NETWORK") from None
    observations: list[dict[str, object]] = []
    with _static_server() as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--disable-gpu"])
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            errors: list[str] = []
            page.on("console", lambda message: errors.append("console") if message.type == "error" else None)
            page.on("pageerror", lambda error: errors.append("page"))
            page.goto(f"{base_url}/", wait_until="networkidle")
            page.locator("#byomPanel summary").click()
            page.locator("#modelInput").set_input_files(str(model))
            page.wait_for_function(
                "document.querySelector('#modelLabel').textContent.trim() === 'Local ONNX model ready'",
                timeout=60_000,
            )
            for record in records:
                recipe = RECIPE_BY_ID[record["candidateId"]]
                image = _safe_child(review, str(record["image"]["reviewName"]))  # type: ignore[index]
                if not image.is_file():
                    raise GalleryError("GALLERY_RECORD")
                page.locator("#fileInput").set_input_files(str(image))
                page.wait_for_function("!document.querySelector('#detectBtn').disabled", timeout=30_000)
                page.screenshot(path=str(_safe_child(screenshots, f"{recipe.candidate_id}-original.png")), full_page=True)
                page.locator("#detectBtn").click()
                page.wait_for_function("document.querySelector('#status').dataset.kind === 'success'", timeout=90_000)
                page.screenshot(path=str(_safe_child(screenshots, f"{recipe.candidate_id}-result.png")), full_page=True)
                observations.append(_candidate_result(page, recipe.candidate_id, recipe.category))
            if errors:
                raise GalleryError("GALLERY_NETWORK")
        finally:
            browser.close()
    payload: dict[str, object] = {"schemaVersion": 1, "threshold": DEFAULT_CONFIDENCE, "modelSha256": MODEL_SHA256, "candidates": observations}
    validate_observations(payload)
    if {item["candidateId"] for item in observations} != {record["candidateId"] for record in records}:
        raise GalleryError("GALLERY_OBSERVATION")
    report.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--screenshot-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_smoke(args.review_root, args.model, args.report, args.screenshot_dir)
    except Exception:
        print("[FAIL] GALLERY_SMOKE")
        return 1
    print("[OK] GALLERY_SMOKE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
