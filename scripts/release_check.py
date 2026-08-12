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
        "README.en.md": ("28,853", "456", "2.43", "100%"),
        "README.md": ("28,853", "456", "2.43", "100%"),
        "docs/analysis_results.md": ("28853", "456", "ground-truth geometry"),
    },
    "browser-scope": {
        "README.en.md": ("user-supplied", "yolo26m-obb", "does not represent"),
        "README.md": ("使用者自行提供", "yolo26m-obb", "不代表"),
        "docs/model_card.md": ("user-supplied", "yolo26m-obb", "does not represent"),
        "demo/space-static/README.md": ("user-supplied", "yolo26m-obb", "does not represent"),
        "demo/space/README.md": ("使用者自行提供", "yolo26m-obb", "不代表"),
    },
}

PUBLIC_PRESENTATION_FILES = (
    "README.md",
    "README.en.md",
    "README.zh-TW.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/training_results.md",
    "docs/analysis_results.md",
    "docs/model_card.md",
    "docs/per_class_metrics.json",
    "release/evidence.json",
    "demo/space-static/README.md",
    "demo/space-static/index.html",
    "demo/space-static/app.js",
    "demo/space/README.md",
    "demo/space/app.py",
    "notebooks/03_recover_per_class_metrics_colab.py",
)
OWNER_HF_ARTIFACT_RE = re.compile(
    r"(?:https://huggingface\.co/(?:spaces/)?)?steven0226/"
    r"(?:aerial-obb-lab-model-archive|yolo26m-obb-dota|yolo26-obb-aerial-detection)",
    re.I,
)

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
TEXT_SUFFIXES = {".css", ".html", ".js", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
FORBIDDEN_MODEL_SUFFIXES = {".pt", ".onnx", ".engine", ".torchscript", ".tflite", ".mlpackage"}
DOTA_DERIVED_VISUAL_RE = re.compile(r"^assets/hbb_vs_obb_.*\.(?:jpg|jpeg|png)$", re.I)
DEMO_MODEL_SOURCE_FILES = ("demo/app.py", "demo/space/app.py")
GRADIO_ENTRYPOINT_FILES = ("demo/app.py", "demo/space/app.py")
GRADIO_UI_SOURCE_FILES = ("demo/gradio_ui.py",)
FORBIDDEN_DEMO_MODEL_PATTERNS = (
    (re.compile(r"\b(?:hf_hub_download|HF_MODEL_REPO)\b"), "Hugging Face model download"),
    (re.compile(r'''YOLO\s*\(\s*["']yolo26[^"']*["']''', re.I), "named model fallback"),
    (re.compile(r"\.export\s*\("), "implicit model export"),
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def verify_evidence(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    evidence = load_json(root / "release" / "evidence.json")
    metrics = load_json(root / "docs" / "per_class_metrics.json")
    analysis = load_json(root / "docs" / "analysis_results.json")

    if evidence.get("schema_version") != 1:
        errors.append("release/evidence.json: unsupported schema_version")

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
    if manifest.get("distribution_mode") != "code-only-byom":
        errors.append("artifact manifest must declare code-only-byom distribution")
    entries = manifest.get("bundled_third_party_artifacts", [])
    paths = [entry.get("path") for entry in entries]
    if len(paths) != len(set(paths)):
        errors.append("artifact manifest contains duplicate paths")

    for entry in entries:
        relative = entry.get("path", "")
        path = root / relative
        for field in ("kind", "provenance", "source_url", "license", "restrictions"):
            if not entry.get(field):
                errors.append(f"{relative}: missing artifact field {field}")
        if not path.is_file():
            errors.append(f"{relative}: artifact is missing")
            continue
        if path.stat().st_size != entry.get("bytes"):
            errors.append(f"{relative}: byte size differs from manifest")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry.get("sha256"):
            errors.append(f"{relative}: SHA-256 differs from manifest")

    errors.extend(verify_code_only_paths(committed_paths(root), manifest))
    return errors


def verify_code_only_paths(relative_paths: list[str], manifest: dict) -> list[str]:
    """Reject model binaries and known DOTA-derived visuals from public distributions."""
    errors: list[str] = []
    normalized = {path.replace("\\", "/") for path in relative_paths}
    for relative in sorted(normalized):
        if Path(relative).suffix.casefold() in FORBIDDEN_MODEL_SUFFIXES:
            errors.append(f"code-only release contains model binary: {relative}")
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
    files = [root / relative for relative in relative_paths]
    manifest = load_json(root / "release" / "artifact-manifest.json")
    binary_paths = [
        root / entry["path"]
        for entry in manifest.get("bundled_third_party_artifacts", [])
    ]
    return (
        verify_privacy_paths(relative_paths)
        + verify_text_privacy(root, files)
        + verify_binary_privacy(root, binary_paths)
    )


def demo_model_source_errors(sources: dict[str, str]) -> list[str]:
    """Reject automatic model acquisition in published Python demo entry points."""
    errors: list[str] = []
    for relative in sorted(sources):
        text = sources[relative]
        for pattern, label in FORBIDDEN_DEMO_MODEL_PATTERNS:
            if pattern.search(text):
                errors.append(f"{relative}: {label}")
    return errors


def verify_demo_model_sources(root: Path = ROOT) -> list[str]:
    sources = {
        relative: (root / relative).read_text(encoding="utf-8")
        for relative in DEMO_MODEL_SOURCE_FILES
    }
    errors = demo_model_source_errors(sources)
    for relative, text in sources.items():
        if "require_model_path()" not in text:
            errors.append(f"{relative}: explicit local model path is not required")
        if 'os.environ.get("MODEL_DEVICE", "cpu")' not in text:
            errors.append(f"{relative}: CPU is not the default model device")
    return errors


def gradio_interaction_source_errors(sources: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for relative in sorted(sources):
        if re.search(r"\.upload\s*\(", sources[relative]):
            errors.append(f"{relative}: upload-triggered inference")
    return errors


def verify_gradio_interaction_sources(root: Path = ROOT) -> list[str]:
    sources = {
        relative: (root / relative).read_text(encoding="utf-8")
        for relative in GRADIO_ENTRYPOINT_FILES + GRADIO_UI_SOURCE_FILES
    }
    errors = gradio_interaction_source_errors(sources)
    for relative in GRADIO_ENTRYPOINT_FILES:
        text = sources[relative]
        if "build_demo(" not in text or "gr.Blocks(" in text:
            errors.append(f"{relative}: shared Gradio builder is not used")
    ui_text = sources["demo/gradio_ui.py"]
    if (
        ".click(" not in ui_text
        or ".failure(" not in ui_text
        or "interactive=False" not in ui_text
    ):
        errors.append("demo/gradio_ui.py: explicit disabled Detect flow is missing")
    return errors


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
    compatibility = (root / "README.zh-TW.md").read_text(encoding="utf-8")
    if not canonical.startswith("正體中文 | [English](README.en.md)"):
        errors.append("README.md: canonical zh-TW language navigation is missing")
    if not english.startswith("[正體中文](README.md) | English"):
        errors.append("README.en.md: English language navigation is missing")
    if (
        len(compatibility) >= 500
        or "[README.md](README.md)" not in compatibility
        or "<!-- claim:" in compatibility
    ):
        errors.append("README.zh-TW.md: expected a short canonical-README pointer")
    return errors


def verify_public_links(root: Path = ROOT) -> list[str]:
    """Keep public product presentation independent of the owner's HF artifacts."""
    errors: list[str] = []
    for relative in PUBLIC_PRESENTATION_FILES:
        text = (root / relative).read_text(encoding="utf-8")
        if OWNER_HF_ARTIFACT_RE.search(text):
            errors.append(f"{relative}: owner Hugging Face artifact reference")
    return errors


def main() -> int:
    errors = (
        verify_evidence(ROOT)
        + verify_claims(ROOT)
        + verify_readme_language_structure(ROOT)
        + verify_public_links(ROOT)
        + verify_artifacts(ROOT)
        + verify_demo_model_sources(ROOT)
        + verify_gradio_interaction_sources(ROOT)
        + verify_committed_privacy(ROOT)
    )
    if errors:
        for error in errors:
            print(f"[FAIL] {error}", file=sys.stderr)
        return 1
    print("[OK] Release evidence, claims, artifacts, local-model demos, and committed privacy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
