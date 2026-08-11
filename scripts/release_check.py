"""Offline release evidence and bounded-claim checks.

This module intentionally uses only the Python standard library. It never imports an ML runtime,
opens a remote connection, or reads ignored private files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

CLAIM_FILES = {
    "matched-evaluation": {
        "README.md": ("-0.05", "-0.13", "near-tie"),
        "README.zh-TW.md": ("-0.05", "-0.13", "持平略降"),
        "docs/training_results.md": ("-0.05", "-0.13", "slight regression"),
        "docs/model_card.md": ("-0.05", "-0.13", "slight regression"),
    },
    "export-smoke": {
        "README.md": ("DOTA8", "0.9950", "not full DOTAv1 production certification"),
        "README.zh-TW.md": ("DOTA8", "0.9950", "不是完整 DOTAv1 production certification"),
    },
    "t4-benchmark": {
        "README.md": ("Tesla T4", "batch=1", "1024", "20.22", "49.4", "historical"),
        "README.zh-TW.md": ("Tesla T4", "batch=1", "1024", "20.22", "49.4", "歷史"),
    },
    "analysis": {
        "README.md": ("28,853", "456", "2.43", "100%"),
        "README.zh-TW.md": ("28,853", "456", "2.43", "100%"),
        "docs/analysis_results.md": ("28853", "456", "ground-truth geometry"),
    },
    "browser-scope": {
        "README.md": ("yolo26n-obb", "yolo26m-obb", "does not represent"),
        "README.zh-TW.md": ("yolo26n-obb", "yolo26m-obb", "不代表"),
        "docs/model_card.md": ("yolo26n-obb", "yolo26m-obb", "does not represent"),
        "demo/space-static/README.md": ("yolo26n-obb", "yolo26m-obb", "does not represent"),
        "demo/space/README.md": ("yolo26n-obb", "yolo26m-obb", "does not represent"),
    },
}


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
    return errors


def _claim_block(text: str, claim_id: str) -> str | None:
    start = f"<!-- claim:{claim_id} -->"
    end = f"<!-- /claim:{claim_id} -->"
    if text.count(start) != 1 or text.count(end) != 1:
        return None
    return text.split(start, 1)[1].split(end, 1)[0]


def verify_claims(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    for claim_id, documents in CLAIM_FILES.items():
        for relative, required_tokens in documents.items():
            path = root / relative
            block = _claim_block(path.read_text(encoding="utf-8"), claim_id)
            if block is None:
                errors.append(f"{relative}: missing unique {claim_id} claim block")
                continue
            for token in required_tokens:
                if token not in block:
                    errors.append(f"{relative}: {claim_id} claim lacks {token!r}")
    return errors


def main() -> int:
    errors = verify_evidence(ROOT) + verify_claims(ROOT)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}", file=sys.stderr)
        return 1
    print("[OK] Release evidence and bounded claim blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
