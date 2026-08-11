from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_release_check():
    module_path = ROOT / "scripts" / "release_check.py"
    assert module_path.is_file(), "release checker is missing"
    spec = importlib.util.spec_from_file_location("release_check", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_evidence() -> dict:
    path = ROOT / "release" / "evidence.json"
    assert path.is_file(), "release evidence registry is missing"
    return json.loads(path.read_text(encoding="utf-8"))


def test_matched_fine_tuning_is_a_negative_delta() -> None:
    evidence = load_evidence()

    assert evidence["matched_evaluation"]["delta_percentage_points"] == {
        "mAP50": -0.05,
        "mAP50_95": -0.13,
    }
    assert evidence["matched_evaluation"]["interpretation"] == "near-tie/slight regression"
    assert evidence["matched_evaluation"]["raw_baseline_log_committed"] is False


def test_dota8_and_t4_claims_are_strictly_scoped() -> None:
    evidence = load_evidence()

    assert evidence["export_smoke"]["dataset"] == "DOTA8 val"
    assert evidence["export_smoke"]["production_certification"] is False
    assert set(evidence["export_smoke"]["backend_mAP50"]) == {
        "PyTorch FP32",
        "ONNX",
        "TensorRT FP16",
    }
    assert set(evidence["export_smoke"]["backend_mAP50"].values()) == {0.995}
    assert evidence["t4_benchmark"]["device"] == "NVIDIA Tesla T4"
    assert evidence["t4_benchmark"]["batch"] == 1
    assert evidence["t4_benchmark"]["imgsz"] == 1024
    assert evidence["t4_benchmark"]["universal_performance_claim"] is False


def test_browser_demo_is_not_medium_checkpoint_evidence() -> None:
    browser = load_evidence()["browser_demo"]

    assert browser["model"] == "official yolo26n-obb"
    assert browser["represents_fine_tuned_medium_accuracy"] is False
    assert browser["represents_t4_latency"] is False


def test_release_evidence_and_claim_blocks_verify() -> None:
    release_check = load_release_check()

    assert release_check.verify_evidence(ROOT) == []
    assert release_check.verify_claims(ROOT) == []
