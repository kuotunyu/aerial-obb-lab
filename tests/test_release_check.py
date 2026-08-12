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

    assert browser["distribution_mode"] == "bring-your-own-model"
    assert browser["model"] == "user-supplied compatible YOLO26 OBB ONNX"
    assert "model_sha256" not in browser
    assert "model_bytes" not in browser
    assert "space_revision" not in browser
    assert browser["represents_fine_tuned_medium_accuracy"] is False
    assert browser["represents_t4_latency"] is False


def test_gradio_ui_evidence_is_model_free_and_zh_tw() -> None:
    evidence = load_evidence()["gradio_ui"]
    assert evidence["language"] == "zh-TW"
    assert evidence["layout"] == "wide-workbench-38-62"
    assert evidence["preview_model_loaded"] is False
    assert evidence["model_inference_run"] is False
    assert evidence["desktop_max_width_px"] == 1720
    assert evidence["responsive_breakpoint_px"] == 900


def test_release_evidence_and_claim_blocks_verify() -> None:
    release_check = load_release_check()

    assert release_check.verify_evidence(ROOT) == []
    assert release_check.verify_claims(ROOT) == []


def test_python_demos_require_local_models_without_fallbacks() -> None:
    release_check = load_release_check()

    assert release_check.verify_demo_model_sources(ROOT) == []


def test_demo_source_policy_rejects_remote_and_named_model_acquisition() -> None:
    release_check = load_release_check()

    assert release_check.demo_model_source_errors(
        {
            "remote.py": "from huggingface_hub import hf_hub_download\n",
            "named.py": 'model = YOLO("yolo26n-obb.pt")\n',
            "export.py": 'model.export(format="onnx")\n',
        }
    ) == [
        "export.py: implicit model export",
        "named.py: named model fallback",
        "remote.py: Hugging Face model download",
    ]


def test_public_presentation_omits_owner_hf_artifact_links() -> None:
    release_check = load_release_check()

    assert release_check.verify_public_links(ROOT) == []


def test_public_presentation_rejects_current_private_hf_archive(tmp_path: Path) -> None:
    release_check = load_release_check()
    release_check.PUBLIC_PRESENTATION_FILES = ("README.md",)
    (tmp_path / "README.md").write_text(
        "https://huggingface.co/steven0226/aerial-obb-lab-model-archive",
        encoding="utf-8",
    )

    assert release_check.verify_public_links(tmp_path) == [
        "README.md: owner Hugging Face artifact reference"
    ]


def test_recovery_notebook_requires_owner_supplied_checkpoint() -> None:
    source = (ROOT / "notebooks" / "03_recover_per_class_metrics_colab.py").read_text(
        encoding="utf-8"
    )

    assert "steven0226" not in source
    assert "hf_hub_download" not in source
    assert 'WEIGHTS = Path("/content/best.pt")' in source
    assert "請先上傳" in source


def test_release_checker_rejects_causal_nms_overclaims() -> None:
    release_check = load_release_check()

    assert release_check.unsupported_claim_errors(
        "A horizontal-box detector's NMS sees these as duplicate detections and "
        "suppresses true positives."
    ) == ["unsupported causal NMS outcome claim"]
    assert release_check.unsupported_claim_errors(
        "Ground-truth geometry is a proxy for potential HBB suppression risk."
    ) == []
    assert release_check.unsupported_claim_errors(
        "We quantify why OBB beats horizontal boxes."
    ) == ["unsupported causal NMS outcome claim"]


def test_code_only_manifest_bundles_no_third_party_artifacts() -> None:
    manifest = json.loads(
        (ROOT / "release" / "artifact-manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == 2
    assert manifest["distribution_mode"] == "code-only-byom"
    assert manifest["bundled_third_party_artifacts"] == []
    assert len(manifest["excluded_historical_artifacts"]) == 6


def test_committed_tree_contains_no_model_or_dota_visual() -> None:
    release_check = load_release_check()
    manifest = release_check.load_json(ROOT / "release" / "artifact-manifest.json")

    assert release_check.verify_code_only_paths(
        release_check.committed_paths(ROOT), manifest
    ) == []
    assert release_check.verify_artifacts(ROOT) == []


def test_private_runtime_and_absolute_user_paths_fail_closed(tmp_path: Path) -> None:
    release_check = load_release_check()

    assert release_check.verify_privacy_paths(
        [
            "notes.private.md",
            ".env",
            "interview-prep.md",
            "runs/best.pt",
            "datasets/DOTAv1/image.png",
        ]
    ) == [
        "private release member: notes.private.md",
        "private release member: .env",
        "private release member: interview-prep.md",
        "runtime release member: runs/best.pt",
        "runtime release member: datasets/DOTAv1/image.png",
    ]

    public = tmp_path / "README.md"
    public.write_text("copied from C:\\Users\\alice\\private\\result.csv", encoding="utf-8")
    assert release_check.verify_text_privacy(tmp_path, [public]) == [
        "README.md: absolute local user path"
    ]


def test_redistributed_binaries_contain_no_absolute_user_paths(tmp_path: Path) -> None:
    release_check = load_release_check()
    binary = tmp_path / "artifact.bin"
    binary.write_bytes(b"metadata=/" + b"home/alice/private/model.yaml")

    assert release_check.verify_binary_privacy(
        tmp_path,
        [binary],
    ) == ["artifact.bin: absolute local user path"]


def test_gradio_sources_use_shared_explicit_detect_flow() -> None:
    checker = load_release_check()
    assert checker.verify_gradio_interaction_sources(ROOT) == []


def test_gradio_source_policy_rejects_upload_inference() -> None:
    checker = load_release_check()
    assert checker.gradio_interaction_source_errors(
        {"legacy.py": "inp.upload(detect, [inp], [out])"}
    ) == ["legacy.py: upload-triggered inference"]


def test_preview_source_has_no_ml_or_remote_model_import() -> None:
    text = (ROOT / "demo" / "gradio_preview.py").read_text(encoding="utf-8")
    for forbidden in ("torch", "ultralytics", "huggingface_hub", "MODEL_PATH"):
        assert forbidden not in text
