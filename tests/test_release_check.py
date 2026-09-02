from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

import pytest

from scripts.prepare_demo_assets import OFFICIAL_ASSETS


ROOT = Path(__file__).resolve().parents[1]


def _copy_release_candidate(tmp_path: Path) -> Path:
    target = tmp_path / "candidate"
    manifest = json.loads(
        (ROOT / "release" / "artifact-manifest.json").read_text(encoding="utf-8")
    )
    relative_paths = {
        "release/artifact-manifest.json",
        "release/sample-gallery-sources.json",
        "demo/web/demo-model.json",
        "demo/web/third_party/yolo26n-obb-privacy-sanitization.json",
        *(entry["path"] for entry in manifest["bundled_third_party_artifacts"]),
        *(entry["path"] for entry in manifest["reviewed_public_artifacts"]),
    }
    for relative in relative_paths:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return target


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _refresh_reviewed_digest(root: Path, relative: str) -> None:
    manifest_path = root / "release" / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item for item in manifest["reviewed_public_artifacts"] if item["path"] == relative
    )
    payload = (root / relative).read_bytes()
    if entry.get("digest_mode") == "canonical-lf":
        payload = (
            payload.decode("utf-8")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .encode("utf-8")
        )
    entry["bytes"] = len(payload)
    entry["sha256"] = hashlib.sha256(payload).hexdigest()
    _write_json(manifest_path, manifest)


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


def test_browser_demo_evidence_is_genuine_local_inference_with_privacy_sanitized_derivative() -> None:
    browser = load_evidence()["browser_demo"]

    assert (
        browser["distribution_mode"]
        == "public-agpl-privacy-sanitized-demo-model-plus-byom"
    )
    assert browser["showcase_enabled"] is False
    assert browser["demo_inference_performed"] is True
    assert browser["model_bundled"] is True
    assert browser["default_demo_image"] == "demo/web/samples/harbor.jpg"
    assert (
        browser["demo_model"]
        == "demo/web/models/yolo26n-obb-privacy-sanitized.onnx"
    )
    assert browser["runtime_load"] == "lazy-on-demo-detect-or-byom-selection"
    assert "space_revision" not in browser
    assert browser["represents_fine_tuned_medium_accuracy"] is False
    assert browser["represents_t4_latency"] is False


def test_browser_demo_evidence_records_exact_single_harbor() -> None:
    browser = load_evidence()["browser_demo"]

    assert browser["demo_images"] == ["demo/web/samples/harbor.jpg"]
    assert browser["default_demo_image"] == "demo/web/samples/harbor.jpg"
    assert browser["sample_count"] == 1
    assert browser["sample_selection"] == "fixed-no-selector"
    assert browser["confidence"] == 0.25
    assert browser["per_image_tuning"] is False
    assert browser["precomputed_results"] is False
    assert browser["demo_inference_performed"] is True
    assert browser["model_bundled"] is True
    assert browser["represents_accuracy_evaluation"] is False
    assert browser["represents_t4_latency"] is False


def test_browser_demo_has_one_canonical_real_demo_source_path() -> None:
    browser = load_evidence()["browser_demo"]
    assert browser["source_files"] == [
        "demo/web/THIRD_PARTY_NOTICES.md",
        "demo/web/app.js",
        "demo/web/demo-assets.js",
        "demo/web/demo-model.json",
        "demo/web/fonts/IBM-Plex-OFL.txt",
        "demo/web/fonts/IBMPlexSansCondensed-SemiBold.woff2",
        "demo/web/index.html",
        "demo/web/models/yolo26n-obb-privacy-sanitized.onnx",
        "demo/web/obb.js",
        "demo/web/samples/harbor.jpg",
        "demo/web/style.css",
        "demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt",
        "demo/web/third_party/yolo26n-obb-privacy-sanitization.json",
        "docs/assets/browser-workbench.png",
    ]
    assert (ROOT / "demo" / "web" / "index.html").is_file()
    assert not (ROOT / "demo" / "space-static").exists()


def test_browser_ui_evidence_matches_restored_workbench() -> None:
    evidence = load_evidence()
    browser = evidence["browser_demo"]
    assert browser["model_bundled"] is True
    assert browser["language"] == "zh-TW"
    assert browser["layout"] == "workbench-31-69"
    assert browser["base_font_px"] == 19
    assert browser["minimum_secondary_text_px"] == 15
    assert browser["desktop_max_width_px"] == 1760
    assert browser["responsive_breakpoint_px"] == 960
    assert browser["corner_style"] == "square"
    assert browser["primary_action_first_viewport"] is True
    assert browser["dense_canvas_labels"] is False
    assert "docs/assets/browser-workbench.png" in browser["source_files"]
    assert "gradio_ui" not in evidence


def test_model_card_uses_current_real_demo_and_advanced_byom_path() -> None:
    text = (ROOT / "docs" / "model_card.md").read_text(encoding="utf-8")

    for retired in ("--group demo", "MODEL_PATH", "MODEL_DEVICE", "demo/app.py"):
        assert retired not in text
    assert "demo/web" in text
    assert "python.exe -m http.server 8765 --directory demo/web" in text
    for token in (
        "official aerial original",
        "privacy-sanitized YOLO26n-OBB AGPL derivative",
        "genuine local inference",
        "filters reuse cached output",
        "Advanced BYOM",
        "does not represent the fine-tuned `yolo26m-obb`",
    ):
        assert token in text


def test_owner_hf_artifacts_are_anonymously_private() -> None:
    follow_up = load_evidence()["owner_visibility_follow_up"]

    assert follow_up["historical_model_archive"]["anonymous_http_status"] == 401
    assert follow_up["historical_space"]["anonymous_api_http_status"] == 401
    assert follow_up["historical_space"]["anonymous_page_http_status"] == 401
    assert follow_up["remote_mutation_performed_by_local_workflow"] is False


def test_tracked_text_tree_omits_retired_owner_handle() -> None:
    retired_handle = "steven" + "0226"
    hits: list[str] = []

    for relative in load_release_check().committed_paths(ROOT):
        path = ROOT / relative
        if not path.is_file():
            continue
        if path.suffix.casefold() not in {
            ".css",
            ".html",
            ".js",
            ".json",
            ".md",
            ".py",
            ".toml",
            ".txt",
            ".yaml",
            ".yml",
        }:
            continue
        if retired_handle.casefold() in path.read_text(encoding="utf-8", errors="ignore").casefold():
            hits.append(relative)

    assert hits == []


def test_tracked_text_tree_omits_private_hf_repo_identifiers() -> None:
    private_identifiers = (
        "aerial-obb-lab-" + "model-archive",
        "yolo26m-obb-" + "dota",
        "yolo26-obb-" + "aerial-detection",
        "dotav1-" + "split-cache",
    )
    hits: list[str] = []

    for relative in load_release_check().committed_paths(ROOT):
        path = ROOT / relative
        if not path.is_file():
            continue
        if path.suffix.casefold() not in {
            ".css",
            ".html",
            ".js",
            ".json",
            ".md",
            ".py",
            ".toml",
            ".txt",
            ".yaml",
            ".yml",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for identifier in private_identifiers:
            pattern = re.compile(
                rf"(?<![A-Za-z0-9._-]){re.escape(identifier)}(?![A-Za-z0-9._-])",
                re.I,
            )
            if pattern.search(text):
                hits.append(f"{relative}: {identifier}")

    assert hits == []


def test_release_evidence_and_claim_blocks_verify() -> None:
    release_check = load_release_check()

    assert release_check.verify_evidence(ROOT) == []
    assert release_check.verify_claims(ROOT) == []


def test_public_presentation_omits_owner_hf_artifact_links() -> None:
    release_check = load_release_check()

    assert release_check.verify_public_links(ROOT) == []


def test_changelog_scopes_retired_bundled_model_to_rc1() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert "The static browser demo uses the official nano model" not in changelog
    assert "The rc.1 static browser demo used the official nano model" in changelog


def test_public_presentation_rejects_current_private_hf_archive(tmp_path: Path) -> None:
    release_check = load_release_check()
    release_check.PUBLIC_PRESENTATION_FILES = ("README.md",)
    private_identifier = "aerial-obb-lab-" + "model-archive"
    (tmp_path / "README.md").write_text(
        f"https://huggingface.co/private-owner/{private_identifier}",
        encoding="utf-8",
    )

    assert release_check.verify_public_links(tmp_path) == [
        "README.md: owner Hugging Face artifact reference"
    ]


def test_public_readme_rejects_missing_workflow_badge_target(tmp_path: Path) -> None:
    release_check = load_release_check()
    release_check.PUBLIC_PRESENTATION_FILES = ("README.md",)
    (tmp_path / "README.md").write_text(
        "[![CI](https://github.com/example/project/actions/workflows/ci.yml/badge.svg)]"
        "(https://github.com/example/project/actions/workflows/ci.yml)",
        encoding="utf-8",
    )

    assert release_check.verify_public_links(tmp_path) == [
        "README.md: workflow badge target does not exist: .github/workflows/ci.yml"
    ]


@pytest.mark.parametrize("retired_reference", ["Gradio", "demo/space/", "demo/space-static/"])
def test_public_readme_rejects_retired_demo_surface(
    tmp_path: Path, retired_reference: str
) -> None:
    release_check = load_release_check()
    release_check.PUBLIC_PRESENTATION_FILES = ("README.md",)
    (tmp_path / "README.md").write_text(retired_reference, encoding="utf-8")

    assert release_check.verify_public_links(tmp_path) == [
        f"README.md: retired demo surface reference: {retired_reference}"
    ]


def test_recovery_notebook_requires_owner_supplied_checkpoint() -> None:
    source = (ROOT / "notebooks" / "03_recover_per_class_metrics_colab.py").read_text(
        encoding="utf-8"
    )

    assert "huggingface.co/" not in source
    assert "hf_hub_download" not in source
    assert 'WEIGHTS = Path("/content/best.pt")' in source
    assert "請先上傳" in source


def test_historical_gpu_smoke_has_no_remote_write_path() -> None:
    source = (ROOT / "scripts" / "smoke_test.py").read_text(encoding="utf-8")

    for retired_remote_surface in (
        "--allow-remote-writes",
        "huggingface_hub",
        "hf_checkpoint",
        "HF_PUSH",
    ):
        assert retired_remote_surface not in source
    assert '"--acknowledge-historical-gpu-workflow"' in source


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


def test_analysis_sources_scope_phantom_overlap_as_a_proxy() -> None:
    sources = {
        relative: (ROOT / relative).read_text(encoding="utf-8")
        for relative in (
            "src/obbkit/analysis.py",
            "scripts/obb_analysis.py",
            "docs/DESIGN_NOTES.md",
        )
    }

    assert "exactly the detections" not in sources["src/obbkit/analysis.py"]
    assert "detections an HBB NMS would wrongly suppress" not in sources["scripts/obb_analysis.py"]
    assert "直接對應到 NMS 會不會誤殺" not in sources["docs/DESIGN_NOTES.md"]
    for text in sources.values():
        assert "proxy" in text.casefold()


def test_geometry_overall_mean_is_weighted_and_readme_scoped() -> None:
    evidence = load_evidence()["geometry_analysis"]
    analysis = json.loads(
        (ROOT / "docs" / "analysis_results.json").read_text(encoding="utf-8")
    )["inflation"]
    total = sum(int(row["n"]) for row in analysis.values())
    weighted = sum(int(row["n"]) * float(row["mean"]) for row in analysis.values()) / total

    assert evidence["overall_inflation_mean"] == pytest.approx(weighted)
    assert round(weighted, 2) == 1.76

    canonical = (ROOT / "README.md").read_text(encoding="utf-8")
    english = (ROOT / "README.en.md").read_text(encoding="utf-8")
    analysis_doc = (ROOT / "docs" / "analysis_results.md").read_text(encoding="utf-8")
    generator = (ROOT / "scripts" / "obb_analysis.py").read_text(encoding="utf-8")
    assert "全體 weighted mean" in canonical and "**1.76×**" in canonical
    assert "bridge mean **2.43×**" in english and "overall weighted mean **1.76×**" in english
    assert "Overall weighted mean across all objects: **1.76x**" in analysis_doc
    assert "overall_mean" in generator
    assert '"--acknowledge-dota-academic-use"' in generator
    assert "acknowledge_dota_academic_use" in generator


def test_real_demo_manifest_records_exact_public_artifacts() -> None:
    manifest = json.loads(
        (ROOT / "release" / "artifact-manifest.json").read_text(encoding="utf-8")
    )
    evidence = load_evidence()

    assert manifest["schema_version"] == 2
    assert (
        manifest["release_candidate"]
        == evidence["release_candidate"]
        == "unreleased-pages-candidate"
    )
    assert (
        manifest["distribution_mode"]
        == "public-agpl-privacy-sanitized-demo-model-plus-byom"
    )
    assert manifest["policy"]["commercial_use_cleared"] is False
    bundled = {entry["path"]: entry for entry in manifest["bundled_third_party_artifacts"]}
    assert set(bundled) == {
        "demo/web/fonts/IBMPlexSansCondensed-SemiBold.woff2",
        "demo/web/models/yolo26n-obb-privacy-sanitized.onnx",
        "demo/web/samples/harbor.jpg",
        "demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt",
    }
    model = bundled["demo/web/models/yolo26n-obb-privacy-sanitized.onnx"]
    assert model["bytes"] == 10207127
    assert model["sha256"] == "a0a1a2dd357067e8c6c9f5ce7bb33487188423f9722e813be880da4f9badcd97"
    assert model["modification_status"] == "metadata-only"
    assert model["source_sha256"] == "02f7c539600296d7389341280beb82da810b15dc09c54cf2bc70f7f610331b38"
    assert model["source_sha256"] != model["sha256"]
    reviewed = {entry["path"] for entry in manifest["reviewed_public_artifacts"]}
    assert {
        "demo/web/app.js",
        "demo/web/index.html",
        "demo/web/style.css",
        "demo/web/third_party/yolo26n-obb-privacy-sanitization.json",
        "docs/assets/browser-workbench.png",
    } <= reviewed
    assert len(manifest["excluded_historical_artifacts"]) == 6


def test_real_demo_manifest_records_exact_single_harbor_artifact() -> None:
    release_check = load_release_check()
    assert release_check.verify_artifacts(ROOT) == []
    receipt = json.loads(
        (ROOT / "release/sample-gallery-sources.json").read_text(encoding="utf-8")
    )["samples"]
    demo = json.loads(
        (ROOT / "demo/web/demo-model.json").read_text(encoding="utf-8")
    )["samples"]
    artifacts = json.loads(
        (ROOT / "release/artifact-manifest.json").read_text(encoding="utf-8")
    )["bundled_third_party_artifacts"]
    assert receipt == demo
    assert len(receipt) == 1
    harbor = receipt[0]
    bundled = [entry for entry in artifacts if entry["path"].startswith("demo/web/samples/")]
    assert len(bundled) == 1
    entry = bundled[0]
    assert (entry["path"], entry["sample_id"], entry["sample_title"]) == (
        "demo/web/samples/harbor.jpg", "harbor", "低密度港區航拍範例"
    )
    assert (entry["bytes"], entry["sha256"], entry["alt"]) == (
        harbor["bytes"], harbor["sha256"], harbor["alt"]
    )
    assert entry["derivation"] == harbor["derivation"]
    assert entry["guardrails"] == harbor["guardrails"]
    assert entry["source_product_id"] == harbor["source"]["productId"]


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("source_acquisition_date", 0),
        ("alt", "unreviewed alternative text"),
        ("guardrails", {}),
        (
            "restrictions",
            ["Accuracy evidence; USDA endorsement is implied."],
        ),
        ("provenance", "unreviewed provenance"),
    ],
)
def test_gallery_manifest_rejects_unreviewed_receipt_or_claim_field(
    tmp_path: Path, field: str, mutation: object
) -> None:
    release_check = load_release_check()
    root = _copy_release_candidate(tmp_path)
    manifest_path = root / "release" / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in manifest["bundled_third_party_artifacts"]
        if item["path"] == "demo/web/samples/harbor.jpg"
    )
    entry[field] = mutation
    _write_json(manifest_path, manifest)

    assert release_check.verify_artifacts(root) == [
        "demo/web/samples/harbor.jpg: manifest NAIP record differs from receipt"
    ]


def test_notices_record_single_public_domain_harbor_derivation() -> None:
    root_notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    demo_notice = (ROOT / "demo" / "web" / "THIRD_PARTY_NOTICES.md").read_text(
        encoding="utf-8"
    )

    for notice in (root_notice, demo_notice):
        for token in (
            "低密度港區航拍範例",
            "samples/harbor.jpg",
            "m_3411955_sw_11_060_20220514",
            "2022-05-14",
            "USDA",
            "Public Domain",
            "[-119.216719, 34.14417, -119.200719, 34.15417]",
            "crop/resample/metadata removal",
            "241046",
            "916a8f11717545b0796cf0ca563d6228c2cc14f02124c9d8639dd26a753ea6f0",
        ):
            assert token in notice
    assert "No USGS or USDA endorsement is implied" in demo_notice
    assert "AGPL-3.0-only" in demo_notice


def test_current_readmes_describe_fixed_harbor_detect_journey() -> None:
    chinese = (ROOT / "README.md").read_text(encoding="utf-8")
    english = (ROOT / "README.en.md").read_text(encoding="utf-8")

    for token in ("固定的 public-domain NAIP 港區原圖", "真正 inference", "查看原圖／查看結果", "BYOM"):
        assert token in chinese
    assert "不代表" in chinese
    for token in ("fixed public-domain NAIP harbor original", "genuine local inference", "cached", "BYOM"):
        assert token in english
    assert "does not represent" in english


def test_release_checklist_records_completed_clean_history_publication() -> None:
    checklist = (ROOT / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")

    for token in (
        "clean root commit",
        "admin enforcement and linear history enabled",
        "force pushes",
        "branch deletion are disabled",
        "Historical v1.0.0",
        "privacy-sanitized nano derivative",
        "live-demo browser checks remain a separate authorized remote gate",
    ):
        assert token in checklist
    assert "publish the reviewed code-only tree from a clean root commit" in checklist
    assert "[x] Restore branch protection" in checklist
    for completed in (
        "[x] Current candidate: run the complete pytest suite",
        "[x] Current candidate: build and verify the strict committed clean export",
        "[x] Current candidate: repeat the complete local CPU, browser, artifact, license, privacy, and",
    ):
        assert completed in checklist
    assert (
        "[ ] Current candidate: complete the final branch audit and whole-branch review"
        in checklist
    )
    assert (
        "[ ] Hosted Ubuntu CPU, Windows CPU, and live-demo browser checks"
        in checklist
    )


def test_release_license_source_matches_the_pinned_acquisition_spec() -> None:
    release_check = load_release_check()
    acquisition = next(
        spec for spec in OFFICIAL_ASSETS if spec.asset_id == "ultralytics-license"
    )
    manifest = release_check.load_json(ROOT / "release" / "artifact-manifest.json")
    license_entry = next(
        item
        for item in manifest["bundled_third_party_artifacts"]
        if item["path"] == release_check.APPROVED_MODEL_LICENSE_FILE
    )

    assert release_check.APPROVED_MODEL_LICENSE_SOURCE_URL == acquisition.source_url
    assert license_entry["source_url"] == acquisition.source_url
    assert release_check.verify_artifacts(ROOT) == []


def test_committed_tree_contains_only_the_approved_demo_model_and_no_dota_visual() -> None:
    release_check = load_release_check()
    manifest = release_check.load_json(ROOT / "release" / "artifact-manifest.json")

    assert release_check.verify_code_only_paths(
        release_check.committed_paths(ROOT), manifest
    ) == []
    assert release_check.verify_artifacts(ROOT) == []


def test_model_release_exception_is_exact_manifest_bound_and_source_safe() -> None:
    release_check = load_release_check()
    approved = "demo/web/models/yolo26n-obb-privacy-sanitized.onnx"
    manifest = {
        "bundled_third_party_artifacts": [
            {
                "path": approved,
                "bytes": 10207127,
                "sha256": "a0a1a2dd357067e8c6c9f5ce7bb33487188423f9722e813be880da4f9badcd97",
                "source_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.onnx",
                "source_sha256": "02f7c539600296d7389341280beb82da810b15dc09c54cf2bc70f7f610331b38",
                "modification_status": "metadata-only",
                "modification_date": "2026-08-31",
                "sanitization_record": "demo/web/third_party/yolo26n-obb-privacy-sanitization.json",
                "license": "AGPL-3.0-only",
                "license_file": "demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt",
            }
        ],
        "excluded_historical_artifacts": [],
    }

    assert release_check.verify_code_only_paths([approved], manifest) == []
    assert release_check.verify_code_only_paths(
        [approved, "demo/web/models/second.onnx"], manifest
    ) == ["public release contains unapproved model binary: demo/web/models/second.onnx"]
    manifest["bundled_third_party_artifacts"][0]["sha256"] = release_check.SOURCE_MODEL_SHA256
    assert release_check.verify_code_only_paths([approved], manifest) == [
        f"public release model exception is not exact: {approved}"
    ]


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("license", "MIT"),
        ("license_file", "demo/web/third_party/OTHER.txt"),
        ("sanitization_record", "demo/web/third_party/other.json"),
        ("modification_date", "2026-09-01"),
        ("source_url", "https://example.invalid/model.onnx"),
        ("source_sha256", "0" * 64),
        ("license", None),
        ("source_url", None),
    ],
)
def test_release_artifact_contract_rejects_derivative_identity_mutation(
    tmp_path: Path, field: str, mutation: str | None
) -> None:
    release_check = load_release_check()
    root = _copy_release_candidate(tmp_path)
    manifest_path = root / "release" / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in manifest["bundled_third_party_artifacts"]
        if item["path"] == release_check.APPROVED_DEMO_MODEL
    )
    if mutation is None:
        entry.pop(field)
    else:
        entry[field] = mutation
    _write_json(manifest_path, manifest)

    errors = release_check.verify_artifacts(root)
    assert (
        f"public release model exception is not exact: {release_check.APPROVED_DEMO_MODEL}"
        in errors
    )
    assert (
        f"{release_check.APPROVED_DEMO_MODEL}: exact manifest-bound model entry is missing"
        in errors
    )
    if mutation is None:
        assert f"{release_check.APPROVED_DEMO_MODEL}: missing artifact field {field}" in errors


@pytest.mark.parametrize(
    ("document", "section", "field", "mutation", "expected_error"),
    [
        (
            "demo/web/demo-model.json",
            "model",
            "license",
            "MIT",
            "demo-model.json: model provenance contract differs from artifact manifest",
        ),
        (
            "demo/web/demo-model.json",
            "sanitization",
            "path",
            "third_party/other.json",
            "demo-model.json: sanitization contract differs from artifact manifest",
        ),
        (
            "demo/web/third_party/yolo26n-obb-privacy-sanitization.json",
            "license",
            "spdx",
            "MIT",
            "sanitization receipt: license contract differs from artifact manifest",
        ),
        (
            "demo/web/third_party/yolo26n-obb-privacy-sanitization.json",
            "source",
            "url",
            "https://example.invalid/model.onnx",
            "sanitization receipt: source identity differs from artifact manifest",
        ),
        (
            "demo/web/third_party/yolo26n-obb-privacy-sanitization.json",
            "transformation",
            "modifiedField",
            "ModelProto.graph.name",
            "sanitization receipt: modification record is incomplete",
        ),
    ],
)
def test_release_artifact_contract_rejects_cross_record_identity_mutation(
    tmp_path: Path,
    document: str,
    section: str,
    field: str,
    mutation: str,
    expected_error: str,
) -> None:
    release_check = load_release_check()
    root = _copy_release_candidate(tmp_path)
    document_path = root / document
    payload = json.loads(document_path.read_text(encoding="utf-8"))
    payload[section][field] = mutation
    _write_json(document_path, payload)
    if document.startswith("demo/web/third_party/"):
        _refresh_reviewed_digest(root, document)

    errors = release_check.verify_artifacts(root)
    assert expected_error in errors


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("digest_mode", None),
        ("digest_mode", "binary"),
        ("sha256", "0" * 64),
    ],
)
def test_release_artifact_contract_rejects_license_digest_mutation(
    tmp_path: Path, field: str, mutation: str | None
) -> None:
    release_check = load_release_check()
    root = _copy_release_candidate(tmp_path)
    manifest_path = root / "release" / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in manifest["bundled_third_party_artifacts"]
        if item["path"] == release_check.APPROVED_MODEL_LICENSE_FILE
    )
    if mutation is None:
        entry.pop(field, None)
    else:
        entry[field] = mutation
    _write_json(manifest_path, manifest)

    assert (
        f"{release_check.APPROVED_MODEL_LICENSE_FILE}: canonical-LF license identity is not exact"
        in release_check.verify_artifacts(root)
    )


@pytest.mark.parametrize(
    ("artifact_path", "digest_mode"),
    [
        (
            "demo/web/models/yolo26n-obb-privacy-sanitized.onnx",
            "canonical-lf",
        ),
        (
            "demo/web/models/yolo26n-obb-privacy-sanitized.onnx",
            "unexpected-text-mode",
        ),
        (
            "demo/web/models/yolo26n-obb-privacy-sanitized.onnx",
            None,
        ),
        ("demo/web/samples/harbor.jpg", "canonical-lf"),
    ],
)
def test_release_artifact_contract_rejects_binary_digest_mode_misuse(
    tmp_path: Path, artifact_path: str, digest_mode: str | None
) -> None:
    release_check = load_release_check()
    root = _copy_release_candidate(tmp_path)
    manifest_path = root / "release" / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in manifest["bundled_third_party_artifacts"]
        if item["path"] == artifact_path
    )
    entry["digest_mode"] = digest_mode
    _write_json(manifest_path, manifest)

    assert (
        f"{artifact_path}: binary artifact digest_mode must be absent or raw-binary"
        in release_check.verify_artifacts(root)
    )


def test_release_artifact_contract_accepts_explicit_raw_binary_model_mode(
    tmp_path: Path,
) -> None:
    release_check = load_release_check()
    root = _copy_release_candidate(tmp_path)
    manifest_path = root / "release" / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in manifest["bundled_third_party_artifacts"]
        if item["path"] == release_check.APPROVED_DEMO_MODEL
    )
    entry["digest_mode"] = "raw-binary"
    _write_json(manifest_path, manifest)

    assert release_check.verify_artifacts(root) == []


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


def test_committed_privacy_ignores_paths_deleted_in_the_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_check = load_release_check()
    monkeypatch.setattr(
        release_check,
        "committed_paths",
        lambda _root: ["README.md", "deleted-demo.py"],
    )
    (tmp_path / "README.md").write_text("public documentation", encoding="utf-8")
    (tmp_path / "release").mkdir()
    (tmp_path / "release" / "artifact-manifest.json").write_text(
        '{"bundled_third_party_artifacts": []}', encoding="utf-8"
    )

    assert release_check.verify_committed_privacy(tmp_path) == []


def test_committed_privacy_rejects_token_in_tracked_superpowers_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_check = load_release_check()
    relative = "docs/superpowers/plan.md"
    monkeypatch.setattr(release_check, "committed_paths", lambda _root: [relative])
    document = tmp_path / relative
    document.parent.mkdir(parents=True)
    document.write_text("token=ghp_" + "x" * 24, encoding="utf-8")
    (tmp_path / "release").mkdir()
    (tmp_path / "release" / "artifact-manifest.json").write_text(
        '{"bundled_third_party_artifacts": []}', encoding="utf-8"
    )

    assert release_check.verify_committed_privacy(tmp_path) == [
        "docs/superpowers/plan.md: token-shaped secret"
    ]


def test_redistributed_binaries_contain_no_absolute_user_paths(tmp_path: Path) -> None:
    release_check = load_release_check()
    binary = tmp_path / "artifact.bin"
    binary.write_bytes(b"metadata=/" + b"home/alice/private/model.yaml")

    assert release_check.verify_binary_privacy(
        tmp_path,
        [binary],
    ) == ["artifact.bin: absolute local user path"]
