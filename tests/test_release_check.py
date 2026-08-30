from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re

import pytest


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


def test_browser_showcase_evidence_is_explicit_and_model_free() -> None:
    browser = load_evidence()["browser_demo"]

    assert browser["showcase_enabled"] is True
    assert browser["showcase_fixture"] == "demo/web/fixtures/showcase.svg"
    assert browser["showcase_image"] == "authored synthetic SVG"
    assert browser["showcase_inference_performed"] is False
    assert browser["showcase_runtime_label"] == "N/A · no inference"
    assert browser["showcase_external_runtime_requests"] is False
    assert browser["runtime_load"] == "lazy-on-byom-selection"
    assert "demo/web/showcase-fixture.js" in browser["source_files"]
    assert "demo/web/fixtures/showcase.svg" in browser["source_files"]
    assert "tests/fixtures/browser-smoke.svg" not in browser["source_files"]


def test_browser_demo_has_one_canonical_source_path() -> None:
    browser = load_evidence()["browser_demo"]
    assert browser["source_files"] == [
        "demo/web/app.js",
        "demo/web/fixtures/showcase.svg",
        "demo/web/fonts/IBM-Plex-OFL.txt",
        "demo/web/fonts/IBMPlexSansCondensed-SemiBold.woff2",
        "demo/web/index.html",
        "demo/web/obb.js",
        "demo/web/showcase-fixture.js",
        "demo/web/style.css",
        "docs/assets/browser-workbench.png",
    ]
    assert (ROOT / "demo" / "web" / "index.html").is_file()
    assert not (ROOT / "demo" / "space-static").exists()


def test_browser_ui_evidence_has_no_bundled_model_or_gradio_surface() -> None:
    evidence = load_evidence()
    browser = evidence["browser_demo"]
    assert browser["model_bundled"] is False
    assert browser["language"] == "zh-TW"
    assert browser["layout"] == "workbench-34-66"
    assert browser["base_font_px"] == 19
    assert browser["minimum_secondary_text_px"] == 15
    assert browser["desktop_max_width_px"] == 1760
    assert browser["responsive_breakpoint_px"] == 900
    assert browser["corner_style"] == "square"
    assert browser["primary_action_first_viewport"] is True
    assert browser["dense_canvas_labels"] is False
    assert "docs/assets/browser-workbench.png" in browser["source_files"]
    assert "gradio_ui" not in evidence


def test_model_card_uses_current_browser_byom_path() -> None:
    text = (ROOT / "docs" / "model_card.md").read_text(encoding="utf-8")

    for retired in ("--group demo", "MODEL_PATH", "MODEL_DEVICE", "demo/app.py"):
        assert retired not in text
    assert "demo/web" in text
    assert "python.exe -m http.server 8765 --directory demo/web" in text


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


def test_code_only_manifest_bundles_only_licensed_display_font() -> None:
    manifest = json.loads(
        (ROOT / "release" / "artifact-manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == 2
    assert manifest["distribution_mode"] == "code-only-byom"
    assert manifest["bundled_third_party_artifacts"] == [
        {
            "path": "demo/web/fonts/IBMPlexSansCondensed-SemiBold.woff2",
            "bytes": 66040,
            "sha256": "385a082a1eac88343eab01fb6746be04b7175dacaf4550b17dee76ea0f78126d",
            "kind": "self-hosted web font",
            "provenance": "Unmodified IBM Plex Sans Condensed SemiBold WOFF2 from @ibm/plex-sans-condensed@2.0.0.",
            "source_url": "https://github.com/IBM/plex/releases/tag/%40ibm%2Fplex-sans-condensed%402.0.0",
            "upstream_git_head": "bb3ab6404e1881ea286f8742dc839e09057db6dd",
            "upstream_integrity": "sha512-dzgR4Npf/JJMiTYf6iOBQJpTDQfllZFLN0A0FkW5gtWhNr9JeQNvRrIRwJvbZHfL0I8wae8kIhO/ukYdeXW54g==",
            "license": "OFL-1.1",
            "license_file": "demo/web/fonts/IBM-Plex-OFL.txt",
            "restrictions": [
                "Keep the copyright notice and SIL Open Font License 1.1 with redistributed copies.",
                "Reserved Font Name Plex applies if the font is modified.",
            ],
        }
    ]
    assert len(manifest["excluded_historical_artifacts"]) == 6


def test_release_checklist_records_completed_clean_history_publication() -> None:
    checklist = (ROOT / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")

    for token in (
        "clean root commit",
        "admin enforcement and linear history enabled",
        "force pushes",
        "branch deletion are disabled",
        "Hosted Ubuntu CPU, Windows CPU, and synthetic browser checks pass",
    ):
        assert token in checklist
    assert "[x] Publish the reviewed code-only tree from a clean root commit" in checklist
    assert "[x] Restore branch protection" in checklist


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


def test_redistributed_binaries_contain_no_absolute_user_paths(tmp_path: Path) -> None:
    release_check = load_release_check()
    binary = tmp_path / "artifact.bin"
    binary.write_bytes(b"metadata=/" + b"home/alice/private/model.yaml")

    assert release_check.verify_binary_privacy(
        tmp_path,
        [binary],
    ) == ["artifact.bin: absolute local user path"]
