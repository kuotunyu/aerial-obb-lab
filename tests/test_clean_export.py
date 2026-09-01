from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import zipfile

from scripts.clean_export_check import (
    DEFAULT_OUTPUT,
    REQUIRED_MEMBERS,
    archive_policy_errors,
    distribution_paths,
    verify_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]


def test_archive_policy_rejects_private_and_runtime_paths() -> None:
    assert archive_policy_errors(
        ["README.md", "notes.private.md", "runs/x/best.pt", "datasets/DOTAv1/a.png"]
    ) == [
        "private path: notes.private.md",
        "runtime/model path: runs/x/best.pt",
        "runtime/model path: datasets/DOTAv1/a.png",
    ]


def test_archive_policy_rejects_internal_design_tool_files() -> None:
    assert archive_policy_errors(
        [
            "PRODUCT.md",
            "DESIGN.md",
            ".claude/launch.json",
            ".impeccable/design.json",
            "docs/superpowers/plans/private-plan.md",
        ]
    ) == [
        "internal-only path: PRODUCT.md",
        "internal-only path: DESIGN.md",
        "internal-only path: .claude/launch.json",
        "internal-only path: .impeccable/design.json",
        "internal-only path: docs/superpowers/plans/private-plan.md",
    ]


def test_archive_policy_rejects_obsolete_operational_surfaces() -> None:
    retired = [
        "README.zh-TW.md",
        "docs/OWNER_ACTIONS.md",
        "docs/PLAN.md",
        "src/obbkit/hf_checkpoint.py",
    ]

    assert archive_policy_errors(retired) == [
        f"obsolete public surface: {relative}" for relative in retired
    ]


def test_archive_policy_rejects_unsafe_tar_members() -> None:
    fake_home_path = "C:/" + "Users/alice/file.txt"
    assert archive_policy_errors(["../escape.txt", "/absolute.txt", fake_home_path]) == [
        "unsafe path: ../escape.txt",
        "unsafe path: /absolute.txt",
        f"unsafe path: {fake_home_path}",
    ]


def test_clean_export_has_a_snapshot_rebuild_gate() -> None:
    assert callable(verify_snapshot)
    assert "run_browser" in inspect.signature(verify_snapshot).parameters


def test_archive_policy_rejects_unapproved_model_and_dota_visuals() -> None:
    assert archive_policy_errors(
        [
            "README.md",
            "demo/web/yolo26n-obb.onnx",
            "assets/hbb_vs_obb_1_P0706_ship.jpg",
        ]
    ) == [
        "model binary path: demo/web/yolo26n-obb.onnx",
        "DOTA-derived visual path: assets/hbb_vs_obb_1_P0706_ship.jpg",
    ]


def test_clean_export_admits_only_the_reviewed_derivative_demo_model() -> None:
    approved_demo_model = "demo/web/models/yolo26n-obb-privacy-sanitized.onnx"
    manifest = {
        "bundled_third_party_artifacts": [
            {
                "path": approved_demo_model,
                "bytes": 10207127,
                "sha256": "a0a1a2dd357067e8c6c9f5ce7bb33487188423f9722e813be880da4f9badcd97",
            }
        ]
    }

    assert archive_policy_errors(["README.md", approved_demo_model], manifest) == []
    assert archive_policy_errors(
        ["README.md", approved_demo_model, "demo/web/models/second.onnx"],
        manifest,
    ) == ["model binary path: demo/web/models/second.onnx"]
    assert archive_policy_errors(["README.md", approved_demo_model], {}) == [
        f"model binary path: {approved_demo_model}"
    ]


def test_clean_export_keeps_its_own_gate_and_real_demo_assets() -> None:
    assert {
        "README.en.md",
        "demo/web/app.js",
        "demo/web/fonts/IBM-Plex-OFL.txt",
        "demo/web/fonts/IBMPlexSansCondensed-SemiBold.woff2",
        "demo/web/index.html",
        "demo/web/obb.js",
        "demo/web/style.css",
        "docs/assets/browser-workbench.png",
        "scripts/clean_export_check.py",
        "scripts/browser_smoke.py",
        "demo/web/demo-assets.js",
        "demo/web/demo-model.json",
        "demo/web/models/yolo26n-obb-privacy-sanitized.onnx",
        "demo/web/samples/boats.jpg",
        "demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt",
        "demo/web/third_party/yolo26n-obb-privacy-sanitization.json",
    } <= REQUIRED_MEMBERS
    assert not any("gradio" in member.casefold() for member in REQUIRED_MEMBERS)


def test_release_archive_keeps_real_demo_assets_and_omits_internal_docs(
    tmp_path: Path,
) -> None:
    attributes = ROOT / ".gitattributes"
    assert attributes.is_file(), "release archive attributes are missing"

    repository = tmp_path / "repository"
    repository.mkdir()
    members = {
        ".gitattributes": attributes.read_bytes(),
        "README.md": b"public release\n",
        "demo/web/samples/boats.jpg": b"reviewed image placeholder\n",
        "docs/superpowers/plans/2026-08-31-aerial-obb-pages-showcase.md": b"internal plan\n",
        "docs/superpowers/specs/2026-08-31-aerial-obb-pages-showcase-design.md": b"internal spec\n",
    }
    for relative, payload in members.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Release Test"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "release-test@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repository, check=True)
    archive = tmp_path / "release.zip"
    subprocess.run(
        ["git", "archive", "--format=zip", f"--output={archive}", "HEAD"],
        cwd=repository,
        check=True,
    )

    with zipfile.ZipFile(archive) as bundle:
        archived = {info.filename for info in bundle.infolist() if not info.is_dir()}

    assert "demo/web/samples/boats.jpg" in archived
    assert not any(name.startswith("docs/superpowers/") for name in archived)


def test_clean_export_omits_obsolete_operational_surfaces() -> None:
    for retired in (
        "README.zh-TW.md",
        "docs/OWNER_ACTIONS.md",
        "docs/PLAN.md",
        "src/obbkit/hf_checkpoint.py",
    ):
        assert retired not in REQUIRED_MEMBERS


def test_clean_export_default_is_stable_v1() -> None:
    assert DEFAULT_OUTPUT.name == "aerial-obb-lab-v1.0.0.zip"


def test_distribution_selector_accepts_uv_build_marker_only(tmp_path) -> None:
    (tmp_path / ".gitignore").write_text("*", encoding="utf-8")
    (tmp_path / "package-1.0.tar.gz").write_bytes(b"sdist")
    (tmp_path / "package-1.0-py3-none-any.whl").write_bytes(b"wheel")

    assert [path.name for path in distribution_paths(tmp_path)] == [
        "package-1.0-py3-none-any.whl",
        "package-1.0.tar.gz",
    ]
