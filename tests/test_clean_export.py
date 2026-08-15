from __future__ import annotations

import inspect

from scripts.clean_export_check import (
    DEFAULT_OUTPUT,
    REQUIRED_MEMBERS,
    archive_policy_errors,
    distribution_paths,
    verify_snapshot,
)


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


def test_archive_policy_rejects_model_and_dota_visuals() -> None:
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


def test_archive_policy_accepts_code_only_release_files() -> None:
    assert archive_policy_errors(
        ["README.md", "src/obbkit/__init__.py", "tests/fixtures/browser-smoke.svg"]
    ) == []


def test_clean_export_keeps_its_own_gate_and_browser_fixture() -> None:
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
        "tests/fixtures/browser-smoke.svg",
    } <= REQUIRED_MEMBERS
    assert not any("gradio" in member.casefold() for member in REQUIRED_MEMBERS)


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
