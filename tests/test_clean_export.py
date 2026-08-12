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
            "demo/space-static/yolo26n-obb.onnx",
            "assets/hbb_vs_obb_1_P0706_ship.jpg",
        ]
    ) == [
        "model binary path: demo/space-static/yolo26n-obb.onnx",
        "DOTA-derived visual path: assets/hbb_vs_obb_1_P0706_ship.jpg",
    ]


def test_archive_policy_accepts_code_only_release_files() -> None:
    assert archive_policy_errors(
        ["README.md", "src/obbkit/__init__.py", "tests/fixtures/browser-smoke.svg"]
    ) == []


def test_clean_export_keeps_its_own_gate_and_browser_fixture() -> None:
    assert {
        "README.en.md",
        "demo/gradio.css",
        "demo/gradio_preview.py",
        "demo/gradio_ui.py",
        "demo/model_source.py",
        "demo/space-static/app.js",
        "demo/space-static/index.html",
        "demo/ui_contract.py",
        "docs/OWNER_ACTIONS.md",
        "scripts/clean_export_check.py",
        "scripts/browser_smoke.py",
        "scripts/gradio_ui_smoke.py",
        "tests/fixtures/browser-smoke.svg",
    } <= REQUIRED_MEMBERS


def test_clean_export_default_is_rc2() -> None:
    assert DEFAULT_OUTPUT.name == "yolo26-dota-obb-v1.0.0rc2.zip"


def test_distribution_selector_accepts_uv_build_marker_only(tmp_path) -> None:
    (tmp_path / ".gitignore").write_text("*", encoding="utf-8")
    (tmp_path / "package-1.0.tar.gz").write_bytes(b"sdist")
    (tmp_path / "package-1.0-py3-none-any.whl").write_bytes(b"wheel")

    assert [path.name for path in distribution_paths(tmp_path)] == [
        "package-1.0-py3-none-any.whl",
        "package-1.0.tar.gz",
    ]
