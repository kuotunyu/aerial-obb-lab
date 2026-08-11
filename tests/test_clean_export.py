from __future__ import annotations

from scripts.clean_export_check import archive_policy_errors


def test_archive_policy_rejects_private_and_runtime_paths() -> None:
    assert archive_policy_errors(
        ["README.md", "notes.private.md", "runs/x/best.pt", "datasets/DOTAv1/a.png"]
    ) == [
        "private path: notes.private.md",
        "runtime/model path: runs/x/best.pt",
        "runtime/model path: datasets/DOTAv1/a.png",
    ]


def test_archive_policy_rejects_unsafe_tar_members() -> None:
    assert archive_policy_errors(["../escape.txt", "/absolute.txt", "C:/Users/alice/file.txt"]) == [
        "unsafe path: ../escape.txt",
        "unsafe path: /absolute.txt",
        "unsafe path: C:/Users/alice/file.txt",
    ]


def test_archive_policy_accepts_release_files() -> None:
    assert archive_policy_errors(
        [
            "README.md",
            "src/obbkit/__init__.py",
            "demo/space-static/yolo26n-obb.onnx",
            "assets/hbb_vs_obb_1_P0706_ship.jpg",
        ]
    ) == []
