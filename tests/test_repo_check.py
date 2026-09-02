from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from scripts import repo_check


def test_repository_files_omits_tracked_paths_deleted_from_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    deleted = tmp_path / "retired.json"
    deleted.write_text("{}", encoding="utf-8")
    subprocess.run(["git", "add", "--", "retired.json"], cwd=tmp_path, check=True)
    deleted.unlink()
    monkeypatch.setattr(repo_check, "ROOT", tmp_path)

    assert deleted not in repo_check.repository_files()


def test_markdown_link_gate_ignores_inline_and_fenced_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repo_check, "ROOT", tmp_path)
    document = tmp_path / "docs" / "design.md"
    document.parent.mkdir()
    document.write_text(
        "Use `[English](README.en.md)` as navigation.\n\n"
        "```markdown\n[Traditional Chinese](README.md)\n```\n",
        encoding="utf-8",
    )

    repo_check.check_markdown_links([document])


def test_markdown_link_gate_still_rejects_a_missing_real_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repo_check, "ROOT", tmp_path)
    document = tmp_path / "README.md"
    document.write_text("[Missing](missing.md)\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"README\.md -> missing\.md"):
        repo_check.check_markdown_links([document])


def test_static_demo_http_gate_uses_current_product_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repo_check, "ROOT", tmp_path)
    demo = tmp_path / "demo" / "web"
    demo.mkdir(parents=True)
    (demo / "index.html").write_text(
        '<h1>Aerial OBB Lab</h1><link href="style.css">'
        '<input id="modelInput"><input id="fileInput"><button id="detectBtn"></button>'
        '<script src="obb.js"></script><script src="app.js"></script>'
        + "<!-- padding -->" * 30,
        encoding="utf-8",
    )
    (demo / "app.js").write_text("// browser app\n" * 100, encoding="utf-8")
    (demo / "obb.js").write_text("// geometry\n" * 100, encoding="utf-8")
    (demo / "style.css").write_text("/* workbench */\n" * 40, encoding="utf-8")

    repo_check.check_static_demo()


def test_static_demo_rejects_app_embedded_demo_model_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repo_check, "ROOT", tmp_path)
    demo = tmp_path / "demo" / "web"
    demo.mkdir(parents=True)
    (demo / "index.html").write_text(
        '<h1>Aerial OBB Lab</h1><link href="style.css">'
        '<input id="modelInput"><input id="fileInput"><button id="detectBtn"></button>'
        '<script src="obb.js"></script><script src="app.js"></script>'
        + "<!-- padding -->" * 30,
        encoding="utf-8",
    )
    (demo / "app.js").write_text(
        'const model = "models/yolo26n-obb-privacy-sanitized.onnx";\n' * 40,
        encoding="utf-8",
    )
    (demo / "obb.js").write_text("// geometry\n" * 100, encoding="utf-8")
    (demo / "style.css").write_text("/* workbench */\n" * 40, encoding="utf-8")

    with pytest.raises(RuntimeError, match="must not embed or fetch a model"):
        repo_check.check_static_demo()
