from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_release_check():
    path = ROOT / "scripts" / "release_check.py"
    spec = importlib.util.spec_from_file_location("release_check_language", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readme_language_structure_is_zh_tw_first() -> None:
    checker = load_release_check()
    assert checker.verify_readme_language_structure(ROOT) == []

    canonical = (ROOT / "README.md").read_text(encoding="utf-8")
    english = (ROOT / "README.en.md").read_text(encoding="utf-8")

    assert canonical.startswith("正體中文 | [English](README.en.md)")
    assert english.startswith("[正體中文](README.md) | English")
    assert not (ROOT / "README.zh-TW.md").exists()

    for token in (
        "demo/web/",
        "ONNX Runtime Web",
        "Browser-native",
        "模型與圖片都只在本機瀏覽器處理",
        "Synthetic UI fixture",
    ):
        assert token in canonical
    for forbidden in ("Gradio", "demo/space-static/", "demo/space/"):
        assert forbidden not in canonical

    for token in ("demo/web/", "browser-native", "local browser", "Synthetic UI fixture"):
        assert token in english
    for forbidden in ("Gradio", "demo/space-static/", "demo/space/"):
        assert forbidden not in english


def test_public_readmes_do_not_invite_release_users_to_mutate_hf() -> None:
    canonical = (ROOT / "README.md").read_text(encoding="utf-8")
    english = (ROOT / "README.en.md").read_text(encoding="utf-8")

    for text in (canonical, english):
        assert "HF_TOKEN" not in text
        assert "write permission" not in text
        assert "write 權限" not in text
        assert "Run All" not in text
        assert "全部執行" not in text

    assert "不屬於本 release gate" in canonical
    assert "not part of this release gate" in english


def test_release_checklist_matches_bundled_font_inventory() -> None:
    checklist = (ROOT / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")

    assert "artifact inventory is empty" not in checklist
    assert "one self-hosted OFL display font" in checklist


def test_release_files_record_private_historical_space_verification() -> None:
    evidence = (ROOT / "release" / "evidence.json").read_text(encoding="utf-8")
    checklist = (ROOT / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert '"anonymous_api_http_status": 401' in evidence
    assert '"anonymous_page_http_status": 401' in evidence
    assert "[x] Make the historical Hugging Face Space private" in checklist
    assert "one external owner blocker" not in checklist
    assert "still public and running" not in checklist
    assert "historical Hugging Face Space still public" not in changelog
    assert "owner must make it private" not in changelog
    assert "historical Space is not present" not in checklist
    assert "empty public GitHub repository" not in checklist


def test_release_checklist_matches_current_public_github_state() -> None:
    checklist = (ROOT / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")

    assert "already the default branch" in checklist
    assert "before changing the default branch" not in checklist
    assert "before changing\n  the default branch" not in checklist
