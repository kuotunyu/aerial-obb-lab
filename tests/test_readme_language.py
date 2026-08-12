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
    compatibility = (ROOT / "README.zh-TW.md").read_text(encoding="utf-8")

    assert canonical.startswith("正體中文 | [English](README.en.md)")
    assert english.startswith("[正體中文](README.md) | English")
    assert len(compatibility) < 500
    assert "[README.md](README.md)" in compatibility
    assert "<!-- claim:" not in compatibility


def test_owner_actions_recommends_zh_tw_about_metadata() -> None:
    text = (ROOT / "docs" / "OWNER_ACTIONS.md").read_text(encoding="utf-8")
    assert "Code-only YOLO26 OBB × DOTA 作品集" in text
    assert "deployment benchmark" in text
    assert "BYOM demo" in text
    assert "`zh-tw`" in text
    assert "Website field" in text
