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
    assert (
        "Aerial OBB 工程實驗室：以 YOLO26 與 DOTA 實驗為核心，涵蓋可重現訓練、誠實 "
        "baseline 評估、ONNX／TensorRT export、效能分析、Browser BYOM demo 與 CPU CI "
        "release gates。"
    ) in text
    assert "BYOM demo" in text
    assert "`zh-tw`" in text
    assert "Website field" in text


def test_owner_actions_records_private_historical_space_verification() -> None:
    text = (ROOT / "docs" / "OWNER_ACTIONS.md").read_text(encoding="utf-8")
    checklist = (ROOT / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")

    assert "Space API and page both returned anonymous HTTP `401`" in text
    assert "`private: false`" not in text
    assert "set the existing Space to **Private**" not in text
    assert "the Space is absent" not in text
    assert "[x] Make the historical Hugging Face Space private" in checklist
    assert "one external owner blocker" not in checklist
    assert "still public and running" not in checklist
    assert "historical Space is not present" not in checklist
    assert "empty public GitHub repository" not in checklist
