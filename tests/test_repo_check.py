from __future__ import annotations

from pathlib import Path

import pytest

from scripts import repo_check


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
