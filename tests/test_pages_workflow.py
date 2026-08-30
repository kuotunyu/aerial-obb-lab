from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_workflow_never_deploys_pages() -> None:
    text = (ROOT / ".github/workflows/release-gates.yml").read_text(encoding="utf-8")

    assert "scripts/pages_artifact_check.py" in text
    assert "actions/upload-artifact@v7" in text
    assert "actions/deploy-pages" not in text
    assert "pages: write" not in text


def test_pages_workflow_is_manual_and_about_free() -> None:
    text = (ROOT / ".github/workflows/pages.yml").read_text(encoding="utf-8")
    trigger = text.split("permissions:", 1)[0]

    assert "workflow_dispatch:" in trigger
    assert "push:" not in trigger and "pull_request:" not in trigger
    assert "github.ref == 'refs/heads/main'" in text
    assert "actions/upload-pages-artifact@v5" in text
    assert "actions/deploy-pages@v5" in text
    assert "pages: write" in text and "id-token: write" in text
    assert "gh repo edit" not in text and "--homepage" not in text
