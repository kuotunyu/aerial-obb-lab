from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _job_block(text: str, job_name: str) -> str:
    lines = text.splitlines()
    marker = f"  {job_name}:"
    start = lines.index(marker)
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].startswith("  ")
            and not lines[index].startswith("    ")
            and lines[index].endswith(":")
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def test_release_workflow_never_deploys_pages() -> None:
    text = (ROOT / ".github/workflows/release-gates.yml").read_text(encoding="utf-8")

    assert "scripts/pages_artifact_check.py" in text
    assert "actions/upload-artifact@v7" in text
    assert "actions/deploy-pages" not in text
    assert "pages: write" not in text


def test_release_pages_candidate_is_bound_to_reviewed_inputs() -> None:
    text = (ROOT / ".github/workflows/release-gates.yml").read_text(encoding="utf-8")
    candidate = _job_block(text, "pages-candidate")

    assert "needs: [core-cpu, browser-smoke]" in candidate
    assert "permissions:\n      contents: read" in candidate
    assert "name: aerial-obb-pages-candidate-${{ github.sha }}" in candidate
    assert "path: demo/web" in candidate


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
