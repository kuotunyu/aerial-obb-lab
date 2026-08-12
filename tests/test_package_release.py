from __future__ import annotations

import importlib.util
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "1.0.0rc2"


def test_package_metadata_and_runtime_version_match_release_candidate() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == EXPECTED_VERSION
    assert project["license"]["text"] == "AGPL-3.0-or-later"

    module_path = ROOT / "src" / "obbkit" / "__init__.py"
    spec = importlib.util.spec_from_file_location("obbkit_release_metadata", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__version__ == EXPECTED_VERSION


def test_release_metadata_files_exist() -> None:
    for relative in ("CHANGELOG.md", "CITATION.cff", "LICENSE", "THIRD_PARTY_NOTICES.md"):
        assert (ROOT / relative).is_file(), f"missing release metadata: {relative}"


def test_ci_runs_core_cpu_gates_on_ubuntu_and_windows() -> None:
    workflow = ROOT / ".github" / "workflows" / "release-gates.yml"
    text = workflow.read_text(encoding="utf-8")

    for token in (
        "ubuntu-latest",
        "windows-latest",
        "uv sync --frozen --no-install-project",
        "python -m pytest -q",
        "python scripts/repo_check.py",
        "python scripts/release_check.py",
        "python scripts/clean_export_check.py",
        "CUDA_VISIBLE_DEVICES",
        "uv build",
    ):
        assert token in text
    for forbidden in ("nvidia", "local-ml", "huggingface-token"):
        assert forbidden not in text.casefold()


def test_ci_runs_a_headless_synthetic_browser_smoke() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-gates.yml").read_text(
        encoding="utf-8"
    )
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "playwright>=1.55,<2" in project["dependency-groups"]["dev"]
    for token in (
        "browser-smoke:",
        "ubuntu-latest",
        "playwright install --with-deps chromium",
        "python scripts/browser_smoke.py",
    ):
        assert token in workflow


def test_sdist_explicitly_excludes_demo_models_and_dota_visuals() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    included = set(config["tool"]["hatch"]["build"]["targets"]["sdist"]["include"])

    assert included == {
        "/CHANGELOG.md",
        "/CITATION.cff",
        "/LICENSE",
        "/README.en.md",
        "/README.md",
        "/THIRD_PARTY_NOTICES.md",
        "/pyproject.toml",
        "/src/obbkit",
    }


def test_ui_preview_dependency_group_excludes_ml_runtime() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    groups = project["dependency-groups"]
    assert groups["ui-preview"] == ["gradio==6.20.0"]
    assert {item["include-group"] for item in groups["demo"] if isinstance(item, dict)} == {
        "local-ml",
        "ui-preview",
    }


def test_ci_runs_model_free_gradio_ui_smoke() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-gates.yml").read_text(
        encoding="utf-8"
    )
    for token in (
        "uv sync --frozen --no-install-project --group ui-preview",
        "python scripts/gradio_ui_smoke.py",
        'CUDA_VISIBLE_DEVICES: "-1"',
    ):
        assert token in workflow
