from __future__ import annotations

import importlib.util
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "1.0.0rc2"


def test_release_identity_matches_aerial_obb_lab() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["name"] == "aerial-obb-lab"
    assert project["urls"]["Repository"] == "https://github.com/kuotunyu/aerial-obb-lab"


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


def test_ci_uses_current_node24_action_majors() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-gates.yml").read_text(
        encoding="utf-8"
    )

    assert workflow.count("actions/checkout@v7") == 2
    assert workflow.count("actions/setup-python@v7") == 2
    assert workflow.count("actions/setup-node@v7") == 1


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


def test_release_dependency_graph_has_no_server_ui_framework() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    groups = project["dependency-groups"]
    package_names = {package["name"] for package in lock["package"]}

    assert "ui-preview" not in groups
    assert "demo" not in groups
    assert "gradio" not in package_names


def test_ci_runs_only_the_browser_native_ui_smoke() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-gates.yml").read_text(
        encoding="utf-8"
    )
    assert "python scripts/browser_smoke.py" in workflow
    assert "gradio" not in workflow.casefold()
    assert "--group ui-preview" not in workflow
    assert 'CUDA_VISIBLE_DEVICES: "-1"' in workflow
