from __future__ import annotations

from html.parser import HTMLParser
import importlib.util
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "1.0.0rc2"
ORT_CDN_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js"
OBSOLETE_PUBLIC_PATHS = (
    "README.zh-TW.md",
    "docs/OWNER_ACTIONS.md",
    "docs/PLAN.md",
    "src/obbkit/hf_checkpoint.py",
)


class _ScriptTagParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[dict[str, str | None]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == "script":
            self.scripts.append(dict(attrs))


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


def test_public_release_omits_obsolete_operational_surfaces() -> None:
    for relative in OBSOLETE_PUBLIC_PATHS:
        assert not (ROOT / relative).exists(), f"obsolete public surface remains: {relative}"


def test_default_dependency_graph_has_no_hugging_face_client() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    package_names = {package["name"] for package in lock["package"]}

    assert all(not dependency.startswith("huggingface_hub") for dependency in project["dependencies"])
    assert "huggingface-hub" not in package_names


def test_release_lock_omits_historical_gpu_dependency_stack() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    package_names = {package["name"] for package in lock["package"]}

    assert "local-ml" not in config["dependency-groups"]
    assert package_names.isdisjoint({"torch", "torchvision", "ultralytics"})


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


def test_browser_runtime_is_version_pinned_and_integrity_checked() -> None:
    parser = _ScriptTagParser()
    parser.feed((ROOT / "demo" / "web" / "index.html").read_text(encoding="utf-8"))
    runtime_scripts = [script for script in parser.scripts if script.get("src") == ORT_CDN_URL]

    assert len(runtime_scripts) == 1
    runtime = runtime_scripts[0]
    assert runtime.get("crossorigin") == "anonymous"
    assert re.fullmatch(r"sha384-[A-Za-z0-9+/]{64}", runtime.get("integrity") or "")
