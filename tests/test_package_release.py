from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tomllib

import pytest

from scripts import browser_smoke


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "1.0.0"
OBSOLETE_PUBLIC_PATHS = (
    "README.zh-TW.md",
    "docs/OWNER_ACTIONS.md",
    "docs/PLAN.md",
    "src/obbkit/hf_checkpoint.py",
)


def test_release_identity_matches_aerial_obb_lab() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["name"] == "aerial-obb-lab"
    assert project["urls"]["Repository"] == "https://github.com/kuotunyu/aerial-obb-lab"


def test_package_metadata_and_runtime_version_match_stable_release() -> None:
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

    assert workflow.count("actions/checkout@v7") == 3
    assert workflow.count("actions/setup-python@v7") == 3
    assert workflow.count("actions/setup-node@v7") == 1


def test_ci_names_the_real_demo_browser_smoke() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-gates.yml").read_text(
        encoding="utf-8"
    )
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "playwright>=1.55,<2" in project["dependency-groups"]["dev"]
    for token in (
        "browser-smoke:",
        "Live demo browser smoke / Ubuntu CPU",
        "Exercise the real-image browser demo and BYOM safety paths",
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


ENCODED_TRANSFER_OK = "[OK] Encoded model transfer preserves decoded integrity"


def _chromium_available() -> bool:
    """True only when every browser component Playwright would launch is installed.

    Uses the documented dry-run listing instead of starting the driver, so the probe
    leaves no pending Playwright tasks behind in the test process.
    """
    try:
        listing = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "--dry-run", "chromium"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    locations = [match.strip() for match in re.findall(r"Install location:\s+(.+)", listing.stdout)]
    return listing.returncode == 0 and bool(locations) and all(Path(location).is_dir() for location in locations)


def test_browser_smoke_wires_the_encoded_model_transfer_scenario(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []
    for name in dir(browser_smoke):
        if name.startswith("run_") and callable(getattr(browser_smoke, name)):
            monkeypatch.setattr(
                browser_smoke, name, lambda *args, _name=name, **kwargs: calls.append(_name)
            )

    assert browser_smoke.main([]) == 0
    assert "run_encoded_model_transfer" in calls
    assert browser_smoke.main(["--scenario", "encoded-model-transfer"]) == 0
    assert calls.count("run_encoded_model_transfer") == 2
    assert capsys.readouterr().out.rstrip().endswith(ENCODED_TRANSFER_OK)


def test_encoded_model_transfer_preserves_decoded_integrity_in_a_real_browser() -> None:
    if not _chromium_available():
        pytest.skip("headless Chromium for Playwright is not installed")

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "browser_smoke.py"), "--scenario", "encoded-model-transfer"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert ENCODED_TRANSFER_OK in result.stdout
