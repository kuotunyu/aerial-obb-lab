from __future__ import annotations

import pathlib
import re
import tomllib

import obbkit


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_stable_release_metadata_is_consistent() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_version = pyproject["project"]["version"]

    assert project_version == "1.0.0"
    assert obbkit.__version__ == project_version
    assert pyproject["project"]["urls"]["Release"].endswith("/releases/tag/v1.0.0")

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert 'version: "1.0.0"' in citation

    lock_text = (ROOT / "uv.lock").read_text(encoding="utf-8")
    root_package = re.search(
        r'\[\[package\]\]\nname = "aerial-obb-lab"\nversion = "([^"]+)"',
        lock_text,
    )
    assert root_package is not None
    assert root_package.group(1) == project_version
