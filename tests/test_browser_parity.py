from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess

from obbkit.browser_reference import evaluate_fixture

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "browser_parity.json"
FLOAT32_TOLERANCE = 5e-6


def run_browser_geometry() -> tuple[dict, dict]:
    node = shutil.which("node")
    assert node, "Node.js is required for the browser parity gate"
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    result = subprocess.run(
        [node, "tests/js/browser_parity_runner.js", str(FIXTURE_PATH)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return fixture, json.loads(result.stdout)


def assert_nested_close(actual, expected, *, tolerance: float = FLOAT32_TOLERANCE) -> None:
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_close(actual[key], expected[key], tolerance=tolerance)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            assert_nested_close(actual_item, expected_item, tolerance=tolerance)
    elif isinstance(expected, (int, float)):
        assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)
    else:
        assert actual == expected


def test_letterbox_and_rgba_to_chw_match_literals() -> None:
    fixture, actual = run_browser_geometry()

    assert_nested_close(actual["geometry"], fixture["letterbox"]["expected"])
    assert_nested_close(actual["chw"], fixture["rgba"]["expected_chw"])


def test_python_reference_and_browser_share_the_float32_contract() -> None:
    fixture, browser = run_browser_geometry()
    reference = evaluate_fixture(fixture)

    assert_nested_close(reference["geometry"], fixture["letterbox"]["expected"])
    assert_nested_close(reference["chw"], fixture["rgba"]["expected_chw"])
    assert_nested_close(reference["detections"], browser["detections"])
    assert_nested_close(reference["corners"], browser["corners"])


def test_end_to_end_schema_unletterbox_angle_and_corners_match_literals() -> None:
    fixture, actual = run_browser_geometry()

    assert len(actual["detections"]) == 1
    assert_nested_close(actual["detections"][0], fixture["decode"]["expected_detection"])
    assert_nested_close(actual["corners"], fixture["decode"]["expected_corners"])


def test_malformed_end_to_end_outputs_fail_closed() -> None:
    fixture, actual = run_browser_geometry()

    expected = {item["name"]: item["error"] for item in fixture["invalid_outputs"]}
    assert actual["invalidErrors"] == expected
