from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest


RESULT_PATH = Path(__file__).resolve().parents[1] / "docs" / "per_class_metrics.json"
CSV_PATH = RESULT_PATH.with_suffix(".csv")
EXPECTED_CLASS_NAMES = (
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
)
METRIC_FIELDS = ("precision", "recall", "f1", "mAP50", "mAP50-95")


def _load_results() -> dict:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_per_class_metrics_schema_and_instance_accounting() -> None:
    results = _load_results()
    rows = results["per_class"]

    assert results["schema_version"] == 1
    assert results["review"]["status"] == "PASS"
    assert results["review"]["initial_failure_was_false_negative"] is True
    assert len(rows) == 15
    assert [row["class_id"] for row in rows] == list(range(15))
    assert tuple(row["class"] for row in rows) == EXPECTED_CLASS_NAMES
    assert len({row["class_id"] for row in rows}) == 15
    assert len({row["class"] for row in rows}) == 15

    for row in rows:
        for field in METRIC_FIELDS:
            assert 0.0 <= row[field] <= 1.0
        assert 0 < row["instances"] <= row["raw_label_lines"]
        assert (
            row["raw_label_lines"] - row["instances"]
            == row["deduplicated_or_filtered_lines"]
        )


def test_per_class_ap_means_match_aggregate() -> None:
    results = _load_results()
    rows = results["per_class"]

    assert math.fsum(row["mAP50"] for row in rows) / len(rows) == pytest.approx(
        results["aggregate"]["mAP50"], rel=0.0, abs=1e-15
    )
    assert math.fsum(row["mAP50-95"] for row in rows) / len(rows) == pytest.approx(
        results["aggregate"]["mAP50-95"], rel=0.0, abs=1e-15
    )


def test_recovered_target_class_metrics_are_exact() -> None:
    rows = {row["class"]: row for row in _load_results()["per_class"]}

    assert (rows["plane"]["mAP50"], rows["plane"]["mAP50-95"]) == (
        0.9521473094544077,
        0.8623519396846154,
    )
    assert (rows["ship"]["mAP50"], rows["ship"]["mAP50-95"]) == (
        0.9094483549065466,
        0.7626811846734778,
    )
    assert (
        rows["storage tank"]["mAP50"],
        rows["storage tank"]["mAP50-95"],
    ) == (
        0.8506994253670539,
        0.7166956140619187,
    )


def test_csv_matches_reviewed_json() -> None:
    json_rows = _load_results()["per_class"]
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert len(csv_rows) == len(json_rows)
    for csv_row, json_row in zip(csv_rows, json_rows, strict=True):
        assert int(csv_row["class_id"]) == json_row["class_id"]
        assert csv_row["class"] == json_row["class"]
        for field in (
            "images",
            "instances",
            "raw_label_lines",
            "deduplicated_or_filtered_lines",
        ):
            assert int(csv_row[field]) == json_row[field]
        for field in METRIC_FIELDS:
            assert float(csv_row[field]) == json_row[field]
