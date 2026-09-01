from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import parse_qs, urlsplit

from PIL import Image
import pytest

import scripts.prepare_sample_gallery as gallery
import scripts.sample_gallery_smoke as smoke
from scripts.prepare_sample_gallery import (
    CANDIDATE_RECIPES,
    GalleryError,
    acquire_candidate,
    validate_approved_gallery,
    validate_candidate_record,
)
from scripts.sample_gallery_smoke import (
    _parse_canvas_descriptions,
    byom_model_ready,
    source_valid_pool,
    validate_observations,
)


class FakeNaipTransport:
    def __init__(self) -> None:
        self.image = Image.new("RGBA", (16, 12), (33, 101, 157, 120))

    def __call__(self, request: dict) -> tuple[bytes, str]:
        if request["kind"] == "query":
            point = request["point"]
            return (
                ('{"features":[{"attributes":{"OBJECTID":42,"Name":"USDA NAIP 2022",'
                '"Year":2022,"raster_name":"naip_2022","download_url":'
                '"https://earthexplorer.usgs.gov/downloads/NAIP/2022/item","acquisition_date":'
                '"2022-06-15","agency":"USDA FSA","resolution_value":0.6,'
                '"resolution_units":"meters","band_count":4,"sensor_type":"NAIP"}}]}').encode(),
                "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/query",
            )
        if request["kind"] == "identify":
            query_body, _ = self({"kind": "query", "point": request["point"]})
            attributes = __import__("json").loads(query_body)["features"][0]["attributes"]
            return (
                __import__("json").dumps({
                    "objectId": 0,
                    "catalogItems": {"features": [{"attributes": attributes}]},
                    "catalogItemVisibilities": [1],
                }).encode(),
                "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/identify",
            )
        stream = io.BytesIO()
        self.image.save(stream, format="PNG")
        return stream.getvalue(), "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/exportImage"


class OverlappingNaipTransport(FakeNaipTransport):
    """The catalog query has overlaps; ImageServer identify chooses one mosaic item."""

    def __call__(self, request: dict) -> tuple[bytes, str]:
        if request["kind"] == "query":
            body, url = super().__call__(request)
            payload = __import__("json").loads(body)
            duplicate = deepcopy(payload["features"][0])
            duplicate["attributes"]["OBJECTID"] = 77
            duplicate["attributes"]["Name"] = "HRO mosaic"
            payload["features"].append(duplicate)
            return __import__("json").dumps(payload).encode(), url
        return super().__call__(request)


class QueryBearingServiceTransport(FakeNaipTransport):
    def __call__(self, request: dict) -> tuple[bytes, str]:
        body, url = super().__call__(request)
        return body, url + "?f=known-request-parameter"


class VisibleCatalogIdentityTransport(FakeNaipTransport):
    def __call__(self, request: dict) -> tuple[bytes, str]:
        if request["kind"] == "identify":
            query_body, _ = super().__call__({"kind": "query", "point": (0, 0)})
            attributes = __import__("json").loads(query_body)["features"][0]["attributes"]
            hidden = deepcopy(attributes)
            hidden["OBJECTID"] = 77
            return (
                __import__("json").dumps({
                    "objectId": 0,
                    "catalogItems": {"features": [{"attributes": hidden}, {"attributes": attributes}]},
                    "catalogItemVisibilities": [0, 1],
                }).encode(),
                "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/identify",
            )
        return super().__call__(request)


class DivergentVisibleCatalogTransport(FakeNaipTransport):
    """Default mosaic visibility changes by point while one qualified raster spans all."""

    def __call__(self, request: dict) -> tuple[bytes, str]:
        if request["kind"] == "identify":
            query_body, _ = super().__call__({"kind": "query", "point": request["point"]})
            common = __import__("json").loads(query_body)["features"][0]["attributes"]
            visible = deepcopy(common)
            visible["OBJECTID"] = 70 + int(abs(float(request["point"][0])) * 1000) % 3
            visible["Name"] = "HRO mosaic"
            return (
                __import__("json").dumps({
                    "catalogItems": {"features": [{"attributes": common}, {"attributes": visible}]},
                    "catalogItemVisibilities": [0, 1],
                }).encode(),
                "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/identify",
            )
        return super().__call__(request)


class PoolTransport(FakeNaipTransport):
    def __init__(self, source_rejected: set[tuple[float, float, float, float]] = set(), fatal: bool = False) -> None:
        super().__init__()
        self.source_rejected = source_rejected
        self.fatal = fatal

    def __call__(self, request: dict) -> tuple[bytes, str]:
        if self.fatal:
            raise GalleryError("GALLERY_NETWORK")
        body, url = super().__call__(request)
        if request["kind"] == "query" and request.get("bbox") in self.source_rejected:
            payload = __import__("json").loads(body)
            payload["features"][0]["attributes"]["Name"] = "HRO mosaic"
            return __import__("json").dumps(payload).encode(), url
        return body, url


class ParseFailureTransport(FakeNaipTransport):
    def __call__(self, request: dict) -> tuple[bytes, str]:
        body, url = super().__call__(request)
        return (b"{}", url) if request["kind"] == "query" else (body, url)


class DerivationFailureTransport(FakeNaipTransport):
    def __call__(self, request: dict) -> tuple[bytes, str]:
        body, url = super().__call__(request)
        return (b"not-a-jpeg", url) if request["kind"] == "export" else (body, url)


@pytest.fixture
def fake_naip_transport() -> FakeNaipTransport:
    return FakeNaipTransport()


@pytest.fixture
def valid_candidate_record(tmp_path: Path, fake_naip_transport: FakeNaipTransport) -> dict:
    return acquire_candidate(CANDIDATE_RECIPES[0], tmp_path, fake_naip_transport)


def invalid_source_tuning_and_privacy_mutations(record: dict) -> tuple[dict, ...]:
    cases: list[dict] = []

    def changed(path: tuple[str, ...], value: object) -> None:
        item = deepcopy(record)
        target = item
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        cases.append(item)

    changed(("source", "service"), "https://untrusted.example/ImageServer")
    changed(("source", "agency"), "Commercial imagery")
    changed(("source", "name"), "HRO mosaic")
    changed(("source", "rasterIds"), [42, 43])
    changed(("source", "bboxWgs84"), [-1, 2, 3, 4])
    changed(("derivation", "outputSize"), [640, 400])
    changed(("derivation", "jpegQuality"), 91)
    changed(("derivation", "threshold"), 0.3)
    changed(("derivation", "classFilter"), [1])
    changed(("source", "responseSha256"), "")
    changed(("image", "reviewName"), "C:/private/image.jpg")
    changed(("source", "downloadUrl"), "https://imagery.nationalmap.gov/file?token=secret")
    changed(("source", "rawError"), "private stack")
    extra = deepcopy(record)
    extra["extra"] = "not admitted"
    cases.append(extra)
    return tuple(cases)


@pytest.fixture
def approved_gallery_report(tmp_path: Path, valid_candidate_record: dict) -> dict:
    records = []
    for recipe in (CANDIDATE_RECIPES[0], CANDIDATE_RECIPES[3], CANDIDATE_RECIPES[6]):
        record = deepcopy(valid_candidate_record)
        record["candidateId"] = recipe.candidate_id
        record["category"] = recipe.category
        record["source"]["bboxWgs84"] = list(recipe.bbox_wgs84)
        record["image"]["reviewName"] = f"{recipe.candidate_id}.jpg"
        (tmp_path / str(record["image"]["reviewName"])).write_bytes(
            (tmp_path / str(valid_candidate_record["image"]["reviewName"])).read_bytes()
        )
        records.append(record)
    return {"schemaVersion": 1, "threshold": 0.25, "records": records,
            "visualReview": {record["candidateId"]: "approved" for record in records}}


def test_candidate_recipes_are_exact_conus_naip_only() -> None:
    assert tuple(recipe.candidate_id for recipe in CANDIDATE_RECIPES) == (
        "airfield-watsonville", "airfield-reid-hillview", "airfield-santa-monica",
        "sports-big-league-manteca", "sports-twin-creeks", "sports-ken-mercer",
        "harbor-port-hueneme", "harbor-redwood-city", "harbor-stockton",
    )
    assert {recipe.category for recipe in CANDIDATE_RECIPES} == {
        "airfield", "sports-complex", "harbor"
    }


def test_derivation_is_deterministic_1280_by_800_srgb_jpeg_without_metadata(
    tmp_path: Path, fake_naip_transport: FakeNaipTransport
) -> None:
    first = acquire_candidate(CANDIDATE_RECIPES[0], tmp_path / "one", fake_naip_transport)
    second = acquire_candidate(CANDIDATE_RECIPES[0], tmp_path / "two", fake_naip_transport)
    assert first["image"]["sha256"] == second["image"]["sha256"]
    with Image.open(tmp_path / "one" / first["image"]["reviewName"]) as image:
        assert image.size == (1280, 800)
        assert image.mode == "RGB"
        assert image.getexif() == {}
        assert image.info.get("icc_profile") is None


def test_candidate_record_rebinds_the_recorded_review_bytes_before_use(
    tmp_path: Path, fake_naip_transport: FakeNaipTransport
) -> None:
    review = tmp_path / "review"
    record = acquire_candidate(CANDIDATE_RECIPES[0], review, fake_naip_transport)
    (review / str(record["image"]["reviewName"])).write_bytes(b"not-a-jpeg")
    with pytest.raises(GalleryError, match="GALLERY_RECORD"):
        validate_candidate_record(record, review)


def test_derivation_uses_the_single_mosaic_identity_when_catalog_rows_overlap(tmp_path: Path) -> None:
    record = acquire_candidate(CANDIDATE_RECIPES[0], tmp_path / "review", OverlappingNaipTransport())
    assert record["source"]["rasterIds"] == [42]


def test_derivation_accepts_the_known_service_endpoint_when_its_response_keeps_request_query(tmp_path: Path) -> None:
    record = acquire_candidate(CANDIDATE_RECIPES[0], tmp_path / "review", QueryBearingServiceTransport())
    assert record["candidateId"] == "airfield-watsonville"


def test_derivation_uses_the_visible_identify_catalog_raster_not_the_service_object_id(tmp_path: Path) -> None:
    record = acquire_candidate(CANDIDATE_RECIPES[0], tmp_path / "review", VisibleCatalogIdentityTransport())
    assert record["source"]["rasterIds"] == [42]


def test_derivation_selects_the_one_common_source_qualified_id_despite_divergent_default_visibility(tmp_path: Path) -> None:
    record = acquire_candidate(CANDIDATE_RECIPES[0], tmp_path / "review", DivergentVisibleCatalogTransport())
    assert record["source"]["rasterIds"] == [42]


def test_source_qualified_intersection_requires_exactly_one_raster() -> None:
    recipe = CANDIDATE_RECIPES[0]
    attributes = {
        "OBJECTID": 42, "Name": "California aerial imagery", "Year": 2022,
        "raster_name": "m_3612142", "download_url": "https://earthexplorer.usgs.gov/downloads/NAIP/2022/item",
        "acquisition_date": "2022-06-15", "agency": "USDA FSA", "resolution_value": 0.6,
        "resolution_units": "meters", "band_count": 4, "sensor_type": "NAIP",
    }
    rejected = deepcopy(attributes); rejected["OBJECTID"] = 43; rejected["Name"] = "HRO mosaic"
    alternate = deepcopy(attributes); alternate["OBJECTID"] = 44
    assert gallery._common_source_qualified_id([{42: attributes, 43: rejected}] * 5, recipe) == 42
    for captures in ([{42: attributes}, {44: alternate}] + [{42: attributes}] * 3, [{42: attributes, 44: alternate}] * 5):
        with pytest.raises(GalleryError, match="GALLERY_SOURCE_REJECTED"):
            gallery._common_source_qualified_id(captures, recipe)


def test_acquisition_pool_keeps_two_or_three_source_valid_candidates_per_category(tmp_path: Path) -> None:
    root = tmp_path / "external-review"
    transport = PoolTransport({CANDIDATE_RECIPES[5].bbox_wgs84})
    records = gallery.acquire_all(root, transport)
    counts = {category: sum(record["category"] == category for record in records) for category in ("airfield", "sports-complex", "harbor")}
    assert counts == {"airfield": 3, "sports-complex": 2, "harbor": 3}
    batch = __import__("json").loads((root / "candidate-records.json").read_text(encoding="utf-8"))
    assert len(batch["records"]) == 8

    insufficient = tmp_path / "insufficient-review"
    with pytest.raises(GalleryError, match="GALLERY_RECORD"):
        gallery.acquire_all(insufficient, PoolTransport({recipe.bbox_wgs84 for recipe in CANDIDATE_RECIPES[3:5]}))
    assert not (insufficient / "candidate-records.json").exists()


def test_acquisition_pool_keeps_network_failures_fatal(tmp_path: Path) -> None:
    with pytest.raises(GalleryError, match="GALLERY_NETWORK"):
        gallery.acquire_all(tmp_path / "external-review", PoolTransport(fatal=True))


def test_acquisition_pool_keeps_parse_failures_fatal(tmp_path: Path) -> None:
    root = tmp_path / "external-review"
    with pytest.raises(GalleryError, match="GALLERY_RECORD"):
        gallery.acquire_all(root, ParseFailureTransport())
    assert not (root / "candidate-records.json").exists()


def test_acquisition_pool_keeps_derivation_failures_fatal(tmp_path: Path) -> None:
    root = tmp_path / "external-review"
    with pytest.raises(GalleryError, match="GALLERY_RECORD"):
        gallery.acquire_all(root, DerivationFailureTransport())
    assert not (root / "candidate-records.json").exists()


def test_acquisition_pool_keeps_containment_failures_fatal(tmp_path: Path) -> None:
    root = tmp_path / "external-review"
    transport = FakeNaipTransport()

    def contaminating_transport(request: dict) -> tuple[bytes, str]:
        if request["kind"] == "query":
            (root / ".gallery-stage" / "unrelated").write_bytes(b"x")
        return transport(request)

    with pytest.raises(GalleryError, match="GALLERY_SCOPE"):
        gallery.acquire_all(root, contaminating_transport)
    assert not (root / "candidate-records.json").exists()


def test_acquisition_pool_keeps_write_failures_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "external-review"
    original_write = Path.write_text

    def refusing_batch_write(path: Path, *args: object, **kwargs: object) -> int:
        if path.name == "candidate-records.json":
            raise OSError("write failed")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", refusing_batch_write)
    with pytest.raises(OSError):
        gallery.acquire_all(root, PoolTransport({CANDIDATE_RECIPES[5].bbox_wgs84}))
    assert not (root / "candidate-records.json").exists()


def test_approval_rejects_a_completed_pool_with_the_wrong_model_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = tmp_path / "external-review"
    records = gallery.acquire_all(review, PoolTransport({CANDIDATE_RECIPES[5].bbox_wgs84}))
    observations = {
        "schemaVersion": 1, "threshold": 0.25, "modelSha256": "0" * 64,
        "candidates": [
            {"candidateId": record["candidateId"], "category": record["category"], "runCompleted": True,
             "numericRuntime": 1.0, "detections": [], "visualReview": "unreviewed"}
            for record in records
        ],
    }
    (review / "observations.json").write_text(__import__("json").dumps(observations), encoding="utf-8")
    selections = iter(("airfield-watsonville", "sports-big-league-manteca", "harbor-port-hueneme"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(selections))
    with pytest.raises(GalleryError, match="GALLERY_RECORD"):
        gallery.approve(review, review / "observations.json", tmp_path / "pointer.txt")
    assert not (review / "approved-gallery.json").exists()


def test_observation_report_requires_the_complete_acquired_pool(tmp_path: Path) -> None:
    review = tmp_path / "external-review"
    records = gallery.acquire_all(review, PoolTransport({CANDIDATE_RECIPES[5].bbox_wgs84}))
    report = {
        "schemaVersion": 1, "threshold": 0.25, "modelSha256": gallery.MODEL_SHA256,
        "candidates": [
            {"candidateId": record["candidateId"], "category": record["category"], "runCompleted": True,
             "numericRuntime": 1.0, "detections": [], "visualReview": "unreviewed"}
            for record in records
        ],
    }
    validate_observations(report, records, review)
    report["candidates"].pop()
    with pytest.raises(ValueError, match="GALLERY_OBSERVATION"):
        validate_observations(report, records, review)


def test_observation_report_accepts_the_authoritative_source_pool_tuple(tmp_path: Path) -> None:
    review = tmp_path / "external-review"
    records = gallery.acquire_all(review, PoolTransport({CANDIDATE_RECIPES[5].bbox_wgs84}))
    pool = source_valid_pool(records, review)
    report = {
        "schemaVersion": 1, "threshold": 0.25, "modelSha256": gallery.MODEL_SHA256,
        "candidates": [
            {"candidateId": record["candidateId"], "category": record["category"], "runCompleted": True,
             "numericRuntime": 1.0, "detections": [], "visualReview": "unreviewed"}
            for record in pool
        ],
    }
    validate_observations(report, pool, review)


def test_source_valid_pool_requires_two_or_three_records_per_fixed_category(
    tmp_path: Path, valid_candidate_record: dict
) -> None:
    records = []
    for recipe in (*CANDIDATE_RECIPES[:3], *CANDIDATE_RECIPES[3:5], *CANDIDATE_RECIPES[6:]):
        record = deepcopy(valid_candidate_record)
        record["candidateId"] = recipe.candidate_id
        record["category"] = recipe.category
        record["source"]["bboxWgs84"] = list(recipe.bbox_wgs84)
        record["image"]["reviewName"] = f"{recipe.candidate_id}.jpg"
        (tmp_path / str(record["image"]["reviewName"])).write_bytes(
            (tmp_path / str(valid_candidate_record["image"]["reviewName"])).read_bytes()
        )
        records.append(record)
    pool = source_valid_pool(records, tmp_path)
    assert tuple(record["category"] for record in pool).count("sports-complex") == 2
    with pytest.raises(ValueError, match="GALLERY_OBSERVATION"):
        source_valid_pool([record for record in records if record["category"] != "sports-complex"] + [records[3]], tmp_path)


def test_smoke_cli_starts_when_run_as_a_script() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/sample_gallery_smoke.py"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2


def test_smoke_rejects_output_paths_outside_the_exact_review_root(tmp_path: Path) -> None:
    review = tmp_path / "review"
    review.mkdir()
    model = tmp_path / "model.onnx"
    model.write_bytes(b"")
    with pytest.raises(GalleryError, match="GALLERY_SCOPE"):
        smoke.run_smoke(review, model, tmp_path / "observations.json", review / "screenshots")


def test_browser_smoke_recognizes_the_actual_byom_model_ready_label() -> None:
    assert byom_model_ready("Local ONNX model ready") is True
    assert byom_model_ready("選擇相容的 .onnx model") is False


def test_browser_smoke_parses_the_current_canvas_description_format() -> None:
    description = (
        "class=plane; confidence=0.750; center-x=320.0 px; center-y=240.0 px; "
        "width=100.0 px; height=50.0 px; angle=45.0\N{DEGREE SIGN}. "
        "class=ship; confidence=0.625; center-x=640.0 px; center-y=480.0 px; "
        "width=80.0 px; height=40.0 px; angle=-15.0\N{DEGREE SIGN}."
    )
    assert _parse_canvas_descriptions(description) == [
        {"classId": 0, "confidence": 0.75, "cx": 320.0, "cy": 240.0, "w": 100.0, "h": 50.0, "angle": 45.0},
        {"classId": 1, "confidence": 0.625, "cx": 640.0, "cy": 480.0, "w": 80.0, "h": 40.0, "angle": -15.0},
    ]


def test_browser_smoke_cli_hides_unexpected_runtime_details(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    def failing_run(*args: object) -> None:
        raise RuntimeError("private browser detail")

    monkeypatch.setattr(smoke, "run_smoke", failing_run)
    assert smoke.main([
        "--review-root", str(tmp_path), "--model", str(tmp_path / "model.onnx"),
        "--report", str(tmp_path / "report.json"), "--screenshot-dir", str(tmp_path / "screenshots"),
    ]) == 1
    assert capsys.readouterr().out == "[FAIL] GALLERY_SMOKE\n"


def test_identify_catalog_identity_requires_one_finite_positive_contributor() -> None:
    payload = {
        "objectId": 0,
        "catalogItems": {"features": [
            {"attributes": {"OBJECTID": 77}},
            {"attributes": {"OBJECTID": 42}},
        ]},
        "catalogItemVisibilities": [0, 1],
    }
    body = __import__("json").dumps(payload).encode()
    endpoint = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/identify"
    assert gallery._mosaic_object_id(body, endpoint) == 42
    mutations = []
    missing_catalog = deepcopy(payload); missing_catalog.pop("catalogItems"); mutations.append(missing_catalog)
    null_catalog = deepcopy(payload); null_catalog["catalogItems"] = None; mutations.append(null_catalog)
    mismatched = deepcopy(payload); mismatched["catalogItemVisibilities"] = [1]; mutations.append(mismatched)
    duplicate = deepcopy(payload); duplicate["catalogItems"]["features"][0]["attributes"]["OBJECTID"] = 42; mutations.append(duplicate)
    bool_id = deepcopy(payload); bool_id["catalogItems"]["features"][1]["attributes"]["OBJECTID"] = True; mutations.append(bool_id)
    non_int_id = deepcopy(payload); non_int_id["catalogItems"]["features"][1]["attributes"]["OBJECTID"] = 42.0; mutations.append(non_int_id)
    nonfinite = deepcopy(payload); nonfinite["catalogItemVisibilities"] = [0, float("nan")]; mutations.append(nonfinite)
    negative = deepcopy(payload); negative["catalogItemVisibilities"] = [0, -1]; mutations.append(negative)
    multiple = deepcopy(payload); multiple["catalogItemVisibilities"] = [1, 1]; mutations.append(multiple)
    for mutation in mutations:
        with pytest.raises(GalleryError, match="GALLERY_RECORD"):
            gallery._mosaic_object_id(__import__("json").dumps(mutation).encode(), endpoint)


def test_identify_transport_uses_the_same_bounded_json_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response(io.BytesIO):
        def getcode(self) -> int:
            return 200

        def geturl(self) -> str:
            return "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/identify?f=json"

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: object) -> None:
            self.close()

    monkeypatch.setattr(gallery.urllib.request, "urlopen", lambda *_args, **_kwargs: Response(b'{"objectId":42}'))
    body, final_url = gallery.urlopen_transport({"kind": "identify", "url": "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/identify?f=json"})
    assert body == b'{"objectId":42}'
    assert final_url.endswith("/identify?f=json")


def test_identify_request_declares_a_wgs84_json_point_for_the_image_service() -> None:
    request = gallery._identify_request(CANDIDATE_RECIPES[0], (-121.791682, 36.933656))
    geometry = parse_qs(urlsplit(str(request["url"])).query)["geometry"]
    assert geometry == ['{"x":-121.791682,"y":36.933656,"spatialReference":{"wkid":4326}}']


def _official_naip_path_record(record: dict) -> dict:
    item = deepcopy(record)
    item["source"]["agency"] = "USDA Farm Service Agency"
    item["source"]["name"] = "California aerial imagery"
    item["source"]["rasterName"] = "m_3612142"
    item["source"]["downloadUrl"] = "https://earthexplorer.usgs.gov/downloads/NAIP/2022/item"
    return item


def test_admission_accepts_official_usda_fsa_record_with_naip_download_path(
    tmp_path: Path, valid_candidate_record: dict
) -> None:
    validate_candidate_record(_official_naip_path_record(valid_candidate_record), tmp_path)


def test_admission_accepts_the_exact_official_download_provenance_host(
    tmp_path: Path, valid_candidate_record: dict
) -> None:
    record = deepcopy(valid_candidate_record)
    record["source"]["downloadUrl"] = "https://earthexplorer.usgs.gov/downloads/NAIP/2022/item"
    validate_candidate_record(record, tmp_path)


def test_admission_rejects_ambiguous_naip_path_and_private_or_nonproduct_source_fields(
    tmp_path: Path, valid_candidate_record: dict
) -> None:
    record = _official_naip_path_record(valid_candidate_record)
    mutations = (
        ("downloadUrl", "https://earthexplorer.usgs.gov/downloads/2022/item"),
        ("downloadUrl", "https://earthexplorer.usgs.gov/downloads/NAIPish/item"),
        ("downloadUrl", "https://user@earthexplorer.usgs.gov/downloads/NAIP/item"),
        ("downloadUrl", "https://earthexplorer.usgs.gov/downloads/NAIP/item?token=secret"),
        ("downloadUrl", "https://earthexplorer.usgs.gov/downloads/NAIP/item#fragment"),
        ("name", ""),
        ("rasterName", ""),
        ("name", "HRO imagery"),
        ("rasterName", "commercial mosaic"),
        ("agency", "USDA commercial"),
    )
    for key, value in mutations:
        candidate = deepcopy(record)
        candidate["source"][key] = value
        with pytest.raises(GalleryError, match="GALLERY_RECORD"):
            validate_candidate_record(candidate, tmp_path)


def test_admission_rejects_lookalike_agencies_and_invalid_official_dates(
    tmp_path: Path, valid_candidate_record: dict
) -> None:
    record = _official_naip_path_record(valid_candidate_record)
    mutations = (
        ("agency", "notusda imagery"),
        ("agency", "FSAX"),
        ("year", 0),
        ("year", "2022"),
        ("acquisitionDate", "2022-6-15"),
        ("acquisitionDate", "2027-01-01"),
    )
    for key, value in mutations:
        candidate = deepcopy(record)
        candidate["source"][key] = value
        with pytest.raises(GalleryError, match="GALLERY_RECORD"):
            validate_candidate_record(candidate, tmp_path)


def test_admission_accepts_the_official_epoch_millis_acquisition_date(
    tmp_path: Path, valid_candidate_record: dict
) -> None:
    record = _official_naip_path_record(valid_candidate_record)
    record["source"]["acquisitionDate"] = 1655337600000
    validate_candidate_record(record, tmp_path)


def test_admission_rejects_non_naip_source_hidden_tuning_and_private_fields(
    tmp_path: Path, valid_candidate_record: dict
) -> None:
    for mutation in invalid_source_tuning_and_privacy_mutations(valid_candidate_record):
        with pytest.raises(GalleryError, match="GALLERY_RECORD"):
            validate_candidate_record(mutation, tmp_path)


def test_approved_gallery_requires_one_visually_approved_record_per_fixed_category(
    tmp_path: Path, approved_gallery_report: dict
) -> None:
    records = validate_approved_gallery(approved_gallery_report, tmp_path)
    assert tuple(record["category"] for record in records) == (
        "airfield", "sports-complex", "harbor"
    )
    duplicate = deepcopy(approved_gallery_report)
    duplicate["records"][1]["category"] = "airfield"
    with pytest.raises(GalleryError, match="GALLERY_RECORD"):
        validate_approved_gallery(duplicate, tmp_path)
    missing = deepcopy(approved_gallery_report)
    missing["visualReview"].pop(missing["records"][0]["candidateId"])
    with pytest.raises(GalleryError, match="GALLERY_RECORD"):
        validate_approved_gallery(missing, tmp_path)


def test_publish_writes_exact_three_images_and_public_safe_receipt_atomically(
    tmp_path: Path, approved_gallery_report: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing a managed image or leaking review data must break publication."""
    review_root = tmp_path
    monkeypatch.setattr(gallery, "REPO_ROOT", tmp_path)
    pages_root = tmp_path / "demo" / "web"
    receipt_path = tmp_path / "release" / "receipt.json"
    monkeypatch.setattr(gallery, "_git_worktree_roots", lambda _root: {tmp_path / "repo"})
    gallery.publish_approved_gallery(
        approved_gallery_report, review_root, pages_root, receipt_path
    )
    assert sorted(path.name for path in (pages_root / "samples").iterdir()) == [
        "airfield.jpg", "harbor.jpg", "sports-complex.jpg"
    ]
    receipt = json.loads(receipt_path.read_text("utf-8"))
    serialized = json.dumps(receipt, sort_keys=True).casefold()
    assert "watsonville" not in serialized
    assert "private" not in serialized
    assert "reviewname" not in serialized


def test_demo_manifest_declares_exact_sample_catalog_and_default() -> None:
    """A wrong sample order/default would make the initial workbench misleading."""
    manifest = json.loads((Path(__file__).resolve().parents[1] / "demo/web/demo-model.json").read_text("utf-8"))
    assert manifest["schemaVersion"] == 2
    assert manifest["defaultSampleId"] == "airfield"
    assert [item["id"] for item in manifest["samples"]] == [
        "airfield", "sports-complex", "harbor"
    ]


def test_publish_preserves_the_entire_prior_batch_when_backup_replace_fails(
    tmp_path: Path, approved_gallery_report: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backup failure must not remove the first old public image or receipt."""
    monkeypatch.setattr(gallery, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(gallery, "_git_worktree_roots", lambda _root: {tmp_path / "repo"})
    pages = tmp_path / "demo" / "web" / "samples"; pages.mkdir(parents=True)
    receipt = tmp_path / "release" / "sample-gallery-sources.json"; receipt.parent.mkdir()
    before = {"airfield.jpg": b"old-air", "sports-complex.jpg": b"old-sports", "harbor.jpg": b"old-harbor"}
    for name, body in before.items(): (pages / name).write_bytes(body)
    receipt.write_bytes(b"old-receipt")
    real_replace, calls = gallery.os.replace, 0
    def fail_second_replace(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2: raise OSError("backup failure")
        real_replace(source, destination)  # type: ignore[arg-type]
    monkeypatch.setattr(gallery.os, "replace", fail_second_replace)
    with pytest.raises(GalleryError, match="GALLERY_SCOPE"):
        gallery.publish_approved_gallery(approved_gallery_report, tmp_path, pages.parent, receipt)
    assert {path.name: path.read_bytes() for path in pages.iterdir()} == before
    assert receipt.read_bytes() == b"old-receipt"


def test_publish_rejects_mutated_boats_predecessor_without_deleting_it(
    tmp_path: Path, approved_gallery_report: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unrelated boats leaf must not be treated as the reviewed predecessor."""
    monkeypatch.setattr(gallery, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(gallery, "_git_worktree_roots", lambda _root: {tmp_path / "repo"})
    samples = tmp_path / "demo" / "web" / "samples"; samples.mkdir(parents=True)
    boats = samples / "boats.jpg"; boats.write_bytes(b"not-the-reviewed-predecessor")
    receipt = tmp_path / "release" / "sample-gallery-sources.json"
    with pytest.raises(GalleryError, match="GALLERY_RECORD"):
        gallery.publish_approved_gallery(approved_gallery_report, tmp_path, samples.parent, receipt)
    assert boats.read_bytes() == b"not-the-reviewed-predecessor"


def test_approval_verification_rejects_a_candidate_outside_the_acquired_pool(tmp_path: Path) -> None:
    review = tmp_path / "external-review"
    records = gallery.acquire_all(review, PoolTransport({CANDIDATE_RECIPES[5].bbox_wgs84}))
    rejected = acquire_candidate(CANDIDATE_RECIPES[5], tmp_path / "other-review", FakeNaipTransport())
    (review / str(rejected["image"]["reviewName"])).write_bytes(
        (tmp_path / "other-review" / str(rejected["image"]["reviewName"])).read_bytes()
    )
    choices = [
        next(record for record in records if record["category"] == "airfield"), rejected,
        next(record for record in records if record["category"] == "harbor"),
    ]
    report = {"schemaVersion": 1, "threshold": 0.25, "records": choices,
              "visualReview": {str(record["candidateId"]): "approved" for record in choices}}
    (review / "approved-gallery.json").write_text(__import__("json").dumps(report), encoding="utf-8")
    with pytest.raises(GalleryError, match="GALLERY_RECORD"):
        gallery.verify_approved(review)


def test_observation_report_allows_only_fixed_candidates_and_measured_finite_values() -> None:
    report = {
        "schemaVersion": 1,
        "threshold": 0.25,
        "modelSha256": "a0a1a2dd357067e8c6c9f5ce7bb33487188423f9722e813be880da4f9badcd97",
        "candidates": [{
            "candidateId": "airfield-watsonville", "category": "airfield", "runCompleted": True,
            "numericRuntime": 12.5, "visualReview": "unreviewed",
            "detections": [{"classId": 1, "confidence": 0.5, "cx": 20, "cy": 30, "w": 40, "h": 50, "angle": 0.2}],
        }],
    }
    with pytest.raises(ValueError, match="GALLERY_OBSERVATION"):
        validate_observations(report)
    bad = deepcopy(report)
    bad["candidates"][0]["detections"][0]["confidence"] = float("nan")
    with pytest.raises(ValueError, match="GALLERY_OBSERVATION"):
        validate_observations(bad)
