from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts import prepare_pages_samples as samples


EXPECTED = {
    "aircraft": ((2821, 1885), "de7588b09b184b36ba136eb836cf8585c9242df7d96c2f55ec235fcf0422fe61", (1050, 320, 800, 600)),
    "naval": ((1280, 1224), "15406f875ab3cf74059fd9a554428448e438a7a6001ca0aab4edf258adc1b40a", (560, 250, 650, 600)),
    "port": ((1000, 667), "3a0db266e598cc6e6cea097958277d50dc1ad0e7436c03f79023165f883467fa", (0, 0, 1000, 667)),
}


def _write_rgb(path: Path, size: tuple[int, int] = (12, 8)) -> None:
    image = Image.new("RGB", size)
    image.putdata([(x * 17 % 256, y * 31 % 256, (x + y) * 13 % 256) for y in range(size[1]) for x in range(size[0])])
    image.save(path, format="PNG")


def _write_flat_rgb(path: Path, size: tuple[int, int]) -> None:
    Image.new("RGB", size, (40, 80, 120)).save(path, format="PNG")


def _spec_for(path: Path, *, size: tuple[int, int] = (12, 8), crop: samples.Crop | None = None) -> samples.CandidateSpec:
    return samples.CandidateSpec(
        id="test", label="Public test", agency="Public agency",
        record_url="https://example.invalid/record", rights_url="https://example.invalid/rights",
        acquisition_url="https://example.invalid/image", original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_size=size, crop=crop or samples.Crop(1, 1, 8, 6),
    )


def _output(rows: int = 1) -> dict[str, object]:
    return {"output0": {"dims": [1, rows, 7], "data": [512.0, 512.0, 20.0, 10.0, 0.9, 3.0, 0.0] * rows}}


def test_candidate_specs_pin_public_sources_sizes_digests_and_crops():
    assert set(samples.CANDIDATES) == set(EXPECTED)
    for candidate_id, (size, digest, crop) in EXPECTED.items():
        spec = samples.CANDIDATES[candidate_id]
        assert (spec.original_size, spec.original_sha256, spec.crop.as_tuple()) == (size, digest, crop)
        assert spec.record_url.startswith("https://")
        assert spec.rights_url.startswith("https://")
        assert spec.acquisition_url.startswith("https://")


def test_verify_source_rejects_digest_dimension_and_crop_drift(tmp_path: Path):
    source = tmp_path / "source.png"
    _write_rgb(source)
    spec = _spec_for(source)

    with pytest.raises(samples.PreparationError) as digest:
        samples.verify_source(source, samples.CandidateSpec(**{**spec.__dict__, "original_sha256": "0" * 64}))
    assert digest.value.code == "[SAMPLE_PREP:SOURCE_DIGEST]"

    with pytest.raises(samples.PreparationError) as dimensions:
        samples.verify_source(source, samples.CandidateSpec(**{**spec.__dict__, "original_size": (13, 8)}))
    assert dimensions.value.code == "[SAMPLE_PREP:SOURCE_DIMENSIONS]"

    with pytest.raises(samples.PreparationError) as crop:
        samples.verify_source(source, samples.CandidateSpec(**{**spec.__dict__, "crop": samples.Crop(8, 1, 8, 6)}))
    assert crop.value.code == "[SAMPLE_PREP:CROP_BOUNDS]"


def test_encode_published_webp_is_deterministic_small_and_metadata_free(tmp_path: Path):
    source = tmp_path / "source.png"
    first = tmp_path / "first.webp"
    second = tmp_path / "second.webp"
    _write_flat_rgb(source, (2000, 1200))
    spec = _spec_for(source, size=(2000, 1200), crop=samples.Crop(0, 0, 2000, 1200))

    samples.encode_published_webp(source, spec, first)
    samples.encode_published_webp(source, spec, second)

    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size <= 300 * 1024
    with Image.open(first) as image:
        assert image.mode == "RGB"
        assert max(image.size) == 1600
        assert not image.getexif()
        assert not any(image.info.get(key) for key in ("exif", "icc_profile", "xmp", "XML:com.adobe.xmp"))


def test_validate_output_rejects_wrong_name_dims_length_nonfinite_and_bad_class():
    with pytest.raises(samples.PreparationError) as name:
        samples.validate_output({"wrong": _output()["output0"]})
    assert name.value.code == "[SAMPLE_PREP:OUTPUT_NAME]"

    with pytest.raises(samples.PreparationError) as dims:
        samples.validate_output({"output0": {"dims": [1, 1, 6], "data": [0.0] * 6}})
    assert dims.value.code == "[SAMPLE_PREP:OUTPUT_DIMS]"

    with pytest.raises(samples.PreparationError) as length:
        samples.validate_output({"output0": {"dims": [1, 1, 7], "data": [0.0] * 6}})
    assert length.value.code == "[SAMPLE_PREP:OUTPUT_LENGTH]"

    nonfinite = _output()
    nonfinite["output0"]["data"][0] = float("nan")
    with pytest.raises(samples.PreparationError) as finite:
        samples.validate_output(nonfinite)
    assert finite.value.code == "[SAMPLE_PREP:OUTPUT_FINITE]"

    bad_class = _output()
    bad_class["output0"]["data"][5] = 15.0
    with pytest.raises(samples.PreparationError) as class_id:
        samples.validate_output(bad_class)
    assert class_id.value.code == "[SAMPLE_PREP:OUTPUT_CLASS]"


def test_build_candidate_emits_only_public_schema_and_never_model_identity(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    source = tmp_path / "source.png"
    model = tmp_path / "owner-authorized-SENTINEL-MODEL.onnx"
    review = tmp_path / "review"
    _write_rgb(source)
    model_bytes = b"PRIVATE-MODEL-SENTINEL-METADATA"
    model.write_bytes(model_bytes)
    spec = _spec_for(source)

    produced = samples.build_candidate(source, spec, model, review, capture=lambda *_: _output())
    serialized = produced.tensor_json.read_text(encoding="utf-8")
    record = json.loads(serialized)
    captured = capsys.readouterr()

    assert set(record) == {"candidate", "output0"}
    assert set(record["candidate"]) == {
        "id", "label", "agency", "record_url", "rights_url", "acquisition_url",
        "original_sha256", "original_size", "crop",
    }
    assert set(record["output0"]) == {"dims", "data"}
    assert produced.webp.name == "test.webp"
    assert produced.preview_png.name == "test.preview.png"
    assert all(secret not in serialized + captured.out + captured.err for secret in (
        str(model), model.name, model_bytes.decode("ascii"), "SENTINEL-MODEL",
    ))


def test_cli_rejects_a_model_inside_the_repository_without_echoing_its_name(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    repo_model = Path(__file__).resolve().parents[1] / "private-SENTINEL-model.onnx"
    repo_model.write_bytes(b"not a model")
    try:
        status = samples.main(["--model", str(repo_model), "--output-dir", str(tmp_path / "review")])
    finally:
        repo_model.unlink(missing_ok=True)
    captured = capsys.readouterr()

    assert status == 2
    assert captured.out == ""
    assert captured.err.strip() == "[SAMPLE_PREP:REPOSITORY_MODEL]"
    assert repo_model.name not in captured.err


def test_cli_rejects_a_review_directory_inside_repository_before_creating_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    model = tmp_path / "owner-authorized.onnx"
    review = Path(__file__).resolve().parents[1] / "private-SENTINEL-review"
    model.write_bytes(b"not a model")
    try:
        status = samples.main(["--model", str(model), "--output-dir", str(review)])
    finally:
        if review.exists():
            review.rmdir()
    captured = capsys.readouterr()

    assert status == 2
    assert not review.exists()
    assert captured.out == ""
    assert captured.err.strip() == "[SAMPLE_PREP:REPOSITORY_OUTPUT]"
    assert review.name not in captured.err


def test_cli_malformed_arguments_emit_only_a_fixed_code_without_secret_echo(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    secret = "PRIVATE-CLI-SENTINEL"

    status = samples.main([
        "--model", str(tmp_path / "owner-authorized.onnx"),
        "--output-dir", str(tmp_path / "review"),
        "--unexpected-option", secret,
    ])
    captured = capsys.readouterr()

    assert status == 2
    assert captured.out == ""
    assert captured.err.strip() == "[SAMPLE_PREP:ARGUMENTS]"
    assert secret not in captured.err
    assert "usage:" not in captured.err.lower()
