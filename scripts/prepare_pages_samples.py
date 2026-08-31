"""Prepare deterministic, privacy-safe external review assets for Pages samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


IMGSZ = 1024
MAX_WEBP_BYTES = 300 * 1024
ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js"
ORT_WASM_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/"
ORT_INTEGRITY = "sha384-RPL/K8tc0JVaNWsunkEmCzLeieefvFX2UCRLKLmLVChCI6P+CTKhzqF7VIeCc3Zp"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PreparationError(Exception):
    """A fixed, non-sensitive diagnostic suitable for command-line output."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Crop:
    left: int
    top: int
    width: int
    height: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)


@dataclass(frozen=True)
class CandidateSpec:
    id: str
    label: str
    agency: str
    record_url: str
    rights_url: str
    acquisition_url: str
    original_sha256: str
    original_size: tuple[int, int]
    crop: Crop


@dataclass(frozen=True)
class ProducedCandidate:
    webp: Path
    tensor_json: Path
    preview_png: Path


CANDIDATES = {
    "aircraft": CandidateSpec(
        id="aircraft", label="機場與飛機", agency="U.S. Air Force / National Archives",
        record_url="https://catalog.archives.gov/id/6438938",
        rights_url="https://commons.wikimedia.org/wiki/File:Aircraft_stored_at_the_Aerospace_Maintenance_and_Regeneration_Center,_Davis-Monthan_Air_Force_Base,_Arizona_(USA),_on_1_October_1988_(6438938).jpeg",
        acquisition_url="https://upload.wikimedia.org/wikipedia/commons/e/e1/Aircraft_stored_at_the_Aerospace_Maintenance_and_Regeneration_Center%2C_Davis-Monthan_Air_Force_Base%2C_Arizona_%28USA%29%2C_on_1_October_1988_%286438938%29.jpeg",
        original_sha256="de7588b09b184b36ba136eb836cf8585c9242df7d96c2f55ec235fcf0422fe61",
        original_size=(2821, 1885), crop=Crop(1050, 320, 800, 600),
    ),
    "naval": CandidateSpec(
        id="naval", label="港灣與船艦", agency="U.S. Navy / National Archives",
        record_url="https://www.history.navy.mil/our-collections/photography/numerical-list-of-images/nhhc-series/nh-series/80-G-361000/80-G-361740.html",
        rights_url="https://www.history.navy.mil/our-collections/photography.html",
        acquisition_url="https://www.history.navy.mil/bin/imageDownload?image=/content/dam/nhhc/our-collections/photography/images/80-G-361000/80-G-361740&rendition=cq5dam.web.1280.1280.jpeg",
        original_sha256="15406f875ab3cf74059fd9a554428448e438a7a6001ca0aab4edf258adc1b40a",
        original_size=(1280, 1224), crop=Crop(560, 250, 650, 600),
    ),
    "port": CandidateSpec(
        id="port", label="港區與運輸設施", agency="U.S. Army / DVIDS",
        record_url="https://www.dvidshub.net/image/3156545/16th-cab-black-hawks-soar-port-tacoma",
        rights_url="https://www.dvidshub.net/about/copyright",
        acquisition_url="https://d1ldvf68ux039x.cloudfront.net/thumbs/photos/1702/3156545/1000w_q95.jpg",
        original_sha256="3a0db266e598cc6e6cea097958277d50dc1ad0e7436c03f79023165f883467fa",
        original_size=(1000, 667), crop=Crop(0, 0, 1000, 667),
    ),
}


def _fail(code: str) -> None:
    raise PreparationError(f"[SAMPLE_PREP:{code}]")


class _FixedDiagnosticParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        _fail("ARGUMENTS")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(source_path: Path | str, spec: CandidateSpec) -> Path:
    """Verify public-source bytes and geometry before any crop is decoded."""
    source = Path(source_path)
    try:
        if not stat.S_ISREG(source.stat().st_mode):
            _fail("SOURCE_FILE")
        if _sha256(source) != spec.original_sha256:
            _fail("SOURCE_DIGEST")
        with Image.open(source) as image:
            size = image.size
    except PreparationError:
        raise
    except Exception:
        _fail("SOURCE_FILE")
    if size != spec.original_size:
        _fail("SOURCE_DIMENSIONS")
    crop = spec.crop
    if (
        crop.left < 0
        or crop.top < 0
        or crop.width <= 0
        or crop.height <= 0
        or crop.left + crop.width > size[0]
        or crop.top + crop.height > size[1]
    ):
        _fail("CROP_BOUNDS")
    return source


def _metadata_free(image: Image.Image) -> bool:
    return not image.getexif() and not any(
        image.info.get(key) for key in ("exif", "icc_profile", "xmp", "XML:com.adobe.xmp")
    )


def encode_published_webp(source_path: Path | str, spec: CandidateSpec, destination: Path | str) -> Path:
    """Encode a crop with the fixed public WebP contract."""
    source = verify_source(source_path, spec)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(source) as original:
            image = original.convert("RGB")
        crop = spec.crop
        published = image.crop((crop.left, crop.top, crop.left + crop.width, crop.top + crop.height))
        if max(published.size) > 1600:
            published.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        published.save(output, format="WEBP", quality=82, method=6, exact=True)
    except PreparationError:
        raise
    except Exception:
        _fail("WEBP_ENCODE")
    try:
        if output.stat().st_size > MAX_WEBP_BYTES:
            _fail("WEBP_SIZE")
        with Image.open(output) as encoded:
            if not _metadata_free(encoded):
                _fail("WEBP_METADATA")
    except PreparationError:
        raise
    except Exception:
        _fail("WEBP_ENCODE")
    return output


def validate_output(results: Mapping[str, Any]) -> dict[str, dict[str, list[float] | list[int]]]:
    """Allow only the public end-to-end OBB output contract."""
    if set(results) != {"output0"}:
        _fail("OUTPUT_NAME")
    tensor = results["output0"]
    if not isinstance(tensor, Mapping):
        _fail("OUTPUT_DIMS")
    dims = tensor.get("dims")
    if (
        not isinstance(dims, Sequence)
        or isinstance(dims, (str, bytes))
        or len(dims) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) for value in dims)
        or dims[0] != 1
        or dims[1] < 1
        or dims[2] != 7
    ):
        _fail("OUTPUT_DIMS")
    data = tensor.get("data")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)) or len(data) != dims[0] * dims[1] * dims[2]:
        _fail("OUTPUT_LENGTH")
    values: list[float] = []
    for value in data:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            _fail("OUTPUT_FINITE")
        values.append(float(value))
    for index in range(0, len(values), 7):
        class_id = values[index + 5]
        if not (0 <= int(class_id) < 15 and class_id == int(class_id)):
            _fail("OUTPUT_CLASS")
        if values[index + 2] <= 0 or values[index + 3] <= 0:
            _fail("OUTPUT_BOX")
    return {"output0": {"dims": list(dims), "data": values}}


def _public_candidate(spec: CandidateSpec) -> dict[str, object]:
    return {
        "id": spec.id,
        "label": spec.label,
        "agency": spec.agency,
        "record_url": spec.record_url,
        "rights_url": spec.rights_url,
        "acquisition_url": spec.acquisition_url,
        "original_sha256": spec.original_sha256,
        "original_size": list(spec.original_size),
        "crop": list(spec.crop.as_tuple()),
    }


def _write_preview(webp: Path, destination: Path) -> None:
    try:
        with Image.open(webp) as published:
            published.convert("RGB").save(destination, format="PNG")
    except Exception:
        _fail("PREVIEW_ENCODE")


def build_candidate(
    source_path: Path | str,
    spec: CandidateSpec,
    model_path: Path | str,
    review_dir: Path | str,
    *,
    capture: Callable[[Path, Path], Mapping[str, Any]] | None = None,
) -> ProducedCandidate:
    """Create the three public review artifacts without serializing model details."""
    review = Path(review_dir)
    review.mkdir(parents=True, exist_ok=True)
    webp = encode_published_webp(source_path, spec, review / f"{spec.id}.webp")
    raw_results = capture(Path(model_path), webp) if capture else capture_browser_output(Path(model_path), webp)
    output = validate_output(raw_results)
    tensor_json = review / f"{spec.id}.tensor.json"
    tensor_json.write_text(
        json.dumps({"candidate": _public_candidate(spec), "output0": output["output0"]}, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    preview_png = review / f"{spec.id}.preview.png"
    _write_preview(webp, preview_png)
    return ProducedCandidate(webp=webp, tensor_json=tensor_json, preview_png=preview_png)


def capture_browser_output(model_path: Path, webp_path: Path) -> dict[str, dict[str, list[float] | list[int]]]:
    """Capture only JSON-safe output0 data through the pinned browser contract."""
    try:
        from playwright.sync_api import sync_playwright

        obb_script = REPOSITORY_ROOT / "demo" / "web" / "obb.js"
        with sync_playwright() as runtime:
            browser = runtime.chromium.launch()
            try:
                page = browser.new_page()
                page.set_content('<input id="model" type="file"><input id="image" type="file">')
                page.add_script_tag(path=str(obb_script))
                page.set_input_files("#model", {"name": "model.onnx", "mimeType": "application/octet-stream", "buffer": model_path.read_bytes()})
                page.set_input_files("#image", {"name": "candidate.webp", "mimeType": "image/webp", "buffer": webp_path.read_bytes()})
                serialized = page.evaluate(
                    """async ({ortUrl, wasmBase, integrity, size}) => {
                      const script = document.createElement("script");
                      script.src = ortUrl;
                      script.integrity = integrity;
                      script.crossOrigin = "anonymous";
                      const ready = new Promise((resolve, reject) => {
                        script.onload = resolve;
                        script.onerror = reject;
                      });
                      document.head.appendChild(script);
                      await ready;
                      ort.env.wasm.wasmPaths = wasmBase;
                      const model = document.querySelector("#model").files[0];
                      const imageFile = document.querySelector("#image").files[0];
                      const session = await ort.InferenceSession.create(new Uint8Array(await model.arrayBuffer()), {executionProviders: ["wasm"]});
                      const imageUrl = URL.createObjectURL(imageFile);
                      const image = await new Promise((resolve, reject) => {
                        const element = new Image();
                        element.onload = () => resolve(element);
                        element.onerror = reject;
                        element.src = imageUrl;
                      });
                      try {
                        const geometry = OBB.letterboxGeometry(image.naturalWidth, image.naturalHeight, size);
                        const canvas = document.createElement("canvas");
                        canvas.width = size;
                        canvas.height = size;
                        const context = canvas.getContext("2d");
                        context.fillStyle = "rgb(114,114,114)";
                        context.fillRect(0, 0, size, size);
                        context.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight, geometry.padX, geometry.padY, geometry.newWidth, geometry.newHeight);
                        const chw = OBB.rgbaToChw(context.getImageData(0, 0, size, size).data);
                        const results = await session.run({images: new ort.Tensor("float32", chw, [1, 3, size, size])});
                        return JSON.parse(JSON.stringify({output0: {dims: Array.from(results.output0.dims), data: Array.from(results.output0.data)}}));
                      } finally {
                        URL.revokeObjectURL(imageUrl);
                        if (typeof session.release === "function") await session.release();
                      }
                    }""",
                    {"ortUrl": ORT_URL, "wasmBase": ORT_WASM_BASE, "integrity": ORT_INTEGRITY, "size": IMGSZ},
                )
            finally:
                browser.close()
    except Exception:
        _fail("BROWSER_CAPTURE")
    return validate_output(serialized)


def _inside_repository(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return False
    except OSError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = _FixedDiagnosticParser(add_help=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    for candidate_id in CANDIDATES:
        parser.add_argument(f"--{candidate_id}-source")
    try:
        args = parser.parse_args(argv)
        review = Path(args.output_dir)
        if _inside_repository(review):
            _fail("REPOSITORY_OUTPUT")
        model = Path(args.model)
        if _inside_repository(model):
            _fail("REPOSITORY_MODEL")
        if not model.is_file():
            _fail("MODEL_FILE")
        source_paths = {candidate_id: getattr(args, f"{candidate_id}_source") for candidate_id in CANDIDATES}
        if any(path is None for path in source_paths.values()):
            _fail("SOURCE_ARGUMENT")
        for candidate_id, source_path in source_paths.items():
            build_candidate(Path(source_path), CANDIDATES[candidate_id], model, review)
    except PreparationError as error:
        print(error.code, file=sys.stderr)
        return 2
    except Exception:
        print("[SAMPLE_PREP:FAILED]", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
