"""Fail-closed browser parity for the admitted source and sanitized model."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from threading import Thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys

import onnx
from onnx import TensorProto

if __name__ == "__main__":
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from scripts.prepare_demo_assets import (
    SANITIZED_MODEL_REVIEW_PATH,
    SOURCE_REVIEW_PATHS,
    checked_child,
    is_reparse_point,
    validate_admitted_assets,
)


ERROR_CODES = {
    "scope": "DEMO_MODEL_PARITY_SCOPE",
    "runtime": "DEMO_MODEL_PARITY_RUNTIME",
    "mismatch": "DEMO_MODEL_PARITY_MISMATCH",
    "report": "DEMO_MODEL_PARITY_REPORT",
}
ORT_SCRIPT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js"
ORT_WASM_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/"
ORT_INTEGRITY = "sha384-RPL/K8tc0JVaNWsunkEmCzLeieefvFX2UCRLKLmLVChCI6P+CTKhzqF7VIeCc3Zp"
ROOT = Path(__file__).resolve().parents[1]
OBB_SCRIPT = ROOT / "demo" / "web" / "obb.js"


class ParityError(Exception):
    def __init__(self, category: str) -> None:
        self.code = ERROR_CODES[category]
        super().__init__(self.code)


def _harness_html() -> bytes:
    return (
        "<!doctype html><meta charset=utf-8><title>Parity</title>"
        f'<script src="{ORT_SCRIPT_URL}" integrity="{ORT_INTEGRITY}" crossorigin="anonymous"></script>'
        '<script src="/obb.js"></script>'
    ).encode("utf-8")


class _Server:
    def __init__(self, routes: dict[str, tuple[bytes, str]]) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                route = routes.get(self.path)
                if route is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                payload, media_type = route
                self.send_response(200)
                self.send_header("Content-Type", media_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}/"

    def __exit__(self, *_args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


_BROWSER_SCRIPT = r"""
async () => {
  const fail = (message) => { throw new Error(message); };
  if (!globalThis.ort || !globalThis.OBB) fail("runtime");
  const runtimeVersion = globalThis.ort.env?.versions?.web;
  if (runtimeVersion !== "1.20.1") fail("runtime-version");
  globalThis.ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  globalThis.ort.env.wasm.numThreads = 1;

  const image = new Image();
  image.src = "/sample-image";
  await image.decode();
  const size = 1024;
  const geometry = OBB.letterboxGeometry(image.naturalWidth, image.naturalHeight, size);
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d", {willReadFrequently: true});
  context.fillStyle = "rgb(114,114,114)";
  context.fillRect(0, 0, size, size);
  context.drawImage(
    image, 0, 0, image.naturalWidth, image.naturalHeight,
    geometry.padX, geometry.padY, geometry.newWidth, geometry.newHeight,
  );
  const chw = OBB.rgbaToChw(context.getImageData(0, 0, size, size).data);

  async function run(route) {
    const response = await fetch(route, {cache: "no-store", credentials: "omit"});
    if (!response.ok) fail("model-fetch");
    const bytes = await response.arrayBuffer();
    const session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    try {
      if (session.inputNames.length !== 1 || session.outputNames.length !== 1) {
        fail("contract-names");
      }
      const inputName = session.inputNames[0];
      const outputName = session.outputNames[0];
      if (inputName !== "images" || outputName !== "output0") fail("runtime-names");
      const tensor = new ort.Tensor("float32", chw, [1, 3, size, size]);
      const inputContract = {
        name: inputName,
        type: tensor.type,
        shape: Array.from(tensor.dims),
      };
      if (JSON.stringify(inputContract) !== JSON.stringify({
        name: "images", type: "float32", shape: [1, 3, size, size],
      })) fail("feed-contract");
      const results = await session.run({[inputName]: tensor});
      const output = results[outputName];
      if (!output || !(output.data instanceof Float32Array)) fail("output-type");
      const outputContract = {
        name: outputName,
        type: output.type,
        shape: Array.from(output.dims),
      };
      if (
        outputContract.name !== "output0" ||
        outputContract.type !== "float32" ||
        outputContract.shape.length !== 3 ||
        outputContract.shape[0] !== 1 ||
        !Number.isInteger(outputContract.shape[1]) ||
        outputContract.shape[1] <= 0 ||
        outputContract.shape[2] !== 7
      ) fail("output-contract");
      return {
        input: inputContract,
        output: outputContract,
        bytes: new Uint8Array(
          output.data.buffer,
          output.data.byteOffset,
          output.data.byteLength,
        ).slice(),
        values: output.data.slice(),
      };
    } finally {
      await session.release();
    }
  }

  const source = await run("/source-model");
  const derived = await run("/derived-model");
  const runtimeContractsEqual =
    JSON.stringify(source.input) === JSON.stringify(derived.input) &&
    JSON.stringify(source.output) === JSON.stringify(derived.output);
  if (!runtimeContractsEqual) {
    return {
      runtime: {name: "onnxruntime-web", version: runtimeVersion},
      source_input: source.input,
      derived_input: derived.input,
      source_output: source.output,
      derived_output: derived.output,
      output_bytes_equal: false,
      detections_equal: false,
      accepted_ship: false,
    };
  }
  const outputBytesEqual =
    source.bytes.length === derived.bytes.length &&
    source.bytes.every((value, index) => value === derived.bytes[index]);
  if (!outputBytesEqual) {
    return {
      runtime: {name: "onnxruntime-web", version: runtimeVersion},
      source_input: source.input,
      derived_input: derived.input,
      source_output: source.output,
      derived_output: derived.output,
      output_bytes_equal: false,
      detections_equal: false,
      accepted_ship: false,
    };
  }
  const sourceDetections = OBB.decodeDetections(source.values, geometry, 0.25, new Set(), 15);
  const derivedDetections = OBB.decodeDetections(derived.values, geometry, 0.25, new Set(), 15);
  const detectionsEqual = JSON.stringify(sourceDetections) === JSON.stringify(derivedDetections);
  const acceptedShip = sourceDetections.some((detection) => detection.cls === 1);
  return {
    runtime: {name: "onnxruntime-web", version: runtimeVersion},
    source_input: source.input,
    derived_input: derived.input,
    source_output: source.output,
    derived_output: derived.output,
    output_bytes_equal: outputBytesEqual,
    detections_equal: detectionsEqual,
    accepted_ship: acceptedShip,
  };
}
"""


def _browser_parity(review_root: Path) -> dict[str, object]:
    source_contract = _declared_model_contract(
        checked_child(review_root, Path(SOURCE_REVIEW_PATHS["obb-model"]))
    )
    derived_contract = _declared_model_contract(
        checked_child(review_root, Path(SANITIZED_MODEL_REVIEW_PATH))
    )
    if source_contract != derived_contract:
        raise ParityError("mismatch")
    routes = {
        "/": (_harness_html(), "text/html; charset=utf-8"),
        "/obb.js": (OBB_SCRIPT.read_bytes(), "text/javascript; charset=utf-8"),
        "/source-model": (
            checked_child(review_root, Path(SOURCE_REVIEW_PATHS["obb-model"])).read_bytes(),
            "application/octet-stream",
        ),
        "/derived-model": (
            checked_child(review_root, Path(SANITIZED_MODEL_REVIEW_PATH)).read_bytes(),
            "application/octet-stream",
        ),
        "/sample-image": (
            checked_child(review_root, Path(SOURCE_REVIEW_PATHS["boats-image"])).read_bytes(),
            "image/jpeg",
        ),
    }
    try:
        from playwright.sync_api import sync_playwright

        with _Server(routes) as url, sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, wait_until="networkidle")
                page.wait_for_function("globalThis.ort && globalThis.OBB")
                result = page.evaluate(_BROWSER_SCRIPT)
            finally:
                browser.close()
    except Exception:
        raise ParityError("runtime") from None
    if not isinstance(result, dict):
        raise ParityError("runtime")
    if (
        result.get("source_input") != source_contract["input"]
        or result.get("derived_input") != source_contract["input"]
        or result.get("source_output") != source_contract["output"]
        or result.get("derived_output") != source_contract["output"]
    ):
        raise ParityError("mismatch")
    result["declared_contracts_equal"] = True
    return result


def _value_contract(value: object) -> dict[str, object]:
    try:
        if value.type.WhichOneof("value") != "tensor_type":  # type: ignore[attr-defined]
            raise ValueError
        tensor_type = value.type.tensor_type  # type: ignore[attr-defined]
        if tensor_type.elem_type != TensorProto.FLOAT or not tensor_type.HasField("shape"):
            raise ValueError
        shape: list[int | str] = []
        for dimension in tensor_type.shape.dim:
            kind = dimension.WhichOneof("value")
            if kind == "dim_value":
                shape.append(dimension.dim_value)
            elif kind == "dim_param" and dimension.dim_param:
                shape.append(dimension.dim_param)
            else:
                raise ValueError
        return {"name": value.name, "type": "float32", "shape": shape}  # type: ignore[attr-defined]
    except Exception:
        raise ParityError("mismatch") from None


def _declared_model_contract(path: Path) -> dict[str, dict[str, object]]:
    try:
        model = onnx.load_model(path, load_external_data=False)
        if len(model.graph.input) != 1 or len(model.graph.output) != 1:
            raise ValueError
        contract = {
            "input": _value_contract(model.graph.input[0]),
            "output": _value_contract(model.graph.output[0]),
        }
    except ParityError:
        raise
    except Exception:
        raise ParityError("mismatch") from None
    if (
        contract["input"]
        != {"name": "images", "type": "float32", "shape": [1, 3, 1024, 1024]}
        or contract["output"].get("name") != "output0"
        or contract["output"].get("type") != "float32"
        or not isinstance(contract["output"].get("shape"), list)
        or len(contract["output"]["shape"]) != 3
        or contract["output"]["shape"][0] != 1
        or type(contract["output"]["shape"][1]) is not int
        or contract["output"]["shape"][1] <= 0
        or contract["output"]["shape"][2] != 7
    ):
        raise ParityError("mismatch")
    return contract


def _contract(value: object, expected: dict[str, object]) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"name", "type", "shape"}
        and value == expected
    )


def _closed_report(evidence: dict[str, object]) -> dict[str, object]:
    expected_keys = {
        "runtime",
        "source_input",
        "derived_input",
        "source_output",
        "derived_output",
        "declared_contracts_equal",
        "output_bytes_equal",
        "detections_equal",
        "accepted_ship",
    }
    runtime = evidence.get("runtime")
    source_input = evidence.get("source_input")
    source_output = evidence.get("source_output")
    if (
        set(evidence) != expected_keys
        or runtime != {"name": "onnxruntime-web", "version": "1.20.1"}
        or not _contract(
            source_input,
            {"name": "images", "type": "float32", "shape": [1, 3, 1024, 1024]},
        )
        or evidence.get("derived_input") != source_input
        or not isinstance(source_output, dict)
        or set(source_output) != {"name", "type", "shape"}
        or source_output.get("name") != "output0"
        or source_output.get("type") != "float32"
        or not isinstance(source_output.get("shape"), list)
        or len(source_output["shape"]) != 3
        or source_output["shape"][0] != 1
        or type(source_output["shape"][1]) is not int
        or source_output["shape"][1] <= 0
        or source_output["shape"][2] != 7
        or evidence.get("derived_output") != source_output
        or evidence.get("declared_contracts_equal") is not True
        or evidence.get("output_bytes_equal") is not True
        or evidence.get("detections_equal") is not True
        or evidence.get("accepted_ship") is not True
    ):
        raise ParityError("mismatch")
    return {
        "runtime": runtime,
        "input": source_input,
        "output": source_output,
        "output_bytes_equal": True,
        "detections_equal": True,
        "accepted_ship": True,
        "verdict": "PASS",
    }


def _write_report(report: Path, payload: dict[str, object]) -> None:
    destination = Path(os.path.abspath(report))
    parent = destination.parent
    if not parent.is_dir() or is_reparse_point(parent) or destination.is_symlink():
        raise ParityError("report")
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".parity-report-", dir=parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = -1
            stream.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise ParityError("report") from None


def run_parity(review_root: Path, report: Path) -> None:
    try:
        validate_admitted_assets(review_root)
    except Exception:
        raise ParityError("scope") from None
    payload = _closed_report(_browser_parity(review_root))
    _write_report(report, payload)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if (
            len(arguments) != 4
            or arguments[0] != "--review-root"
            or arguments[2] != "--report"
        ):
            raise ParityError("scope")
        run_parity(Path(arguments[1]), Path(arguments[3]))
        print("[OK] DEMO_MODEL_PARITY")
        return 0
    except ParityError as error:
        print(f"[FAIL] {error.code}")
        return 1
    except Exception:
        print(f"[FAIL] {ERROR_CODES['runtime']}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
