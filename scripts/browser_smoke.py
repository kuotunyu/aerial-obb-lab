"""Headless browser smoke for the real demo and its reusable local session."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import sys
from threading import Thread
from typing import Iterator
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo" / "web"
ORT_CDN_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js"
ORT_WASM_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/"
ORT_INTEGRITY = "sha384-RPL/K8tc0JVaNWsunkEmCzLeieefvFX2UCRLKLmLVChCI6P+CTKhzqF7VIeCc3Zp"
DEMO_MANIFEST_PATH = "/demo-model.json"
DEMO_MODEL_PATH = "/models/yolo26n-obb-privacy-sanitized.onnx"
DEMO_IMAGE_PATH = "/samples/boats.jpg"
DEMO_PROVENANCE = "Ultralytics YOLO26n-OBB · privacy-sanitized AGPL derivative"

ORT_STUB = r"""
globalThis.__ortCreateCount = 0;
globalThis.__demoRunCount = 0;
globalThis.ort = {
  env: { wasm: {} },
  Tensor: class Tensor {
    constructor(type, data, dims) {
      this.type = type;
      this.data = data;
      this.dims = dims;
    }
  },
  InferenceSession: {
    create: async (modelBytes) => {
      globalThis.__ortCreateCount += 1;
      if (!(modelBytes instanceof Uint8Array) || modelBytes.length === 0) {
        throw new Error("invalid model bytes");
      }
      return {
        inputNames: ["images"],
        outputNames: ["output0"],
        release: async () => {},
        run: async () => {
          globalThis.__demoRunCount += 1;
          return {
            output0: {
              dims: [1, 2, 7],
              data: new Float32Array([
                512, 512, 256, 128, 0.9, 1, Math.PI / 2,
                100, 100, 50, 40, 0.2, 2, 0
              ])
            }
          };
        }
      };
    }
  }
};
"""

SRI_STUB_SHIM = f"""
(() => {{
  const appendChild = Node.prototype.appendChild;
  Node.prototype.appendChild = function (child) {{
    const isPinnedRuntime = child instanceof HTMLScriptElement &&
      child.src === {ORT_CDN_URL!r} &&
      child.integrity === {ORT_INTEGRITY!r} &&
      child.crossOrigin === "anonymous";
    if (!isPinnedRuntime) return appendChild.call(this, child);
    child.integrity = "";
    const appended = appendChild.call(this, child);
    child.integrity = {ORT_INTEGRITY!r};
    return appended;
  }};
}})();
"""

REAL_INSTRUMENTATION = """
globalThis.__demoRunCount = 0;
globalThis.__obbStrokedPolygons = [];
let currentPath = [];
const originalBeginPath = CanvasRenderingContext2D.prototype.beginPath;
const originalMoveTo = CanvasRenderingContext2D.prototype.moveTo;
const originalLineTo = CanvasRenderingContext2D.prototype.lineTo;
const originalStroke = CanvasRenderingContext2D.prototype.stroke;
CanvasRenderingContext2D.prototype.beginPath = function (...args) {
  currentPath = [];
  return originalBeginPath.apply(this, args);
};
CanvasRenderingContext2D.prototype.moveTo = function (x, y, ...args) {
  currentPath.push([x, y]);
  return originalMoveTo.call(this, x, y, ...args);
};
CanvasRenderingContext2D.prototype.lineTo = function (x, y, ...args) {
  currentPath.push([x, y]);
  return originalLineTo.call(this, x, y, ...args);
};
CanvasRenderingContext2D.prototype.stroke = function (...args) {
  if (currentPath.length) globalThis.__obbStrokedPolygons.push([...currentPath]);
  return originalStroke.apply(this, args);
};
let runtime;
Object.defineProperty(globalThis, "ort", {
  configurable: true,
  get() { return runtime; },
  set(value) {
    runtime = value;
    const create = value.InferenceSession.create.bind(value.InferenceSession);
    value.InferenceSession.create = async (...args) => {
      const session = await create(...args);
      const run = session.run.bind(session);
      session.run = (...feeds) => {
        globalThis.__demoRunCount += 1;
        return run(...feeds);
      };
      return session;
    };
  },
});
"""


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


@contextmanager
def static_server() -> Iterator[str]:
    handler = partial(QuietHandler, directory=str(DEMO))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request_paths(requests: list[str]) -> list[str]:
    return [urlparse(request).path for request in requests]


def assert_real_demo_initial(page: object, requests: list[str], messages: list[str]) -> None:
    """Assert the first paint is the official original and has no lazy assets."""
    if "Synthetic" in page.locator("body").inner_text():
        raise RuntimeError("real demo initial state still exposes Synthetic-first UI")
    original = page.locator("#demoOriginalImage")
    if original.count() != 1 or not original.is_visible():
        raise RuntimeError("official original image is not visible before Detect")
    if urlparse(original.get_attribute("src") or "").path != DEMO_IMAGE_PATH.lstrip("/"):
        raise RuntimeError("official original image path is not exact")
    if page.locator("#demoFigureLabel").inner_text() != "原圖 · 尚未 Detect":
        raise RuntimeError("original image label is not exact")
    summary = page.evaluate(
        "[summaryCount.textContent, summaryTop.textContent, runtimeValue.textContent, "
        "modeBadge.textContent, provenanceValue.textContent]"
    )
    if summary != ["0", "—", "—", "尚未 Detect", "官方範例 · 尚未執行"]:
        raise RuntimeError(f"real demo initial summary is wrong: {summary!r}")
    if page.locator("#demoDetectBtn").inner_text() != "開始 Detect":
        raise RuntimeError("real demo primary action is not exact")
    if not page.locator("#viewToggleBtn").is_hidden():
        raise RuntimeError("original/result toggle is visible before a completed result")
    if not page.locator("#resultControls").is_hidden():
        raise RuntimeError("result filters are visible before a completed result")
    byom = page.locator("#byomPanel")
    if byom.count() != 1 or byom.get_attribute("open") is not None:
        raise RuntimeError("advanced BYOM controls are not collapsed initially")
    paths = _request_paths(requests)
    if DEMO_MANIFEST_PATH in paths or DEMO_MODEL_PATH in paths or ORT_CDN_URL in requests:
        raise RuntimeError("initial page requested a lazy manifest/runtime/model asset")
    if any(request.startswith(ORT_WASM_BASE) for request in requests):
        raise RuntimeError("initial page requested a lazy WASM asset")
    if messages:
        raise RuntimeError("initial page emitted console or page errors")


def exercise_real_demo_success(page: object, requests: list[str], messages: list[str]) -> int:
    """Run the committed derivative and assert the visible result contract."""
    page.locator("#demoDetectBtn").click()
    page.wait_for_function(
        "document.querySelector('#status').dataset.kind === 'success'",
        timeout=120_000,
    )
    if page.locator("#provenanceValue").inner_text() != DEMO_PROVENANCE:
        raise RuntimeError("real demo provenance is not exact")
    runtime = page.locator("#runtimeValue").inner_text()
    if not re.fullmatch(r"\d+ ms", runtime):
        raise RuntimeError(f"real demo runtime is not numeric: {runtime!r}")
    if page.locator("#modeBadge").inner_text() != "LOCAL BROWSER INFERENCE":
        raise RuntimeError("real demo mode badge is not exact")
    rows = page.locator("#resultsBody tr")
    if rows.count() < 1:
        raise RuntimeError("real demo produced no accepted detection rows")
    row_values = [row.locator("td").all_text_contents() for row in rows.all()]
    ship_rows = [row for row in row_values if row and row[0] == "ship"]
    if not ship_rows:
        raise RuntimeError("real demo produced no accepted ship row")
    polygons = page.evaluate("globalThis.__obbStrokedPolygons")
    if not polygons or any(len(polygon) != 4 for polygon in polygons):
        raise RuntimeError("real demo did not paint oriented polygon pixels")
    description = page.locator("#canvasDescription")
    if description.get_attribute("aria-live") is not None:
        raise RuntimeError("canvas description must remain non-live")
    ship = ship_rows[0]
    description_text = description.inner_text()
    if f"class={ship[0]}" not in description_text or f"confidence={ship[1]}" not in description_text:
        raise RuntimeError("visible table and canvas description are not synchronized")
    if page.locator("#demoDetectBtn").inner_text() != "再次 Detect":
        raise RuntimeError("completed demo primary action is not exact")
    if page.locator("#viewToggleBtn").inner_text() != "查看原圖":
        raise RuntimeError("completed demo toggle is not exact")
    if page.locator("#resultControls").is_hidden():
        raise RuntimeError("completed demo filters remain hidden")
    paths = _request_paths(requests)
    if paths.count(DEMO_MANIFEST_PATH) != 1 or paths.count(DEMO_MODEL_PATH) != 1:
        raise RuntimeError("real demo did not request the exact manifest/model once")
    if requests.count(ORT_CDN_URL) != 1:
        raise RuntimeError("real demo did not request the pinned runtime once")
    if not any(request.startswith(ORT_WASM_BASE) for request in requests):
        raise RuntimeError("real demo did not request pinned WASM")
    if messages:
        raise RuntimeError("real demo emitted console or page errors")
    run_count = page.evaluate("globalThis.__demoRunCount")
    if run_count != 1:
        raise RuntimeError(f"real demo run count is not one: {run_count!r}")
    return run_count


def assert_original_result_toggle(
    page: object, requests: list[str], run_counter: int
) -> None:
    """Switch views without fetching assets or running inference again."""
    request_count = len(requests)
    page.locator("#viewToggleBtn").click()
    if page.locator("#viewToggleBtn").inner_text() != "查看結果":
        raise RuntimeError("original view did not expose the result action")
    if page.locator("#demoFigureLabel").inner_text() != "原圖":
        raise RuntimeError("original view label is not exact")
    page.locator("#viewToggleBtn").click()
    if page.locator("#viewToggleBtn").inner_text() != "查看原圖":
        raise RuntimeError("result view did not restore the original action")
    if page.locator("#demoFigureLabel").inner_text() != "Detect 結果":
        raise RuntimeError("result view label is not exact")
    if page.evaluate("globalThis.__demoRunCount") != run_counter:
        raise RuntimeError("original/result switching ran inference again")
    if len(requests) != request_count:
        raise RuntimeError("original/result switching requested another asset")


def assert_demo_cached_filters(page: object, run_counter: int) -> None:
    """Re-filter the completed output without another session.run."""
    runtime = page.locator("#runtimeValue").inner_text()
    page.locator("#confSlider").evaluate(
        "slider => { slider.value = '0.90'; slider.dispatchEvent(new Event('input', { bubbles: true })); }"
    )
    page.locator("#confSlider").evaluate(
        "slider => { slider.value = '0.25'; slider.dispatchEvent(new Event('input', { bubbles: true })); }"
    )
    ship = page.locator('.class-cb[value="1"]')
    ship.check()
    if page.locator("#resultsBody tr").count() < 1:
        raise RuntimeError("cached ship filter lost the admitted ship result")
    ship.uncheck()
    if page.evaluate("globalThis.__demoRunCount") != run_counter:
        raise RuntimeError("cached filters ran inference again")
    if page.locator("#runtimeValue").inner_text() != runtime:
        raise RuntimeError("cached filters changed the completed runtime")


def assert_malformed_frozen_manifest_is_closed(page: object) -> None:
    """Reject arbitrary frozen inputs with a fixed public error code."""
    error = page.evaluate(
        """
        async () => {
          const path = Object.freeze({
            toString() { throw new Error("native-url-exception"); },
          });
          const malformed = Object.freeze({
            model: Object.freeze({ path }),
          });
          try {
            await DemoAssets.fetchVerifiedModel(malformed);
            return "NO_ERROR";
          } catch (failure) {
            return failure.message;
          }
        }
        """
    )
    if error not in {"DEMO_MANIFEST", "DEMO_MODEL_URL"}:
        raise RuntimeError(f"malformed frozen manifest escaped fixed diagnostics: {error!r}")


def _launch_options(executable_path: Path | None) -> dict[str, object]:
    options: dict[str, object] = {"headless": True, "args": ["--disable-gpu"]}
    if executable_path is not None:
        options["executable_path"] = str(executable_path)
    return options


def _record_errors(page: object, requests: list[str], messages: list[str]) -> None:
    page.on("request", lambda request: requests.append(request.url))
    page.on(
        "console",
        lambda message: messages.append(message.text) if message.type == "error" else None,
    )
    page.on("pageerror", lambda error: messages.append(str(error)))


def run_real_demo_success(
    executable_path: Path | None = None,
    base_url: str | None = None,
    screenshot: Path | None = None,
    mobile_screenshot: Path | None = None,
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required; run the locked development environment") from exc

    server = nullcontext(base_url) if base_url else static_server()
    with server as served_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(**_launch_options(executable_path))
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            requests: list[str] = []
            messages: list[str] = []
            page.add_init_script(REAL_INSTRUMENTATION)
            _record_errors(page, requests, messages)
            page.goto(f"{str(served_url).rstrip('/')}/", wait_until="networkidle")
            assert_real_demo_initial(page, requests, messages)
            run_counter = exercise_real_demo_success(page, requests, messages)
            assert_original_result_toggle(page, requests, run_counter)
            assert_demo_cached_filters(page, run_counter)
            if screenshot is not None:
                screenshot.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshot), full_page=True)
            if mobile_screenshot is not None:
                page.set_viewport_size({"width": 390, "height": 844})
                mobile_screenshot.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(mobile_screenshot), full_page=True)
        finally:
            browser.close()


def run_stubbed_cache(
    executable_path: Path | None = None,
    base_url: str | None = None,
) -> None:
    try:
        from playwright.sync_api import Route, sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required; run the locked development environment") from exc

    server = nullcontext(base_url) if base_url else static_server()
    with server as served_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(**_launch_options(executable_path))
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            requests: list[str] = []
            messages: list[str] = []
            page.add_init_script(SRI_STUB_SHIM)
            _record_errors(page, requests, messages)

            def stub_ort(route: Route) -> None:
                route.fulfill(
                    status=200,
                    content_type="application/javascript",
                    headers={"Access-Control-Allow-Origin": "*"},
                    body=ORT_STUB,
                )

            page.route(ORT_CDN_URL, stub_ort)
            page.goto(f"{str(served_url).rstrip('/')}/", wait_until="networkidle")
            assert_real_demo_initial(page, requests, messages)
            assert_malformed_frozen_manifest_is_closed(page)
            page.locator("#demoDetectBtn").click()
            page.wait_for_function(
                "document.querySelector('#status').dataset.kind === 'success'",
                timeout=30_000,
            )
            if page.evaluate("[globalThis.__ortCreateCount, globalThis.__demoRunCount]") != [1, 1]:
                raise RuntimeError("first stubbed demo run did not create and run one session")
            page.locator("#demoDetectBtn").click()
            page.wait_for_function("globalThis.__demoRunCount === 2", timeout=30_000)
            page.wait_for_function("document.querySelector('#status').dataset.kind === 'success'")
            if page.evaluate("globalThis.__ortCreateCount") != 1:
                raise RuntimeError("repeated Detect did not reuse the valid demo session")
            assert_demo_cached_filters(page, 2)
            if requests.count(ORT_CDN_URL) != 1:
                raise RuntimeError("stubbed repeated Detect requested the runtime again")
            if _request_paths(requests).count(DEMO_MODEL_PATH) != 1:
                raise RuntimeError("stubbed repeated Detect requested the model again")
            if messages:
                raise RuntimeError("stubbed cache scenario emitted console or page errors")
        finally:
            browser.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable-path", type=Path)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--mobile-screenshot", type=Path)
    parser.add_argument("--base-url", help="use an already-running static server")
    parser.add_argument("--scenario", choices=("real-demo-success", "stubbed-cache"))
    args = parser.parse_args(argv)
    try:
        if args.scenario == "real-demo-success":
            run_real_demo_success(
                args.executable_path,
                args.base_url,
                args.screenshot,
                args.mobile_screenshot,
            )
        elif args.scenario == "stubbed-cache":
            run_stubbed_cache(args.executable_path, args.base_url)
        else:
            run_real_demo_success(
                args.executable_path,
                args.base_url,
                args.screenshot,
                args.mobile_screenshot,
            )
            run_stubbed_cache(args.executable_path, args.base_url)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    if args.scenario == "real-demo-success":
        print("[OK] Real demo browser smoke: genuine local derivative inference")
    elif args.scenario == "stubbed-cache":
        print("[OK] Real demo cache smoke: one verified model and reusable session")
    else:
        print("[OK] Real demo browser smoke: genuine inference and cached session behavior")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
