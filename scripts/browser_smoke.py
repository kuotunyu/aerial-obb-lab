"""Headless browser smoke for the real demo and its reusable local session."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
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
DEMO_IMAGE_PATH = "/samples/airfield.jpg"
DEMO_PROVENANCE = "Ultralytics YOLO26n-OBB · privacy-sanitized AGPL derivative"
FIXED_CONSOLE_DIAGNOSTICS = frozenset(
    f"[AERIAL_OBB:{code}]"
    for code in (
        "DEMO_MANIFEST",
        "DEMO_MODEL_FETCH",
        "DEMO_MODEL_SIZE",
        "DEMO_MODEL_DIGEST",
        "DEMO_MODEL_URL",
        "RUNTIME_LOAD",
        "MODEL_CONTRACT",
        "IMAGE_DECODE",
        "INFERENCE_RUN",
        "OUTPUT_SCHEMA",
        "RENDER_RESULT",
    )
)

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


def _scenario_ort_stub(
    *,
    input_names: tuple[str, ...] = ("images",),
    output_names: tuple[str, ...] = ("output0",),
    run_mode: str = "success",
    create_mode: str = "immediate",
    lifecycle: bool = False,
) -> str:
    output = """
      return {
        output0: {
          dims: [1, 2, 7],
          data: new Float32Array([
            512, 512, 256, 128, 0.9, 1, Math.PI / 2,
            100, 100, 50, 40, 0.2, 2, 0
          ])
        }
      };
    """
    if run_mode == "failure":
        run_body = "throw new Error(globalThis.__privacyProbe.rawException);"
    elif run_mode == "output":
        run_body = """
          return {output0: {dims: [1, 1, 6], data: new Float32Array(6)}};
        """
    elif run_mode == "delayed":
        run_body = f"""
          return new Promise((resolve) => {{
            globalThis.__resolveDemoRun = () => resolve((() => {{ {output} }})());
          }});
        """
    else:
        run_body = output
    lifecycle_setup = "globalThis.__sessionLifecycle = [];" if lifecycle else ""
    lifecycle_create = (
        "globalThis.__sessionLifecycle.push(`candidate:${id}`);"
        if lifecycle
        else ""
    )
    lifecycle_release = (
        "globalThis.__sessionLifecycle.push(`release:${id}`);"
        if lifecycle
        else ""
    )
    candidate_names = json.dumps(list(input_names))
    candidate_outputs = json.dumps(list(output_names))
    delayed_create = ""
    if create_mode == "delayed-first":
        delayed_create = """
      if (id === 1) {
        await new Promise((resolve) => {
          globalThis.__resolveCandidateCreate = resolve;
        });
      }
        """
    return f"""
globalThis.__ortCreateCount = 0;
globalThis.__demoRunCount = 0;
globalThis.__releaseCount = 0;
{lifecycle_setup}
globalThis.ort = {{
  env: {{ wasm: {{}} }},
  Tensor: class Tensor {{
    constructor(type, data, dims) {{
      this.type = type;
      this.data = data;
      this.dims = dims;
    }}
  }},
  InferenceSession: {{
    create: async (modelBytes) => {{
      const id = ++globalThis.__ortCreateCount;
      if (!(modelBytes instanceof Uint8Array) || modelBytes.length === 0) {{
        throw new Error(globalThis.__privacyProbe.rawException);
      }}
      {delayed_create}
      {lifecycle_create}
      const badCandidate = Boolean(globalThis.__failNextCandidate);
      globalThis.__failNextCandidate = false;
      return {{
        inputNames: badCandidate ? ["wrong-input"] : {candidate_names},
        outputNames: {candidate_outputs},
        release: async () => {{
          globalThis.__releaseCount += 1;
          {lifecycle_release}
        }},
        run: async () => {{
          globalThis.__demoRunCount += 1;
          {run_body}
        }}
      }};
    }}
  }}
}};
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

CANVAS_INSTRUMENTATION = """
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
  if (currentPath.length) {
    globalThis.__obbStrokedPolygons.push({
      points: [...currentPath],
      strokeStyle: this.strokeStyle,
    });
  }
  return originalStroke.apply(this, args);
};
"""

REAL_INSTRUMENTATION = CANVAS_INSTRUMENTATION + """
globalThis.__demoRunCount = 0;
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


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request: object, client_address: object) -> None:
        pass


@contextmanager
def static_server() -> Iterator[str]:
    handler = partial(QuietHandler, directory=str(DEMO))
    server = QuietThreadingHTTPServer(("127.0.0.1", 0), handler)
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


def _assert_box_matches_viewport(page: object, selector: str, label: str) -> None:
    viewport = page.locator("#resultViewport").bounding_box()
    layer = page.locator(selector).bounding_box()
    if viewport is None or layer is None:
        raise RuntimeError(f"{label} lacks measurable viewport bounds")
    for edge in ("x", "y", "width", "height"):
        if abs(viewport[edge] - layer[edge]) > 1:
            raise RuntimeError(
                f"{label} does not share the result viewport bounds: "
                f"viewport={viewport!r}, layer={layer!r}"
            )


def _assert_empty_disabled_result_state(page: object) -> None:
    controls = page.locator("#resultControls")
    if not controls.is_visible():
        raise RuntimeError("result filters are not visible before Detect")
    if not page.locator("#confSlider").is_disabled():
        raise RuntimeError("confidence filter is enabled without cached output")
    if page.locator(".class-cb:not(:disabled)").count() != 0:
        raise RuntimeError("class filters are enabled without cached output")
    disabled_label_style = page.locator(".class-list label").first.evaluate(
        "label => [getComputedStyle(label).cursor, getComputedStyle(label).opacity]"
    )
    if disabled_label_style[0] != "not-allowed" or float(disabled_label_style[1]) > 0.62:
        raise RuntimeError("disabled class filter labels still look interactive")
    if page.locator("#resultsBody tr[data-empty='true']").count() != 1:
        raise RuntimeError("initial table lacks one explicit empty state")


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
    if summary != ["0", "—", "—", "尚未 Detect", "USGS／USDA NAIP · 尚未執行"]:
        raise RuntimeError(f"real demo initial summary is wrong: {summary!r}")
    if page.locator("#demoDetectBtn").inner_text() != "開始 Detect":
        raise RuntimeError("real demo primary action is not exact")
    if page.locator("#sampleState").inner_text() != "Original · ready":
        raise RuntimeError("official sample initial state is not exact")
    if not page.locator("#viewToggleBtn").is_hidden():
        raise RuntimeError("original/result toggle is visible before a completed result")
    _assert_empty_disabled_result_state(page)
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


def assert_sample_gallery_initial(page: object, requests: list[str], messages: list[str]) -> None:
    """Exercise the published selector before it can trigger any lazy model work."""
    options = page.locator("#sampleSelector .sample-option")
    if options.count() != 3:
        raise RuntimeError("real sample selector does not expose exactly three options")
    if page.locator('.sample-option[aria-pressed="true"]').get_attribute("data-sample-id") != "airfield":
        raise RuntimeError("airfield is not the exact initial sample")
    if page.locator("#demoOriginalImage").get_attribute("src") != "samples/airfield.jpg":
        raise RuntimeError("initial original is not the admitted airfield image")
    if any(path in _request_paths(requests) for path in (DEMO_MANIFEST_PATH, DEMO_MODEL_PATH)):
        raise RuntimeError("sample gallery loaded model resources before Detect")
    if messages:
        raise RuntimeError("sample gallery initial state emitted console or page errors")


def run_sample_gallery(
    executable_path: Path | None = None,
    base_url: str | None = None,
    screenshot: Path | None = None,
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
            _record_errors(page, requests, messages)
            page.goto(f"{str(served_url).rstrip('/')}/", wait_until="networkidle")
            assert_sample_gallery_initial(page, requests, messages)
            for sample_id in ("sports-complex", "harbor", "airfield"):
                page.locator(f'.sample-option[data-sample-id="{sample_id}"]').click()
                page.wait_for_function(
                    "([id]) => document.querySelector('#demoOriginalImage').getAttribute('src') === `samples/${id}.jpg`",
                    arg=[sample_id],
                )
                if page.locator('.sample-option[aria-pressed="true"]').get_attribute("data-sample-id") != sample_id:
                    raise RuntimeError("sample selection did not expose the active semantic state")
                if page.locator("#sampleState").inner_text() != "Original · ready" or page.locator("#summaryCount").inner_text() != "0":
                    raise RuntimeError("sample selection did not clear the former result state")
            if screenshot is not None:
                screenshot.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshot), full_page=True)
        finally:
            browser.close()


def run_held_decode(
    executable_path: Path | None = None, base_url: str | None = None,
) -> None:
    """A pending replacement decode must expose no stale result surface."""
    from playwright.sync_api import Route, sync_playwright
    server = nullcontext(base_url) if base_url else static_server()
    with server as served_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(**_launch_options(executable_path))
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.add_init_script(SRI_STUB_SHIM)
            page.route(ORT_CDN_URL, lambda route: route.fulfill(status=200, content_type="application/javascript", headers={"Access-Control-Allow-Origin": "*"}, body=ORT_STUB))
            held_images: list[Route] = []
            page.route("**/samples/sports-complex.jpg", lambda route: held_images.append(route))
            page.goto(f"{str(served_url).rstrip('/')}/", wait_until="domcontentloaded")
            page.wait_for_function("demoOriginalImage.complete && demoOriginalImage.naturalWidth > 0")
            page.locator("#demoDetectBtn").click()
            page.wait_for_function("document.querySelector('#status').dataset.kind === 'success'")
            page.locator('[data-sample-id="sports-complex"]').click()
            page.wait_for_function("() => document.querySelector('#sampleState').textContent.includes('Loading')")
            held_state = page.evaluate("[summaryCount.textContent, runtimeValue.textContent, demoDetectBtn.disabled, demoDetectBtn.textContent, sampleState.textContent, canvasFrame.hidden, viewToggleBtn.hidden]")
            if held_state[0] != "0" or held_state[2] is not True or held_state[5:] != [True, True]:
                raise RuntimeError(f"held decode leaves stale result or active Detect: {held_state!r}")
            if page.locator("#resultsBody tr[data-empty='true']").count() != 1 or not page.locator("#confSlider").is_disabled():
                raise RuntimeError("held decode did not clear result controls")
            if not held_images:
                raise RuntimeError("held-decode scenario did not intercept the selected sample response")
            for route in held_images:
                route.continue_()
            page.wait_for_function("document.querySelector('#sampleState').textContent === 'Original · ready'")
        finally:
            browser.close()


def assert_workbench_initial_layout(page: object) -> None:
    """Assert the initial desktop controls and viewport form a compact workbench."""
    if page.locator("#demoDetectBtn").evaluate(
        "node => node.closest('#controlRail')?.id || ''"
    ) != "controlRail":
        raise RuntimeError("demo action is not inside the compact control rail")
    if page.locator("#sampleCard").count() != 1:
        raise RuntimeError("official sample card is missing")
    rail = page.locator("#controlRail").bounding_box()
    viewport = page.locator("#resultViewport").bounding_box()
    if rail is None or viewport is None or viewport["x"] <= rail["x"] + rail["width"]:
        raise RuntimeError("desktop viewport is not to the right of the control rail")
    ratio = rail["width"] / (rail["width"] + viewport["width"])
    if not 0.27 <= ratio <= 0.35:
        raise RuntimeError(f"desktop workbench is not approximately 31/69: {ratio!r}")
    if page.locator(".demo-intro, .demo-action-zone").count() != 0:
        raise RuntimeError("retired full-width demo layout is still present")


def exercise_real_demo_success(page: object, requests: list[str], messages: list[str]) -> int:
    """Run the committed derivative and assert the visible result contract."""
    page.locator("#demoDetectBtn").click()
    page.wait_for_function(
        "document.querySelector('#status').dataset.kind === 'success'",
        timeout=120_000,
    )
    if page.locator("#status").inner_text() != "完成 · 可調整 filters。":
        raise RuntimeError("completed demo live status is not count-neutral")
    if page.locator("#provenanceValue").inner_text() != f"{DEMO_PROVENANCE} · 小型機場航拍範例":
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
    accepted_rows = [row for row in row_values if row and row[0] and row[0] != "尚未執行 Detect。"]
    if not accepted_rows:
        raise RuntimeError("real demo produced no accepted result row")
    polygons = page.evaluate("globalThis.__obbStrokedPolygons")
    if not polygons or any(len(polygon["points"]) != 4 for polygon in polygons):
        raise RuntimeError("real demo did not paint oriented polygon pixels")
    description = page.locator("#canvasDescription")
    if description.get_attribute("aria-live") is not None:
        raise RuntimeError("canvas description must remain non-live")
    visible_row = accepted_rows[0]
    description_text = description.inner_text()
    if f"class={visible_row[0]}" not in description_text or f"confidence={visible_row[1]}" not in description_text:
        raise RuntimeError("visible table and canvas description are not synchronized")
    if page.locator("#demoDetectBtn").inner_text() != "再次 Detect":
        raise RuntimeError("completed demo primary action is not exact")
    if page.locator("#sampleState").inner_text() != "Result · ready":
        raise RuntimeError("official sample result state is not exact")
    if page.locator("#viewToggleBtn").inner_text() != "查看原圖":
        raise RuntimeError("completed demo toggle is not exact")
    if page.locator("#resultControls").is_hidden():
        raise RuntimeError("completed demo filters remain hidden")
    if page.locator("#confSlider").is_disabled():
        raise RuntimeError("completed confidence filter remains disabled")
    if page.locator(".class-cb:not(:disabled)").count() == 0:
        raise RuntimeError("completed class filters remain disabled")
    if page.locator("#canvasFrame").is_hidden() or page.locator("#canvas").is_hidden():
        raise RuntimeError("completed demo result canvas is not visible")
    _assert_box_matches_viewport(page, "#canvas", "result canvas")
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
    visible_original = page.locator("#demoFigure img:visible")
    if visible_original.count() != 1:
        raise RuntimeError("original view does not expose exactly one original layer")
    _assert_box_matches_viewport(page, "#demoFigure img:visible", "active original layer")
    page.locator("#viewToggleBtn").click()
    if page.locator("#viewToggleBtn").inner_text() != "查看原圖":
        raise RuntimeError("result view did not restore the original action")
    if page.locator("#demoFigureLabel").inner_text() != "Detect 結果":
        raise RuntimeError("result view label is not exact")
    for width, height in ((640, 360), (1280, 500)):
        page.set_viewport_size({"width": width, "height": height})
        _assert_box_matches_viewport(page, "#canvas", "short-viewport result canvas")
        page.locator("#viewToggleBtn").click()
        _assert_box_matches_viewport(
            page, "#demoFigure img:visible", "short-viewport original layer"
        )
        page.locator("#viewToggleBtn").click()
    page.set_viewport_size({"width": 1280, "height": 720})
    if page.evaluate("globalThis.__demoRunCount") != run_counter:
        raise RuntimeError("original/result switching ran inference again")
    if len(requests) != request_count:
        raise RuntimeError("original/result switching requested another asset")


def assert_demo_cached_filters(
    page: object, requests: list[str], run_counter: int
) -> None:
    """Re-filter the completed output without another session.run."""
    runtime = page.locator("#runtimeValue").inner_text()
    request_count = len(requests)
    completion_status = page.locator("#status").inner_text()
    initial_count = int(page.locator("#summaryCount").inner_text())
    if completion_status != "完成 · 可調整 filters。" or initial_count <= 0:
        raise RuntimeError("cached-filter baseline lacks count-neutral completed state")
    page.locator("#confSlider").evaluate(
        "slider => { slider.value = '0.90'; slider.dispatchEvent(new Event('input', { bubbles: true })); }"
    )
    plane = page.locator('.class-cb[value="0"]')
    page.evaluate("globalThis.__obbStrokedPolygons = []")
    plane.check()
    if page.locator("#resultsBody tr[data-empty='true']").count() != 1:
        raise RuntimeError("empty cached filter lacks one explicit empty state")
    if page.locator("#canvasDescription").inner_text() != (
        "目前篩選條件下沒有 detections；canvas 沒有 oriented polygons。"
    ):
        raise RuntimeError("empty cached table and canvas description are not synchronized")
    if page.locator("#summaryCount").inner_text() != "0":
        raise RuntimeError("empty cached filter did not synchronize the summary count")
    if page.evaluate("globalThis.__obbStrokedPolygons") != []:
        raise RuntimeError("empty cached filter retained rendered polygons")
    if page.locator("#status").inner_text() != completion_status:
        raise RuntimeError("cached empty filter changed the count-neutral live status")
    plane.uncheck()
    page.locator("#confSlider").evaluate(
        "slider => { slider.value = '0.25'; slider.dispatchEvent(new Event('input', { bubbles: true })); }"
    )
    first_class = page.locator("#resultsBody tr:not([data-empty='true']) td").first.inner_text()
    class_index = page.evaluate(
        "([name]) => Array.from(document.querySelectorAll('.class-cb')).findIndex((box) => box.parentElement.textContent === name)",
        arg=[first_class],
    )
    if not isinstance(class_index, int) or class_index < 0:
        raise RuntimeError("cached filter could not locate the visible class")
    ship = page.locator(f'.class-cb[value="{class_index}"]')
    page.evaluate("globalThis.__obbStrokedPolygons = []")
    ship.check()
    rows = page.locator("#resultsBody tr:not([data-empty='true'])")
    if rows.count() < 1:
        raise RuntimeError("cached selected-class filter lost the admitted result")
    row_values = [row.locator("td").all_text_contents() for row in rows.all()]
    if any(not row or row[0] != first_class for row in row_values):
        raise RuntimeError("cached class filter retained a different table row")
    if int(page.locator("#summaryCount").inner_text()) != len(row_values):
        raise RuntimeError("cached class filter did not synchronize the summary count")
    description = page.locator("#canvasDescription").inner_text()
    if any(
        f"class={row[0]}" not in description or f"confidence={row[1]}" not in description
        for row in row_values
    ):
        raise RuntimeError("cached table and canvas description are not synchronized")
    polygons = page.evaluate("globalThis.__obbStrokedPolygons")
    if len(polygons) != len(row_values) or any(
        len(polygon["points"]) != 4 for polygon in polygons
    ):
        raise RuntimeError("cached table and rendered polygons are not synchronized")
    described = re.findall(
        r"class=[^;]+; confidence=[0-9.]+; center-x=([0-9.]+) px; "
        r"center-y=([0-9.]+) px; width=([0-9.]+) px; height=([0-9.]+) px; "
        r"angle=(-?[0-9.]+)°\.",
        description,
    )
    if len(described) != len(polygons):
        raise RuntimeError("cached polygon geometry lacks matching described detections")
    for values, polygon in zip(described, polygons):
        cx, cy, width, height, angle_degrees = map(float, values)
        angle = math.radians(angle_degrees)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        expected = [
            (
                cx + x * cos_angle - y * sin_angle,
                cy + x * sin_angle + y * cos_angle,
            )
            for x, y in (
                (-width / 2, -height / 2),
                (width / 2, -height / 2),
                (width / 2, height / 2),
                (-width / 2, height / 2),
            )
        ]
        if any(
            abs(actual_axis - expected_axis) > 0.5
            for actual, wanted in zip(polygon["points"], expected)
            for actual_axis, expected_axis in zip(actual, wanted)
        ):
            raise RuntimeError("cached polygon geometry or class color is out of sync")
    if page.evaluate("globalThis.__demoRunCount") != run_counter:
        raise RuntimeError("cached filters ran inference again")
    if len(requests) != request_count:
        raise RuntimeError("cached filters requested another asset")
    if page.locator("#runtimeValue").inner_text() != runtime:
        raise RuntimeError("cached filters changed the completed runtime")
    if page.locator("#status").inner_text() != completion_status:
        raise RuntimeError("cached filters changed the count-neutral live status")
    if page.locator("#viewToggleBtn").inner_text() != "查看原圖":
        raise RuntimeError("cached filters changed the current result view")
    if page.locator("#canvasFrame").is_hidden():
        raise RuntimeError("cached filters hid the current result view")


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


def _privacy_sentinels() -> tuple[str, ...]:
    drive_path = "C" + ":" + "\\" + "Users" + "\\" + "private-owner" + "\\" + "model.onnx"
    return (
        drive_path,
        "private-" + "local-model.onnx",
        "private-" + "response-body",
        "private-" + "model-metadata",
        "private-" + "native-exception",
        "private-" + "stack-frame",
        "private-" + "access-token",
        "signature=" + "private-query",
    )


def _record_evidence(
    page: object,
    requests: list[str],
    console_messages: list[str],
    page_errors: list[str],
) -> None:
    page.on("request", lambda request: requests.append(request.url))
    page.on("console", lambda message: console_messages.append(message.text))
    page.on("pageerror", lambda error: page_errors.append(str(error)))


def _assert_failure_cleanup(
    page: object, expected_sample_state: str = "Retry · available"
) -> None:
    if not page.locator("#demoOriginalImage").is_visible():
        raise RuntimeError("failure did not restore the official original image")
    if page.locator("#demoFigureLabel").inner_text() != "原圖 · 尚未 Detect":
        raise RuntimeError("failure did not restore the pre-Detect original label")
    summary = page.evaluate(
        "[summaryCount.textContent, summaryTop.textContent, runtimeValue.textContent]"
    )
    if summary != ["0", "—", "—"]:
        raise RuntimeError("failure retained completed summary state")
    if not page.locator("#canvasFrame").is_hidden():
        raise RuntimeError("failure retained the result canvas")
    _assert_empty_disabled_result_state(page)
    if not page.locator("#viewToggleBtn").is_hidden():
        raise RuntimeError("failure retained the original/result toggle")
    if page.locator("#canvasDescription").inner_text() != "尚無 detection result。":
        raise RuntimeError("failure retained the completed canvas description")
    if "LOCAL BROWSER INFERENCE" in page.locator("#modeBadge").inner_text():
        raise RuntimeError("failure retained the completed mode badge")
    if page.locator("#sampleState").inner_text() != expected_sample_state:
        raise RuntimeError(
            "failure exposed the wrong official sample state: "
            f"{page.locator('#sampleState').inner_text()!r}"
        )
    status = page.locator("#status").inner_text()
    if not any(
        action in status
        for action in ("重試", "重新整理", "重新執行", "BYOM", "改選", "選擇", "改用")
    ):
        raise RuntimeError("failure did not provide an actionable recovery")


def _assert_privacy_surfaces(
    page: object,
    requests: list[str],
    console_messages: list[str],
    page_errors: list[str],
    sentinels: tuple[str, ...],
) -> None:
    if page_errors:
        raise RuntimeError("failure evidence retained a page error or stack")
    if any(message not in FIXED_CONSOLE_DIAGNOSTICS for message in console_messages):
        raise RuntimeError("failure evidence retained a non-fixed console diagnostic")
    text_surfaces = "\n".join([
        page.locator("html").evaluate("node => node.outerHTML"),
        page.locator("body").inner_text(),
        page.evaluate("(globalThis.__privacyRenderedText || []).join('\\n')"),
        *requests,
        *console_messages,
        *page_errors,
    ])
    screenshot_metadata_bytes = page.screenshot()
    for sentinel in sentinels:
        if sentinel in text_surfaces:
            raise RuntimeError("failure evidence exposed a private sentinel")
        if sentinel.encode("utf-8") in screenshot_metadata_bytes:
            raise RuntimeError("screenshot metadata retained a private sentinel")
    forbidden_patterns = (
        r"(?i)\b[a-z]:[\\/](?:users|documents and settings)[\\/]",
        r"(?i)/(?:home|users)/[^/\s]+/",
        r"(?i)\bfile://",
        r"(?i)(?:authorization|bearer|access[_-]?token|api[_-]?key)\s*[:=]",
        r"(?i)[?&](?:sig|signature|token|key)=[^&\s]+",
        r"(?i)(?:traceback \(most recent call last\)|\bat\s+\S+\s*\(|\.js:\d+:\d+)",
    )
    if any(re.search(pattern, text_surfaces) for pattern in forbidden_patterns):
        raise RuntimeError("failure evidence matched a closed privacy diagnostic pattern")
    if any(urlparse(request).query for request in requests):
        raise RuntimeError("failure network report retained a signed query")


def _assert_privacy_oracle_rejects_open_diagnostics(
    page: object,
    requests: list[str],
    sentinels: tuple[str, ...],
) -> None:
    unsafe_cases = (
        (["unexpected browser diagnostic"], []),
        ([], ["Error: unexpected browser failure"]),
        (["at render (app.js:1:1)"], []),
        (["authorization=" + "unexpected-value"], []),
    )
    for console_messages, page_errors in unsafe_cases:
        try:
            _assert_privacy_surfaces(
                page, requests, console_messages, page_errors, sentinels
            )
        except RuntimeError:
            continue
        raise RuntimeError("privacy oracle accepted an open browser diagnostic")
    page.evaluate(
        "value => canvas.getContext('2d').fillText(value, 1, 12)", sentinels[1]
    )
    try:
        _assert_privacy_surfaces(page, requests, [], [], sentinels)
    except RuntimeError:
        page.evaluate("globalThis.__privacyRenderedText = []")
    else:
        raise RuntimeError("privacy oracle accepted user-visible canvas text")


def run_manifest_failure(
    executable_path: Path | None = None,
    base_url: str | None = None,
) -> None:
    try:
        from playwright.sync_api import Route, sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required; run the locked development environment") from exc

    sentinels = _privacy_sentinels()
    manifest = json.loads((DEMO / "demo-model.json").read_text(encoding="utf-8"))
    server = nullcontext(base_url) if base_url else static_server()
    with server as served_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(**_launch_options(executable_path))
        try:
            for variant in ("fetch", "status", "schema"):
                page = browser.new_page(viewport={"width": 1280, "height": 720})
                requests: list[str] = []
                console_messages: list[str] = []
                page_errors: list[str] = []
                _record_evidence(page, requests, console_messages, page_errors)

                def fail_manifest(route: Route, selected: str = variant) -> None:
                    if selected == "fetch":
                        route.abort("failed")
                    elif selected == "status":
                        route.fulfill(status=503, body=sentinels[2])
                    else:
                        malformed = dict(manifest)
                        malformed["privateProbe"] = sentinels[3]
                        route.fulfill(
                            status=200,
                            content_type="application/json",
                            body=json.dumps(malformed),
                        )

                page.route(f"**{DEMO_MANIFEST_PATH}", fail_manifest)
                page.goto(f"{str(served_url).rstrip('/')}/", wait_until="networkidle")
                _install_privacy_probe(page, sentinels)
                page.locator("#demoDetectBtn").click()
                page.wait_for_function(
                    "document.querySelector('#status').dataset.kind === 'error'",
                    timeout=30_000,
                )
                _assert_failure_cleanup(page)
                _assert_privacy_surfaces(
                    page, requests, console_messages, page_errors, sentinels
                )
                _assert_privacy_oracle_rejects_open_diagnostics(
                    page, requests, sentinels
                )
                page.close()
        finally:
            browser.close()


def _install_privacy_probe(page: object, sentinels: tuple[str, ...]) -> None:
    page.evaluate(
        """
        values => {
          globalThis.__privacyProbe = {
            localPath: values[0],
            localFilename: values[1],
            responseBody: values[2],
            modelMetadata: values[3],
            rawException: values[4],
            stackFrame: values[5],
            token: values[6],
            signedQuery: values[7],
          };
          globalThis.__privacyRenderedText = [];
          for (const method of ['fillText', 'strokeText']) {
            const original = CanvasRenderingContext2D.prototype[method];
            CanvasRenderingContext2D.prototype[method] = function (text, ...args) {
              globalThis.__privacyRenderedText.push(String(text));
              return original.call(this, text, ...args);
            };
          }
        }
        """,
        list(sentinels),
    )


def _fulfill_runtime(route: object, body: str) -> None:
    route.fulfill(
        status=200,
        content_type="application/javascript",
        headers={"Access-Control-Allow-Origin": "*"},
        body=body,
    )


def run_model_digest_failure(
    executable_path: Path | None = None,
    base_url: str | None = None,
) -> None:
    try:
        from playwright.sync_api import Route, sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required; run the locked development environment") from exc

    sentinels = _privacy_sentinels()
    admitted = (DEMO / DEMO_MODEL_PATH.lstrip("/")).read_bytes()
    variants = (
        ("truncated", admitted[:-1], "大小驗證失敗"),
        ("changed", bytes([admitted[0] ^ 1]) + admitted[1:], "完整性驗證失敗"),
    )
    server = nullcontext(base_url) if base_url else static_server()
    with server as served_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(**_launch_options(executable_path))
        try:
            for _variant, model_bytes, expected_copy in variants:
                page = browser.new_page(viewport={"width": 1280, "height": 720})
                requests: list[str] = []
                console_messages: list[str] = []
                page_errors: list[str] = []
                page.add_init_script(SRI_STUB_SHIM)
                _record_evidence(page, requests, console_messages, page_errors)
                page.route(ORT_CDN_URL, lambda route: _fulfill_runtime(route, ORT_STUB))
                def serve_model(route: Route) -> None:
                    route.fulfill(
                        status=200,
                        content_type="application/onnx",
                        body=model_bytes,
                    )

                page.route(f"**{DEMO_MODEL_PATH}", serve_model)
                page.goto(f"{str(served_url).rstrip('/')}/", wait_until="networkidle")
                _install_privacy_probe(page, sentinels)
                page.locator("#demoDetectBtn").click()
                page.wait_for_function(
                    "document.querySelector('#status').dataset.kind === 'error'",
                    timeout=30_000,
                )
                if expected_copy not in page.locator("#status").inner_text():
                    raise RuntimeError("model integrity failure used the wrong fixed recovery")
                _assert_failure_cleanup(page)
                _assert_privacy_surfaces(
                    page, requests, console_messages, page_errors, sentinels
                )
                page.close()
        finally:
            browser.close()


def run_runtime_failure(
    executable_path: Path | None = None,
    base_url: str | None = None,
) -> None:
    try:
        from playwright.sync_api import Route, sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required; run the locked development environment") from exc

    sentinels = _privacy_sentinels()
    server = nullcontext(base_url) if base_url else static_server()
    with server as served_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(**_launch_options(executable_path))
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            requests: list[str] = []
            console_messages: list[str] = []
            page_errors: list[str] = []
            attempts = {"count": 0}
            page.add_init_script(SRI_STUB_SHIM)
            _record_evidence(page, requests, console_messages, page_errors)

            def runtime_route(route: Route) -> None:
                attempts["count"] += 1
                if attempts["count"] == 1:
                    _fulfill_runtime(route, "/* runtime unavailable */")
                else:
                    _fulfill_runtime(route, ORT_STUB)

            page.route(ORT_CDN_URL, runtime_route)
            page.goto(f"{str(served_url).rstrip('/')}/", wait_until="networkidle")
            _install_privacy_probe(page, sentinels)
            page.locator("#demoDetectBtn").click()
            if page.locator("#sampleState").inner_text() != "Loading · local browser":
                raise RuntimeError("runtime attempt did not mark the official sample loading")
            page.wait_for_function(
                "document.querySelector('#status').dataset.kind === 'error'",
                timeout=30_000,
            )
            _assert_failure_cleanup(page)
            if page.locator("#runtimeRetryBtn").is_hidden():
                raise RuntimeError("runtime failure did not expose the focused retry action")
            if requests.count(ORT_CDN_URL) != 1:
                raise RuntimeError("runtime failure made an unexpected pinned request count")
            page.locator("#runtimeRetryBtn").click()
            if page.locator("#sampleState").inner_text() != "Loading · local browser":
                raise RuntimeError("runtime retry did not restore the official sample loading state")
            page.wait_for_function(
                "document.querySelector('#status').dataset.kind === 'success'",
                timeout=30_000,
            )
            if requests.count(ORT_CDN_URL) != 2:
                raise RuntimeError("runtime retry did not make one new pinned request")
            if page.evaluate("globalThis.__demoRunCount") != 1:
                raise RuntimeError("runtime retry did not complete one genuine pipeline run")
            if page.locator("#sampleState").inner_text() != "Result · ready":
                raise RuntimeError("runtime retry did not restore the official sample result state")
            _assert_privacy_surfaces(
                page, requests, console_messages, page_errors, sentinels
            )
        finally:
            browser.close()


def _run_stubbed_failure(
    scenario: str,
    stub: str,
    *,
    executable_path: Path | None = None,
    base_url: str | None = None,
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required; run the locked development environment") from exc

    sentinels = _privacy_sentinels()
    server = nullcontext(base_url) if base_url else static_server()
    with server as served_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(**_launch_options(executable_path))
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            requests: list[str] = []
            console_messages: list[str] = []
            page_errors: list[str] = []
            page.add_init_script(SRI_STUB_SHIM)
            _record_evidence(page, requests, console_messages, page_errors)
            page.route(ORT_CDN_URL, lambda route: _fulfill_runtime(route, stub))
            page.goto(f"{str(served_url).rstrip('/')}/", wait_until="networkidle")
            _install_privacy_probe(page, sentinels)
            if scenario == "render":
                page.evaluate(
                    """
                    () => {
                      OBB.rotatedCorners = () => {
                        throw new Error(globalThis.__privacyProbe.stackFrame);
                      };
                    }
                    """
                )
            page.locator("#demoDetectBtn").click()
            page.wait_for_function(
                "['error', 'success'].includes(document.querySelector('#status').dataset.kind)",
                timeout=30_000,
            )
            if page.locator("#status").get_attribute("data-kind") != "error":
                raise RuntimeError(f"{scenario} failure was overwritten by completed status")
            _assert_failure_cleanup(page)
            if scenario == "session" and page.evaluate("globalThis.__releaseCount") != 1:
                raise RuntimeError("invalid candidate session was not released")
            _assert_privacy_surfaces(
                page, requests, console_messages, page_errors, sentinels
            )
        finally:
            browser.close()


def run_session_failure(
    executable_path: Path | None = None,
    base_url: str | None = None,
) -> None:
    _run_stubbed_failure(
        "session",
        _scenario_ort_stub(input_names=("wrong-input",)),
        executable_path=executable_path,
        base_url=base_url,
    )


def run_run_failure(
    executable_path: Path | None = None,
    base_url: str | None = None,
) -> None:
    _run_stubbed_failure(
        "run",
        _scenario_ort_stub(run_mode="failure"),
        executable_path=executable_path,
        base_url=base_url,
    )


def run_output_failure(
    executable_path: Path | None = None,
    base_url: str | None = None,
) -> None:
    _run_stubbed_failure(
        "output",
        _scenario_ort_stub(run_mode="output"),
        executable_path=executable_path,
        base_url=base_url,
    )


def run_render_failure(
    executable_path: Path | None = None,
    base_url: str | None = None,
) -> None:
    _run_stubbed_failure(
        "render",
        _scenario_ort_stub(),
        executable_path=executable_path,
        base_url=base_url,
    )


def _set_model_file(page: object, name: str = "candidate.onnx") -> None:
    page.locator("#byomPanel").evaluate("panel => { panel.open = true; }")
    page.locator("#modelInput").set_input_files(
        {"name": name, "mimeType": "application/octet-stream", "buffer": b"candidate"}
    )


def run_stale_generation(
    executable_path: Path | None = None,
    base_url: str | None = None,
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required; run the locked development environment") from exc

    sentinels = _privacy_sentinels()
    server = nullcontext(base_url) if base_url else static_server()
    with server as served_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(**_launch_options(executable_path))
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            requests: list[str] = []
            console_messages: list[str] = []
            page_errors: list[str] = []
            page.add_init_script(SRI_STUB_SHIM)
            _record_evidence(page, requests, console_messages, page_errors)
            page.route(
                ORT_CDN_URL,
                lambda route: _fulfill_runtime(route, _scenario_ort_stub()),
            )
            page.goto(f"{str(served_url).rstrip('/')}/", wait_until="networkidle")
            _install_privacy_probe(page, sentinels)
            page.evaluate(
                """
                () => {
                  const actualFetch = globalThis.fetch.bind(globalThis);
                  globalThis.__modelFetchStarted = false;
                  globalThis.__modelFetchAborted = false;
                  globalThis.fetch = (input, init = {}) => {
                    const url = String(input instanceof Request ? input.url : input);
                    if (!url.endsWith('/models/yolo26n-obb-privacy-sanitized.onnx')) {
                      return actualFetch(input, init);
                    }
                    globalThis.__modelFetchStarted = true;
                    return new Promise((_resolve, reject) => {
                      init.signal?.addEventListener('abort', () => {
                        globalThis.__modelFetchAborted = true;
                        reject(new DOMException('aborted', 'AbortError'));
                      }, {once: true});
                    });
                  };
                }
                """
            )
            page.locator("#demoDetectBtn").click()
            page.wait_for_function("globalThis.__modelFetchStarted === true")
            _set_model_file(page)
            page.wait_for_function(
                "document.querySelector('#modelLabel').textContent === 'Local ONNX model ready'"
            )
            page.wait_for_timeout(100)
            if not page.evaluate("globalThis.__modelFetchAborted"):
                raise RuntimeError("BYOM transition did not abort the stale demo model fetch")
            _assert_privacy_surfaces(
                page, requests, console_messages, page_errors, sentinels
            )
            page.close()

            page = browser.new_page(viewport={"width": 1280, "height": 720})
            requests = []
            console_messages = []
            page_errors = []
            page.add_init_script(SRI_STUB_SHIM)
            _record_evidence(page, requests, console_messages, page_errors)
            page.route(
                ORT_CDN_URL,
                lambda route: _fulfill_runtime(
                    route, _scenario_ort_stub(run_mode="delayed")
                ),
            )
            page.goto(f"{str(served_url).rstrip('/')}/", wait_until="networkidle")
            _install_privacy_probe(page, sentinels)
            page.locator("#demoDetectBtn").click()
            page.wait_for_function("typeof globalThis.__resolveDemoRun === 'function'")
            _set_model_file(page)
            page.wait_for_function(
                "document.querySelector('#modelLabel').textContent === 'Local ONNX model ready'"
            )
            page.evaluate("globalThis.__resolveDemoRun()")
            page.wait_for_timeout(100)
            if page.locator("#provenanceValue").inner_text() == DEMO_PROVENANCE:
                raise RuntimeError("stale demo completion replaced the BYOM selection")
            _assert_empty_disabled_result_state(page)
            _assert_privacy_surfaces(
                page, requests, console_messages, page_errors, sentinels
            )
            page.close()

            page = browser.new_page(viewport={"width": 1280, "height": 720})
            requests = []
            console_messages = []
            page_errors = []
            page.add_init_script(SRI_STUB_SHIM)
            _record_evidence(page, requests, console_messages, page_errors)
            page.route(
                ORT_CDN_URL,
                lambda route: _fulfill_runtime(
                    route,
                    _scenario_ort_stub(
                        create_mode="delayed-first", lifecycle=True
                    ),
                ),
            )
            page.goto(f"{str(served_url).rstrip('/')}/", wait_until="networkidle")
            _install_privacy_probe(page, sentinels)
            page.locator("#demoDetectBtn").click()
            page.wait_for_function(
                "typeof globalThis.__resolveCandidateCreate === 'function'"
            )
            _set_model_file(page)
            page.wait_for_function(
                "document.querySelector('#modelLabel').textContent === 'Local ONNX model ready'"
            )
            if page.evaluate("globalThis.__releaseCount") != 0:
                raise RuntimeError("active work was released before the stale candidate resolved")
            page.evaluate("globalThis.__resolveCandidateCreate()")
            page.wait_for_timeout(100)
            lifecycle = page.evaluate("globalThis.__sessionLifecycle")
            if "release:1" not in lifecycle:
                raise RuntimeError("stale delayed candidate was not released")
            if lifecycle[-1] != "release:1":
                raise RuntimeError("stale delayed candidate was installed instead of released")
            if page.locator("#provenanceValue").inner_text() == DEMO_PROVENANCE:
                raise RuntimeError("stale delayed candidate replaced the BYOM selection")
            _assert_empty_disabled_result_state(page)
            _assert_privacy_surfaces(
                page, requests, console_messages, page_errors, sentinels
            )
        finally:
            browser.close()


def run_byom_transition(
    executable_path: Path | None = None,
    base_url: str | None = None,
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required; run the locked development environment") from exc

    sentinels = (*_privacy_sentinels(), "private-selected-image.jpg")
    server = nullcontext(base_url) if base_url else static_server()
    with server as served_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(**_launch_options(executable_path))
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            requests: list[str] = []
            console_messages: list[str] = []
            page_errors: list[str] = []
            page.add_init_script(SRI_STUB_SHIM)
            _record_evidence(page, requests, console_messages, page_errors)
            page.route(
                ORT_CDN_URL,
                lambda route: _fulfill_runtime(
                    route, _scenario_ort_stub(lifecycle=True)
                ),
            )
            page.goto(f"{str(served_url).rstrip('/')}/", wait_until="networkidle")
            _install_privacy_probe(page, sentinels)
            _assert_empty_disabled_result_state(page)
            page.locator("#demoDetectBtn").click()
            page.wait_for_function(
                "document.querySelector('#status').dataset.kind === 'success'"
            )
            page.evaluate("globalThis.__failNextCandidate = true")
            _set_model_file(page, sentinels[1])
            page.wait_for_function(
                "document.querySelector('#status').dataset.kind === 'error'"
            )
            lifecycle = page.evaluate("globalThis.__sessionLifecycle")
            if "release:1" in lifecycle or "release:2" not in lifecycle:
                raise RuntimeError("invalid candidate did not preserve the active demo session")
            _assert_failure_cleanup(page, "Original · ready")

            _set_model_file(page, "candidate-valid.onnx")
            page.wait_for_function(
                "document.querySelector('#modelLabel').textContent === 'Local ONNX model ready'"
            )
            lifecycle = page.evaluate("globalThis.__sessionLifecycle")
            if lifecycle.index("candidate:3") > lifecycle.index("release:1"):
                raise RuntimeError("old session released before the valid candidate was ready")
            if not page.locator("#detectBtn").is_disabled():
                raise RuntimeError("BYOM model selection reused the demo image as local input")
            _assert_empty_disabled_result_state(page)

            page.locator("#fileInput").set_input_files(
                {
                    "name": sentinels[1],
                    "mimeType": "image/png",
                    "buffer": ("invalid-" + sentinels[2]).encode("utf-8"),
                }
            )
            page.wait_for_function(
                "document.querySelector('#status').dataset.kind === 'error'"
            )
            _assert_failure_cleanup(page, "Original · ready")

            valid_image_name = sentinels[-1]
            page.locator("#fileInput").set_input_files(
                {
                    "name": valid_image_name,
                    "mimeType": "image/jpeg",
                    "buffer": (DEMO / DEMO_IMAGE_PATH.lstrip("/")).read_bytes(),
                }
            )
            page.wait_for_function(
                "document.querySelector('#detectBtn').disabled === false"
            )
            byom_original = page.locator("#viewportByomImage")
            if not byom_original.is_visible():
                raise RuntimeError("selected BYOM image is not the visible original layer")
            _assert_box_matches_viewport(page, "#viewportByomImage", "BYOM original layer")
            if valid_image_name in (byom_original.get_attribute("alt") or ""):
                raise RuntimeError("BYOM original alt text exposes the local filename")
            if valid_image_name in page.locator("body").inner_text():
                raise RuntimeError("BYOM selection exposes the local filename")
            page.locator("#detectBtn").click()
            page.wait_for_function(
                "document.querySelector('#status').dataset.kind === 'success'"
            )
            if page.locator("#provenanceValue").inner_text() != "Local files":
                raise RuntimeError("BYOM run did not use local-file provenance")
            if page.locator("#demoDetectBtn").is_disabled():
                raise RuntimeError("BYOM completion left return-to-demo disabled")
            page.locator("#demoDetectBtn").click()
            page.wait_for_function(
                "document.querySelector('#provenanceValue').textContent === "
                + json.dumps(f"{DEMO_PROVENANCE} · 小型機場航拍範例", ensure_ascii=False)
            )
            if page.locator("#status").get_attribute("data-kind") != "success":
                raise RuntimeError("return to demo did not complete a local inference")
            _assert_privacy_surfaces(
                page, requests, console_messages, page_errors, sentinels
            )
        finally:
            browser.close()


def run_accessibility(
    executable_path: Path | None = None,
    base_url: str | None = None,
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
            page.add_init_script(SRI_STUB_SHIM)
            page.route(ORT_CDN_URL, lambda route: _fulfill_runtime(route, ORT_STUB))
            page.goto(f"{str(served_url).rstrip('/')}/", wait_until="networkidle")
            page.evaluate("document.activeElement.blur()")
            page.keyboard.press("Tab")
            if page.evaluate("document.activeElement?.className") != "skip-link":
                raise RuntimeError("skip link is not the first keyboard focus target")
            page.keyboard.press("Enter")
            if page.evaluate("document.activeElement?.id") != "mainContent":
                raise RuntimeError("skip link did not focus main#mainContent")
            if not page.evaluate(
                """
                () => Boolean(
                  claimBoundary.compareDocumentPosition(demoDetectBtn) &
                  Node.DOCUMENT_POSITION_FOLLOWING
                )
                """
            ):
                raise RuntimeError("claim notice does not precede the first primary control")
            claim_box = page.locator("#claimBoundary").bounding_box()
            action_box = page.locator("#demoDetectBtn").bounding_box()
            if claim_box is None or action_box is None or claim_box["y"] >= action_box["y"]:
                raise RuntimeError("claim notice lost visual priority over the first primary control")
            names = page.evaluate(
                """
                () => ({
                  model: [modelInput.name, modelInput.labels.length],
                  image: [fileInput.name, fileInput.labels.length],
                  confidence: [confSlider.name, confSlider.labels.length],
                  classes: [...document.querySelectorAll('.class-cb')].map(
                    item => [item.name, item.labels.length]
                  ),
                })
                """
            )
            if names["model"] != ["model", 1] or names["image"] != ["image", 1]:
                raise RuntimeError("file inputs lost stable names or labels")
            if names["confidence"] != ["confidence", 1]:
                raise RuntimeError("confidence input lost its stable name or label")
            if any(item != ["class-filter", 1] for item in names["classes"]):
                raise RuntimeError("class inputs lost stable names or labels")
            headings = page.evaluate(
                """
                () => ({
                  page: document.querySelector('h1')?.tagName,
                  controls: controlsTitle.tagName,
                  result: resultTitle.tagName,
                  sample: sampleTitle.tagName,
                  table: tableTitle.tagName,
                  byom: byomPanel.querySelector('summary')?.tagName,
                })
                """
            )
            if headings != {
                "page": "H1",
                "controls": "H2",
                "result": "H2",
                "sample": "H3",
                "table": "H3",
                "byom": "SUMMARY",
            }:
                raise RuntimeError(f"workbench heading hierarchy is wrong: {headings!r}")
            description = page.locator("#canvasDescription")
            if (
                page.locator("#canvas").get_attribute("aria-describedby")
                != "canvasDescription"
                or page.locator("#canvasDescription").count() != 1
            ):
                raise RuntimeError("canvas lost its single textual-description target")
            if description.get_attribute("aria-live") is not None:
                raise RuntimeError("canvas description unexpectedly became live")
            if description.inner_text() != "尚無 detection result。":
                raise RuntimeError("empty canvas description is stale")
            status = page.locator("#status")
            polite_regions = page.locator('[aria-live="polite"]')
            if (
                status.get_attribute("aria-live") != "polite"
                or polite_regions.count() != 1
                or polite_regions.first.get_attribute("id") != "status"
            ):
                raise RuntimeError("status is not the only deliberate polite live region")

            page.locator("main#mainContent").focus()
            initial_focus_order = []
            for _ in range(20):
                page.keyboard.press("Tab")
                focused = page.evaluate(
                    """
                    () => {
                      const active = document.activeElement;
                      if (!active) return '';
                      if (active.id) return `#${active.id}`;
                      if (active.matches('#byomPanel summary')) return '#byomPanel summary';
                      if (active.matches('.source-links a')) return '.source-links a';
                      return active.tagName.toLowerCase();
                    }
                    """
                )
                initial_focus_order.append(focused)
                if focused == ".source-links a":
                    break
            try:
                detect_index = initial_focus_order.index("#demoDetectBtn")
                byom_index = initial_focus_order.index("#byomPanel summary")
                source_index = initial_focus_order.index(".source-links a")
            except ValueError as exc:
                raise RuntimeError(
                    f"initial keyboard order misses a required stop: {initial_focus_order!r}"
                ) from exc
            if not detect_index < byom_index < source_index:
                raise RuntimeError(
                    f"initial keyboard order leaves the workbench sequence: {initial_focus_order!r}"
                )
            if not page.evaluate(
                """
                () => Boolean(
                  resultControls.compareDocumentPosition(byomPanel) &
                  Node.DOCUMENT_POSITION_FOLLOWING
                )
                """
            ):
                raise RuntimeError("BYOM disclosure no longer follows the filters")

            page.locator("#demoDetectBtn").click()
            page.wait_for_function(
                "document.querySelector('#status').dataset.kind === 'success'"
            )
            row = page.locator("#resultsBody tr").first.locator("td").all_text_contents()
            description_text = description.inner_text()
            if f"class={row[0]}" not in description_text or f"confidence={row[1]}" not in description_text:
                raise RuntimeError("accessible description diverged from the sorted visible table")
            if "confidence=" in status.inner_text() or "center-x=" in status.inner_text():
                raise RuntimeError("live status duplicates the full detection announcement")

            if page.locator("#confSlider").is_disabled():
                raise RuntimeError("successful Detect left confidence outside keyboard workflow")
            if page.locator(".class-cb:not(:disabled)").count() == 0:
                raise RuntimeError("successful Detect left class filters outside keyboard workflow")
            if page.evaluate("document.activeElement?.id") != "demoDetectBtn":
                raise RuntimeError("successful Detect moved focus away from the demo action")
            success_focus_order = ["#demoDetectBtn"]
            for _ in range(64):
                page.keyboard.press("Tab")
                focused = page.evaluate(
                    """
                    () => {
                      const active = document.activeElement;
                      if (!active) return '';
                      if (active.matches('.class-cb')) return '.class-cb';
                      if (active.matches('#byomPanel summary')) return '#byomPanel summary';
                      if (active.matches('.table-scroll')) return '.table-scroll';
                      if (active.matches('.source-links a')) return '.source-links a';
                      if (active.id) return `#${active.id}`;
                      return active.tagName.toLowerCase();
                    }
                    """
                )
                success_focus_order.append(focused)
                if focused == ".source-links a":
                    break
            required_success_stops = [
                "#demoDetectBtn",
                "#confSlider",
                ".class-cb",
                "#byomPanel summary",
                ".table-scroll",
                ".source-links a",
            ]
            try:
                success_indices = [
                    success_focus_order.index(stop) for stop in required_success_stops
                ]
            except ValueError as exc:
                raise RuntimeError(
                    "successful keyboard order misses an enabled workbench stop: "
                    f"{success_focus_order!r}"
                ) from exc
            if success_indices != sorted(success_indices):
                raise RuntimeError(
                    "successful keyboard order leaves the workbench sequence: "
                    f"{success_focus_order!r}"
                )
            if not page.locator("#viewToggleBtn").is_hidden():
                try:
                    view_index = success_focus_order.index("#viewToggleBtn")
                except ValueError as exc:
                    raise RuntimeError(
                        "successful keyboard order misses the visible original/result action"
                    ) from exc
                if not success_indices[0] < view_index < success_indices[1]:
                    raise RuntimeError(
                        "original/result action is outside the successful keyboard sequence"
                    )

            for selector in (
                "#demoDetectBtn",
                "#byomPanel summary",
                ".table-scroll",
                ".source-links a",
            ):
                target = page.locator(selector).first
                target.focus()
                page.keyboard.press("Tab")
                page.keyboard.press("Shift+Tab")
                focus = target.evaluate(
                    """
                    element => {
                      const style = getComputedStyle(element);
                      return [style.outlineStyle, parseFloat(style.outlineWidth) || 0];
                    }
                    """
                )
                if focus[0] == "none" or focus[1] < 3:
                    raise RuntimeError(f"keyboard focus indicator is not visible for {selector}")
            normal_motion = page.evaluate(
                """
                () => ({
                  animation: getComputedStyle(canvas).animationName,
                  transition: getComputedStyle(
                    document.querySelector('.file-control')
                  ).transitionDuration,
                })
                """
            )
            if normal_motion["animation"] != "result-reveal":
                raise RuntimeError("reduced-motion test lacks the normal result animation")
            page.emulate_media(reduced_motion="reduce")
            reduced_motion = page.evaluate(
                """
                () => ({
                  animation: getComputedStyle(canvas).animationName,
                  transition: getComputedStyle(
                    document.querySelector('.file-control')
                  ).transitionDuration,
                })
                """
            )
            durations = [
                float(value.removesuffix("ms")) if value.endswith("ms") else float(value.removesuffix("s")) * 1000
                for value in reduced_motion["transition"].split(", ")
            ]
            if reduced_motion["animation"] != "none":
                raise RuntimeError("reduced-motion preference retains the result animation")
            if any(value > 20 for value in durations):
                raise RuntimeError("reduced-motion preference retains a visible transition")
            source_links = {
                text.strip(): href
                for text, href in page.locator(".source-links a").evaluate_all(
                    "items => items.map(item => [item.textContent, item.getAttribute('href')])"
                )
            }
            if "Source" not in source_links or "AGPL-3.0-or-later" not in source_links:
                raise RuntimeError("source or code-license link is not readable")
            if not any(
                href == "third_party/yolo26n-obb-privacy-sanitization.json"
                for href in source_links.values()
            ):
                raise RuntimeError("privacy-sanitization record lacks a readable direct link")
            byom = page.locator("#byomPanel")
            if byom.get_attribute("open") is not None or "進階" not in byom.locator("summary").inner_text():
                raise RuntimeError("advanced BYOM is no longer secondary")
            if byom.evaluate("panel => panel.closest('#controlRail')?.id || ''") != "controlRail":
                raise RuntimeError("advanced BYOM is not inside the compact control rail")
        finally:
            browser.close()


def _assert_responsive_layout(page: object, label: str) -> None:
    overflow = page.evaluate(
        "Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) - innerWidth"
    )
    if overflow > 1:
        offenders = page.evaluate(
            """
            () => [...document.querySelectorAll('body *')]
              .filter(element => {
                const rect = element.getBoundingClientRect();
                return rect.left < -1 || rect.right > innerWidth + 1;
              })
              .slice(0, 8)
              .map(element => {
                const rect = element.getBoundingClientRect();
                const name = element.id
                  ? `#${element.id}`
                  : `${element.tagName.toLowerCase()}.${element.className || ''}`;
                return `${name}:${Math.round(rect.left)}..${Math.round(rect.right)}`;
              })
            """
        )
        raise RuntimeError(f"{label} layout has horizontal overflow: {offenders!r}")
    for selector in (
        "#demoOriginalImage",
        "#demoDetectBtn",
        "#status",
        '.source-links a[href*="github.com/kuotunyu/aerial-obb-lab"]',
        '.source-links a[href*="LICENSE"]',
        '.source-links a[href*="sanitization"]',
    ):
        target = page.locator(selector).first
        if target.count() != 1 or not target.is_visible():
            raise RuntimeError(
                f"{label} layout hides required demo or source element {selector}"
            )
        box = target.bounding_box()
        if box is None or box["x"] < -1 or box["x"] + box["width"] > page.evaluate("innerWidth") + 1:
            raise RuntimeError(
                f"{label} layout clips required demo or source element {selector}"
            )
    rail_box = page.locator("#controlRail").bounding_box()
    viewport_box = page.locator("#resultViewport").bounding_box()
    if label == "desktop":
        if (
            rail_box is None
            or viewport_box is None
            or viewport_box["x"] <= rail_box["x"] + rail_box["width"]
        ):
            raise RuntimeError("desktop result viewport is not right of the control rail")
        return

    ordered = [
        "#sampleCard",
        "#resultViewport",
        ".result-summary",
        "#resultControls",
        ".detections",
        "#byomPanel",
    ]
    boxes = [page.locator(selector).bounding_box() for selector in ordered]
    if any(box is None for box in boxes):
        raise RuntimeError(f"{label} layout hides an ordered workbench section")
    tops = [box["y"] for box in boxes if box is not None]
    if tops != sorted(tops):
        raise RuntimeError(f"{label} workbench visual order is wrong: {tops!r}")


def run_responsive(
    width: int,
    height: int,
    label: str,
    *,
    executable_path: Path | None = None,
    base_url: str | None = None,
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required; run the locked development environment") from exc

    server = nullcontext(base_url) if base_url else static_server()
    with server as served_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(**_launch_options(executable_path))
        try:
            page = browser.new_page(viewport={"width": width, "height": height})
            page.goto(f"{str(served_url).rstrip('/')}/", wait_until="networkidle")
            _assert_responsive_layout(page, label)
            if label == "desktop":
                page.set_viewport_size({"width": width // 2, "height": height // 2})
                _assert_responsive_layout(page, "desktop 200% zoom")
        finally:
            browser.close()


def run_desktop(
    executable_path: Path | None = None,
    base_url: str | None = None,
) -> None:
    run_responsive(
        1280, 720, "desktop", executable_path=executable_path, base_url=base_url
    )


def run_workbench_layout(
    executable_path: Path | None = None,
    base_url: str | None = None,
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required; run the locked development environment") from exc

    server = nullcontext(base_url) if base_url else static_server()
    with server as served_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(**_launch_options(executable_path))
        try:
            loading_page = browser.new_page(viewport={"width": 1280, "height": 720})
            held_image_routes = []
            loading_page.route(
                f"**{DEMO_IMAGE_PATH}", lambda route: held_image_routes.append(route)
            )
            loading_page.goto(
                f"{str(served_url).rstrip('/')}/", wait_until="domcontentloaded"
            )
            try:
                if loading_page.evaluate("demoOriginalImage.complete"):
                    raise RuntimeError("preload empty-state test did not hold the demo image open")
                _assert_empty_disabled_result_state(loading_page)
            finally:
                for route in held_image_routes:
                    route.abort()
                loading_page.close()

            page = browser.new_page(viewport={"width": 1280, "height": 720})
            requests: list[str] = []
            messages: list[str] = []
            _record_errors(page, requests, messages)
            page.goto(f"{str(served_url).rstrip('/')}/", wait_until="networkidle")
            assert_real_demo_initial(page, requests, messages)
            assert_workbench_initial_layout(page)
        finally:
            browser.close()


def run_mobile(
    executable_path: Path | None = None,
    base_url: str | None = None,
) -> None:
    run_responsive(
        390, 844, "mobile", executable_path=executable_path, base_url=base_url
    )


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
            assert_demo_cached_filters(page, requests, run_counter)
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
            page.add_init_script(CANVAS_INSTRUMENTATION)
            _record_errors(page, requests, messages)

            def stub_ort(route: Route) -> None:
                route.fulfill(
                    status=200,
                    content_type="application/javascript",
                    headers={"Access-Control-Allow-Origin": "*"},
                    body=_scenario_ort_stub(run_mode="delayed"),
                )

            page.route(ORT_CDN_URL, stub_ort)
            page.goto(f"{str(served_url).rstrip('/')}/", wait_until="networkidle")
            assert_real_demo_initial(page, requests, messages)
            assert_malformed_frozen_manifest_is_closed(page)
            page.locator("#demoDetectBtn").click()
            page.wait_for_function("typeof globalThis.__resolveDemoRun === 'function'")
            if page.locator("#sampleState").inner_text() != "Loading · local browser":
                raise RuntimeError("held demo work did not keep the official sample loading")
            page.evaluate("globalThis.__resolveDemoRun()")
            page.wait_for_function(
                "document.querySelector('#status').dataset.kind === 'success'",
                timeout=30_000,
            )
            if page.evaluate("[globalThis.__ortCreateCount, globalThis.__demoRunCount]") != [1, 1]:
                raise RuntimeError("first stubbed demo run did not create and run one session")
            if page.locator("#sampleState").inner_text() != "Result · ready":
                raise RuntimeError("held demo work did not finish in the official sample result state")
            page.evaluate("globalThis.__resolveDemoRun = null")
            page.locator("#demoDetectBtn").click()
            page.wait_for_function(
                "globalThis.__demoRunCount === 2 && "
                "typeof globalThis.__resolveDemoRun === 'function'",
                timeout=30_000,
            )
            if page.locator("#sampleState").inner_text() != "Loading · local browser":
                raise RuntimeError("repeated held demo work did not restore the loading state")
            page.evaluate("globalThis.__resolveDemoRun()")
            page.wait_for_function("document.querySelector('#status').dataset.kind === 'success'")
            if page.evaluate("globalThis.__ortCreateCount") != 1:
                raise RuntimeError("repeated Detect did not reuse the valid demo session")
            assert_demo_cached_filters(page, requests, 2)
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
    parser.add_argument(
        "--scenario",
        choices=(
            "sample-gallery",
            "held-decode",
            "real-demo-success",
            "stubbed-cache",
            "manifest-failure",
            "model-digest-failure",
            "runtime-failure",
            "session-failure",
            "run-failure",
            "output-failure",
            "render-failure",
            "stale-generation",
            "byom-transition",
            "accessibility",
            "desktop",
            "workbench-layout",
            "mobile",
        ),
    )
    args = parser.parse_args(argv)
    try:
        if args.scenario == "sample-gallery":
            run_sample_gallery(args.executable_path, args.base_url, args.screenshot)
        elif args.scenario == "held-decode":
            run_held_decode(args.executable_path, args.base_url)
        elif args.scenario == "real-demo-success":
            run_real_demo_success(
                args.executable_path,
                args.base_url,
                args.screenshot,
                args.mobile_screenshot,
            )
        elif args.scenario == "stubbed-cache":
            run_stubbed_cache(args.executable_path, args.base_url)
        elif args.scenario == "manifest-failure":
            run_manifest_failure(args.executable_path, args.base_url)
        elif args.scenario == "model-digest-failure":
            run_model_digest_failure(args.executable_path, args.base_url)
        elif args.scenario == "runtime-failure":
            run_runtime_failure(args.executable_path, args.base_url)
        elif args.scenario == "session-failure":
            run_session_failure(args.executable_path, args.base_url)
        elif args.scenario == "run-failure":
            run_run_failure(args.executable_path, args.base_url)
        elif args.scenario == "output-failure":
            run_output_failure(args.executable_path, args.base_url)
        elif args.scenario == "render-failure":
            run_render_failure(args.executable_path, args.base_url)
        elif args.scenario == "stale-generation":
            run_stale_generation(args.executable_path, args.base_url)
        elif args.scenario == "byom-transition":
            run_byom_transition(args.executable_path, args.base_url)
        elif args.scenario == "accessibility":
            run_accessibility(args.executable_path, args.base_url)
        elif args.scenario == "desktop":
            run_desktop(args.executable_path, args.base_url)
        elif args.scenario == "workbench-layout":
            run_workbench_layout(args.executable_path, args.base_url)
        elif args.scenario == "mobile":
            run_mobile(args.executable_path, args.base_url)
        else:
            run_sample_gallery(args.executable_path, args.base_url, args.screenshot)
            run_held_decode(args.executable_path, args.base_url)
            run_real_demo_success(
                args.executable_path,
                args.base_url,
                args.screenshot,
                args.mobile_screenshot,
            )
            run_stubbed_cache(args.executable_path, args.base_url)
            run_manifest_failure(args.executable_path, args.base_url)
            run_model_digest_failure(args.executable_path, args.base_url)
            run_runtime_failure(args.executable_path, args.base_url)
            run_session_failure(args.executable_path, args.base_url)
            run_run_failure(args.executable_path, args.base_url)
            run_output_failure(args.executable_path, args.base_url)
            run_render_failure(args.executable_path, args.base_url)
            run_stale_generation(args.executable_path, args.base_url)
            run_byom_transition(args.executable_path, args.base_url)
            run_accessibility(args.executable_path, args.base_url)
            run_desktop(args.executable_path, args.base_url)
            run_workbench_layout(args.executable_path, args.base_url)
            run_mobile(args.executable_path, args.base_url)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    if args.scenario == "sample-gallery":
        print("[OK] Real sample gallery initial state")
    elif args.scenario == "held-decode":
        print("[OK] Held real sample decode clears stale result state")
    elif args.scenario == "real-demo-success":
        print("[OK] Real demo browser smoke: genuine local derivative inference")
    elif args.scenario == "stubbed-cache":
        print("[OK] Real demo cache smoke: one verified model and reusable session")
    elif args.scenario == "manifest-failure":
        print("[OK] Real demo manifest failures are closed and recoverable")
    elif args.scenario == "model-digest-failure":
        print("[OK] Real demo model integrity failures are closed and recoverable")
    elif args.scenario == "runtime-failure":
        print("[OK] Real demo runtime retry is pinned and recoverable")
    elif args.scenario == "session-failure":
        print("[OK] Real demo candidate-session failure is atomic")
    elif args.scenario == "run-failure":
        print("[OK] Real demo run failure is closed and recoverable")
    elif args.scenario == "output-failure":
        print("[OK] Real demo output failure is closed and recoverable")
    elif args.scenario == "render-failure":
        print("[OK] Real demo render failure is closed and recoverable")
    elif args.scenario == "stale-generation":
        print("[OK] Real demo stale work is aborted or ignored")
    elif args.scenario == "byom-transition":
        print("[OK] Real demo and BYOM transitions preserve session safety")
    elif args.scenario == "accessibility":
        print("[OK] Real demo accessibility contract")
    elif args.scenario == "desktop":
        print("[OK] Real demo desktop and 200% zoom layout")
    elif args.scenario == "workbench-layout":
        print("[OK] Real demo compact semantic workbench layout")
    elif args.scenario == "mobile":
        print("[OK] Real demo mobile layout")
    else:
        print("[OK] Real demo browser smoke: genuine inference and cached session behavior")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
