"""Headless BYOM demo smoke using synthetic model bytes, image, and ONNX output.

This script tests local model selection, lazy runtime loading, browser wiring,
preprocessing, strict output selection, OBB decoding, drawing, and result rendering
without model inference or an external network request. The pinned CDN request is
fulfilled directly with the synthetic runtime stub.
"""

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


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo" / "web"
FIXTURE = DEMO / "fixtures" / "showcase.svg"
EXPECTED_ROW = ["ship", "0.900", "100.0", "50.0", "90.0"]
ORT_CDN_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js"
ORT_WASM_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/"
ORT_INTEGRITY = "sha384-RPL/K8tc0JVaNWsunkEmCzLeieefvFX2UCRLKLmLVChCI6P+CTKhzqF7VIeCc3Zp"
SENSITIVE_IMAGE_NAME = "customer-alpha-private-aerial-image.svg"
ERROR_COPY = {
    "SHOWCASE_ASSET": "Synthetic fixture 無法載入。請重新整理頁面，或改用 BYOM。",
    "RUNTIME_LOAD": "Browser runtime 無法載入。請檢查網路或 content blocker 後重試；Synthetic Showcase 仍可使用。",
    "MODEL_CONTRACT": "請選擇使用 images [1,3,1024,1024] 與 output0 [1,N,7] 的相容 ONNX。",
    "IMAGE_DECODE": "Browser 無法解碼影像。請改選 PNG、JPEG 或 WebP。",
    "INFERENCE_RUN": "推論未完成。請確認模型 contract、重新選擇影像後再試。",
    "OUTPUT_SCHEMA": "模型輸出不符合 output0 [1,N,7]。請改用相容的 end-to-end OBB export。",
    "RENDER_RESULT": "結果無法呈現。請重新載入 Synthetic Showcase，或重新執行 Detect。",
}
SENSITIVE_TOKENS = (
    "C:\\Users\\alice\\private-model.onnx",
    SENSITIVE_IMAGE_NAME,
    "tenant=omega",
    "raw create failure",
    "flight=classified",
    "raw inference failure",
    "expected output0 shape [1,N,7]",
    "tile=restricted",
    "raw renderer failure",
)
ORT_STUB = r"""
globalThis.__ortCreateCount = 0;
globalThis.__ortReleaseCount = 0;
globalThis.__failInferenceRun = false;
globalThis.__invalidInferenceOutput = false;
globalThis.__ortActiveSessionIdsAtRelease = [];
globalThis.__ortReleasedSessionIds = [];
globalThis.__ortRunSessionIds = [];
globalThis.__ortCreatedSessionIds = [];
globalThis.__ortDelayedCreateResolvers = [];
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
      const sessionId = ++globalThis.__ortCreateCount;
      globalThis.__ortCreatedSessionIds.push(sessionId);
      if (!(modelBytes instanceof Uint8Array) || modelBytes.length === 0) {
        throw new Error("expected non-empty local model bytes");
      }
      if (modelBytes[0] === 0) {
        throw new Error(
          "C:\\Users\\alice\\private-model.onnx | tenant=omega | raw create failure"
        );
      }
      if (modelBytes[0] === 1) {
        await new Promise((resolve) => {
          globalThis.__ortDelayedCreateResolvers.push(resolve);
        });
      }
      const behavior = modelBytes[0];
      let released = false;
      return {
        __sessionId: sessionId,
        inputNames: ["images"],
        outputNames: behavior === 3 ? ["unexpected"] : ["output0"],
        release: async () => {
          if (released) throw new Error("session released twice");
          released = true;
          globalThis.__ortReleaseCount += 1;
          globalThis.__ortReleasedSessionIds.push(sessionId);
          globalThis.__ortActiveSessionIdsAtRelease.push(
            state.session?.__sessionId ?? null
          );
        },
        run: async () => {
          if (released) throw new Error("released session was used");
          globalThis.__ortRunSessionIds.push(sessionId);
          if (behavior === 4 || globalThis.__failInferenceRun) {
            throw new Error("flight=classified | raw inference failure");
          }
          if (behavior === 2 || globalThis.__invalidInferenceOutput) {
            return {
              output0: {
                dims: [1, 2, 6],
                data: new Float32Array(12)
              }
            };
          }
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

# The intercepted stub cannot share the production bundle's SRI digest. Bypass
# hashing only when production already supplied the exact pinned attributes,
# then restore the integrity value before the app can observe the script.
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


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


def assert_fixed_failure(page: object, messages: list[str], code: str) -> None:
    status = page.locator("#status").inner_text()
    if status != ERROR_COPY[code]:
        raise RuntimeError(f"wrong {code} copy: {status!r}")
    if page.locator("#status").get_attribute("data-kind") != "error":
        raise RuntimeError(f"{code} has no semantic error state")
    if f"[AERIAL_OBB:{code}]" not in messages:
        raise RuntimeError(f"{code} did not emit its fixed diagnostic code")
    retry = page.locator("#runtimeRetryBtn")
    if retry.count() != 1:
        raise RuntimeError("runtime retry control is missing")
    if retry.is_hidden() != (code != "RUNTIME_LOAD"):
        raise RuntimeError(f"runtime retry visibility is wrong for {code}")
    rendered_and_console = " | ".join([page.locator("body").inner_text(), *messages])
    leaked = [token for token in SENSITIVE_TOKENS if token in rendered_and_console]
    if leaked:
        raise RuntimeError(f"{code} leaked sensitive exception data: {leaked!r}")


def assert_result_cleared(page: object, base_canvas: str, code: str) -> None:
    if page.locator("#resultsBody tr").count() != 0:
        raise RuntimeError(f"{code} left a stale result row visible")
    summary = page.evaluate(
        "[summaryCount.textContent, summaryTop.textContent, runtimeValue.textContent, "
        "modeBadge.textContent, provenanceValue.textContent]"
    )
    if summary != ["0", "—", "—", "NO RESULT", "—"]:
        raise RuntimeError(f"{code} left stale result metadata visible: {summary!r}")
    if page.locator("#canvasFrame").evaluate("el => el.classList.contains('has-results')"):
        raise RuntimeError(f"{code} left the completed-result state visible")
    current_canvas = page.locator("#canvas").evaluate("canvas => canvas.toDataURL()")
    if current_canvas != base_canvas:
        raise RuntimeError(f"{code} left a stale polygon painted on the canvas")


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


def run_smoke(
    executable_path: Path | None = None,
    screenshot: Path | None = None,
    base_url: str | None = None,
    mobile_screenshot: Path | None = None,
) -> None:
    try:
        from playwright.sync_api import Route, sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required; run the locked development environment") from exc

    browser_errors: list[str] = []
    browser_messages: list[str] = []
    requested_urls: list[str] = []
    server = nullcontext(base_url) if base_url else static_server()
    with server as served_url, sync_playwright() as playwright:
        launch_options: dict[str, object] = {"headless": True, "args": ["--disable-gpu"]}
        if executable_path is not None:
            launch_options["executable_path"] = str(executable_path)
        browser = playwright.chromium.launch(**launch_options)
        try:
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.add_init_script(SRI_STUB_SHIM)
            page.add_init_script(
                """
                globalThis.__obbFillTextCalls = 0;
                globalThis.__obbStrokedPolygons = [];
                let currentPath = [];
                const originalFillText = CanvasRenderingContext2D.prototype.fillText;
                const originalBeginPath = CanvasRenderingContext2D.prototype.beginPath;
                const originalMoveTo = CanvasRenderingContext2D.prototype.moveTo;
                const originalLineTo = CanvasRenderingContext2D.prototype.lineTo;
                const originalStroke = CanvasRenderingContext2D.prototype.stroke;
                CanvasRenderingContext2D.prototype.fillText = function (...args) {
                  globalThis.__obbFillTextCalls += 1;
                  return originalFillText.apply(this, args);
                };
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
                  if (globalThis.__failResultRender) {
                    throw new Error("tile=restricted | raw renderer failure");
                  }
                  if (currentPath.length) globalThis.__obbStrokedPolygons.push([...currentPath]);
                  return originalStroke.apply(this, args);
                };
                """
            )
            page.on("request", lambda request: requested_urls.append(request.url))
            page.on("pageerror", lambda error: browser_errors.append(str(error)))
            page.on("console", lambda message: browser_messages.append(message.text))
            page.on(
                "console",
                lambda message: browser_errors.append(message.text)
                if message.type == "error"
                else None,
            )

            def stub_ort(route: Route) -> None:
                route.fulfill(
                    status=200,
                    content_type="application/javascript",
                    headers={"Access-Control-Allow-Origin": "*"},
                    body=ORT_STUB,
                )

            entry_url = f"{str(served_url).rstrip('/')}/"
            page.route(ORT_CDN_URL, stub_ort)
            page.goto(entry_url, wait_until="networkidle")
            if requested_urls.count(ORT_CDN_URL) != 0:
                raise RuntimeError("ORT must not be requested during initial page load")
            if page.locator("html").get_attribute("lang") != "zh-Hant-TW":
                raise RuntimeError("browser workbench must declare zh-Hant-TW")
            header = page.locator("header").inner_text()
            for token in ("Aerial OBB Lab", "Browser", "WASM", "Local files"):
                if token not in header:
                    raise RuntimeError(f"browser workbench header is missing {token!r}")
            body_text = page.locator("body").inner_text()
            if "模型與影像不會上傳" not in body_text:
                raise RuntimeError("browser workbench must disclose local-only file handling")
            if "output0 [1,N,7]" not in page.locator("footer").inner_text():
                raise RuntimeError("browser workbench footer is missing the output contract")
            if page.locator("h1").count() != 1:
                raise RuntimeError("browser workbench must contain exactly one h1")

            initial_result_presentation = page.evaluate(
                "[modeBadge.textContent, provenanceValue.textContent]"
            )

            notice = page.locator("#claimBoundary")
            control = page.locator("#showcaseBtn")
            assert notice.count() == 1 and control.count() == 1
            assert "沒有執行模型推論" in notice.inner_text()
            assert notice.evaluate(
                "(notice, control) => Boolean(notice.compareDocumentPosition(control) & Node.DOCUMENT_POSITION_FOLLOWING)",
                control.element_handle(),
            )
            notice_box, control_box = notice.bounding_box(), control.bounding_box()
            assert notice_box and control_box and notice_box["y"] < control_box["y"]

            body_size = page.locator("body").evaluate("el => getComputedStyle(el).fontSize")
            title_size = page.locator("h1").evaluate("el => getComputedStyle(el).fontSize")
            title_family = page.locator("h1").evaluate(
                "el => getComputedStyle(el).fontFamily"
            )
            button_height = page.locator("#detectBtn").evaluate(
                "el => el.getBoundingClientRect().height"
            )
            if body_size != "19px":
                raise RuntimeError(f"unexpected body font size: {body_size}")
            if float(str(title_size).removesuffix("px")) < 38:
                raise RuntimeError(f"unexpected title font size: {title_size}")
            if "IBM Plex Sans Condensed" not in str(title_family):
                raise RuntimeError(f"unexpected display font: {title_family}")
            if float(button_height) < 44:
                raise RuntimeError(f"Detect control is too short: {button_height}")

            compact_metrics = page.evaluate(
                """
                () => {
                  const box = (selector) => document.querySelector(selector).getBoundingClientRect();
                  const font = (selector) => parseFloat(
                    getComputedStyle(document.querySelector(selector)).fontSize
                  );
                  const radius = (selector) => getComputedStyle(
                    document.querySelector(selector)
                  ).borderRadius;
                  return {
                    headerHeight: box('.product-header').height,
                    fileControlHeight: box('.file-control').height,
                    detectBottom: box('#detectBtn').bottom,
                    viewportHeight: window.innerHeight,
                    smallType: {
                      fileKind: font('.file-kind'),
                      fileHelp: font('.file-control small'),
                      rangeScale: font('.range-scale'),
                      filterHelp: font('.class-filter > p'),
                      filterChoice: font('.class-list label'),
                      metricLabel: font('.result-summary dt'),
                      tableMeta: font('.table-heading span'),
                      tableHeader: font('th'),
                      footer: font('footer')
                    },
                    radii: {
                      trustSignal: radius('.trust-signals li'),
                      controlRail: radius('.control-rail'),
                      fileControl: radius('.file-control'),
                      thresholdValue: radius('#confVal'),
                      classFilter: radius('.class-filter'),
                      filterChoice: radius('.class-list label'),
                      status: radius('.status'),
                      detect: radius('#detectBtn'),
                      viewport: radius('.canvas-frame'),
                      detections: radius('.detections')
                    }
                  };
                }
                """
            )
            if compact_metrics["headerHeight"] > 80:
                raise RuntimeError(f"header wastes vertical space: {compact_metrics!r}")
            if compact_metrics["fileControlHeight"] > 90:
                raise RuntimeError(f"file controls are too tall: {compact_metrics!r}")
            if compact_metrics["detectBottom"] > compact_metrics["viewportHeight"]:
                raise RuntimeError(f"primary action is below the first viewport: {compact_metrics!r}")
            undersized = {
                name: size
                for name, size in compact_metrics["smallType"].items()
                if size < 15
            }
            if undersized:
                raise RuntimeError(f"secondary UI text is too small: {undersized!r}")
            rounded = {
                name: value
                for name, value in compact_metrics["radii"].items()
                if value != "0px"
            }
            if rounded:
                raise RuntimeError(f"nonessential rounded borders remain: {rounded!r}")

            controls = page.locator("#controlRail").bounding_box()
            results = page.locator("#resultWorkspace").bounding_box()
            if controls is None or results is None:
                raise RuntimeError("canonical workbench regions are missing")
            if abs(controls["y"] - results["y"]) > 2:
                raise RuntimeError("desktop workbench regions do not align")
            if results["width"] <= controls["width"]:
                raise RuntimeError("desktop result workspace must dominate the control rail")
            if results["height"] < controls["height"] - 2:
                raise RuntimeError(
                    "desktop result workspace leaves ineffective lower whitespace; "
                    f"controls={controls!r}; results={results!r}"
                )

            page.locator("#modelInput").focus()
            focus_width = page.locator("#modelDrop").evaluate(
                "el => parseFloat(getComputedStyle(el).outlineWidth)"
            )
            if float(focus_width) < 2:
                raise RuntimeError("model picker has no visible keyboard focus")
            if not page.locator("#detectBtn").is_disabled():
                raise RuntimeError("Detect must be disabled before model and image selection")
            page.locator("#confSlider").evaluate(
                "slider => slider.dispatchEvent(new Event('input', { bubbles: true }))"
            )
            if page.locator("#status").get_attribute("data-kind") == "error":
                raise RuntimeError("filtering before a result must remain a no-op")

            page.locator("#showcaseBtn").click()
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('Synthetic fixture')",
                timeout=5_000,
            )
            if requested_urls.count(ORT_CDN_URL) != 0:
                raise RuntimeError("synthetic showcase must not request ORT")
            if page.locator("#modeBadge").inner_text() != "SYNTHETIC FIXTURE · NO INFERENCE":
                raise RuntimeError("showcase mode badge is not exact")
            if page.locator("#provenanceValue").inner_text() != "Committed synthetic fixture":
                raise RuntimeError("showcase provenance is not exact")
            if page.locator("#runtimeValue").inner_text() != "N/A · no inference":
                raise RuntimeError("showcase runtime must disclose that inference did not run")
            showcase_row = page.locator("#resultsBody tr").first.locator("td").all_text_contents()
            if showcase_row != EXPECTED_ROW:
                raise RuntimeError(f"unexpected synthetic showcase row: {showcase_row!r}")
            if page.locator("#resultTitle").evaluate("el => document.activeElement === el") is not True:
                raise RuntimeError("showcase activation must move focus to the result title")
            polygon = page.evaluate("globalThis.__obbStrokedPolygons.at(-1)")
            expected_polygon = [[225, 50], [225, 150], [175, 150], [175, 50]]
            if polygon is None or len(polygon) != len(expected_polygon) or any(
                abs(actual - expected) > 0.01
                for actual_corner, expected_corner in zip(polygon, expected_polygon, strict=True)
                for actual, expected in zip(actual_corner, expected_corner, strict=True)
            ):
                raise RuntimeError(f"synthetic showcase did not draw expected polygon: {polygon!r}")
            page.locator("#confSlider").evaluate("(slider) => { slider.value = '0.95'; slider.dispatchEvent(new Event('input', { bubbles: true })); }")
            if page.locator("#resultsBody tr").count() != 0:
                raise RuntimeError("confidence changes must re-filter cached showcase output")
            if page.locator("#summaryCount").inner_text() != "0":
                raise RuntimeError("cached showcase re-filter must update the summary")
            if page.locator("#runtimeValue").inner_text() != "N/A · no inference":
                raise RuntimeError("synthetic confidence refilter lost no-inference runtime")
            page.locator("#confSlider").evaluate("(slider) => { slider.value = '0.25'; slider.dispatchEvent(new Event('input', { bubbles: true })); }")
            if page.locator("#runtimeValue").inner_text() != "N/A · no inference":
                raise RuntimeError("synthetic confidence restore lost no-inference runtime")
            plane = page.locator('.class-cb[value="0"]')
            plane.check()
            if page.locator("#resultsBody tr").count() != 0:
                raise RuntimeError("synthetic class refilter did not hide the fixture row")
            if page.locator("#runtimeValue").inner_text() != "N/A · no inference":
                raise RuntimeError("synthetic class refilter lost no-inference runtime")
            plane.uncheck()
            if page.locator("#runtimeValue").inner_text() != "N/A · no inference":
                raise RuntimeError("synthetic class restore lost no-inference runtime")

            page.locator("#modelInput").set_input_files(
                files=[
                    {
                        "name": "customer-alpha-private-model.onnx",
                        "mimeType": "application/octet-stream",
                        "buffer": b"synthetic-not-a-real-model",
                    }
                ]
            )
            try:
                page.wait_for_function(
                    "document.querySelector('#status').dataset.kind === 'success'",
                    timeout=5_000,
                )
            except Exception as exc:
                current_status = page.locator("#status").inner_text()
                raise RuntimeError(
                    "model selection did not reach success state; "
                    f"status={current_status!r}; browser_errors={browser_errors!r}"
                ) from exc
            sensitive_model_name = "customer-alpha-private-model.onnx"
            visible_and_console = " | ".join(
                [page.locator("body").inner_text(), *browser_messages]
            )
            if sensitive_model_name in visible_and_console:
                raise RuntimeError("successful model selection exposed a local filename")
            if page.locator("#modelLabel").inner_text() != "Local ONNX model ready":
                raise RuntimeError("successful model selection must use fixed neutral copy")
            if requested_urls.count(ORT_CDN_URL) != 1:
                raise RuntimeError(
                    "first BYOM model selection must request ORT exactly once; "
                    f"requests={requested_urls.count(ORT_CDN_URL)}"
                )
            runtime_scripts = page.locator(f'script[src="{ORT_CDN_URL}"]')
            if runtime_scripts.count() != 1:
                raise RuntimeError("lazy ORT loader must append exactly one runtime script")
            runtime_attributes = runtime_scripts.evaluate(
                "script => ({ integrity: script.integrity, crossOrigin: script.crossOrigin })"
            )
            if runtime_attributes != {
                "integrity": ORT_INTEGRITY,
                "crossOrigin": "anonymous",
            }:
                raise RuntimeError(
                    f"lazy ORT script security attributes are wrong: {runtime_attributes!r}"
                )
            if page.evaluate("globalThis.ort.env.wasm.wasmPaths") != ORT_WASM_BASE:
                raise RuntimeError("lazy ORT loader did not pin the WASM asset base")
            if page.evaluate("[globalThis.__ortCreateCount, globalThis.__ortReleaseCount]") != [1, 0]:
                raise RuntimeError("first validated model must become active without a release")
            if page.locator("#modeBadge").inner_text() != "NO RESULT":
                raise RuntimeError("BYOM selection must clear synthetic mode state")
            if page.locator("#provenanceValue").inner_text() != "—":
                raise RuntimeError("BYOM selection must clear synthetic provenance")
            canvas_is_clear = page.locator("#canvas").evaluate(
                "canvas => canvas.getContext('2d').getImageData(200, 100, 1, 1).data[3] === 0"
            )
            if not canvas_is_clear or page.locator("#canvasFrame").evaluate(
                "el => el.classList.contains('has-results')"
            ):
                raise RuntimeError("BYOM selection must clear the synthetic canvas result")
            if not page.locator("#detectBtn").is_disabled():
                raise RuntimeError("Detect must remain disabled until an image is selected")
            page.locator("#fileInput").set_input_files(
                files=[
                    {
                        "name": SENSITIVE_IMAGE_NAME,
                        "mimeType": "image/svg+xml",
                        "buffer": FIXTURE.read_bytes(),
                    }
                ]
            )
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('影像已載入')"
            )
            visible_and_console = " | ".join(
                [page.locator("body").inner_text(), *browser_messages]
            )
            if SENSITIVE_IMAGE_NAME in visible_and_console:
                raise RuntimeError("successful image selection exposed a local filename")
            if page.locator("#fileLabel").inner_text() != "Local image ready":
                raise RuntimeError("successful image selection must use fixed neutral copy")
            if page.locator("#detectBtn").is_disabled():
                raise RuntimeError("Detect must be enabled after local model and image selection")
            base_canvas = page.locator("#canvas").evaluate("canvas => canvas.toDataURL()")
            page.locator("#detectBtn").click()
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('完成')"
            )

            completed_byom_presentation = page.evaluate(
                "[modeBadge.textContent, provenanceValue.textContent]"
            )
            presentation_failures = []
            if initial_result_presentation != ["NO RESULT", "—"]:
                presentation_failures.append(
                    "initial result presentation must be exact "
                    f"['NO RESULT', '—']; got {initial_result_presentation!r}"
                )
            if completed_byom_presentation != [
                "BYOM · LOCAL BROWSER INFERENCE",
                "Local files",
            ]:
                presentation_failures.append(
                    "successful BYOM result presentation must be exact "
                    "['BYOM · LOCAL BROWSER INFERENCE', 'Local files']; "
                    f"got {completed_byom_presentation!r}"
                )
            if presentation_failures:
                raise RuntimeError("; ".join(presentation_failures))
            row = page.locator("#resultsBody tr").first.locator("td").all_text_contents()
            if row != EXPECTED_ROW:
                raise RuntimeError(f"unexpected browser result row: {row!r}")
            canvas_size = page.locator("#canvas").evaluate(
                "element => [element.width, element.height]"
            )
            if canvas_size != [400, 200]:
                raise RuntimeError(f"unexpected canvas size: {canvas_size!r}")
            status = page.locator("#status").inner_text()
            if "完成" not in status:
                raise RuntimeError(f"unexpected browser status: {status!r}")
            if page.locator("#status").get_attribute("data-kind") != "success":
                raise RuntimeError("successful inference has no semantic success state")
            if page.locator("#summaryCount").inner_text() != "1":
                raise RuntimeError("detection count summary was not updated")
            if page.locator("#summaryTop").inner_text() != "0.900":
                raise RuntimeError("top-confidence summary was not updated")
            if page.locator("#resultsBody tr").count() != 1:
                raise RuntimeError("result table must contain the decoded row")
            if not page.locator("#canvasFrame").evaluate(
                "el => el.classList.contains('has-results')"
            ):
                raise RuntimeError("result viewport has no authored completion reveal state")
            if page.evaluate("globalThis.__obbFillTextCalls") != 0:
                raise RuntimeError("dense result labels must not be drawn on the canvas")
            byom_runtime = page.locator("#runtimeValue").inner_text()
            if not re.fullmatch(r"\d+ ms", byom_runtime):
                raise RuntimeError(f"BYOM runtime is not numeric: {byom_runtime!r}")
            page.locator("#confSlider").evaluate(
                "slider => { slider.value = '0.95'; slider.dispatchEvent(new Event('input', { bubbles: true })); }"
            )
            if page.locator("#runtimeValue").inner_text() != byom_runtime:
                raise RuntimeError("BYOM confidence refilter changed measured runtime")
            page.locator("#confSlider").evaluate(
                "slider => { slider.value = '0.25'; slider.dispatchEvent(new Event('input', { bubbles: true })); }"
            )

            for failure_flag, failure_code in (
                ("__failInferenceRun", "INFERENCE_RUN"),
                ("__invalidInferenceOutput", "OUTPUT_SCHEMA"),
                ("__failResultRender", "RENDER_RESULT"),
            ):
                page.evaluate(f"globalThis.{failure_flag} = true")
                page.locator("#detectBtn").click()
                page.wait_for_function(
                    "document.querySelector('#status').dataset.kind === 'error'"
                )
                assert_fixed_failure(page, browser_messages, failure_code)
                assert_result_cleared(page, base_canvas, failure_code)
                page.evaluate(f"globalThis.{failure_flag} = false")
                page.locator("#detectBtn").click()
                page.wait_for_function(
                    "document.querySelector('#status').textContent.includes('完成')"
                )

            page.locator("#modelInput").set_input_files(
                files=[
                    {
                        "name": "replacement.onnx",
                        "mimeType": "application/octet-stream",
                        "buffer": b"replacement-model",
                    }
                ]
            )
            page.wait_for_function(
                "globalThis.__ortCreateCount === 2 && globalThis.__ortReleaseCount === 1"
            )
            if requested_urls.count(ORT_CDN_URL) != 1:
                raise RuntimeError("cached ORT loader must not request the runtime twice")
            if page.evaluate("[globalThis.__ortCreateCount, globalThis.__ortReleaseCount]") != [2, 1]:
                raise RuntimeError("validated replacement must release exactly the previous session")
            if page.evaluate("globalThis.__ortActiveSessionIdsAtRelease") != [2]:
                raise RuntimeError("previous session was released before replacement assignment")

            page.locator("#modelInput").set_input_files(
                files=[
                    {
                        "name": "invalid-contract.onnx",
                        "mimeType": "application/octet-stream",
                        "buffer": b"\x03invalid-contract",
                    }
                ]
            )
            page.wait_for_function(
                "document.querySelector('#status').dataset.kind === 'error'"
            )
            if page.locator("#modelLabel").inner_text() != "Local ONNX model ready":
                raise RuntimeError("invalid candidate changed the neutral active-model label")
            if page.evaluate("[globalThis.__ortCreateCount, globalThis.__ortReleaseCount]") != [3, 2]:
                raise RuntimeError("invalid candidate was not released while preserving the active session")
            page.locator("#detectBtn").click()
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('完成')"
            )
            if page.evaluate("globalThis.__ortRunSessionIds.at(-1)") != 2:
                raise RuntimeError("invalid candidate replaced the last validated session")

            page.locator("#modelInput").set_input_files(
                files=[
                    {
                        "name": "delayed-model-a.onnx",
                        "mimeType": "application/octet-stream",
                        "buffer": b"\x01delayed-candidate",
                    }
                ]
            )
            page.wait_for_function(
                "globalThis.__ortCreateCount === 4 && globalThis.__ortDelayedCreateResolvers.length === 1"
            )
            page.locator("#modelInput").set_input_files(
                files=[
                    {
                        "name": "model-b.onnx",
                        "mimeType": "application/octet-stream",
                        "buffer": b"model-b",
                    }
                ]
            )
            page.wait_for_function(
                "globalThis.__ortCreateCount === 5 && globalThis.__ortReleaseCount === 3 && state.session?.__sessionId === 5"
            )
            page.evaluate("globalThis.__ortDelayedCreateResolvers.shift()()")
            page.wait_for_function(
                "globalThis.__ortReleaseCount === 4 && globalThis.__ortReleasedSessionIds.includes(4)"
            )
            if page.locator("#modelLabel").inner_text() != "Local ONNX model ready":
                raise RuntimeError("stale candidate changed the neutral active-model label")
            if page.evaluate("state.session?.__sessionId") != 5:
                raise RuntimeError("delayed model A replaced newer model B")
            page.locator("#detectBtn").click()
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('完成')"
            )
            if page.evaluate("globalThis.__ortRunSessionIds.at(-1)") != 5:
                raise RuntimeError("Detect did not keep newer model B active")

            page.locator("#modelInput").set_input_files(
                files=[
                    {
                        "name": "inference-failure.onnx",
                        "mimeType": "application/octet-stream",
                        "buffer": b"\x04inference-failure",
                    }
                ]
            )
            page.wait_for_function(
                "globalThis.__ortCreateCount === 6 && document.querySelector('#status').dataset.kind === 'success'"
            )
            page.locator("#detectBtn").click()
            page.wait_for_function(
                "document.querySelector('#status').dataset.kind === 'error'"
            )
            assert_fixed_failure(page, browser_messages, "INFERENCE_RUN")

            page.locator("#modelInput").set_input_files(
                files=[
                    {
                        "name": "invalid-output.onnx",
                        "mimeType": "application/octet-stream",
                        "buffer": b"\x02invalid-output",
                    }
                ]
            )
            page.wait_for_function(
                "globalThis.__ortCreateCount === 7 && document.querySelector('#status').dataset.kind === 'success'"
            )
            page.locator("#detectBtn").click()
            page.wait_for_function(
                "document.querySelector('#status').dataset.kind === 'error'"
            )
            assert_fixed_failure(page, browser_messages, "OUTPUT_SCHEMA")

            page.locator("#modelInput").set_input_files(
                files=[
                    {
                        "name": "render-recovery.onnx",
                        "mimeType": "application/octet-stream",
                        "buffer": b"render-recovery",
                    }
                ]
            )
            page.wait_for_function(
                "globalThis.__ortCreateCount === 8 && document.querySelector('#status').dataset.kind === 'success'"
            )
            page.evaluate("globalThis.__failResultRender = true")
            page.locator("#detectBtn").click()
            page.wait_for_function(
                "document.querySelector('#status').dataset.kind === 'error'"
            )
            assert_fixed_failure(page, browser_messages, "RENDER_RESULT")
            page.evaluate("globalThis.__failResultRender = false")
            page.locator("#detectBtn").click()
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('完成')"
            )

            page.locator("#fileInput").set_input_files(
                files=[
                    {
                        "name": "broken-image.png",
                        "mimeType": "image/png",
                        "buffer": b"not-an-image",
                    }
                ]
            )
            page.wait_for_function(
                "document.querySelector('#status').dataset.kind === 'error'"
            )
            assert_fixed_failure(page, browser_messages, "IMAGE_DECODE")
            page.locator("#fileInput").set_input_files(str(FIXTURE))
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('影像已載入')"
            )

            page.locator("#showcaseBtn").click()
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('Synthetic fixture')"
            )
            if not page.locator("#detectBtn").is_disabled():
                raise RuntimeError("showcase activation must disable BYOM Detect")
            if page.locator("#modelLabel").inner_text() != "選擇相容的 .onnx model":
                raise RuntimeError("showcase activation left the BYOM model label ready")
            if page.locator("#fileLabel").inner_text() != "選擇或拖放一張影像":
                raise RuntimeError("showcase activation left the BYOM image label ready")
            if page.locator("#modelInput").input_value() or page.locator("#fileInput").input_value():
                raise RuntimeError("showcase activation retained stale BYOM file selections")
            if requested_urls.count(ORT_CDN_URL) != 1:
                raise RuntimeError("session lifecycle changes must reuse the cached ORT runtime")
            if screenshot is not None:
                screenshot.parent.mkdir(parents=True, exist_ok=True)
                page.wait_for_timeout(350)
                page.screenshot(path=str(screenshot), full_page=True)

            page.set_viewport_size({"width": 820, "height": 1100})
            page.wait_for_timeout(100)
            controls = page.locator("#controlRail").bounding_box()
            results = page.locator("#resultWorkspace").bounding_box()
            button = page.locator("#detectBtn").bounding_box()
            if controls is None or results is None or button is None:
                raise RuntimeError("mobile workbench regions are missing")
            if results["y"] < controls["y"] + controls["height"]:
                raise RuntimeError("mobile result workspace does not stack below controls")
            if button["width"] < controls["width"] * 0.95:
                raise RuntimeError("mobile Detect action is not full width")
            if mobile_screenshot is not None:
                mobile_screenshot.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(mobile_screenshot), full_page=True)

            invalid_page = browser.new_page(viewport={"width": 1200, "height": 800})
            invalid_messages: list[str] = []
            try:
                invalid_page.add_init_script(SRI_STUB_SHIM)
                invalid_page.on(
                    "console", lambda message: invalid_messages.append(message.text)
                )
                invalid_page.on(
                    "pageerror", lambda error: invalid_messages.append(str(error))
                )
                invalid_page.route(ORT_CDN_URL, stub_ort)
                invalid_page.goto(entry_url, wait_until="networkidle")
                invalid_page.locator("#modelInput").set_input_files(
                    files=[
                        {
                            "name": "invalid.onnx",
                            "mimeType": "application/octet-stream",
                            "buffer": b"\x00invalid-model",
                        }
                    ]
                )
                invalid_page.wait_for_function(
                    "document.querySelector('#status').dataset.kind === 'error'"
                )
                assert_fixed_failure(invalid_page, invalid_messages, "MODEL_CONTRACT")
                invalid_page.locator("#modelInput").set_input_files(
                    files=[
                        {
                            "name": "recovered.onnx",
                            "mimeType": "application/octet-stream",
                            "buffer": b"recovered-model",
                        }
                    ]
                )
                invalid_page.wait_for_function(
                    "document.querySelector('#status').dataset.kind === 'success'"
                )
            finally:
                invalid_page.close()

            runtime_page = browser.new_page(viewport={"width": 1200, "height": 800})
            runtime_messages: list[str] = []
            runtime_request_count = 0
            try:
                runtime_page.add_init_script(SRI_STUB_SHIM)
                runtime_page.on(
                    "console", lambda message: runtime_messages.append(message.text)
                )
                runtime_page.on(
                    "pageerror", lambda error: runtime_messages.append(str(error))
                )

                def flaky_ort(route: Route) -> None:
                    nonlocal runtime_request_count
                    runtime_request_count += 1
                    if runtime_request_count == 1:
                        route.abort("failed")
                    else:
                        stub_ort(route)

                runtime_page.route(ORT_CDN_URL, flaky_ort)
                runtime_page.goto(entry_url, wait_until="networkidle")
                runtime_page.locator("#modelInput").set_input_files(
                    files=[
                        {
                            "name": "runtime-retry.onnx",
                            "mimeType": "application/octet-stream",
                            "buffer": b"runtime-retry",
                        }
                    ]
                )
                runtime_page.wait_for_function(
                    "document.querySelector('#status').dataset.kind === 'error'"
                )
                assert_fixed_failure(runtime_page, runtime_messages, "RUNTIME_LOAD")
                runtime_page.locator("#runtimeRetryBtn").click()
                runtime_page.wait_for_function(
                    "document.querySelector('#status').dataset.kind === 'success'"
                )
                if runtime_page.locator("#runtimeRetryBtn").is_visible():
                    raise RuntimeError("runtime retry remained visible after recovery")
                if runtime_request_count != 2:
                    raise RuntimeError("runtime retry did not perform exactly one fresh request")
            finally:
                runtime_page.close()

            showcase_page = browser.new_page(viewport={"width": 1200, "height": 800})
            showcase_messages: list[str] = []
            showcase_request_count = 0
            showcase_ort_request_count = 0
            try:
                showcase_page.add_init_script(
                    """
                    (() => {
                      const source = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, "src");
                      let fixtureLoad = 0;
                      Object.defineProperty(HTMLImageElement.prototype, "src", {
                        configurable: true,
                        enumerable: source.enumerable,
                        get() { return source.get.call(this); },
                        set(value) {
                          const url = new URL(value, location.href);
                          if (url.pathname.endsWith("/fixtures/showcase.svg")) {
                            url.searchParams.set("smoke-showcase-load", String(++fixtureLoad));
                            source.set.call(this, url.href);
                            return;
                          }
                          source.set.call(this, value);
                        },
                      });
                    })();
                    """
                )
                showcase_page.on(
                    "console", lambda message: showcase_messages.append(message.text)
                )
                showcase_page.on(
                    "pageerror", lambda error: showcase_messages.append(str(error))
                )

                def flaky_showcase(route: Route) -> None:
                    nonlocal showcase_request_count
                    showcase_request_count += 1
                    if showcase_request_count == 2:
                        route.abort("failed")
                    else:
                        route.continue_()

                def count_showcase_ort(route: Route) -> None:
                    nonlocal showcase_ort_request_count
                    showcase_ort_request_count += 1
                    route.abort("failed")

                showcase_page.route("**/fixtures/showcase.svg*", flaky_showcase)
                showcase_page.route(ORT_CDN_URL, count_showcase_ort)
                showcase_page.goto(entry_url, wait_until="networkidle")
                showcase_page.locator("#showcaseBtn").click()
                showcase_page.wait_for_function(
                    "document.querySelector('#status').textContent.includes('Synthetic fixture')"
                )
                if showcase_page.locator("#runtimeValue").inner_text() != "N/A · no inference":
                    raise RuntimeError("initial flaky showcase result lost no-inference runtime")
                showcase_page.locator("#showcaseBtn").click()
                showcase_page.wait_for_function(
                    "document.querySelector('#status').dataset.kind === 'error'"
                )
                assert_fixed_failure(showcase_page, showcase_messages, "SHOWCASE_ASSET")
                if showcase_page.locator("#runtimeValue").inner_text() != "—":
                    raise RuntimeError("synthetic asset failure did not clear runtime before retry")
                if showcase_page.locator("#resultsBody tr").count() != 0:
                    raise RuntimeError("synthetic asset failure left stale result rows")
                if not showcase_page.locator("#canvas").evaluate(
                    "canvas => canvas.getContext('2d').getImageData(200, 100, 1, 1).data[3] === 0"
                ):
                    raise RuntimeError("synthetic asset failure left stale canvas pixels")
                if showcase_ort_request_count != 0:
                    raise RuntimeError("synthetic retry requested ORT")
                showcase_page.locator("#showcaseBtn").click()
                showcase_page.wait_for_function(
                    "document.querySelector('#status').textContent.includes('Synthetic fixture')"
                )
                if showcase_page.locator("#runtimeValue").inner_text() != "N/A · no inference":
                    raise RuntimeError("synthetic retry did not restore no-inference runtime")
                if showcase_request_count != 3:
                    raise RuntimeError("showcase retry did not issue exactly three fixture requests")
            finally:
                showcase_page.close()
        finally:
            browser.close()
    if browser_errors:
        raise RuntimeError("browser console/page errors: " + " | ".join(browser_errors))
    unexpected = [
        url
        for url in requested_urls
        if url != ORT_CDN_URL and not url.startswith(str(served_url))
    ]
    if unexpected:
        raise RuntimeError("unexpected external browser requests: " + " | ".join(unexpected))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable-path", type=Path)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--mobile-screenshot", type=Path)
    parser.add_argument("--base-url", help="use an already-running static server")
    args = parser.parse_args(argv)
    try:
        run_smoke(
            args.executable_path,
            args.screenshot,
            args.base_url,
            args.mobile_screenshot,
        )
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print("[OK] Headless BYOM UI smoke: local synthetic model bytes, stubbed [1,N,7], no inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
