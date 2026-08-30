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
ORT_STUB = r"""
globalThis.__ortCreateCount = 0;
globalThis.__ortReleaseCount = 0;
globalThis.__ortActiveSessionIdsAtRelease = [];
globalThis.__ortRunSessionIds = [];
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
      if (!(modelBytes instanceof Uint8Array) || modelBytes.length === 0) {
        throw new Error("expected non-empty local model bytes");
      }
      if (modelBytes[0] === 0) {
        throw new Error("C:\\Users\\alice\\private-model.onnx");
      }
      if (modelBytes[0] === 2) {
        await new Promise((resolve) => setTimeout(resolve, 150));
      }
      let released = false;
      return {
        __sessionId: sessionId,
        inputNames: ["images"],
        outputNames: modelBytes[0] === 1 ? ["unexpected"] : ["output0"],
        release: async () => {
          if (released) throw new Error("session released twice");
          released = true;
          globalThis.__ortReleaseCount += 1;
          globalThis.__ortActiveSessionIdsAtRelease.push(
            state.session?.__sessionId ?? null
          );
        },
        run: async () => {
          if (released) throw new Error("released session was used");
          globalThis.__ortRunSessionIds.push(sessionId);
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
            page.locator("#confSlider").evaluate("(slider) => { slider.value = '0.25'; slider.dispatchEvent(new Event('input', { bubbles: true })); }")

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
            page.locator("#fileInput").set_input_files(str(FIXTURE))
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('影像已載入')"
            )
            if page.locator("#detectBtn").is_disabled():
                raise RuntimeError("Detect must be enabled after local model and image selection")
            page.locator("#detectBtn").click()
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('完成')"
            )

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
                        "buffer": b"\x01invalid-contract",
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
                        "name": "stale.onnx",
                        "mimeType": "application/octet-stream",
                        "buffer": b"\x02stale-candidate",
                    }
                ]
            )
            page.locator("#showcaseBtn").click()
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('Synthetic fixture')"
            )
            page.wait_for_function(
                "globalThis.__ortCreateCount === 4 && globalThis.__ortReleaseCount === 4"
            )
            if page.locator("#modelLabel").inner_text() != "Local ONNX model ready":
                raise RuntimeError("stale candidate changed the neutral active-model label")
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
                error_status = invalid_page.locator("#status").inner_text()
                expected_error = "模型載入失敗，請確認 ONNX 格式與 output contract。"
                if error_status != expected_error:
                    raise RuntimeError(f"unsafe model error copy: {error_status!r}")
                visible_and_console = " | ".join([error_status, *invalid_messages])
                if "alice" in visible_and_console or "private-model" in visible_and_console:
                    raise RuntimeError("model error leaked a private path to the UI or console")
            finally:
                invalid_page.close()
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
