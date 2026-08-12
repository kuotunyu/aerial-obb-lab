"""Headless BYOM demo smoke using synthetic model bytes, image, and ONNX output.

This script tests local model selection, browser wiring, preprocessing, strict output
selection, OBB decoding, drawing, and result rendering without model inference or an
external network request. The committed CDN tag is checked separately by unit tests;
the smoke response replaces only that tag with the synthetic runtime stub.
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
FIXTURE = ROOT / "tests" / "fixtures" / "browser-smoke.svg"
EXPECTED_ROW = ["ship", "0.900", "100.0", "50.0", "90.0"]
ORT_CDN_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js"
ORT_SCRIPT_RE = re.compile(
    rf'<script\b(?=[^>]*\bsrc="{re.escape(ORT_CDN_URL)}")[^>]*></script>',
    flags=re.IGNORECASE | re.DOTALL,
)
ORT_STUB = r"""
globalThis.ort = {
  Tensor: class Tensor {
    constructor(type, data, dims) {
      this.type = type;
      this.data = data;
      this.dims = dims;
    }
  },
  InferenceSession: {
    create: async (modelBytes) => {
      if (!(modelBytes instanceof Uint8Array) || modelBytes.length === 0) {
        throw new Error("expected non-empty local model bytes");
      }
      if (modelBytes[0] === 0) {
        throw new Error("C:\\Users\\alice\\private-model.onnx");
      }
      return {
        run: async () => ({
          output0: {
            dims: [1, 2, 7],
            data: new Float32Array([
              512, 512, 256, 128, 0.9, 1, Math.PI / 2,
              100, 100, 50, 40, 0.2, 2, 0
            ])
          }
        })
      };
    }
  }
};
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
    requested_urls: list[str] = []
    server = nullcontext(base_url) if base_url else static_server()
    with server as served_url, sync_playwright() as playwright:
        launch_options: dict[str, object] = {"headless": True, "args": ["--disable-gpu"]}
        if executable_path is not None:
            launch_options["executable_path"] = str(executable_path)
        browser = playwright.chromium.launch(**launch_options)
        try:
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.add_init_script(
                """
                globalThis.__obbFillTextCalls = 0;
                const originalFillText = CanvasRenderingContext2D.prototype.fillText;
                CanvasRenderingContext2D.prototype.fillText = function (...args) {
                  globalThis.__obbFillTextCalls += 1;
                  return originalFillText.apply(this, args);
                };
                """
            )
            page.on("request", lambda request: requested_urls.append(request.url))
            page.on("pageerror", lambda error: browser_errors.append(str(error)))
            page.on(
                "console",
                lambda message: browser_errors.append(message.text)
                if message.type == "error"
                else None,
            )

            def stub_ort(route: Route) -> None:
                response = route.fetch()
                body, replacements = ORT_SCRIPT_RE.subn(
                    f"<script>{ORT_STUB}</script>", response.text(), count=1
                )
                if replacements != 1:
                    raise RuntimeError("pinned ONNX Runtime Web script tag was not found")
                route.fulfill(response=response, body=body)

            entry_url = f"{str(served_url).rstrip('/')}/"
            page.route(entry_url, stub_ort)
            page.goto(entry_url, wait_until="networkidle")
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
            page.locator("#modelInput").set_input_files(
                files=[
                    {
                        "name": "synthetic-model.onnx",
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
                invalid_page.on(
                    "console", lambda message: invalid_messages.append(message.text)
                )
                invalid_page.on(
                    "pageerror", lambda error: invalid_messages.append(str(error))
                )
                invalid_page.route(entry_url, stub_ort)
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
        if not url.startswith(str(served_url))
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
