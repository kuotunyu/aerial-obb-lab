"""Headless static-demo smoke using a synthetic image and deterministic ONNX stub.

The real bundled ONNX file is checked and hashed by the release gates.  This script
tests browser wiring, preprocessing, strict output selection, OBB decoding, drawing,
and result rendering without performing model inference or contacting the network.
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
DEMO = ROOT / "demo" / "space-static"
FIXTURE = ROOT / "tests" / "fixtures" / "browser-smoke.svg"
EXPECTED_ROW = ["ship", "0.900", "100.0", "50.0", "90.0"]
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
    create: async () => ({
      run: async () => ({
        output0: {
          dims: [1, 2, 7],
          data: new Float32Array([
            512, 512, 256, 128, 0.9, 1, Math.PI / 2,
            100, 100, 50, 40, 0.2, 2, 0
          ])
        }
      })
    })
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
) -> None:
    try:
        from playwright.sync_api import Route, sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required; run the locked development environment") from exc

    browser_errors: list[str] = []
    server = nullcontext(base_url) if base_url else static_server()
    with server as served_url, sync_playwright() as playwright:
        launch_options: dict[str, object] = {"headless": True, "args": ["--disable-gpu"]}
        if executable_path is not None:
            launch_options["executable_path"] = str(executable_path)
        browser = playwright.chromium.launch(**launch_options)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.on("pageerror", lambda error: browser_errors.append(str(error)))
            page.on(
                "console",
                lambda message: browser_errors.append(message.text)
                if message.type == "error"
                else None,
            )

            def stub_ort(route: Route) -> None:
                route.fulfill(status=200, content_type="application/javascript", body=ORT_STUB)

            page.route(re.compile(r"https://cdn\.jsdelivr\.net/.*/ort\.min\.js"), stub_ort)
            page.goto(str(served_url), wait_until="networkidle")
            page.wait_for_function("document.querySelector('#status').textContent === 'model ready'")
            page.locator("#fileInput").set_input_files(str(FIXTURE))
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('image loaded')"
            )
            page.locator("#detectBtn").click()
            page.wait_for_function(
                "document.querySelector('#status').textContent.includes('done in')"
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
            if "1 detection(s)" not in status:
                raise RuntimeError(f"unexpected browser status: {status!r}")
            if screenshot is not None:
                screenshot.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshot), full_page=True)
        finally:
            browser.close()
    if browser_errors:
        raise RuntimeError("browser console/page errors: " + " | ".join(browser_errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable-path", type=Path)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--base-url", help="use an already-running static server")
    args = parser.parse_args(argv)
    try:
        run_smoke(args.executable_path, args.screenshot, args.base_url)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print("[OK] Headless browser UI smoke: synthetic fixture, stubbed [1,N,7] output, no inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
