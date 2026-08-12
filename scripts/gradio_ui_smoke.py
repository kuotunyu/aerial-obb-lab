"""Headless smoke for the loopback-only, model-free Gradio UI preview."""

from __future__ import annotations

import argparse
import binascii
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


def png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = binascii.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def synthetic_png(width: int = 640, height: int = 360) -> bytes:
    import zlib

    pixel = bytes((34, 96, 180, 255))
    scanlines = b"".join(b"\x00" + pixel * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(scanlines))
        + png_chunk(b"IEND", b"")
    )


def reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_for_server(process: subprocess.Popen[str], url: str) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"Gradio preview exited before startup ({process.returncode}):\n{stdout}\n{stderr}"
            )
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            time.sleep(0.2)
    raise RuntimeError("Gradio preview did not become ready within 30 seconds")


def is_loopback_request(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme in {"blob", "data"}:
        return True
    return parsed.scheme in {"http", "https", "ws", "wss"} and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }


def run_smoke(screenshot: Path | None = None, executable_path: Path | None = None) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required; install the locked development group") from exc

    port = reserve_loopback_port()
    url = f"http://127.0.0.1:{port}"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["GRADIO_ANALYTICS_ENABLED"] = "False"
    process = subprocess.Popen(
        [sys.executable, "demo/gradio_preview.py", "--port", str(port)],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    browser_errors: list[str] = []
    requested_urls: list[str] = []
    server_stderr = ""
    try:
        wait_for_server(process, url)
        with sync_playwright() as playwright:
            launch_options: dict[str, object] = {
                "headless": True,
                "args": ["--disable-gpu"],
            }
            if executable_path is not None:
                launch_options["executable_path"] = str(executable_path)
            browser = playwright.chromium.launch(**launch_options)
            try:
                page = browser.new_page(viewport={"width": 1920, "height": 1080})
                page.on("request", lambda request: requested_urls.append(request.url))
                page.on(
                    "response",
                    lambda response: browser_errors.append(
                        f"HTTP {response.status}: {response.url}"
                    )
                    if response.status >= 400
                    else None,
                )
                page.on("pageerror", lambda error: browser_errors.append(str(error)))
                page.on(
                    "console",
                    lambda message: browser_errors.append(message.text)
                    if message.type == "error"
                    else None,
                )
                page.goto(url, wait_until="domcontentloaded")
                page.locator("#app-header h1").wait_for(state="visible")

                header_text = page.locator("#app-header").inner_text()
                if "UI-only preview" not in header_text or "CPU" not in header_text:
                    raise RuntimeError(f"unexpected preview header: {header_text!r}")
                container_width = page.locator(".gradio-container").evaluate(
                    "element => element.getBoundingClientRect().width"
                )
                if not 1600 <= float(container_width) <= 1740:
                    raise RuntimeError(f"unexpected desktop container width: {container_width}")
                heading_size = page.locator("#app-header h1").evaluate(
                    "element => getComputedStyle(element).fontSize"
                )
                if heading_size != "32px":
                    raise RuntimeError(f"unexpected heading size: {heading_size}")

                detect_button = page.locator("#detect-button")
                if not detect_button.is_disabled():
                    raise RuntimeError("Detect must be disabled before image selection")
                input_box = page.locator("#input-panel").bounding_box()
                result_box = page.locator("#result-panel").bounding_box()
                if input_box is None or result_box is None:
                    raise RuntimeError("workbench panels are not visible")
                if abs(input_box["y"] - result_box["y"]) > 2:
                    raise RuntimeError("desktop input and result panels do not align")
                if result_box["width"] <= input_box["width"]:
                    raise RuntimeError("desktop result panel must be wider than input panel")
                if result_box["height"] <= input_box["height"] + 20:
                    raise RuntimeError("input panel is stretched into unused vertical space")

                page.locator("#input-panel input[type=file]").first.set_input_files(
                    files={
                        "name": "synthetic.png",
                        "mimeType": "image/png",
                        "buffer": synthetic_png(),
                    }
                )
                page.locator("#input-image img").first.wait_for(state="visible")
                page.wait_for_function(
                    "document.querySelector('#app-status').textContent.includes('UI-only preview')"
                )
                if not detect_button.is_disabled():
                    raise RuntimeError("preview Detect must stay disabled after image selection")
                summary = page.locator("#detection-summary").inner_text()
                if "偵測數量：0" not in summary.replace("**", ""):
                    raise RuntimeError(f"preview fabricated a detection summary: {summary!r}")

                if screenshot is not None:
                    screenshot.parent.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(screenshot), full_page=True)

                page.set_viewport_size({"width": 820, "height": 1100})
                page.wait_for_timeout(100)
                input_box = page.locator("#input-panel").bounding_box()
                result_box = page.locator("#result-panel").bounding_box()
                if input_box is None or result_box is None:
                    raise RuntimeError("responsive workbench panels are not visible")
                if result_box["y"] < input_box["y"] + input_box["height"]:
                    raise RuntimeError("responsive result panel does not stack below input panel")
            finally:
                browser.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if process.stderr is not None:
            server_stderr = process.stderr.read()

    if browser_errors:
        server_tail = server_stderr[-2_000:].strip()
        details = " | ".join(browser_errors)
        if server_tail:
            details += f" | preview stderr: {server_tail}"
        raise RuntimeError("browser console/page errors: " + details)
    unexpected = [url for url in requested_urls if not is_loopback_request(url)]
    if unexpected:
        raise RuntimeError("unexpected external browser requests: " + " | ".join(unexpected))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--executable-path", type=Path)
    args = parser.parse_args(argv)
    try:
        run_smoke(args.screenshot, args.executable_path)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print("[OK] Gradio UI smoke: zh-TW wide workbench, responsive stack, no model inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
