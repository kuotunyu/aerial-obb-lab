from __future__ import annotations

from playwright.sync_api import sync_playwright

from scripts.browser_smoke import static_server


def test_detect_action_stays_within_the_1600_by_1000_first_viewport() -> None:
    with static_server() as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--disable-gpu"])
        try:
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.goto(f"{base_url}/", wait_until="networkidle")

            detect = page.locator("#detectBtn").bounding_box()

            assert detect is not None
            assert detect["y"] + detect["height"] <= 1000
        finally:
            browser.close()
