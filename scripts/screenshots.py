"""Capture README screenshots of each dashboard tab. Run with the app on :8517."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8517"
OUT = Path(__file__).resolve().parent.parent / "docs"
OUT.mkdir(exist_ok=True)
TABS = ["Overview", "1 · Comfort model", "2 · Building physics model", "3 · Control & demand response",
        "4 · Safety & deployment"]

with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge")
    page = b.new_page(viewport={"width": 1500, "height": 1000})
    page.goto(URL)
    page.get_by_text("What this demonstrates").wait_for(timeout=240_000)
    page.wait_for_timeout(1500)
    for i, name in enumerate(TABS):
        page.get_by_role("tab", name=name).click()
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT / f"tab{i}.png"), full_page=True)
        print("saved", name)
    b.close()
