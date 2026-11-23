"""Capture real screenshots of the running application for the README/report/slides.

Prerequisites: backend on :8000 and frontend on :5173 are running, with demo data
(python scripts/seed_demo.py). Uses Playwright with the locally installed
Microsoft Edge or Google Chrome (no browser download needed):

    pip install playwright
    python scripts/capture_screenshots.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
BASE = "http://127.0.0.1:5173"
API = "http://127.0.0.1:8000"
QUESTION = "From which university did the 2024 Nobel Prize in Physics laureate from the University of Toronto receive his PhD?"


def latest_query_id() -> str | None:
    with urllib.request.urlopen(f"{API}/api/qa/history?limit=20") as r:
        for q in json.loads(r.read())["queries"]:
            if q["status"] == "completed" and q["mode"] == "multi_document" and not q["insufficient_evidence"]:
                return q["id"]
    return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = None
        for channel in ("msedge", "chrome"):
            try:
                browser = p.chromium.launch(channel=channel, headless=True)
                break
            except Exception:
                continue
        if browser is None:
            print("No Edge/Chrome found; run `playwright install chromium`.")
            return 1
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5,
                                color_scheme="dark")

        def shot(name: str, url: str, wait_text: str | None = None, full: bool = False, delay: int = 2500) -> None:
            page.goto(BASE + url, wait_until="networkidle")
            if wait_text:
                page.get_by_text(wait_text, exact=False).first.wait_for(timeout=60000)
            page.wait_for_timeout(delay)
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=full)
            print("saved", name)

        shot("01_dashboard", "/", "GraphRAG-QA")
        shot("02_documents", "/documents", "nobel_prizes_2024")
        # ask a fresh question through the UI
        page.goto(BASE + "/qa", wait_until="networkidle")
        page.locator("textarea").first.fill(QUESTION)
        page.get_by_role("button", name="Ask").first.click()
        # wait until the request is finished (button no longer says "Answering…") and the answer has rendered
        page.wait_for_function(
            "() => !document.body.innerText.includes('Answering…') && !document.body.innerText.includes('Answering...')"
            " && document.body.innerText.includes('Final answer')",
            timeout=180000)
        page.get_by_text("University of Edinburgh", exact=False).first.wait_for(timeout=30000)
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT / "03_question_answering.png"), full_page=False)
        page.screenshot(path=str(OUT / "03b_question_answering_full.png"), full_page=True)
        print("saved 03_question_answering")
        qid = latest_query_id()
        if qid:
            shot("04_graph_evidence_chain", f"/graph?query={qid}&view=chain", None, delay=4000)
            shot("05_graph_question_subgraph", f"/graph?query={qid}&view=subgraph", None, delay=4000)
        shot("06_graph_full", "/graph", None, delay=4000)
        shot("07_analytics", "/analytics", None, full=True, delay=4000)
        shot("08_evaluation", "/evaluation", None, full=True, delay=3000)
        shot("09_settings", "/settings", None, full=True)
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
