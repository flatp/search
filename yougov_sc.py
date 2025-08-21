#!/usr/bin/env python3
"""
YouGov Ratings scraper (full list)

Scrapes a ranking page like:
- https://yougov.co.uk/ratings/consumer/popularity/fashion-clothing-brands/all

and extracts: rank, name, fame (%), popularity (%), quarter/period and writes a CSV.

This version uses Playwright to click "Load More" until the full list is loaded.

Usage:
  pip install playwright bs4
  playwright install
  python yougov_ratings_scraper.py https://yougov.co.uk/ratings/consumer/popularity/fashion-clothing-brands/all --out brands.csv

Respect YouGov's terms of use when scraping.
"""
from __future__ import annotations

import re
import sys
import csv
import time
from dataclasses import dataclass
from typing import List

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

LINE_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s+(\d{1,3})%\s+(\d{1,3})%\s*$")
PERIOD_RE = re.compile(r"\((Q\d\s+\d{4})\)")

@dataclass
class Row:
    url: str
    period: str | None
    rank: int
    name: str
    fame: int
    popularity: int


def _guess_period(soup: BeautifulSoup) -> str | None:
    h1 = soup.find(["h1", "h2"])
    if not h1:
        return None
    m = PERIOD_RE.search(h1.get_text(" ", strip=True))
    return m.group(1) if m else None


def _parse_list_items(soup: BeautifulSoup, url: str) -> List[Row]:
    text = "\n".join(li.get_text(" ", strip=True) for li in soup.find_all("li"))
    period = _guess_period(soup)
    rows: List[Row] = []
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        rank = int(m.group(1))
        name = m.group(2)
        fame = int(m.group(3))
        popularity = int(m.group(4))
        rows.append(Row(url=url, period=period, rank=rank, name=name, fame=fame, popularity=popularity))
    return rows


def scrape_full_with_playwright(url: str) -> List[Row]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # headless=False にするとブラウザが見える
        page = browser.new_page()
        page.goto(url, timeout=60000)

        try:
            cookie_btn = page.locator("#onetrust-accept-btn-handler")
            if cookie_btn.is_visible():
                print("クッキー同意ボタンをクリック")
                cookie_btn.click()
                page.wait_for_timeout(2000)
        except Exception as e:
            print("クッキー押下で例外:", e)

        # ページ最初の li の数を確認
        print("初期 li 件数:", page.locator("li").count())

        # 「and more」ボタンを一度押す
        try:
            btn = page.locator("button[name='rankings-load-more-entities']")
            if btn.is_visible():
                print("ボタンが見つかったのでクリック")
                before_count = page.locator("li").count()
                btn.click()
                page.wait_for_function(
                    "(before) => document.querySelectorAll('li').length > before",
                    arg=before_count,
                    timeout=10000
                )
                print("クリック後 li 件数:", page.locator("li").count())
        except Exception as e:
            print("ボタン押下で例外:", e)

        # ページ最下部までスクロールしながら li の数を確認
        scroll_attempts = 0
        max_attempts_without_increase = 50  # 5回連続で増えなければ終了
        prev_count = 0

        while scroll_attempts < max_attempts_without_increase:
            current_count = page.locator("li").count()
            if current_count > prev_count:
                prev_count = current_count
                scroll_attempts = 0  # 件数が増えたらカウントリセット
            else:
                scroll_attempts += 1

                # 小刻みにスクロール
                page.evaluate("window.scrollBy(0, 200)")
                page.wait_for_timeout(500)

        html = page.content()
        browser.close()

    soup = BeautifulSoup(html, "html.parser")
    return _parse_list_items(soup, url)



def write_csv(all_rows: List[Row], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "fame_pct", "popularity_pct"])
        for r in all_rows:
            w.writerow([r.name, r.fame, r.popularity])


def main(url: str, out_path: str) -> None:
    rows = scrape_full_with_playwright(url)
    print(f"Scraped {len(rows)} rows from {url}")
    write_csv(rows, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python yougov_ratings_scraper.py <url> [--out out.csv]")
        sys.exit(0)
    url = sys.argv[1]
    out = "yougov_ratings.csv"
    if "--out" in sys.argv:
        idx = sys.argv.index("--out")
        out = sys.argv[idx + 1]
    main(url, out)
