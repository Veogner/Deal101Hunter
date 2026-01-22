import json, textwrap, os, re, sqlite3, hashlib, datetime, time, pathlib

base = pathlib.Path("/mnt/data")

deal_hunter_code = r'''#!/usr/bin/env python3
"""
Deal Hunter (RSS-first) — runs nicely on PythonAnywhere.

What it does:
- Pulls deals from RSS feeds (Slickdeals + Reddit by default).
- Filters by your watchlists (keywords, merchants, max price).
- Dedupes using SQLite (so you don't get spammed by the same deal).
- Sends alerts to a Discord webhook.

Why RSS-first:
- Retailer pages (Walmart/Amazon/BestBuy) often block headless scraping.
- RSS feeds are stable, cheap, and ToS-friendly compared to scraping.

Usage:
  python deal_hunter.py --config config.json --once
  python deal_hunter.py --config config.json --loop 900
  python deal_hunter.py --config config.json --reset-seen
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import feedparser  # pip install feedparser
import requests    # pip install requests


PRICE_RE = re.compile(r"(?<!\w)\$([0-9]{1,5}(?:\.[0-9]{1,2})?)")  # grabs "$489" "$489.99" etc


@dataclass(frozen=True)
class Deal:
    source: str
    title: str
    url: str
    published: Optional[str] = None
    summary: Optional[str] = None
    merchant: Optional[str] = None
    price: Optional[float] = None
    guid: Optional[str] = None  # feed entry id


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _extract_price(text: str) -> Optional[float]:
    if not text:
        return None
    m = PRICE_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _guess_merchant(title: str) -> Optional[str]:
    """
    Common deal format: "Item thing $123 @ Walmart" or "... - $123 (Walmart)"
    """
    t = title or ""
    # " @ Merchant"
    if "@" in t:
        maybe = t.split("@")[-1].strip()
        # keep it short-ish
        if 2 <= len(maybe) <= 40:
            return maybe
    # "(Merchant)"
    m = re.search(r"\(([^)]+)\)\s*$", t)
    if m:
        maybe = m.group(1).strip()
        if 2 <= len(maybe) <= 40:
            return maybe
    return None


class SeenDB:
    def __init__(self, path: str):
        self.path = path
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS seen (id TEXT PRIMARY KEY, first_seen_utc TEXT NOT NULL)"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def reset(self) -> None:
        self._conn.execute("DROP TABLE IF EXISTS seen")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS seen (id TEXT PRIMARY KEY, first_seen_utc TEXT NOT NULL)"
        )
        self._conn.commit()

    def has(self, deal_id: str) -> bool:
        cur = self._conn.execute("SELECT 1 FROM seen WHERE id = ? LIMIT 1", (deal_id,))
        return cur.fetchone() is not None

    def add(self, deal_id: str) -> None:
        now = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self._conn.execute("INSERT OR IGNORE INTO seen (id, first_seen_utc) VALUES (?, ?)", (deal_id, now))
        self._conn.commit()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Environment variable override (keeps webhook out of files if you want)
    env_webhook = os.getenv("DISCORD_WEBHOOK_URL")
    if env_webhook:
        cfg["discord_webhook_url"] = env_webhook

    if not cfg.get("discord_webhook_url"):
        raise ValueError("Missing discord_webhook_url in config.json (or DISCORD_WEBHOOK_URL env var).")

    cfg.setdefault("user_agent", "DealHunter/1.0 (+RSS)")
    cfg.setdefault("timeout_seconds", 20)
    cfg.setdefault("max_deals_per_run", 25)
    cfg.setdefault("db_path", "seen.sqlite3")

    # validate
    if not isinstance(cfg.get("sources", []), list) or not cfg["sources"]:
        raise ValueError("config.json must include non-empty 'sources' list.")
    if not isinstance(cfg.get("watchlists", []), list) or not cfg["watchlists"]:
        raise ValueError("config.json must include non-empty 'watchlists' list.")
    return cfg


def fetch_deals_from_sources(cfg: Dict[str, Any]) -> List[Deal]:
    headers = {"User-Agent": cfg["user_agent"]}
    timeout = int(cfg["timeout_seconds"])

    deals: List[Deal] = []
    for src in cfg["sources"]:
        name = src.get("name", "source")
        url = src["url"]
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            for entry in feed.entries[:200]:
                title = entry.get("title", "").strip()
                link = entry.get("link", "").strip()
                if not title or not link:
                    continue
                summary = entry.get("summary", "") or entry.get("description", "")
                published = entry.get("published") or entry.get("updated")
                guid = entry.get("id") or entry.get("guid")
                merchant = _guess_merchant(title)
                price = _extract_price(title) or _extract_price(summary or "")
                deals.append(
                    Deal(
                        source=name,
                        title=title,
                        url=link,
                        published=published,
                        summary=summary,
                        merchant=merchant,
                        price=price,
                        guid=guid,
                    )
                )
        except Exception as e:
            print(f"[warn] source '{name}' failed: {e}", file=sys.stderr)
            continue

    # de-dupe within this run (same url/title)
    uniq: Dict[str, Deal] = {}
    for d in deals:
        k = hashlib.sha256((_norm(d.url) + "|" + _norm(d.title)).encode("utf-8")).hexdigest()
        if k not in uniq:
            uniq[k] = d
    return list(uniq.values())


def deal_matches_watchlist(deal: Deal, wl: Dict[str, Any]) -> bool:
    hay = _norm(deal.title + " " + (deal.summary or ""))

    includes = [ _norm(x) for x in wl.get("include", []) if str(x).strip() ]
    excludes = [ _norm(x) for x in wl.get("exclude", []) if str(x).strip() ]

    # include rules
    if includes:
        if not any(k in hay for k in includes):
            return False

    # exclude rules
    if excludes:
        if any(k in hay for k in excludes):
            return False

    # merchant preference filter (optional hard filter)
    merchants = wl.get("merchants") or wl.get("preferred_merchants") or []
    if wl.get("only_these_merchants") and merchants:
        m = _norm(deal.merchant or "")
        if not any(_norm(x) in m for x in merchants):
            return False

    # price ceiling
    max_price = wl.get("max_price")
    if max_price is not None and deal.price is not None:
        try:
            if float(deal.price) > float(max_price):
                return False
        except Exception:
            pass

    return True


def score_deal(deal: Deal, wl: Dict[str, Any]) -> int:
    """
    Very simple scoring so you see "your stores" first.
    """
    score = 0
    merchants = wl.get("preferred_merchants") or []
    if merchants and deal.merchant:
        m = _norm(deal.merchant)
        for i, pref in enumerate(merchants):
            if _norm(pref) in m:
                score += max(1, 10 - i)  # earlier in list => higher score
                break
    if deal.price is not None:
        score += 1
    if "open box" in _norm(deal.title):
        score += 1
    return score


def format_discord_messages(found: Dict[str, List[Deal]], tz_label: str = "UTC") -> List[str]:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"🔥 **New deals** ({now})"

    lines: List[str] = [header]
    for wl_name, deals in found.items():
        if not deals:
            continue
        lines.append(f"\n**{wl_name}**")
        for d in deals:
            price_txt = f" — **${d.price:.2f}**" if d.price is not None else ""
            merchant_txt = f" ({d.merchant})" if d.merchant else ""
            lines.append(f"• {d.title}{price_txt}{merchant_txt}\n  {d.url}")

    # Discord hard limit is 2000 chars per message; chunk it.
    msgs: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for ln in lines:
        add_len = len(ln) + 1
        if cur and cur_len + add_len > 1800:
            msgs.append("\n".join(cur))
            cur = []
            cur_len = 0
        cur.append(ln)
        cur_len += add_len
    if cur:
        msgs.append("\n".join(cur))
    return msgs


def post_to_discord(webhook_url: str, content: str, timeout: int = 20) -> None:
    r = requests.post(webhook_url, json={"content": content}, timeout=timeout)
    r.raise_for_status()


def run_once(cfg: Dict[str, Any]) -> int:
    db = SeenDB(cfg["db_path"])
    try:
        all_deals = fetch_deals_from_sources(cfg)

        matched: Dict[str, List[Deal]] = {}
        new_count = 0

        for wl in cfg["watchlists"]:
            wl_name = wl.get("name", "Watchlist")
            hits: List[Deal] = []
            for d in all_deals:
                if not deal_matches_watchlist(d, wl):
                    continue

                # stable id for dedupe across runs
                deal_id = d.guid or hashlib.sha256((_norm(d.url)).encode("utf-8")).hexdigest()

                if db.has(deal_id):
                    continue
                hits.append(d)

            # sort by score then title
            hits.sort(key=lambda x: (-score_deal(x, wl), _norm(x.title)))

            # cap per watchlist if you want
            cap = wl.get("max_results", 10)
            hits = hits[: int(cap)]

            # mark seen
            for d in hits:
                deal_id = d.guid or hashlib.sha256((_norm(d.url)).encode("utf-8")).hexdigest()
                db.add(deal_id)
                new_count += 1

            matched[wl_name] = hits

        if new_count == 0:
            print("[ok] no new matches")
            return 0

        messages = format_discord_messages(matched)
        for msg in messages:
            post_to_discord(cfg["discord_webhook_url"], msg, timeout=int(cfg["timeout_seconds"]))
            time.sleep(1)  # small spacing so Discord doesn't hate you

        print(f"[ok] sent {new_count} new deal(s)")
        return new_count
    finally:
        db.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json", help="Path to config.json")
    ap.add_argument("--once", action="store_true", help="Run a single check and exit (default).")
    ap.add_argument("--loop", type=int, default=0, help="Run forever; sleep N seconds between runs.")
    ap.add_argument("--reset-seen", action="store_true", help="Wipe the seen DB (you'll get alerts again).")
    args = ap.parse_args()

    cfg = load_config(args.config)

    db = SeenDB(cfg["db_path"])
    if args.reset_seen:
        db.reset()
        db.close()
        print("[ok] seen database reset")
        return 0
    db.close()

    if args.loop and args.loop > 0:
        print(f"[ok] loop mode every {args.loop}s")
        while True:
            try:
                run_once(cfg)
            except Exception as e:
                print(f"[warn] run failed: {e}", file=sys.stderr)
            time.sleep(args.loop)
        # unreachable
    else:
        return 0 if run_once(cfg) >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''

config_example = {
  "discord_webhook_url": "PUT-YOUR-DISCORD-WEBHOOK-HERE",
  "db_path": "seen.sqlite3",
  "user_agent": "DealHunter/1.0 (+rss)",
  "timeout_seconds": 20,
  "max_deals_per_run": 25,
  "sources": [
    {
      "name": "Slickdeals Frontpage",
      "url": "https://slickdeals.net/newsearch.php?mode=frontpage&searcharea=deals&searchin=first&rss=1"
    },
    {
      "name": "Reddit r/buildapcsales (new)",
      "url": "https://www.reddit.com/r/buildapcsales/new.rss"
    },
    {
      "name": "Reddit r/deals (new)",
      "url": "https://www.reddit.com/r/deals/new.rss"
    }
  ],
  "watchlists": [
    {
      "name": "GPU",
      "include": ["rtx 5070", "5070 ti", "rtx 5060 ti", "5060ti", "gddr7"],
      "exclude": ["prebuilt", "laptop", "case"],
      "max_price": 650,
      "preferred_merchants": ["Walmart", "Best Buy", "Newegg", "Micro Center", "B&H"],
      "max_results": 8
    },
    {
      "name": "Kitchen",
      "include": ["air fryer", "instant pot", "rice cooker", "blender", "keurig", "toaster"],
      "exclude": ["filter", "replacement"],
      "max_price": 150,
      "preferred_merchants": ["Walmart", "Best Buy", "Target", "Costco"],
      "max_results": 8
    },
    {
      "name": "Everyday",
      "include": ["aa batteries", "paper towels", "dish soap", "toothpaste", "laundry detergent"],
      "exclude": [],
      "max_price": 50,
      "preferred_merchants": ["Walmart", "Target", "Costco"],
      "max_results": 8
    }
  ]
}

requirements_txt = """feedparser==6.0.11
requests==2.32.3
"""

# Write files
(deal_file := base/"deal_hunter.py").write_text(deal_hunter_code, encoding="utf-8")
(config_file := base/"config.example.json").write_text(json.dumps(config_example, indent=2), encoding="utf-8")
(req_file := base/"requirements.txt").write_text(requirements_txt, encoding="utf-8")

(str(deal_file), str(config_file), str(req_file))
