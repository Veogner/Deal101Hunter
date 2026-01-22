#!/usr/bin/env python3
"""
Deal Hunter (RSS-first) — runs nicely on GitHub Actions.

What it does:
- Pulls deals from RSS feeds (Slickdeals + Reddit by default).
- Filters by your watchlists (keywords, merchants, max price).
- Dedupes using SQLite (so you don't get spammed by the same deal).
- Sends alerts to a Discord webhook.

Usage:
  python deal_hunter.py --config config.json --once
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
from typing import Any, Dict, List, Optional

import feedparser  # pip install feedparser
import requests    # pip install requests

PRICE_RE = re.compile(r"(?<!\w)\$([0-9]{1,5}(?:\.[0-9]{1,2})?)")


@dataclass(frozen=True)
class Deal:
    source: str
    title: str
    url: str
    published: Optional[str] = None
    summary: Optional[str] = None
    merchant: Optional[str] = None
    price: Optional[float] = None
    guid: Optional[str] = None


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
    t = title or ""
    if "@" in t:
        maybe = t.split("@")[-1].strip()
        if 2 <= len(maybe) <= 40:
            return maybe
    m = re.search(r"\(([^)]+)\)\s*$", t)
    if m:
        maybe = m.group(1).strip()
        if 2 <= len(maybe) <= 40:
            return maybe
    return None


class SeenDB:
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path)
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
        self._conn.execute(
            "INSERT OR IGNORE INTO seen (id, first_seen_utc) VALUES (?, ?)",
            (deal_id, now),
        )
        self._conn.commit()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    env_webhook = os.getenv("DISCORD_WEBHOOK_URL")
    if env_webhook:
        cfg["discord_webhook_url"] = env_webhook

    if not cfg.get("discord_webhook_url"):
        raise ValueError("Missing discord_webhook_url in config.json (or DISCORD_WEBHOOK_URL env var).")

    cfg.setdefault("user_agent", "DealHunter/1.0 (+RSS)")
    cfg.setdefault("timeout_seconds", 20)
    cfg.setdefault("db_path", "seen.sqlite3")

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
                title = (entry.get("title") or "").strip()
                link = (entry.get("link") or "").strip()
                if not title or not link:
                    continue
                summary = entry.get("summary") or entry.get("description") or ""
                published = entry.get("published") or entry.get("updated")
                guid = entry.get("id") or entry.get("guid")
                merchant = _guess_merchant(title)
                price = _extract_price(title) or _extract_price(summary)
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

    # de-dupe within this run (same url/title)
    uniq: Dict[str, Deal] = {}
    for d in deals:
        k = hashlib.sha256((_norm(d.url) + "|" + _norm(d.title)).encode("utf-8")).hexdigest()
        if k not in uniq:
            uniq[k] = d
    return list(uniq.values())


def deal_matches_watchlist(deal: Deal, wl: Dict[str, Any]) -> bool:
    hay = _norm(deal.title + " " + (deal.summary or ""))

    includes = [_norm(x) for x in wl.get("include", []) if str(x).strip()]
    excludes = [_norm(x) for x in wl.get("exclude", []) if str(x).strip()]

    if includes and not any(k in hay for k in includes):
        return False
    if excludes and any(k in hay for k in excludes):
        return False

    merchants = wl.get("preferred_merchants") or []
    if wl.get("only_these_merchants") and merchants:
        m = _norm(deal.merchant or "")
        if not any(_norm(x) in m for x in merchants):
            return False

    max_price = wl.get("max_price")
    if max_price is not None and deal.price is not None:
        try:
            if float(deal.price) > float(max_price):
                return False
        except Exception:
            pass

    return True


def score_deal(deal: Deal, wl: Dict[str, Any]) -> int:
    score = 0
    merchants = wl.get("preferred_merchants") or []
    if merchants and deal.merchant:
        m = _norm(deal.merchant)
        for i, pref in enumerate(merchants):
            if _norm(pref) in m:
                score += max(1, 10 - i)
                break
    if deal.price is not None:
        score += 1
    if "open box" in _norm(deal.title):
        score += 1
    return score


def format_discord_messages(found: Dict[str, List[Deal]]) -> List[str]:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = [f"🔥 **New deals** ({now})"]

    for wl_name, deals in found.items():
        if not deals:
            continue
        lines.append(f"\n**{wl_name}**")
        for d in deals:
            price_txt = f" — **${d.price:.2f}**" if d.price is not None else ""
            merchant_txt = f" ({d.merchant})" if d.merchant else ""
            lines.append(f"• {d.title}{price_txt}{merchant_txt}\n  {d.url}")

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

                deal_id = d.guid or hashlib.sha256(_norm(d.url).encode("utf-8")).hexdigest()
                if db.has(deal_id):
                    continue
                hits.append(d)

            hits.sort(key=lambda x: (-score_deal(x, wl), _norm(x.title)))
            cap = int(wl.get("max_results", 10))
            hits = hits[:cap]

            for d in hits:
                deal_id = d.guid or hashlib.sha256(_norm(d.url).encode("utf-8")).hexdigest()
                db.add(deal_id)
                new_count += 1

            matched[wl_name] = hits

        if new_count == 0:
            print("[ok] no new matches")
            return 0

        for msg in format_discord_messages(matched):
            post_to_discord(cfg["discord_webhook_url"], msg, timeout=int(cfg["timeout_seconds"]))
            time.sleep(1)

        print(f"[ok] sent {new_count} new deal(s)")
        return new_count
    finally:
        db.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--reset-seen", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.reset_seen:
        db = SeenDB(cfg["db_path"])
        db.reset()
        db.close()
        print("[ok] seen database reset")
        return 0

    run_once(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
