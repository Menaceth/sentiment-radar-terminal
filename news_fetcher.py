"""
news_fetcher.py
---------------
Ingests financial headlines from free Indian market RSS feeds,
scores each article with VADER, and returns a clean DataFrame.

Decorated with @st.cache_data(ttl=90) so Streamlit only re-fetches
every 90 seconds even when the app auto-refreshes.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import feedparser
import pandas as pd
import streamlit as st

from sentiment import score as sentiment_score

# ── RSS feed catalogue ──────────────────────────────────────────────────────
RSS_FEEDS: list[dict] = [
    {
        "name": "Economic Times",
        "url": "https://economictimes.indiatimes.com/markets/rss.cms",
        "logo": "ET",
    },
    {
        "name": "Moneycontrol",
        "url": "https://www.moneycontrol.com/rss/marketsnews.xml",
        "logo": "MC",
    },
    {
        "name": "NDTV Profit",
        "url": "https://feeds.feedburner.com/ndtvprofit-latest",
        "logo": "NDTV",
    },
    {
        "name": "Mint Markets",
        "url": "https://www.livemint.com/rss/markets",
        "logo": "MINT",
    },
    {
        "name": "Business Standard",
        "url": "https://www.business-standard.com/rss/markets-106.rss",
        "logo": "BS",
    },
]

# Source badge colours for the UI
SOURCE_COLORS: dict[str, str] = {
    "Economic Times": "#ff6600",
    "Moneycontrol": "#003087",
    "NDTV Profit": "#e60000",
    "Mint Markets": "#1a1a2e",
    "Business Standard": "#c0392b",
}


def _parse_date(entry: feedparser.FeedParserDict) -> datetime:
    """Extract and normalise the published date from a feed entry."""
    try:
        import time
        t = entry.get("published_parsed") or entry.get("updated_parsed")
        if t:
            return datetime.fromtimestamp(time.mktime(t), tz=timezone.utc)
    except Exception:
        pass
    return datetime.now(tz=timezone.utc)


def _article_id(title: str, url: str) -> str:
    """Stable unique ID for deduplication."""
    return hashlib.md5(f"{title}{url}".encode()).hexdigest()[:12]


def _fetch_feed(feed: dict) -> list[dict]:
    """Fetch and parse one RSS feed. Returns a list of article dicts."""
    articles = []
    try:
        parsed = feedparser.parse(feed["url"])
        for entry in parsed.entries:
            title = entry.get("title", "").strip()
            url = entry.get("link", "").strip()
            summary = entry.get("summary", entry.get("description", "")).strip()

            if not title or not url:
                continue

            # Combine title + summary for richer sentiment signal
            text_for_scoring = f"{title}. {summary}"[:512]
            sent = sentiment_score(text_for_scoring)

            articles.append(
                {
                    "id": _article_id(title, url),
                    "title": title,
                    "summary": summary[:200] if summary else "",
                    "url": url,
                    "source": feed["name"],
                    "logo": feed["logo"],
                    "source_color": SOURCE_COLORS.get(feed["name"], "#64748b"),
                    "published": _parse_date(entry),
                    "compound": sent["compound"],
                    "label": sent["label"],
                    "emoji": sent["emoji"],
                    "color": sent["color"],
                    "pos": sent["pos"],
                    "neg": sent["neg"],
                    "neu": sent["neu"],
                }
            )
    except Exception as exc:
        # Silently swallow individual feed errors so one bad feed
        # doesn't break the whole dashboard.
        st.toast(f"⚠️ Could not fetch {feed['name']}: {exc}", icon="⚠️")

    return articles


@st.cache_data(ttl=90, show_spinner=False)
def fetch_news(keyword: str = "") -> pd.DataFrame:
    """
    Fetch news from all RSS feeds, deduplicate, and optionally filter
    by *keyword*.

    Parameters
    ----------
    keyword : str
        If non-empty, only articles whose title or summary contains
        the keyword (case-insensitive) are returned.

    Returns
    -------
    pd.DataFrame sorted by published timestamp (newest first).
    """
    all_articles: list[dict] = []
    seen_ids: set[str] = set()

    for feed in RSS_FEEDS:
        for article in _fetch_feed(feed):
            if article["id"] not in seen_ids:
                seen_ids.add(article["id"])
                all_articles.append(article)

    if not all_articles:
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)
    df = df.sort_values("published", ascending=False).reset_index(drop=True)

    # Keyword filter (case-insensitive, searches title + summary)
    if keyword.strip():
        kw = keyword.strip().lower()
        mask = (
            df["title"].str.lower().str.contains(kw, na=False)
            | df["summary"].str.lower().str.contains(kw, na=False)
        )
        df = df[mask].reset_index(drop=True)

    return df


def hourly_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment counts per hour for the bar chart.

    Returns columns: hour_label, Bullish, Bearish, Neutral, avg_compound
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["hour"] = pd.to_datetime(df["published"]).dt.floor("h")
    grp = df.groupby("hour")

    rows = []
    for hour, group in grp:
        rows.append(
            {
                "hour": hour,
                "hour_label": hour.strftime("%d %b %H:%M"),
                "Bullish": (group["label"] == "Bullish").sum(),
                "Bearish": (group["label"] == "Bearish").sum(),
                "Neutral": (group["label"] == "Neutral").sum(),
                "avg_compound": round(group["compound"].mean(), 4),
                "total": len(group),
            }
        )

    return pd.DataFrame(rows).sort_values("hour")
