"""
sentiment.py
------------
VADER-based sentiment analyser extended with a custom Indian
financial market lexicon.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Custom Indian financial lexicon ─────────────────────────────────────────
# Score range: -4 (very bearish) to +4 (very bullish).
# VADER's default is −4..+4 so we stay consistent.
INDIA_FINANCE_LEXICON: dict[str, float] = {
    # ── Bullish signals ──────────────────────────────────────────────────
    "fii buying": 2.5,
    "dii buying": 2.0,
    "promoter buying": 2.0,
    "rbi rate cut": 2.8,
    "repo rate cut": 2.8,
    "bumper ipo": 1.8,
    "oversubscribed": 2.0,
    "nifty all-time high": 3.0,
    "sensex all-time high": 3.0,
    "bull run": 2.5,
    "strong results": 2.2,
    "record profit": 2.5,
    "record revenue": 2.2,
    "beat estimates": 2.0,
    "beat expectations": 2.0,
    "robust growth": 1.8,
    "gdp growth": 1.5,
    "foreign inflow": 1.8,
    "capital inflow": 1.8,
    "dividend declared": 1.5,
    "bonus shares": 1.5,
    "buyback": 1.5,
    "upper circuit": 2.5,
    "breakout": 1.8,
    "rally": 1.8,
    "surge": 1.5,
    "jumps": 1.5,
    "soars": 2.0,
    "gains": 1.2,
    "upside": 1.2,
    "positive outlook": 1.5,
    "upgrade": 1.5,
    "disinvestment": 0.5,
    "infrastructure push": 1.2,
    "make in india": 0.8,
    "capex boost": 1.2,
    "gst collection record": 1.5,
    "export growth": 1.5,

    # ── Bearish signals ──────────────────────────────────────────────────
    "fii selling": -2.5,
    "dii selling": -2.0,
    "promoter selling": -2.0,
    "rbi rate hike": -2.8,
    "repo rate hike": -2.8,
    "sebi ban": -3.0,
    "sebi penalty": -2.5,
    "circuit breaker": -2.5,
    "lower circuit": -2.5,
    "npa": -2.0,
    "bad loans": -2.0,
    "profit warning": -2.5,
    "earnings miss": -2.5,
    "miss estimates": -2.2,
    "miss expectations": -2.2,
    "global selloff": -2.8,
    "market crash": -3.5,
    "recession fears": -2.5,
    "inflation surge": -2.0,
    "oil surge": -2.0,
    "crude surge": -2.0,
    "rupee falls": -1.8,
    "rupee weakness": -1.8,
    "current account deficit": -1.5,
    "trade deficit": -1.5,
    "foreign outflow": -2.0,
    "capital outflow": -2.0,
    "downgrade": -2.0,
    "slowdown": -1.8,
    "correction": -1.5,
    "selloff": -2.0,
    "slumps": -2.0,
    "tumbles": -2.2,
    "plunges": -2.5,
    "drops": -1.2,
    "falls": -1.0,
    "bearish": -2.0,
    "hindenburg": -3.0,
    "fraud": -3.0,
    "scam": -3.0,
    "default": -2.5,
    "insolvency": -2.5,
    "nclt": -1.5,
    "ib notice": -1.8,
    "it raid": -2.5,
    "ed raid": -2.5,

    # ── Neutral / context signals ────────────────────────────────────────
    "ipo": 0.3,
    "fpo": 0.0,
    "qip": 0.2,
    "rbi": 0.0,
    "sebi": 0.0,
    "nse": 0.0,
    "bse": 0.0,
    "results": 0.0,
    "earnings": 0.0,
    "budget": 0.2,
}


def _build_analyser() -> SentimentIntensityAnalyzer:
    """Create and return a VADER analyser extended with the India lexicon."""
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(INDIA_FINANCE_LEXICON)
    return sia


# Single shared instance (module-level, thread-safe for reads)
_analyser = _build_analyser()


def score(text: str) -> dict:
    """
    Analyse sentiment of *text*.

    Returns
    -------
    dict with keys:
        compound : float  – polarity score in [-1, +1]
        label    : str    – "Bullish", "Bearish", or "Neutral"
        pos      : float
        neg      : float
        neu      : float
        emoji    : str
        color    : str    – hex colour for UI
    """
    scores = _analyser.polarity_scores(text.lower())
    compound = round(scores["compound"], 4)

    if compound >= 0.05:
        label, emoji, color = "Bullish", "🟢", "#00d26a"
    elif compound <= -0.05:
        label, emoji, color = "Bearish", "🔴", "#ef4444"
    else:
        label, emoji, color = "Neutral", "⚪", "#94a3b8"

    return {
        "compound": compound,
        "label": label,
        "emoji": emoji,
        "color": color,
        "pos": round(scores["pos"], 4),
        "neg": round(scores["neg"], 4),
        "neu": round(scores["neu"], 4),
    }
