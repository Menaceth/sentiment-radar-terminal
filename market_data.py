"""
market_data.py
--------------
Fetches live intraday price data for Indian market indices via yfinance.

Supported symbols:
    ^NSEI  → Nifty 50
    ^BSESN → BSE Sensex
    RELIANCE.NS, TCS.NS, INFY.NS …  (any NSE ticker with .NS suffix)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import yfinance as yf

# ── Symbol catalogue ────────────────────────────────────────────────────────
INDICES: dict[str, str] = {
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
}

STOCKS: dict[str, str] = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "L&T": "LT.NS",
    "State Bank of India": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Bank": "KOTAKBANK.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Wipro": "WIPRO.NS",
    "HCL Tech": "HCLTECH.NS",
    "NTPC": "NTPC.NS",
    "Power Grid": "POWERGRID.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Titan Company": "TITAN.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
}

NSE_OPEN_HOUR_IST = 9
NSE_CLOSE_HOUR_IST = 15
NSE_CLOSE_MINUTE_IST = 30


def market_status() -> dict:
    """Return whether NSE is currently open and a human-readable label."""
    now_ist = datetime.now(timezone.utc).astimezone()
    weekday = now_ist.weekday()
    hour = now_ist.hour
    minute = now_ist.minute

    is_open = (
        weekday < 5  # Mon–Fri
        and (
            hour > NSE_OPEN_HOUR_IST
            or (hour == NSE_OPEN_HOUR_IST and minute >= 15)
        )
        and (
            hour < NSE_CLOSE_HOUR_IST
            or (hour == NSE_CLOSE_HOUR_IST and minute <= 30)
        )
    )
    return {
        "is_open": is_open,
        "label": "🟢 NSE Open" if is_open else "🔴 NSE Closed",
        "color": "#00d26a" if is_open else "#ef4444",
    }


@st.cache_data(ttl=90, show_spinner=False)
def fetch_price_series(symbol: str = "^NSEI") -> pd.DataFrame:
    """
    Download intraday OHLCV for *symbol* for the last trading day.

    Returns a DataFrame with columns: datetime, open, high, low, close, volume
    sorted ascending by datetime.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1d", interval="5m")

        if df.empty:
            # Fallback: try last 5 days in case of holiday / weekend
            df = ticker.history(period="5d", interval="5m")
            if not df.empty:
                last_date = df.index.date[-1]
                df = df[df.index.date == last_date]

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        # Normalise datetime column name (yfinance v0.2 → "datetime")
        if "datetime" not in df.columns and "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})

        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
        df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
        df = df.sort_values("datetime").reset_index(drop=True)
        return df

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=90, show_spinner=False)
def fetch_quote(symbol: str = "^NSEI") -> dict:
    """
    Return the latest price, change, and % change for *symbol*.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info

        price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
        prev_close = getattr(info, "previous_close", None) or getattr(info, "regularMarketPreviousClose", None)

        if price is None or not isinstance(price, (int, float)):
            # Fallback via history
            hist = ticker.history(period="5d", interval="1d")
            if len(hist) >= 1:
                price = float(hist["Close"].iloc[-1])
                prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price

        if isinstance(price, (int, float)) and isinstance(prev_close, (int, float)):
            change = round(price - prev_close, 2)
            pct = round((change / prev_close) * 100, 2) if prev_close else 0
            price_val = round(price, 2)
        else:
            change, pct, price_val = 0, 0, "N/A"

        return {
            "price": price_val,
            "change": change,
            "pct_change": pct,
            "direction": "▲" if change >= 0 else "▼",
            "color": "#00d26a" if change >= 0 else "#ef4444",
        }
    except Exception:
        return {"price": "N/A", "change": 0, "pct_change": 0, "direction": "–", "color": "#94a3b8"}