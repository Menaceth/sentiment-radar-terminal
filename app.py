"""
app.py
------
Live Market News Sentiment Radar — Indian Stock Market Dashboard
Built with Streamlit + Plotly + VADER NLP
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

import market_data as md
import news_fetcher as nf

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Radar | Terminal Mode",
    page_icon="RADAR",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st_autorefresh(interval=90_000, key="auto_refresh")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .stApp {
        background: #06090e;
        color: #cbd5e1;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 98% !important;
    }
    section[data-testid="stSidebar"] {
        background: #0a0f18;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stMetric"] {
        background: #0b1320 !important;
        border: 1px solid #1e293b !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
    }
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 800 !important;
        font-size: 1.15rem !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .news-row {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .news-row:hover {
        border-color: #38bdf8;
        background: rgba(15, 23, 42, 0.95);
    }
    .news-row-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: #f1f5f9;
        text-decoration: none;
    }
    .news-row-meta {
        font-size: 0.7rem;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 6px;
    }
    .pill {
        display: inline-flex;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.68rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #090e17;
        padding: 4px;
        border-radius: 6px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        height: 32px;
        border-radius: 4px;
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .stTabs [aria-selected="true"] {
        background: #1e293b !important;
        color: #38bdf8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color="#64748b", size=10),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.03)",
        zerolinecolor="rgba(255,255,255,0.03)",
        tickfont=dict(size=9, color="#64748b"),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.03)",
        zerolinecolor="rgba(255,255,255,0.03)",
        tickfont=dict(size=9, color="#64748b"),
    ),
    legend=dict(
        bgcolor="rgba(11,19,32,0.8)",
        bordercolor="#1e293b",
        borderwidth=1,
        font=dict(size=10, color="#94a3b8"),
    ),
    hoverlabel=dict(
        bgcolor="#0f172a",
        bordercolor="#38bdf8",
        font=dict(family="JetBrains Mono, monospace", size=11, color="#f8fafc"),
    ),
)


# ── Chart 1: Fixed Reaction Line Chart ───────────────────────────────────────
# ── Chart 1: Fixed Reaction Line Chart ───────────────────────────────────────
def make_reaction_chart(price_df: pd.DataFrame, news_df: pd.DataFrame, symbol_label: str) -> go.Figure:
    fig = go.Figure()

    # 1. Calculate explicit dynamic Y-axis bounds for the price line
    if not price_df.empty and "close" in price_df.columns:
        p_min = price_df["close"].min()
        p_max = price_df["close"].max()
        p_padding = (p_max - p_min) * 0.15 if p_max != p_min else p_max * 0.01
        price_range = [p_min - p_padding, p_max + p_padding]

        fig.add_trace(
            go.Scatter(
                x=price_df["datetime"],
                y=price_df["close"],
                name=f"{symbol_label} Price",
                line=dict(color="#38bdf8", width=2),
                fill="tozeroy",
                fillcolor="rgba(56,189,248,0.04)",
                yaxis="y1",
                hovertemplate="<b>%{x|%H:%M}</b><br>Price: ₹%{y:,.2f}<extra></extra>",
            )
        )
    else:
        price_range = None

    # 2. Resample sentiment over continuous 15-minute intervals across market hours
    if not news_df.empty and not price_df.empty:
        df_s = news_df.copy()
        df_s["pub_dt"] = pd.to_datetime(df_s["published"]).dt.tz_localize(None)

        # Resample news sentiment onto 15-minute grid
        sent_series = (
            df_s.set_index("pub_dt")["compound"]
            .resample("15min")
            .mean()
            .reset_index()
            .rename(columns={"pub_dt": "datetime", "compound": "avg_sentiment"})
        )

        # Merge with price timeline to align timestamps
        merged_s = pd.merge_asof(
            price_df[["datetime"]].sort_values("datetime"),
            sent_series.sort_values("datetime"),
            on="datetime",
            direction="backward",
        )
        
        # Smooth and fill gaps across trading hours
        merged_s["avg_sentiment"] = merged_s["avg_sentiment"].ffill().bfill().fillna(0)

        fig.add_trace(
            go.Scatter(
                x=merged_s["datetime"],
                y=merged_s["avg_sentiment"],
                name="Sentiment Impulse",
                line=dict(color="#f59e0b", width=1.5, dash="dot"),
                yaxis="y2",
                hovertemplate="<b>%{x|%H:%M}</b><br>Sentiment: %{y:+.3f}<extra></extra>",
            )
        )

    layout = dict(**PLOTLY_LAYOUT)
    layout.update(
        height=340,
        title=dict(text=f"{symbol_label} INTRADAY PRICE vs SENTIMENT IMPULSE", font=dict(size=11, color="#38bdf8")),
        yaxis=dict(
            range=price_range,   # Force explicit dynamic price range [min - padding, max + padding]
            autorange=False if price_range else True,
            zeroline=False,
            gridcolor="rgba(255,255,255,0.03)",
            title=dict(text="Price (₹)", font=dict(color="#38bdf8", size=9)),
            tickfont=dict(color="#38bdf8", size=9),
        ),
        yaxis2=dict(
            title=dict(text="Sentiment Index", font=dict(color="#f59e0b", size=9)),
            tickfont=dict(color="#f59e0b", size=9),
            overlaying="y",
            side="right",
            range=[-1.05, 1.05],
            zeroline=True,
            zerolinecolor="rgba(245,158,11,0.2)",
        ),
        hovermode="x unified",
    )
    fig.update_layout(**layout)
    return fig


# ── Chart 2: Stock Performance Treemap (Heatmap) ─────────────────────────────
def make_heatmap(ticker_quotes: dict[str, dict]) -> go.Figure:
    data = []
    for name, q in ticker_quotes.items():
        if isinstance(q.get("pct_change"), (int, float)):
            data.append({
                "stock": name,
                "pct_change": q["pct_change"],
                "price": f"₹{q['price']:,}" if isinstance(q['price'], (int, float)) else str(q['price']),
                "group": "Market Movers",
                "abs_change": abs(q["pct_change"]) + 0.1,
            })

    if not data:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, height=230)
        return fig

    df_map = pd.DataFrame(data)
    fig = px.treemap(
        df_map,
        path=["group", "stock"],
        values="abs_change",
        color="pct_change",
        color_continuous_scale=[[0, "#ef4444"], [0.5, "#334155"], [1, "#10b981"]],
        color_continuous_midpoint=0,
        custom_data=["price", "pct_change"],
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Price: %{customdata[0]}<br>Change: %{customdata[1]:+.2f}%<extra></extra>",
        texttemplate="<b>%{label}</b><br>%{customdata[1]:+.2f}%",
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=230, coloraxis_showscale=False)
    return fig


# ── Chart 3: Headline Keywords Bar Chart ─────────────────────────────────────
def make_keywords_chart(news_df: pd.DataFrame) -> go.Figure:
    if news_df.empty:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, height=230)
        return fig

    all_titles = " ".join(news_df["title"].tolist()).lower()
    keywords = ["rbi", "nifty", "sensex", "reliance", "tcs", "adani", "hdfc", "ipo", "inflation", "fii", "budget", "sebi"]
    counts = [{"term": k.upper(), "count": all_titles.count(k)} for k in keywords if all_titles.count(k) > 0]
    df_kw = pd.DataFrame(counts).sort_values("count", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df_kw["count"],
            y=df_kw["term"],
            orientation="h",
            marker=dict(color="#38bdf8"),
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=230, title="HEADLINE KEYWORD FREQUENCY")
    return fig


# ── Gauge Indicator ──────────────────────────────────────────────────────────
def make_gauge(compound: float) -> go.Figure:
    bar_color = "#10b981" if compound >= 0.05 else "#ef4444" if compound <= -0.05 else "#f59e0b"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=compound,
            number=dict(font=dict(size=30, color="#f8fafc", family="JetBrains Mono"), valueformat=".3f"),
            gauge=dict(
                axis=dict(range=[-1, 1], tickwidth=1, tickcolor="#334155", tickfont=dict(size=8, color="#64748b"), dtick=0.5),
                bar=dict(color=bar_color, thickness=0.22),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[-1.0, -0.05], color="rgba(239,68,68,0.12)"),
                    dict(range=[-0.05, 0.05], color="rgba(148,163,184,0.05)"),
                    dict(range=[0.05, 1.0], color="rgba(16,185,129,0.12)"),
                ],
            ),
            title=dict(text="MARKET MOOD INDEX", font=dict(size=7, color="#64748b", family="JetBrains Mono")),
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=200)
    return fig


# ── Render Single News Card Row ───────────────────────────────────────────────
def render_news_card(row: pd.Series) -> str:
    border_color = row["color"]
    score = row["compound"]
    pub = pd.to_datetime(row["published"])
    try:
        pub_str = pub.strftime("%d %b · %H:%M")
    except Exception:
        pub_str = "—"
    source_color = row.get("source_color", "#64748b")

    return f"""
    <div class="news-row" style="border-left: 3px solid {border_color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 6px;">
                <span class="pill" style="background:{source_color}22; color:{source_color};">{row['source']}</span>
                <span class="pill" style="background:{border_color}18; color:{border_color};">{row['label']}</span>
            </div>
            <span class="pill" style="background:rgba(255,255,255,0.04); color:#94a3b8;">{score:+.3f}</span>
        </div>
        <div style="margin-top: 6px;">
            <a href="{row['url']}" target="_blank" class="news-row-title">{row['title']}</a>
        </div>
        <div class="news-row-meta"><span>{row['source']}</span> • <span>{pub_str} IST</span></div>
    </div>
    """


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-family:JetBrains Mono; font-size:1.1rem; font-weight:800; color:#38bdf8;'>TERMINAL CONTROLS</div>", unsafe_allow_html=True)
    status = md.market_status()
    clean_label = status['label'].replace("🟢 ", "").replace("🔴 ", "")
    st.markdown(f"<div style='background:{status['color']}15; border:1px solid {status['color']}35; border-radius:6px; padding:6px; text-align:center; font-size:0.75rem; font-weight:700; color:{status['color']}; margin:10px 0;'>{clean_label}</div>", unsafe_allow_html=True)

    keyword = st.text_input("Search Keyword", placeholder="Type keyword...", label_visibility="collapsed")
    chips = ["Nifty", "Sensex", "RBI", "Reliance", "TCS", "Adani", "HDFC", "Infosys", "IPO"]
    chip_cols = st.columns(3)
    for i, chip in enumerate(chips):
        if chip_cols[i % 3].button(chip, key=f"chip_{chip}"):
            keyword = chip

    st.divider()
    index_choice = st.selectbox("Market Index", list(md.INDICES.keys()))
    symbol = md.INDICES[index_choice]
    stock_choice = st.selectbox("Watchlist Stock", ["— None —"] + list(md.STOCKS.keys()))


# ── Data Ingestion ────────────────────────────────────────────────────────────
with st.spinner("Syncing Feed Telemetry..."):
    news_df = nf.fetch_news(keyword=keyword)
    price_df = md.fetch_price_series(symbol)

    stock_symbol = md.STOCKS.get(stock_choice, None) if stock_choice != "— None —" else None
    stock_price_df = md.fetch_price_series(stock_symbol) if stock_symbol else pd.DataFrame()

    ticker_quotes: dict[str, dict] = {}
    for name, sym in list(md.INDICES.items()) + list(md.STOCKS.items())[:6]:
        ticker_quotes[name] = md.fetch_quote(sym)

    hourly_df = nf.hourly_sentiment(news_df) if not news_df.empty else pd.DataFrame()


# ── Top Header Bar ───────────────────────────────────────────────────────────
hdr_col, ts_col = st.columns([3, 1])
with hdr_col:
    title_keyword = f" [ FILTER: \"{keyword}\" ]" if keyword.strip() else ""
    st.markdown(f"<div style='font-family:JetBrains Mono;'><span style='font-size:1.3rem; font-weight:800; color:#f8fafc;'>SENTIMENT RADAR TERMINAL</span> <span style='color:#38bdf8; font-size:1rem;'>{title_keyword}</span></div>", unsafe_allow_html=True)
with ts_col:
    from datetime import datetime
    now_str = datetime.now().strftime("%d %b %Y • %H:%M:%S IST")
    st.markdown(f"<div style='text-align:right; font-family:JetBrains Mono; font-size:0.75rem; color:#38bdf8; font-weight:700;'>SYNC: {now_str}</div>", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Top Tickers Grid ──
t_cols = st.columns(len(ticker_quotes))
for i, (name, q) in enumerate(ticker_quotes.items()):
    with t_cols[i]:
        price_val = q["price"]
        price_display = f"₹{price_val:,}" if isinstance(price_val, (int, float)) else str(price_val)
        pct_val = q.get("pct_change", 0)
        delta_val = f"{pct_val:+.2f}%" if isinstance(pct_val, (int, float)) else None
        st.metric(label=name, value=price_display, delta=delta_val)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ── Main Layout Split ─────────────────────────────────────────────────────────
chart_df = stock_price_df if (stock_symbol and not stock_price_df.empty) else price_df
chart_label = stock_choice if (stock_symbol and not stock_price_df.empty) else index_choice

col_left, col_right = st.columns([1.8, 1.2])

with col_left:
    # 1. Main Reaction Chart
    st.plotly_chart(make_reaction_chart(chart_df, news_df, chart_label), width="stretch", config={"displayModeBar": False})

    # 2. Tabs
    tab_stream, tab_sources, tab_heatmap, tab_keywords = st.tabs(
        ["LIVE FEED", "SOURCE ANALYTICS", "MARKET HEATMAP", "KEYWORD MATRIX"]
    )

    with tab_stream:
        if news_df.empty:
            st.info("No matching feed items found.")
        else:
            cards_html = "".join(render_news_card(row) for _, row in news_df.head(25).iterrows())
            st.markdown(cards_html, unsafe_allow_html=True)

    with tab_sources:
        if not news_df.empty:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                source_sent = (
                    news_df.groupby("source")["compound"]
                    .mean()
                    .reset_index()
                    .sort_values("compound", ascending=True)
                )
                fig_src = go.Figure(
                    go.Bar(
                        x=source_sent["compound"],
                        y=source_sent["source"],
                        orientation="h",
                        marker=dict(
                            color=[
                                "#10b981" if v >= 0.05 else "#ef4444" if v <= -0.05 else "#94a3b8"
                                for v in source_sent["compound"]
                            ],
                        ),
                        text=[f"{v:+.3f}" for v in source_sent["compound"]],
                        textposition="outside",
                    )
                )
                fig_src.update_layout(**PLOTLY_LAYOUT, height=230, title="Avg Sentiment by Source")
                st.plotly_chart(fig_src, width="stretch", config={"displayModeBar": False})

            with col_s2:
                bull = (news_df["label"] == "Bullish").sum()
                bear = (news_df["label"] == "Bearish").sum()
                neut = (news_df["label"] == "Neutral").sum()
                fig_pie = go.Figure(
                    go.Pie(
                        labels=["Bullish", "Bearish", "Neutral"],
                        values=[bull, bear, neut],
                        hole=0.6,
                        marker=dict(colors=["#10b981", "#ef4444", "#475569"]),
                    )
                )
                fig_pie.update_layout(**PLOTLY_LAYOUT, height=230, title="Overall Mood Mix")
                st.plotly_chart(fig_pie, width="stretch", config={"displayModeBar": False})

    with tab_heatmap:
        st.plotly_chart(make_heatmap(ticker_quotes), width="stretch", config={"displayModeBar": False})

    with tab_keywords:
        st.plotly_chart(make_keywords_chart(news_df), width="stretch", config={"displayModeBar": False})

with col_right:
    # Right Analytics Panel
    compound_avg = float(news_df["compound"].mean()) if not news_df.empty else 0.0
    st.plotly_chart(make_gauge(compound_avg), width="stretch", config={"displayModeBar": False})

    if not news_df.empty:
        total = len(news_df)
        bull = (news_df["label"] == "Bullish").sum()
        bear = (news_df["label"] == "Bearish").sum()
        bull_pct = bull / total * 100
        bear_pct = bear / total * 100

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Total Stories", total)
            st.metric("Bullish Ratio", f"{bull_pct:.0f}%")
        with m2:
            st.metric("Bearish Ratio", f"{bear_pct:.0f}%")
            st.metric("Neutral Stories", f"{total - bull - bear}")