"""
Options Unusual Volume Scanner — Streamlit Dashboard
Target: Monday Feb 23, 2026 expiry (configurable)

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from fetcher import YFinanceClient, BROAD_SCAN_TICKERS, VIX_TICKER
from analyzer import compute_signals, put_call_summary, top_unusual, AnalysisConfig

st.set_page_config(
    page_title="Options Unusual Volume Scanner",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Options Unusual Volume Scanner")
st.caption("Scan for unusual options activity on a target expiration date — powered by Yahoo Finance (yfinance, 15-min delayed)")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # ── CBOE 2026 expiration calendar ─────────────────────────────────────────
    today = date.today()

    # Standard monthly (3rd Friday), VIX Wednesday, Quarter-end
    CBOE_2026 = [
        ("Jan 16 — Monthly (3rd Fri)",   date(2026,  1, 16)),
        ("Feb 20 — Monthly (3rd Fri)",   date(2026,  2, 20)),
        ("Feb 18 — VIX Expiry (Wed)",    date(2026,  2, 18)),
        ("Mar 18 — VIX Expiry (Wed)",    date(2026,  3, 18)),
        ("Mar 20 — Monthly (3rd Fri)",   date(2026,  3, 20)),
        ("Mar 31 — Quarter-End",         date(2026,  3, 31)),
        ("Apr 15 — VIX Expiry (Wed)",    date(2026,  4, 15)),
        ("Apr 17 — Monthly (3rd Fri)",   date(2026,  4, 17)),
        ("May 15 — Monthly (3rd Fri)",   date(2026,  5, 15)),
        ("May 20 — VIX Expiry (Wed)",    date(2026,  5, 20)),
        ("Jun 17 — VIX Expiry (Wed)",    date(2026,  6, 17)),
        ("Jun 19 — Monthly (3rd Fri)",   date(2026,  6, 19)),
        ("Jun 30 — Quarter-End",         date(2026,  6, 30)),
        ("Jul 15 — VIX Expiry (Wed)",    date(2026,  7, 15)),
        ("Jul 17 — Monthly (3rd Fri)",   date(2026,  7, 17)),
        ("Aug 19 — VIX Expiry (Wed)",    date(2026,  8, 19)),
        ("Aug 21 — Monthly (3rd Fri)",   date(2026,  8, 21)),
        ("Sep 16 — VIX Expiry (Wed)",    date(2026,  9, 16)),
        ("Sep 18 — Monthly (3rd Fri)",   date(2026,  9, 18)),
        ("Sep 30 — Quarter-End",         date(2026,  9, 30)),
        ("Oct 16 — Monthly (3rd Fri)",   date(2026, 10, 16)),
        ("Oct 21 — VIX Expiry (Wed)",    date(2026, 10, 21)),
        ("Nov 18 — VIX Expiry (Wed)",    date(2026, 11, 18)),
        ("Nov 20 — Monthly (3rd Fri)",   date(2026, 11, 20)),
        ("Dec 16 — VIX Expiry (Wed)",    date(2026, 12, 16)),
        ("Dec 18 — Monthly (3rd Fri)",   date(2026, 12, 18)),
        ("Dec 31 — Quarter-End",         date(2026, 12, 31)),
    ]

    # Next standard Friday default
    days_to_friday = (4 - today.weekday()) % 7 or 7
    next_friday = today + timedelta(days=days_to_friday)

    upcoming = [(label, d) for label, d in sorted(CBOE_2026, key=lambda x: x[1]) if d >= today]
    cboe_labels = [label for label, _ in upcoming] + ["📅 Custom date…"]
    cboe_dates  = {label: d for label, d in upcoming}

    # Pre-select the nearest upcoming CBOE date
    default_label = next((label for label, d in upcoming if d >= today), cboe_labels[0])
    selected_label = st.selectbox(
        "Target Expiration Date",
        cboe_labels,
        index=cboe_labels.index(default_label),
        help="CBOE 2026 standard monthly (3rd Fri), VIX Wednesday, and quarter-end dates.",
    )

    if selected_label == "📅 Custom date…":
        target_expiry = st.date_input("Custom expiry", value=next_friday)
    else:
        target_expiry = cboe_dates[selected_label]
        st.caption(f"📆 {target_expiry.strftime('%A, %B %-d, %Y')}")

    expiry_str = target_expiry.strftime("%Y-%m-%d")

    st.divider()

    st.subheader("Tickers to Scan")
    scan_spy_qqq = st.checkbox("SPY / QQQ / IWM", value=True)
    scan_vix = st.checkbox("VIX Options", value=True)
    scan_broad = st.checkbox("Broad Scan (top 50)", value=False)
    custom_raw = st.text_input("Custom tickers (comma-separated)", placeholder="NVDA, TSLA")

    st.divider()

    st.subheader("Signal Thresholds")
    vol_oi_thresh = st.slider("Vol/OI Ratio threshold", 0.1, 5.0, 0.5, 0.1)
    zscore_thresh = st.slider("Volume Z-Score threshold", 1.0, 5.0, 2.0, 0.25)
    block_pct = st.slider("Large block percentile", 50, 99, 90, 1)
    min_vol = st.number_input("Min Volume", min_value=1, value=100, step=10)
    min_oi = st.number_input("Min Open Interest", min_value=0, value=10, step=10)

    run_scan = st.button("🔍 Run Scan", type="primary", use_container_width=True)

# ── Build ticker list ──────────────────────────────────────────────────────────
tickers = []
if scan_spy_qqq:
    tickers += ["SPY", "QQQ", "IWM"]
if scan_vix:
    tickers += [VIX_TICKER]
if scan_broad:
    tickers += [t for t in BROAD_SCAN_TICKERS if t not in tickers]
if custom_raw.strip():
    tickers += [t.strip().upper() for t in custom_raw.split(",") if t.strip() and t.strip().upper() not in tickers]
tickers = list(dict.fromkeys(tickers))

# ── Pre-scan info ──────────────────────────────────────────────────────────────
if not run_scan:
    st.info(
        f"**Target expiry:** `{expiry_str}`  |  "
        f"**Tickers queued:** {', '.join(tickers) if tickers else 'none selected'}\n\n"
        "Configure settings in the sidebar and click **Run Scan**.",
        icon="ℹ️",
    )
    with st.expander("ℹ️ How to get a free Tradier API key"):
        st.markdown("""
1. Go to **https://developer.tradier.com/**
2. Click **Get an API Token** → sign up free
3. Copy your **sandbox token** from the dashboard
4. Paste it in the sidebar with **Use Sandbox** ON
5. For live data, use a production token from a Tradier brokerage account
        """)
    with st.expander("📖 Signal definitions"):
        st.markdown("""
| Signal | Definition |
|---|---|
| **Vol/OI Ratio** | Volume ÷ Open Interest ≥ threshold. >0.5 = heavy same-day activity vs existing positioning. |
| **Volume Z-Score** | Std deviations above mean volume for that option type. ≥2σ = unusual. |
| **Large Block** | Volume ≥ Nth percentile of all scanned contracts. Top 10% by default. |
| **Signal Score** | 0–3 composite. Score of 3 = all three signals firing simultaneously. |
| **P/C Vol Ratio** | Put vol ÷ Call vol per underlying. >1.2 = bearish tilt, <0.8 = bullish. |
        """)
    st.stop()

# ── Validate ───────────────────────────────────────────────────────────────────
if not tickers:
    st.warning("No tickers selected.")
    st.stop()

# ── Fetch ──────────────────────────────────────────────────────────────────────
cfg = AnalysisConfig(
    vol_oi_ratio_threshold=vol_oi_thresh,
    zscore_threshold=zscore_thresh,
    large_block_percentile=float(block_pct),
    min_volume=int(min_vol),
    min_oi=int(min_oi),
)

client = YFinanceClient()
progress = st.progress(0, text="Fetching options chains…")
frames, errors = [], []

for i, ticker in enumerate(tickers):
    progress.progress((i + 1) / len(tickers), text=f"Fetching {ticker}…")
    try:
        df = client.get_option_chain(ticker, expiry_str)
        if not df.empty:
            frames.append(df)
        else:
            errors.append(f"{ticker}: no chain data for {expiry_str}")
    except Exception as e:
        errors.append(f"{ticker}: {e}")

progress.empty()

if errors:
    with st.expander(f"⚠️ {len(errors)} fetch warning(s)"):
        for e in errors:
            st.warning(e)

if not frames:
    st.error(
        "No data returned. Possible causes:\n"
        "- Expiration date doesn't exist for these tickers\n"
        "- Invalid API key\n"
        "- Tradier sandbox not returning data for this date"
    )
    st.stop()

raw_df = pd.concat(frames, ignore_index=True)

# ── Analyze ────────────────────────────────────────────────────────────────────
analyzed_df = compute_signals(raw_df, cfg)
pc_df = put_call_summary(analyzed_df)
unusual_df = top_unusual(analyzed_df, n=200)

# ── Summary metrics ────────────────────────────────────────────────────────────
total_contracts = len(analyzed_df)
total_unusual = int(analyzed_df.get("is_unusual", pd.Series(dtype=bool)).sum())
total_volume = int(analyzed_df["volume"].sum()) if "volume" in analyzed_df.columns else 0
score3 = int((analyzed_df.get("signal_score", pd.Series(0)) == 3).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Contracts Scanned", f"{total_contracts:,}")
c2.metric("Unusual Flags", f"{total_unusual:,}", f"{total_unusual/max(total_contracts,1)*100:.1f}% of total")
c3.metric("Total Volume", f"{total_volume:,}")
c4.metric("Max Signal (3/3)", f"{score3:,}", "🔴 ALERT" if score3 > 0 else "none")

st.divider()

# ── Put/Call Summary ───────────────────────────────────────────────────────────
st.subheader("📈 Put/Call Summary by Underlying")

if not pc_df.empty:
    def _bias_icon(b):
        return {"BEARISH": "🔴 BEARISH", "BULLISH": "🟢 BULLISH"}.get(b, "⚪ NEUTRAL")

    display_pc = pc_df.copy()
    if "bias" in display_pc.columns:
        display_pc["bias"] = display_pc["bias"].apply(_bias_icon)

    st.dataframe(
        display_pc,
        use_container_width=True,
        hide_index=True,
        column_config={
            "pc_vol_ratio": st.column_config.NumberColumn("P/C Vol Ratio", format="%.3f"),
            "pc_oi_ratio": st.column_config.NumberColumn("P/C OI Ratio", format="%.3f"),
            "call_volume": st.column_config.NumberColumn("Call Vol", format="%d"),
            "put_volume": st.column_config.NumberColumn("Put Vol", format="%d"),
            "total_volume": st.column_config.NumberColumn("Total Vol", format="%d"),
        },
    )

    if "underlying" in pc_df.columns and "pc_vol_ratio" in pc_df.columns:
        fig_pc = px.bar(
            pc_df.dropna(subset=["pc_vol_ratio"]),
            x="underlying", y="pc_vol_ratio",
            color="pc_vol_ratio",
            color_continuous_scale=["green", "white", "red"],
            color_continuous_midpoint=1.0,
            title=f"Put/Call Volume Ratio — {expiry_str} expiry",
            labels={"pc_vol_ratio": "P/C Ratio", "underlying": "Ticker"},
        )
        fig_pc.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Neutral (1.0)")
        fig_pc.add_hline(y=1.2, line_dash="dot", line_color="red", annotation_text="Bearish (1.2)")
        fig_pc.add_hline(y=0.8, line_dash="dot", line_color="green", annotation_text="Bullish (0.8)")
        st.plotly_chart(fig_pc, use_container_width=True)

st.divider()

# ── Unusual Contracts Table ────────────────────────────────────────────────────
st.subheader(f"🚨 Unusual Contracts — Expiring {expiry_str}")

if unusual_df.empty:
    st.info("No unusual contracts found. Try lowering the Vol/OI or Z-Score thresholds.")
else:
    min_score = st.radio("Minimum signal score:", [0, 1, 2, 3], horizontal=True, index=0)
    filtered = unusual_df[unusual_df["signal_score"] >= min_score] if "signal_score" in unusual_df.columns else unusual_df

    col_cfg = {}
    if "vol_oi_ratio" in filtered.columns:
        col_cfg["vol_oi_ratio"] = st.column_config.NumberColumn("Vol/OI", format="%.2f")
    if "vol_zscore" in filtered.columns:
        col_cfg["vol_zscore"] = st.column_config.NumberColumn("Z-Score", format="%.2f")
    if "mid_iv" in filtered.columns:
        col_cfg["mid_iv"] = st.column_config.NumberColumn("IV", format="%.1%")
    if "delta" in filtered.columns:
        col_cfg["delta"] = st.column_config.NumberColumn("Delta", format="%.3f")
    if "signal_score" in filtered.columns:
        col_cfg["signal_score"] = st.column_config.ProgressColumn("Score", min_value=0, max_value=3)

    st.dataframe(filtered, use_container_width=True, hide_index=True, column_config=col_cfg)
    st.caption(f"Showing {len(filtered)} unusual contracts (score ≥ {min_score})")

st.divider()

# ── Volume by Strike Heatmap ───────────────────────────────────────────────────
st.subheader("🗺️ Volume by Strike")
heatmap_ticker = st.selectbox("Select underlying:", tickers)
hm = analyzed_df[analyzed_df["underlying"] == heatmap_ticker].copy()

if not hm.empty and "strike" in hm.columns and "option_type" in hm.columns:
    pivot = hm.pivot_table(index="strike", columns="option_type", values="volume", aggfunc="sum", fill_value=0).reset_index()
    fig_hm = go.Figure()
    for col in [c for c in pivot.columns if c != "strike"]:
        fig_hm.add_trace(go.Bar(
            name=col.upper(), x=pivot["strike"], y=pivot[col],
            marker_color="red" if col == "put" else "green", opacity=0.75,
        ))
    fig_hm.update_layout(
        barmode="group",
        title=f"{heatmap_ticker} — Volume by Strike ({expiry_str})",
        xaxis_title="Strike", yaxis_title="Volume",
    )
    st.plotly_chart(fig_hm, use_container_width=True)
else:
    st.info("Not enough data for this ticker.")

# ── Vol/OI Scatter ─────────────────────────────────────────────────────────────
st.subheader("🔵 Vol/OI Ratio Scatter — All Contracts")
scatter = analyzed_df.dropna(subset=["vol_oi_ratio", "volume", "strike"]).copy()
if not scatter.empty:
    fig_sc = px.scatter(
        scatter, x="strike", y="vol_oi_ratio", size="volume",
        color="underlying", symbol="option_type",
        hover_data=["underlying", "strike", "option_type", "volume", "open_interest", "vol_oi_ratio"],
        title=f"Vol/OI Ratio by Strike — {expiry_str}",
        labels={"vol_oi_ratio": "Vol/OI Ratio", "strike": "Strike"},
        size_max=40,
    )
    fig_sc.add_hline(y=vol_oi_thresh, line_dash="dash", line_color="orange",
                     annotation_text=f"Threshold ({vol_oi_thresh})")
    st.plotly_chart(fig_sc, use_container_width=True)

# ── Raw data ───────────────────────────────────────────────────────────────────
with st.expander("🗃️ Raw Chain Data"):
    st.dataframe(analyzed_df, use_container_width=True)
    st.download_button(
        "⬇️ Download CSV",
        analyzed_df.to_csv(index=False),
        f"options_scan_{expiry_str}.csv",
        "text/csv",
    )
