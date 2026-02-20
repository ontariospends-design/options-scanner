"""
Dark pool & institutional flow module.

Data sources (all free, no API key):
  1. yfinance — short interest, institutional ownership, days-to-cover
  2. Options chain large blocks (from analyzer.py is_large_block flag)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
from dataclasses import dataclass


# ── Known ATS / dark pool operator MPID → human name ─────────────────────────
ATS_NAMES = {
    "CDRG": "Citadel Connect",
    "NITE": "KCG / Virtu",
    "VRTU": "Virtu BEX",
    "JNST": "Jane Street ATS",
    "IATS": "Interactive Brokers ATS",
    "UBSA": "UBS ATS",
    "MSPL": "Morgan Stanley ATS",
    "JPBX": "JP Morgan ATS",
    "MLIX": "Merrill Lynch ATS",
    "GSCO": "Goldman Sachs ATS",
    "CODA": "Coda ATS",
    "BIDS": "BIDS Trading",
    "LATS": "Liquidnet ATS",
    "POSIT": "ITG POSIT",
}


@dataclass
class InstitutionalSnapshot:
    ticker: str
    short_pct_float: float        # short interest as % of float
    short_ratio: float            # days to cover
    shares_short: int
    short_change_pct: float       # month-over-month change in short interest
    institutional_pct: float      # % held by institutions
    insider_pct: float
    float_shares: int
    smart_money_bias: str         # BEARISH / BULLISH / NEUTRAL
    signal: str                   # human label


def fetch_institutional_flow(tickers: list, delay: float = 0.15) -> pd.DataFrame:
    """
    Fetch short interest and institutional ownership for a list of tickers.
    ETFs (SPY, QQQ, etc.) don't have short interest — they'll have partial data.
    Returns DataFrame sorted by short_pct_float descending.
    """
    rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            shares_short     = int(info.get("sharesShort") or 0)
            shares_short_pm  = int(info.get("sharesShortPriorMonth") or 0)
            float_shares     = int(info.get("floatShares") or 0)
            short_pct        = float(info.get("shortPercentOfFloat") or 0)
            short_ratio      = float(info.get("shortRatio") or 0)
            inst_pct         = float(info.get("heldPercentInstitutions") or 0)
            insider_pct      = float(info.get("heldPercentInsiders") or 0)

            short_change_pct = 0.0
            if shares_short_pm > 0:
                short_change_pct = (shares_short - shares_short_pm) / shares_short_pm * 100

            # Smart money bias heuristic
            if short_pct > 0.05 and short_change_pct > 5:
                bias   = "BEARISH"
                signal = "🔴 Short interest rising"
            elif short_pct < 0.02 and inst_pct > 0.7:
                bias   = "BULLISH"
                signal = "🟢 High inst. ownership, low short"
            elif short_change_pct < -10:
                bias   = "BULLISH"
                signal = "🟢 Short covering (squeeze risk)"
            elif short_pct > 0.10:
                bias   = "BEARISH"
                signal = "🔴 High short interest"
            else:
                bias   = "NEUTRAL"
                signal = "⚪ No strong signal"

            rows.append({
                "ticker":            ticker,
                "short_pct_float":   round(short_pct * 100, 2),
                "short_ratio":       round(short_ratio, 1),
                "shares_short":      shares_short,
                "short_change_pct":  round(short_change_pct, 1),
                "institutional_pct": round(inst_pct * 100, 1),
                "insider_pct":       round(insider_pct * 100, 2),
                "float_shares":      float_shares,
                "smart_money_bias":  bias,
                "signal":            signal,
            })
        except Exception:
            rows.append({
                "ticker":            ticker,
                "short_pct_float":   None,
                "short_ratio":       None,
                "shares_short":      None,
                "short_change_pct":  None,
                "institutional_pct": None,
                "insider_pct":       None,
                "float_shares":      None,
                "smart_money_bias":  "UNKNOWN",
                "signal":            "⚠️ No data",
            })
        time.sleep(delay)

    df = pd.DataFrame(rows)
    return df.sort_values("short_pct_float", ascending=False, na_position="last").reset_index(drop=True)


def large_block_summary(analyzed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate large block options activity per underlying.
    Returns: ticker, call_block_vol, put_block_vol, block_pc_ratio,
             top_strike_call, top_strike_put, notional_est, block_bias.
    """
    if analyzed_df.empty or "is_large_block" not in analyzed_df.columns:
        return pd.DataFrame()

    blocks = analyzed_df[analyzed_df["is_large_block"] == True].copy()
    if blocks.empty:
        return pd.DataFrame()

    rows = []
    for ticker, grp in blocks.groupby("underlying"):
        calls = grp[grp["option_type"] == "call"]
        puts  = grp[grp["option_type"] == "put"]

        c_vol = int(calls["volume"].sum())
        p_vol = int(puts["volume"].sum())

        # Biggest single-strike block
        top_call_row = calls.loc[calls["volume"].idxmax()] if not calls.empty else None
        top_put_row  = puts.loc[puts["volume"].idxmax()]  if not puts.empty else None

        top_call_strike = float(top_call_row["strike"]) if top_call_row is not None else None
        top_put_strike  = float(top_put_row["strike"])  if top_put_row  is not None else None

        # Notional estimate: volume × mid × 100 shares
        notional = float((grp["volume"].fillna(0) * grp.get("mid", pd.Series(0, index=grp.index)).fillna(0) * 100).sum())

        pc = round(p_vol / c_vol, 3) if c_vol > 0 else None
        bias = (
            "🔴 PUT pressure" if pc and pc > 1.3
            else "🟢 CALL pressure" if pc and pc < 0.7
            else "⚪ Mixed"
        )

        rows.append({
            "ticker":           ticker,
            "call_block_vol":   c_vol,
            "put_block_vol":    p_vol,
            "total_block_vol":  c_vol + p_vol,
            "block_pc_ratio":   pc,
            "top_call_strike":  top_call_strike,
            "top_put_strike":   top_put_strike,
            "notional_est":     round(notional),
            "block_bias":       bias,
        })

    df = pd.DataFrame(rows)
    return df.sort_values("total_block_vol", ascending=False).reset_index(drop=True)


def combined_score(inst_df: pd.DataFrame, block_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge institutional snapshot with options block data into one ranked table.
    """
    if inst_df.empty and block_df.empty:
        return pd.DataFrame()

    merged = inst_df.copy() if not inst_df.empty else pd.DataFrame()

    if not block_df.empty:
        block_df = block_df.rename(columns={"ticker": "ticker"})
        if merged.empty:
            merged = block_df
        else:
            merged = merged.merge(block_df, on="ticker", how="outer")

    return merged
