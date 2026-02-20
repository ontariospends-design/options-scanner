"""
Unusual volume / activity analyzer for options chains.

Signals:
  - vol_oi_ratio:   Volume / Open Interest  (>= threshold = unusual)
  - vol_zscore:     (Volume - mean_vol) / std_vol across all strikes same type/expiry
  - is_large_block: Volume in top N% of all contracts scanned
  - pc_vol_ratio:   Put Volume / Call Volume per underlying (bearish/bullish bias)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    vol_oi_ratio_threshold: float = 0.5
    zscore_threshold: float = 2.0
    large_block_percentile: float = 90.0
    min_volume: int = 100
    min_oi: int = 10


NUMERIC_COLS = [
    "volume", "open_interest", "last", "bid", "ask",
    "strike", "delta", "gamma", "theta", "vega", "iv", "mid_iv",
]


def clean_chain(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = df["volume"].fillna(0).astype(int)
    df["open_interest"] = df["open_interest"].fillna(0).astype(int)
    if "bid" in df.columns and "ask" in df.columns:
        df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2
    if "mid_iv" not in df.columns and "iv" in df.columns:
        df["mid_iv"] = df["iv"]
    return df


def compute_signals(df: pd.DataFrame, cfg: AnalysisConfig = AnalysisConfig()) -> pd.DataFrame:
    if df.empty:
        return df
    df = clean_chain(df.copy())

    # Vol / OI Ratio
    df["vol_oi_ratio"] = np.where(
        df["open_interest"] > 0,
        df["volume"] / df["open_interest"],
        np.nan,
    )

    # Volume Z-Score within same underlying + option_type group
    def _zscore_group(g):
        mu = g["volume"].mean()
        sigma = g["volume"].std(ddof=0)
        g["vol_zscore"] = 0.0 if (sigma == 0 or np.isnan(sigma)) else (g["volume"] - mu) / sigma
        return g

    group_cols = [c for c in ["underlying", "option_type"] if c in df.columns]
    if group_cols:
        df = df.groupby(group_cols, group_keys=False).apply(_zscore_group)
    else:
        df["vol_zscore"] = 0.0

    # Large block flag (top N%)
    vol_threshold = np.percentile(df["volume"].dropna(), cfg.large_block_percentile)
    df["is_large_block"] = df["volume"] >= vol_threshold

    # Signal flags
    df["flag_vol_oi"] = df["vol_oi_ratio"] >= cfg.vol_oi_ratio_threshold
    df["flag_zscore"] = df["vol_zscore"] >= cfg.zscore_threshold
    df["flag_block"] = df["is_large_block"]

    # Composite score 0–3
    df["signal_score"] = (
        df["flag_vol_oi"].astype(int)
        + df["flag_zscore"].astype(int)
        + df["flag_block"].astype(int)
    )

    df["is_unusual"] = (
        (df["flag_vol_oi"] | df["flag_zscore"] | df["flag_block"])
        & (df["volume"] >= cfg.min_volume)
        & (df["open_interest"] >= cfg.min_oi)
    )

    return df


def put_call_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "option_type" not in df.columns:
        return pd.DataFrame()

    def _agg(g):
        calls = g[g["option_type"] == "call"]
        puts = g[g["option_type"] == "put"]
        c_vol = calls["volume"].sum()
        p_vol = puts["volume"].sum()
        c_oi = calls["open_interest"].sum()
        p_oi = puts["open_interest"].sum()
        pc_vol = round(p_vol / c_vol, 3) if c_vol > 0 else np.nan
        pc_oi = round(p_oi / c_oi, 3) if c_oi > 0 else np.nan
        bias = "BEARISH" if pc_vol and pc_vol > 1.2 else ("BULLISH" if pc_vol and pc_vol < 0.8 else "NEUTRAL")
        return pd.Series({
            "call_volume": int(c_vol),
            "put_volume": int(p_vol),
            "total_volume": int(c_vol + p_vol),
            "pc_vol_ratio": pc_vol,
            "call_oi": int(c_oi),
            "put_oi": int(p_oi),
            "pc_oi_ratio": pc_oi,
            "bias": bias,
        })

    group_col = "underlying" if "underlying" in df.columns else df.index
    return df.groupby(group_col).apply(_agg).reset_index()


def top_unusual(df: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    if df.empty or "is_unusual" not in df.columns:
        return pd.DataFrame()
    unusual = df[df["is_unusual"]].copy()
    unusual = unusual.sort_values(["signal_score", "volume"], ascending=[False, False])
    display_cols = [c for c in [
        "underlying", "option_type", "strike", "expiration_date",
        "volume", "open_interest", "vol_oi_ratio", "vol_zscore",
        "mid", "mid_iv", "delta", "gamma",
        "flag_vol_oi", "flag_zscore", "flag_block", "signal_score",
        "description",
    ] if c in unusual.columns]
    return unusual[display_cols].head(n).reset_index(drop=True)
