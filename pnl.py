"""
Black-Scholes P&L calculator for options contracts.
No external dependencies beyond math/numpy.
"""

from math import log, sqrt, exp, erf, pi as _pi
import numpy as np
import pandas as pd
from datetime import date


# ── Math helpers ───────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * _pi)


def time_to_expiry(expiry_str: str) -> float:
    """Fraction of a year from today to expiry_str (YYYY-MM-DD). Min 0."""
    try:
        exp_date = date.fromisoformat(expiry_str)
        return max((exp_date - date.today()).days / 365.0, 0.0)
    except Exception:
        return 0.0


# ── Core Black-Scholes ─────────────────────────────────────────────────────────

def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Theoretical B-S option price. Returns intrinsic value when T≤0 or sigma≤0."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    try:
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        if option_type == "call":
            return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
        else:
            return K * exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    except Exception:
        return 0.0


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> dict:
    """Return delta, gamma, theta (per day), vega (per 1% IV move)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    try:
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        pdf_d1 = _norm_pdf(d1)
        delta = _norm_cdf(d1) if option_type == "call" else _norm_cdf(d1) - 1.0
        gamma = pdf_d1 / (S * sigma * sqrt(T))
        if option_type == "call":
            theta = ((-S * pdf_d1 * sigma / (2.0 * sqrt(T))) - r * K * exp(-r * T) * _norm_cdf(d2)) / 365.0
        else:
            theta = ((-S * pdf_d1 * sigma / (2.0 * sqrt(T))) + r * K * exp(-r * T) * _norm_cdf(-d2)) / 365.0
        vega = S * pdf_d1 * sqrt(T) / 100.0
        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}
    except Exception:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}


# ── P&L profile ────────────────────────────────────────────────────────────────

def pnl_at_expiry(K: float, premium: float, option_type: str, S_range) -> pd.DataFrame:
    """P&L per 1-contract lot (×100 shares) across underlying prices at expiry."""
    rows = []
    for S in S_range:
        payoff = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        rows.append({"underlying_price": round(S, 2), "pnl": round((payoff - premium) * 100, 2)})
    return pd.DataFrame(rows)


def breakeven_prices(K: float, premium: float, option_type: str) -> list:
    if option_type == "call":
        return [round(K + premium, 2)]
    else:
        return [round(K - premium, 2)]


def pnl_summary(S: float, K: float, T: float, r: float, sigma: float,
                option_type: str, market_mid: float) -> dict:
    """Full P&L summary dict for one contract."""
    theory = bs_price(S, K, T, r, sigma, option_type)
    greeks = bs_greeks(S, K, T, r, sigma, option_type)
    be = breakeven_prices(K, market_mid, option_type)[0]
    edge_pct = ((theory - market_mid) / market_mid * 100) if market_mid > 0 else 0.0

    # P&L curve: ±40% range around current price
    lo, hi = S * 0.60, S * 1.40
    S_range = np.linspace(lo, hi, 120)
    curve_df = pnl_at_expiry(K, market_mid, option_type, S_range)

    return {
        "bs_price": round(theory, 4),
        "market_mid": round(market_mid, 4),
        "edge_pct": round(edge_pct, 2),
        "breakeven": be,
        "max_loss_per_contract": round(-market_mid * 100, 2),
        "greeks": greeks,
        "curve_df": curve_df,
    }
