"""
Rules-based strategy engine.
Screens unusual-volume contracts for edge (BS underpricing + signal score),
sizes positions, and tracks a paper-trading ledger.
"""

from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import List, Optional
import pandas as pd

from pnl import bs_price, time_to_expiry, breakeven_prices


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class TradeCandidate:
    underlying: str
    option_type: str
    strike: float
    expiration_date: str
    signal_score: int
    vol_oi_ratio: float
    vol_zscore: float
    volume: int
    open_interest: int
    market_mid: float
    mid_iv: float
    bs_price: float
    edge_pct: float        # (bs - market) / market × 100  — positive = underpriced
    breakeven: float
    contracts: int
    estimated_cost: float
    rationale: str


@dataclass
class PaperTrade:
    id: int
    timestamp: str
    underlying: str
    option_type: str
    strike: float
    expiration_date: str
    contracts: int
    entry_price: float      # mid at entry
    total_cost: float
    signal_score: int
    rationale: str
    status: str = "OPEN"    # OPEN | CLOSED
    exit_price: float = 0.0
    pnl: float = 0.0


# ── Strategy engine ────────────────────────────────────────────────────────────

class StrategyEngine:
    """
    Screens unusual-volume contracts and generates trade candidates.

    Edge criteria (any one qualifies, score-3 overrides edge filter):
      • BS theoretical price > market mid by min_edge_pct
      • Signal score >= min_score

    Position sizing:
      • Risk per trade = capital × risk_per_trade_pct
      • Contracts = floor(risk_budget / (mid × 100)), min 1
    """

    def __init__(
        self,
        capital: float,
        risk_per_trade_pct: float = 0.03,
        min_score: int = 2,
        min_edge_pct: float = 5.0,   # percent
        max_iv: float = 2.0,         # skip insane IV (>200%)
        min_volume: int = 200,
        risk_free_rate: float = 0.045,
    ):
        self.capital = capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.min_score = min_score
        self.min_edge_pct = min_edge_pct
        self.max_iv = max_iv
        self.min_volume = min_volume
        self.r = risk_free_rate

    def screen(self, unusual_df: pd.DataFrame, spot_prices: dict) -> List[TradeCandidate]:
        """
        Filter and rank trade candidates from unusual contracts DataFrame.
        spot_prices: {ticker: float} — current underlying price.
        """
        if unusual_df.empty:
            return []

        candidates = []
        for _, row in unusual_df.iterrows():
            ticker    = str(row.get("underlying", ""))
            opt_type  = str(row.get("option_type", "call")).lower()
            K         = float(row.get("strike") or 0)
            iv        = float(row.get("mid_iv") or 0)
            mid       = float(row.get("mid") or 0)
            expiry    = str(row.get("expiration_date", ""))
            volume    = int(row.get("volume") or 0)
            oi        = int(row.get("open_interest") or 0)
            score     = int(row.get("signal_score") or 0)
            vol_oi    = float(row.get("vol_oi_ratio") or 0)
            zscore    = float(row.get("vol_zscore") or 0)

            S = spot_prices.get(ticker)
            if not S or S <= 0:
                continue
            if score < self.min_score:
                continue
            if iv <= 0 or iv > self.max_iv:
                continue
            if mid <= 0:
                continue
            if volume < self.min_volume:
                continue

            T = time_to_expiry(expiry)
            if T <= 0:
                continue

            theory = bs_price(S, K, T, self.r, iv, opt_type)
            edge_pct = (theory - mid) / mid * 100 if mid > 0 else 0.0

            # Must have edge OR be a score-3 signal
            if edge_pct < self.min_edge_pct and score < 3:
                continue

            # Position sizing
            risk_budget = self.capital * self.risk_per_trade_pct
            contracts   = max(1, int(risk_budget / (mid * 100)))
            cost        = round(contracts * mid * 100, 2)

            be = breakeven_prices(K, mid, opt_type)[0]

            # Human-readable rationale
            flags = []
            if row.get("flag_vol_oi"):  flags.append(f"Vol/OI {vol_oi:.1f}x")
            if row.get("flag_zscore"):  flags.append(f"Z {zscore:.1f}σ")
            if row.get("flag_block"):   flags.append("Large block")
            if edge_pct > 0:            flags.append(f"BS edge +{edge_pct:.0f}%")
            rationale = " · ".join(flags) if flags else "Signal score"

            candidates.append(TradeCandidate(
                underlying=ticker,
                option_type=opt_type,
                strike=K,
                expiration_date=expiry,
                signal_score=score,
                vol_oi_ratio=round(vol_oi, 3),
                vol_zscore=round(zscore, 3),
                volume=volume,
                open_interest=oi,
                market_mid=round(mid, 4),
                mid_iv=round(iv, 4),
                bs_price=round(theory, 4),
                edge_pct=round(edge_pct, 2),
                breakeven=be,
                contracts=contracts,
                estimated_cost=cost,
                rationale=rationale,
            ))

        candidates.sort(key=lambda c: (c.signal_score, c.edge_pct), reverse=True)
        return candidates

    def to_dataframe(self, candidates: List[TradeCandidate]) -> pd.DataFrame:
        if not candidates:
            return pd.DataFrame()
        return pd.DataFrame([asdict(c) for c in candidates])


# ── Paper trading ledger ───────────────────────────────────────────────────────

class PaperLedger:
    """In-memory paper trade tracker (persisted via Streamlit session_state)."""

    def __init__(self, trades: Optional[List[dict]] = None):
        self._trades: List[PaperTrade] = []
        if trades:
            for t in trades:
                self._trades.append(PaperTrade(**t))

    def enter(self, candidate: TradeCandidate) -> PaperTrade:
        trade = PaperTrade(
            id=len(self._trades) + 1,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            underlying=candidate.underlying,
            option_type=candidate.option_type,
            strike=candidate.strike,
            expiration_date=candidate.expiration_date,
            contracts=candidate.contracts,
            entry_price=candidate.market_mid,
            total_cost=candidate.estimated_cost,
            signal_score=candidate.signal_score,
            rationale=candidate.rationale,
        )
        self._trades.append(trade)
        return trade

    def close(self, trade_id: int, exit_price: float):
        for t in self._trades:
            if t.id == trade_id and t.status == "OPEN":
                t.exit_price = exit_price
                t.pnl = round((exit_price - t.entry_price) * t.contracts * 100, 2)
                t.status = "CLOSED"
                break

    def to_dataframe(self) -> pd.DataFrame:
        if not self._trades:
            return pd.DataFrame()
        rows = [asdict(t) for t in self._trades]
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        df = self.to_dataframe()
        if df.empty:
            return {"total_trades": 0, "open": 0, "closed": 0,
                    "realized_pnl": 0.0, "total_cost": 0.0, "win_rate": 0.0}
        closed = df[df["status"] == "CLOSED"]
        wins   = closed[closed["pnl"] > 0]
        return {
            "total_trades": len(df),
            "open":  int((df["status"] == "OPEN").sum()),
            "closed": len(closed),
            "realized_pnl": round(closed["pnl"].sum(), 2),
            "total_cost":   round(df["total_cost"].sum(), 2),
            "win_rate":     round(len(wins) / len(closed) * 100, 1) if len(closed) > 0 else 0.0,
        }

    def to_serializable(self) -> List[dict]:
        return [asdict(t) for t in self._trades]
