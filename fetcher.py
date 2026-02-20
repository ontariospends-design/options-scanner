"""
yfinance-based data fetcher for options chains.
No API key required — uses Yahoo Finance data (15-min delayed).
"""

import yfinance as yf
import pandas as pd
import time

# Top 50 most actively traded options tickers for broad scan
BROAD_SCAN_TICKERS = [
    # Core ETFs
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "XLF", "XLE", "XLK",
    "XLV", "XLU", "XLI", "XLY", "XLC", "EEM", "EFA", "GDX", "USO", "ARKK",
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD", "PLTR", "NFLX",
    "AVGO", "ORCL", "CRM", "NOW", "SNOW", "CRWD", "PANW", "NET", "DDOG", "ZS",
    # Financials
    "BAC", "JPM", "GS", "MS", "C", "WFC", "V", "MA", "PYPL", "COIN",
    # Energy / Materials
    "XOM", "CVX", "OXY", "SLB", "HAL",
    # Consumer / Retail
    "WMT", "COST", "TGT", "NKE", "DIS",
    # Healthcare / Biotech
    "UNH", "LLY", "ABBV", "PFE", "MRNA",
    # Semis / Hardware
    "MU", "INTC", "TSM", "SMCI", "ARM",
    # Mobility / Transport
    "UBER", "LYFT", "GM", "F", "RIVN",
    # Crypto-adjacent / High-vol
    "MSTR", "MARA", "RIOT", "HUT", "CLSK",
    # Meme / Retail favorites
    "GME", "AMC", "BABA", "LCID", "NIO",
    # Fintech / Social
    "SOFI", "HOOD", "SQ", "SNAP", "RBLX",
    # Additional high-volume names
    "SHOP", "BA", "CAT", "ABNB", "DASH", "LULU", "ADBE", "SPOT", "MELI", "TTD",
]

VIX_TICKER = "^VIX"


class YFinanceClient:
    def get_option_expirations(self, symbol: str) -> list:
        """Return list of expiration date strings for a symbol."""
        try:
            return list(yf.Ticker(symbol).options)
        except Exception:
            return []

    def get_option_chain(self, symbol: str, expiration: str) -> pd.DataFrame:
        """Fetch full options chain for symbol/expiration. Returns DataFrame."""
        try:
            chain = yf.Ticker(symbol).option_chain(expiration)
        except Exception:
            return pd.DataFrame()

        calls = chain.calls.copy()
        calls["option_type"] = "call"
        puts = chain.puts.copy()
        puts["option_type"] = "put"

        df = pd.concat([calls, puts], ignore_index=True)
        if df.empty:
            return df

        # Normalize column names to match analyzer expectations
        df = df.rename(columns={
            "openInterest": "open_interest",
            "impliedVolatility": "mid_iv",
            "lastPrice": "last",
            "contractSymbol": "description",
        })

        df["underlying"] = symbol
        df["expiration_date"] = expiration

        if "bid" in df.columns and "ask" in df.columns:
            df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2

        return df

    def get_quotes(self, symbols: list) -> pd.DataFrame:
        """Fetch spot quotes for a list of symbols."""
        rows = []
        for sym in symbols:
            try:
                info = yf.Ticker(sym).fast_info
                rows.append({"symbol": sym, "last": info.last_price})
            except Exception:
                pass
        return pd.DataFrame(rows)

    def scan_tickers(self, tickers: list, expiration: str, delay: float = 0.1) -> pd.DataFrame:
        """Fetch chains for multiple tickers and concatenate."""
        frames = []
        for ticker in tickers:
            try:
                df = self.get_option_chain(ticker, expiration)
                if not df.empty:
                    frames.append(df)
            except Exception:
                pass
            time.sleep(delay)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
