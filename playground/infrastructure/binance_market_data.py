"""Binance public market-data adapter.

Fetches historical and newly completed OHLCV candles.
Normalizes symbols, retries with bounded backoff, and validates payloads.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from playground.domain.market import Candle, CandleSource, Symbol, Timeframe
from playground.infrastructure.configuration import BinanceConfig, MarketDataConfig


class BinanceMarketDataError(Exception):
    """Errors from the Binance market-data adapter."""
    pass


class BinanceRateLimitError(BinanceMarketDataError):
    """429 or rate-limit header detected."""
    pass


class BinanceInvalidPayloadError(BinanceMarketDataError):
    """Response payload failed validation."""
    pass


class BinanceMarketDataAdapter:
    """Fetches OHLCV candles from Binance public endpoints.

    Does NOT use API keys for public market data.
    """

    KLINE_ENDPOINT = "/api/v3/klines"
    MAX_LIMIT = 1000

    def __init__(
        self,
        binance_config: BinanceConfig | None = None,
        market_config: MarketDataConfig | None = None,
    ) -> None:
        self._binance = binance_config or BinanceConfig()
        self._market = market_config or MarketDataConfig()
        self._base_url = self._binance.public_endpoint.rstrip("/")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_historical_candles(
        self, symbol: str, timeframe: str, limit: int | None = None
    ) -> list[Candle]:
        """Fetch the most recent `limit` completed candles."""
        limit = limit or self._market.historical_candle_limit
        raw = self._get_klines(symbol, timeframe, limit=min(limit, self.MAX_LIMIT))
        return [self._parse_kline(k, symbol, timeframe, CandleSource.HISTORICAL) for k in raw]

    def fetch_candles_since(
        self, symbol: str, timeframe: str, start_time: datetime,
        limit: int | None = None,
    ) -> list[Candle]:
        """Fetch candles from start_time up to now."""
        limit = limit or self._market.historical_candle_limit
        raw = self._get_klines(
            symbol, timeframe,
            start_time_ms=int(start_time.timestamp() * 1000),
            limit=min(limit, self.MAX_LIMIT),
        )
        return [self._parse_kline(k, symbol, timeframe, CandleSource.LIVE) for k in raw]

    def fetch_candles_range(
        self, symbol: str, timeframe: str,
        start_time: datetime, end_time: datetime,
    ) -> list[Candle]:
        """Fetch candles in a specific time range."""
        raw = self._get_klines(
            symbol, timeframe,
            start_time_ms=int(start_time.timestamp() * 1000),
            end_time_ms=int(end_time.timestamp() * 1000),
            limit=self.MAX_LIMIT,
        )
        return [self._parse_kline(k, symbol, timeframe, CandleSource.BACKFILL) for k in raw]

    def fetch_latest_completed_candle(
        self, symbol: str, timeframe: str,
    ) -> Optional[Candle]:
        """Fetch the single most recently completed candle."""
        candles = self.fetch_historical_candles(symbol, timeframe, limit=2)
        if not candles:
            return None
        # The last candle returned might still be forming; take the one before
        for c in reversed(candles):
            if c.is_complete:
                return c
        return None

    # ------------------------------------------------------------------
    # Symbol normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Convert display format (BNB/USDT) to exchange format (BNBUSDT)."""
        return symbol.replace("/", "").upper()

    @staticmethod
    def denormalize_symbol(exchange_symbol: str) -> str:
        """Convert exchange format (BNBUSDT) to display format (BNB/USDT)."""
        # Common quote assets
        for quote in ["USDT", "USDC", "BUSD", "BTC", "ETH", "BNB", "TUSD", "DAI"]:
            if exchange_symbol.endswith(quote) and len(exchange_symbol) > len(quote):
                base = exchange_symbol[:-len(quote)]
                return f"{base}/{quote}"
        return exchange_symbol

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_klines(
        self, symbol: str, timeframe: str,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 500,
    ) -> list[list]:
        """Raw GET /api/v3/klines with retry and backoff."""
        params = {
            "symbol": self.normalize_symbol(symbol),
            "interval": timeframe,
            "limit": limit,
        }
        if start_time_ms:
            params["startTime"] = start_time_ms
        if end_time_ms:
            params["endTime"] = end_time_ms

        url = f"{self._base_url}{self.KLINE_ENDPOINT}?{urlencode(params)}"

        last_error: Optional[Exception] = None
        for attempt in range(self._market.max_retries + 1):
            try:
                req = Request(url, headers={"Accept": "application/json"})
                with urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())

                if not isinstance(data, list):
                    raise BinanceInvalidPayloadError(
                        f"Expected list, got {type(data).__name__}: {data}"
                    )

                self._validate_klines(data)
                return data

            except HTTPError as e:
                if e.code == 429:
                    last_error = BinanceRateLimitError(f"Rate limited on attempt {attempt + 1}")
                elif e.code >= 500:
                    last_error = BinanceMarketDataError(f"Server error {e.code}")
                else:
                    raise BinanceMarketDataError(f"HTTP {e.code}: {e.reason}") from e

            except URLError as e:
                last_error = BinanceMarketDataError(f"Network error: {e.reason}")

            except json.JSONDecodeError as e:
                last_error = BinanceInvalidPayloadError(f"Invalid JSON: {e}")

            if attempt < self._market.max_retries:
                backoff = min(
                    self._market.retry_backoff_base_seconds * (2 ** attempt),
                    self._market.retry_backoff_max_seconds,
                )
                time.sleep(backoff)

        raise last_error or BinanceMarketDataError("Max retries exceeded")

    def _validate_klines(self, data: list[list]) -> None:
        """Validate kline payload structure and ordering."""
        if not data:
            return

        prev_open_time: Optional[int] = None
        for i, k in enumerate(data):
            if not isinstance(k, list) or len(k) < 12:
                raise BinanceInvalidPayloadError(
                    f"Kline {i} is not a valid array: {k}"
                )

            open_time = k[0]
            if not isinstance(open_time, (int, float)):
                raise BinanceInvalidPayloadError(
                    f"Kline {i} open_time is not numeric: {open_time}"
                )

            if prev_open_time is not None and open_time <= prev_open_time:
                raise BinanceInvalidPayloadError(
                    f"Klines not in ascending order: {open_time} <= {prev_open_time}"
                )
            prev_open_time = int(open_time)

    def _parse_kline(
        self, k: list, symbol: str, timeframe: str, source: CandleSource,
    ) -> Candle:
        """Parse a raw Binance kline array into a Candle domain model."""
        return Candle(
            symbol=Symbol(symbol),
            timeframe=Timeframe(timeframe),
            open_time=datetime.utcfromtimestamp(k[0] / 1000.0),
            open=float(k[1]),
            high=float(k[2]),
            low=float(k[3]),
            close=float(k[4]),
            volume=float(k[5]),
            close_time=datetime.utcfromtimestamp(k[6] / 1000.0),
            quote_asset_volume=float(k[7]),
            number_of_trades=int(k[8]),
            taker_buy_base_volume=float(k[9]),
            taker_buy_quote_volume=float(k[10]),
            source=source,
            ingested_at=datetime.utcnow(),
        )


# ------------------------------------------------------------------
# Helper: detect newly completed candle
# ------------------------------------------------------------------

def is_candle_complete(candle: Candle, now: datetime | None = None) -> bool:
    """Check if a candle is complete (close_time is in the past)."""
    now = now or datetime.utcnow()
    return candle.close_time <= now


def get_next_candle_open(candle: Candle) -> datetime:
    """Compute the open_time of the next candle given timeframe."""
    tf_minutes = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360,
        "8h": 480, "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080,
    }
    delta = timedelta(minutes=tf_minutes.get(str(candle.timeframe), 60))
    return candle.open_time + delta
