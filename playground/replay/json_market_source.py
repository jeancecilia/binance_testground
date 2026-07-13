"""JSON market source for replay mode.

Reads historical candles from a local JSON file and presents them
as a market data source compatible with the same interface.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Iterator, Optional, Sequence

from playground.domain.market import Candle, CandleSource, Symbol, Timeframe


class JsonMarketSource:
    """Reads historical candles from a JSON file for replay.

    Expected JSON format:
    [
        {
            "symbol": "BNB/USDT",
            "timeframe": "1h",
            "open_time": "2026-07-01T00:00:00Z",
            "open": 300.0,
            "high": 305.0,
            "low": 298.0,
            "close": 302.0,
            "volume": 1234.56,
            "close_time": "2026-07-01T01:00:00Z",
            "quote_asset_volume": 370000.0,
            "number_of_trades": 500,
            "taker_buy_base_volume": 600.0,
            "taker_buy_quote_volume": 180000.0
        },
        ...
    ]
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._candles: list[Candle] = []
        self._index = 0
        self._dataset_identifier = ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all candles from the JSON file."""
        with open(self.file_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            self._dataset_identifier = data.get("dataset_identifier", "")
            candles_data = data.get("candles", data.get("data", []))
        else:
            candles_data = data

        self._candles = [
            self._parse_candle(c) for c in candles_data
        ]
        self._candles.sort(key=lambda c: c.open_time)
        self._index = 0

    @property
    def dataset_identifier(self) -> str:
        return self._dataset_identifier

    @property
    def candle_count(self) -> int:
        return len(self._candles)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def has_next(self) -> bool:
        return self._index < len(self._candles)

    def next_candle(self) -> Optional[Candle]:
        """Return the next chronological candle and advance."""
        if not self.has_next():
            return None
        candle = self._candles[self._index]
        self._index += 1
        return candle

    def peek_candle(self) -> Optional[Candle]:
        """Return the next candle without advancing."""
        if not self.has_next():
            return None
        return self._candles[self._index]

    def reset(self) -> None:
        """Reset the iterator to the beginning."""
        self._index = 0

    # ------------------------------------------------------------------
    # Bulk access
    # ------------------------------------------------------------------

    def get_candles_up_to(self, timestamp: datetime) -> list[Candle]:
        """Get all candles with open_time <= timestamp."""
        return [c for c in self._candles if c.open_time <= timestamp]

    def get_all_candles(self) -> list[Candle]:
        return list(self._candles)

    def get_candles_for_symbol_timeframe(
        self, symbol: str, timeframe: str,
    ) -> list[Candle]:
        """Filter candles by symbol and timeframe."""
        sym = str(Symbol(symbol))
        return [
            c for c in self._candles
            if str(c.symbol) == sym and str(c.timeframe) == timeframe
        ]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_candle(data: dict) -> Candle:
        """Parse a JSON dict into a Candle domain model."""
        # Handle Binance-style array format
        if isinstance(data, list):
            return Candle(
                symbol=Symbol(data.get("symbol", "BNB/USDT")),
                timeframe=Timeframe(data.get("timeframe", "1h")),
                open_time=datetime.utcfromtimestamp(data[0] / 1000.0),
                open=float(data[1]),
                high=float(data[2]),
                low=float(data[3]),
                close=float(data[4]),
                volume=float(data[5]),
                close_time=datetime.utcfromtimestamp(data[6] / 1000.0),
                quote_asset_volume=float(data[7]),
                number_of_trades=int(data[8]),
                taker_buy_base_volume=float(data[9]),
                taker_buy_quote_volume=float(data[10]),
                source=CandleSource.REPLAY,
            )

        # Handle dict format
        return Candle(
            symbol=Symbol(data["symbol"]),
            timeframe=Timeframe(data.get("timeframe", "1h")),
            open_time=_parse_dt(data["open_time"]),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"]),
            close_time=_parse_dt(data["close_time"]),
            quote_asset_volume=float(data.get("quote_asset_volume", 0)),
            number_of_trades=int(data.get("number_of_trades", 0)),
            taker_buy_base_volume=float(data.get("taker_buy_base_volume", 0)),
            taker_buy_quote_volume=float(data.get("taker_buy_quote_volume", 0)),
            source=CandleSource.REPLAY,
        )


def _parse_dt(value: str | int | float) -> datetime:
    """Parse a datetime from ISO string or Unix timestamp."""
    if isinstance(value, (int, float)):
        return datetime.utcfromtimestamp(value / 1000.0 if value > 1e12 else value)
    return datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))
