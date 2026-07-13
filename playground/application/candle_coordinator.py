"""Candle-close coordinator: processes newly completed candles.

Determines the latest completed candle, compares with stored checkpoints,
skips already-processed candles, and processes missed candles chronologically.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from playground.domain.market import Candle, Symbol, Timeframe
from playground.domain.positions import EngineCheckpoint
from playground.infrastructure.sqlite_repository import SQLiteRepository
from playground.infrastructure.system_clock import Clock, SystemClock


class CandleCoordinator:
    """Coordinates candle-close processing.

    Determines which candles need processing based on checkpoints
    and the current time. Handles recovery after interruption.
    """

    def __init__(
        self,
        repository: SQLiteRepository,
        clock: Clock | None = None,
    ) -> None:
        self._repo = repository
        self._clock = clock or SystemClock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_pending_candles(
        self, symbol: str, timeframe: str, mode: str,
    ) -> list[Candle]:
        """Get candles that have completed but haven't been processed yet.

        Args:
            symbol: Normalized symbol
            timeframe: Timeframe string
            mode: Runtime mode (replay, shadow, testnet)

        Returns:
            List of unprocessed candles in chronological order.
        """
        # Get the checkpoint
        checkpoint = self._repo.get_checkpoint(symbol, timeframe, mode)

        # Get the latest completed candle time
        latest_candle_time = self._repo.get_latest_candle_time(symbol, timeframe)
        if latest_candle_time is None:
            return []  # No candles at all

        # Determine start time for query
        if checkpoint is not None:
            # Start from the candle after the last processed one
            start_time = self._get_next_candle_time(
                checkpoint.last_processed_candle, timeframe
            )
        else:
            # No checkpoint — process all available completed candles
            start_time = None

        # Fetch unprocessed candles
        now = self._clock.now()
        all_candles = self._repo.get_candles(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
        )

        # Filter to only completed candles (close_time <= now)
        # This excludes the currently forming candle
        pending = [
            c for c in all_candles
            if c.close_time <= now
        ]

        return pending

    def has_pending(self, symbol: str, timeframe: str, mode: str) -> bool:
        """Check if there are pending candles to process."""
        pending = self.get_pending_candles(symbol, timeframe, mode)
        return len(pending) > 0

    def update_checkpoint(
        self, run_id: str, symbol: str, timeframe: str,
        mode: str, last_processed: datetime,
        engine_version: str = "0.1.0",
    ) -> None:
        """Update the engine checkpoint after successful processing."""
        checkpoint = EngineCheckpoint(
            run_id=run_id,
            symbol=symbol,
            timeframe=timeframe,
            last_processed_candle=last_processed,
            mode=mode,
            engine_version=engine_version,
        )
        self._repo.upsert_checkpoint(checkpoint)

    def get_checkpoint(
        self, symbol: str, timeframe: str, mode: str,
    ) -> Optional[EngineCheckpoint]:
        """Get the current checkpoint."""
        return self._repo.get_checkpoint(symbol, timeframe, mode)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_next_candle_time(
        current: datetime, timeframe: str,
    ) -> datetime:
        """Get the open time of the next candle after the given time."""
        from datetime import timedelta

        tf_minutes = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360,
            "8h": 480, "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080,
        }
        delta = timedelta(minutes=tf_minutes.get(timeframe, 60))
        return current + delta
