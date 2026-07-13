"""Market pipeline: orchestrates candle ingestion, indicator calculation,
regime detection, and stability overlay for a single symbol/timeframe.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Sequence

from playground.domain.market import (
    Candle, IndicatorSnapshot, MarketContext, Symbol, Timeframe,
)
from playground.domain.indicators import IndicatorConfig, InsufficientHistoryRejection
from playground.domain.regimes import RegimeConfig, RegimeDecision, StabilityDecision
from playground.core.indicator_engine import IndicatorEngine, InsufficientHistoryError
from playground.core.regime_detector import RegimeDetector
from playground.core.stability_overlay import StabilityOverlay
from playground.infrastructure.binance_market_data import BinanceMarketDataAdapter
from playground.infrastructure.configuration import (
    AppConfig, MarketDataConfig,
)
from playground.infrastructure.sqlite_repository import SQLiteRepository
from playground.infrastructure.system_clock import Clock, SystemClock

logger = logging.getLogger(__name__)


class MarketPipeline:
    """Orchestrates market data flow: fetch → indicators → regime → stability."""

    def __init__(
        self,
        repository: SQLiteRepository,
        market_adapter: BinanceMarketDataAdapter,
        indicator_engine: IndicatorEngine,
        regime_detector: RegimeDetector,
        stability_overlay: StabilityOverlay,
        clock: Clock | None = None,
    ) -> None:
        self._repo = repository
        self._market = market_adapter
        self._indicators = indicator_engine
        self._regime = regime_detector
        self._stability = stability_overlay
        self._clock = clock or SystemClock()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_historical(
        self, symbol: str, timeframe: str,
    ) -> int:
        """Fetch and persist historical candles. Returns count inserted."""
        candles = self._market.fetch_historical_candles(symbol, timeframe)
        inserted = 0
        for candle in candles:
            if self._repo.insert_candle(candle):
                inserted += 1
        logger.info(
            "Historical ingestion complete", extra={
                "symbol": symbol, "timeframe": timeframe,
                "fetched": len(candles), "inserted": inserted,
            }
        )
        return inserted

    def ingest_missing(
        self, symbol: str, timeframe: str,
        start: datetime, end: datetime,
    ) -> int:
        """Detect and backfill missing candle intervals."""
        missing = self._repo.find_missing_intervals(symbol, timeframe, start, end)
        if not missing:
            return 0

        logger.info(
            "Backfilling missing candles", extra={
                "symbol": symbol, "timeframe": timeframe,
                "missing_count": len(missing),
                "first": missing[0].isoformat(),
                "last": missing[-1].isoformat(),
            }
        )

        inserted = 0
        for open_time in missing:
            candles = self._market.fetch_candles_range(
                symbol, timeframe, open_time, open_time,
            )
            for candle in candles:
                if self._repo.insert_candle(candle):
                    inserted += 1
        return inserted

    # ------------------------------------------------------------------
    # Processing pipeline for a single candle
    # ------------------------------------------------------------------

    def process_candle(
        self, symbol: str, timeframe: str, candle: Candle,
    ) -> Optional[MarketContext]:
        """Run the full pipeline for one candle: indicators → regime → stability.

        Returns MarketContext if successful, None if insufficient history.
        """
        sym = Symbol(symbol)
        tf = Timeframe(timeframe)

        # 1. Get candle history for indicator calculation
        candles = self._repo.get_candles(
            symbol, timeframe,
            end_time=candle.open_time,
        )

        # 2. Calculate indicators
        try:
            snapshot = self._indicators.create_snapshot(sym, tf, candles)
        except InsufficientHistoryError as e:
            logger.warning(
                "Insufficient history for indicators", extra={
                    "symbol": symbol, "timeframe": timeframe,
                    "candle_timestamp": candle.open_time.isoformat(),
                    "available": len(candles),
                }
            )
            return None

        # Persist indicator snapshot
        self._repo.insert_indicator_snapshot(snapshot)

        # 3. Raw regime detection
        raw_regime = self._regime.detect(symbol, timeframe, snapshot)
        self._repo.insert_regime_decision(raw_regime)

        # 4. Get recent regime decisions for stability overlay
        recent_decisions = self._get_recent_regime_decisions(symbol, timeframe)

        # 5. Stability overlay
        stability = self._stability.evaluate(raw_regime, recent_decisions)
        self._repo.insert_stability_decision(stability)

        # 6. Build MarketContext
        context = MarketContext(
            symbol=sym,
            timeframe=tf,
            candle=candle,
            indicators=snapshot,
            regime=stability.final_regime,
        )

        logger.debug(
            "Candle processed", extra={
                "symbol": symbol, "timeframe": timeframe,
                "candle_timestamp": candle.open_time.isoformat(),
                "raw_regime": raw_regime.regime,
                "final_regime": stability.final_regime,
                "confidence": stability.confidence_score,
            }
        )

        return context

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_recent_regime_decisions(
        self, symbol: str, timeframe: str, lookback: int = 20,
    ) -> list[RegimeDecision]:
        """Get recent raw regime decisions for stability calculation."""
        # We need to query directly since SQLiteRepository doesn't have
        # a direct method for this — but regime_decisions are in the DB
        conn = self._repo.conn
        rows = conn.execute(
            """SELECT * FROM regime_decisions
            WHERE symbol = ? AND timeframe = ?
            ORDER BY candle_timestamp DESC
            LIMIT ?""",
            (symbol, timeframe, lookback),
        ).fetchall()

        import json
        decisions = []
        for row in reversed(rows):  # chronological order
            decisions.append(RegimeDecision(
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                candle_timestamp=datetime.fromisoformat(row["candle_timestamp"]),
                regime=row["regime"],
                indicator_values=json.loads(row["indicator_values"]),
                applied_thresholds=json.loads(row["applied_thresholds"]),
                config_version=row["config_version"],
                decision_reason=row["decision_reason"],
                created_at=datetime.fromisoformat(row["created_at"]),
            ))
        return decisions
