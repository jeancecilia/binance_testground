"""Strategy pipeline: evaluates strategies, enforces idempotency,
and records all evaluations (signals and rejections).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from playground.domain.market import MarketContext
from playground.domain.signals import (
    SignalId, SignalRejection, SignalRejectionReason, StrategySignal,
)
from playground.core.specialist_registry import Strategy, StrategyRegistry
from playground.infrastructure.sqlite_repository import SQLiteRepository

logger = logging.getLogger(__name__)


class StrategyPipeline:
    """Evaluates registered strategies against market contexts.

    Enforces signal idempotency and records every evaluation.
    """

    def __init__(
        self,
        registry: StrategyRegistry,
        repository: SQLiteRepository,
    ) -> None:
        self._registry = registry
        self._repo = repository

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self, context: MarketContext,
    ) -> List[Tuple[Strategy, StrategySignal | SignalRejection]]:
        """Evaluate all strategies against a market context.

        Returns list of (strategy, result) pairs.
        Each result is either a StrategySignal or SignalRejection.
        Duplicate signals are replaced with rejections so callers
        never receive an executable signal that already exists.
        """
        raw_results = self._registry.evaluate_all(context)
        clean_results: List[Tuple[Strategy, StrategySignal | SignalRejection]] = []

        for strategy, result in raw_results:
            cleaned = self._record_evaluation(strategy, context, result)
            clean_results.append((strategy, cleaned))

        return clean_results

    def evaluate_single(
        self, strategy_id: str, context: MarketContext,
    ) -> Optional[StrategySignal | SignalRejection]:
        """Evaluate a single strategy by ID."""
        key = None
        for k, s in self._registry.strategies.items():
            if s.meta.strategy_id == strategy_id:
                key = k
                strategy = s
                break

        if key is None:
            logger.warning(
                "Strategy not found", extra={"strategy_id": strategy_id}
            )
            return None

        result = strategy.evaluate(context)
        self._record_evaluation(strategy, context, result)
        return result

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _record_evaluation(
        self, strategy: Strategy, context: MarketContext,
        result: StrategySignal | SignalRejection,
    ) -> StrategySignal | SignalRejection:
        """Record every strategy evaluation in the database.

        Returns the result after idempotency checking. If the signal already
        exists in the database, a SignalRejection is returned instead.
        """
        meta = strategy.meta
        is_signal = isinstance(result, StrategySignal)

        if is_signal:
            signal: StrategySignal = result
            # Check idempotency
            signal_id_str = str(signal.signal_id)
            if self._repo.signal_exists(signal_id_str):
                logger.info(
                    "Duplicate signal skipped (idempotency)", extra={
                        "signal_id": signal_id_str,
                    }
                )
                # Record as a rejection for audit trail
                dup_rejection = SignalRejection(
                    strategy_id=meta.strategy_id,
                    strategy_version=meta.strategy_version,
                    symbol=str(context.symbol),
                    timeframe=str(context.timeframe),
                    candle_timestamp=context.candle.open_time,
                    direction=meta.direction,
                    reason=SignalRejectionReason.DUPLICATE_SIGNAL,
                    detail=f"Signal {signal_id_str} already exists",
                )
                self._repo.insert_strategy_evaluation(
                    strategy_id=meta.strategy_id,
                    strategy_version=meta.strategy_version,
                    symbol=str(context.symbol),
                    timeframe=str(context.timeframe),
                    candle_timestamp=context.candle.open_time,
                    direction=meta.direction.value,
                    regime=context.regime,
                    result_type="rejection",
                    score=None,
                    signal_id=signal_id_str,
                    rejection_reason=SignalRejectionReason.DUPLICATE_SIGNAL.value,
                    detail=dup_rejection.detail,
                )
                return dup_rejection

            # New signal — persist it
            self._repo.insert_signal(signal)
            self._repo.insert_strategy_evaluation(
                strategy_id=meta.strategy_id,
                strategy_version=meta.strategy_version,
                symbol=str(context.symbol),
                timeframe=str(context.timeframe),
                candle_timestamp=context.candle.open_time,
                direction=meta.direction.value,
                regime=context.regime,
                result_type="signal",
                score=signal.score,
                signal_id=signal_id_str,
                rejection_reason=None,
                detail=f"Score: {signal.score}",
            )

            logger.info(
                "Signal generated", extra={
                    "signal_id": signal_id_str,
                    "strategy_id": meta.strategy_id,
                    "score": signal.score,
                }
            )
            return signal
        else:
            rejection: SignalRejection = result
            self._repo.insert_signal_rejection(rejection)
            self._repo.insert_strategy_evaluation(
                strategy_id=meta.strategy_id,
                strategy_version=meta.strategy_version,
                symbol=str(context.symbol),
                timeframe=str(context.timeframe),
                candle_timestamp=context.candle.open_time,
                direction=meta.direction.value,
                regime=context.regime,
                result_type="rejection",
                score=None,
                signal_id=None,
                rejection_reason=rejection.reason.value,
                detail=rejection.detail,
            )

            logger.debug(
                "Signal rejected", extra={
                    "strategy_id": meta.strategy_id,
                    "reason": rejection.reason.value,
                    "detail": rejection.detail,
                }
            )
            return rejection
