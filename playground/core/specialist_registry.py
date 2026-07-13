"""Strategy interface and registry.

Each strategy must declare its ID, version, supported symbols/timeframes/regimes,
direction, required indicators, entry/exit conditions, and risk configuration.

Strategies receive a MarketContext and return either a StrategySignal or a
structured SignalRejection. Strategies must NOT access infrastructure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from playground.domain.market import MarketContext
from playground.domain.signals import (
    Direction, SignalId, SignalRejection, SignalRejectionReason,
    StrategySignal,
)


@dataclass(frozen=True)
class StrategyMeta:
    """Metadata that every strategy must declare."""

    strategy_id: str
    strategy_version: str
    supported_symbols: tuple[str, ...]
    supported_timeframes: tuple[str, ...]
    supported_regimes: tuple[str, ...]
    direction: Direction
    required_indicators: tuple[str, ...]
    min_score: float = 50.0  # Minimum confidence score for a valid signal


class Strategy(ABC):
    """Abstract strategy interface.

    Concrete strategies implement `evaluate()` which receives a MarketContext
    and returns either a StrategySignal or a structured SignalRejection.

    Strategies must NOT:
    - Call exchange APIs
    - Read from the database
    - Access environment variables
    - Submit orders
    - Read the system clock directly
    """

    @property
    @abstractmethod
    def meta(self) -> StrategyMeta:
        """Strategy metadata."""
        ...

    @abstractmethod
    def evaluate(self, context: MarketContext) -> StrategySignal | SignalRejection:
        """Evaluate the market context and produce a signal or rejection.

        Args:
            context: Prepared market context with candle, indicators, regime.

        Returns:
            StrategySignal if entry conditions are met, otherwise SignalRejection.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience methods for subclasses
    # ------------------------------------------------------------------

    def _check_prerequisites(
        self, context: MarketContext,
    ) -> Optional[SignalRejection]:
        """Validate that the strategy should even evaluate this context.

        Returns None if all checks pass, otherwise a SignalRejection.
        """
        m = self.meta

        if context.symbol not in m.supported_symbols:
            return SignalRejection(
                strategy_id=m.strategy_id,
                strategy_version=m.strategy_version,
                symbol=str(context.symbol),
                timeframe=str(context.timeframe),
                candle_timestamp=context.candle.open_time,
                direction=m.direction,
                reason=SignalRejectionReason.WRONG_SYMBOL,
                detail=f"Symbol {context.symbol} not in {m.supported_symbols}",
            )

        if str(context.timeframe) not in m.supported_timeframes:
            return SignalRejection(
                strategy_id=m.strategy_id,
                strategy_version=m.strategy_version,
                symbol=str(context.symbol),
                timeframe=str(context.timeframe),
                candle_timestamp=context.candle.open_time,
                direction=m.direction,
                reason=SignalRejectionReason.WRONG_TIMEFRAME,
                detail=f"Timeframe {context.timeframe} not in {m.supported_timeframes}",
            )

        if context.regime not in m.supported_regimes:
            return SignalRejection(
                strategy_id=m.strategy_id,
                strategy_version=m.strategy_version,
                symbol=str(context.symbol),
                timeframe=str(context.timeframe),
                candle_timestamp=context.candle.open_time,
                direction=m.direction,
                reason=SignalRejectionReason.WRONG_REGIME,
                detail=f"Regime '{context.regime}' not in {m.supported_regimes}",
            )

        return None

    def _make_signal_id(self, context: MarketContext) -> SignalId:
        """Create a deterministic signal ID for the current context."""
        m = self.meta
        return SignalId(
            strategy_id=m.strategy_id,
            strategy_version=m.strategy_version,
            symbol=str(context.symbol),
            timeframe=str(context.timeframe),
            candle_timestamp=context.candle.open_time,
            direction=m.direction,
        )

    def _make_signal(
        self, context: MarketContext, score: float,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> StrategySignal:
        """Create a StrategySignal with the deterministic ID."""
        sig_id = self._make_signal_id(context)
        return StrategySignal(
            signal_id=sig_id,
            strategy_id=self.meta.strategy_id,
            strategy_version=self.meta.strategy_version,
            symbol=str(context.symbol),
            timeframe=str(context.timeframe),
            candle_timestamp=context.candle.open_time,
            direction=self.meta.direction,
            regime=context.regime,
            score=score,
            entry_price=entry_price or context.candle.close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata or {},
        )

    def _reject(
        self, context: MarketContext, reason: SignalRejectionReason, detail: str = "",
    ) -> SignalRejection:
        """Create a structured rejection."""
        return SignalRejection(
            strategy_id=self.meta.strategy_id,
            strategy_version=self.meta.strategy_version,
            symbol=str(context.symbol),
            timeframe=str(context.timeframe),
            candle_timestamp=context.candle.open_time,
            direction=self.meta.direction,
            reason=reason,
            detail=detail,
        )


class StrategyRegistry:
    """Registry of all active strategies.

    Evaluates all registered strategies against a MarketContext and
    collects signals and rejections.
    """

    def __init__(self) -> None:
        self._strategies: Dict[str, Strategy] = {}

    def register(self, strategy: Strategy) -> None:
        """Register a strategy instance."""
        key = f"{strategy.meta.strategy_id}:{strategy.meta.strategy_version}"
        if key in self._strategies:
            raise ValueError(f"Strategy '{key}' is already registered")
        self._strategies[key] = strategy

    def unregister(self, strategy_id: str, version: str) -> None:
        key = f"{strategy_id}:{version}"
        self._strategies.pop(key, None)

    @property
    def strategies(self) -> Dict[str, Strategy]:
        return dict(self._strategies)

    def evaluate_all(
        self, context: MarketContext,
    ) -> List[Tuple[Strategy, StrategySignal | SignalRejection]]:
        """Evaluate all registered strategies against the context.

        Returns a list of (strategy, result) tuples.
        """
        results: List[Tuple[Strategy, StrategySignal | SignalRejection]] = []
        for strategy in self._strategies.values():
            result = strategy.evaluate(context)
            results.append((strategy, result))
        return results
