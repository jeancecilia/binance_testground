"""Risk engine: validates every signal before execution.

Minimum controls:
- Max open positions / positions per symbol
- Max exposure per symbol / total exposure
- Position sizing
- Max daily loss / max drawdown
- Entry cooldown
- Max spread / min depth / max slippage
- One entry per strategy per candle
- Kill switch

Every signal results in either an approved RiskDecision or a persisted rejection.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Sequence

from playground.domain.orders import RiskDecision
from playground.domain.positions import Position
from playground.domain.signals import SignalRejectionReason, StrategySignal
from playground.infrastructure.configuration import RiskEngineConfig


class RiskEngine:
    """Independent risk engine between strategy evaluation and execution.

    Strategies must never call the broker directly.
    """

    def __init__(
        self,
        config: RiskEngineConfig | None = None,
        positions: Dict[str, Position] | None = None,
        daily_pnl: float = 0.0,
        active_signal_ids: set[str] | None = None,
    ) -> None:
        self.config = config or RiskEngineConfig()
        self._positions: Dict[str, Position] = positions or {}
        self._daily_pnl = daily_pnl
        self._last_entry_time: Dict[str, datetime] = {}
        self._active_signal_ids: set[str] = active_signal_ids or set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        signal: StrategySignal,
        current_price: float,
        spread_pct: float = 0.0,
        market_depth_usdt: float = 0.0,
        estimated_slippage_pct: float = 0.0,
    ) -> RiskDecision:
        """Evaluate a strategy signal against all risk controls.

        Returns RiskDecision with approved=True/False and detailed results.
        """
        checks_passed: list[str] = []
        checks_failed: list[str] = []

        # 1. Kill switch
        if self.config.kill_switch:
            checks_failed.append("kill_switch_engaged")
            return self._make_decision(
                signal, False, checks_passed, checks_failed,
                SignalRejectionReason.KILL_SWITCH_ENGAGED,
            )

        # 2. One entry per strategy per candle
        if str(signal.signal_id) in self._active_signal_ids:
            checks_failed.append("duplicate_signal")
            return self._make_decision(
                signal, False, checks_passed, checks_failed,
                SignalRejectionReason.DUPLICATE_SIGNAL,
            )
        checks_passed.append("unique_signal")

        # 3. Max open positions
        open_positions = sum(1 for p in self._positions.values() if p.is_open)
        if open_positions >= self.config.max_open_positions:
            checks_failed.append("max_open_positions")
            return self._make_decision(
                signal, False, checks_passed, checks_failed,
                SignalRejectionReason.MAX_OPEN_POSITIONS_REACHED,
            )
        checks_passed.append("open_positions_ok")

        # 4. Max positions per symbol
        symbol_positions = sum(
            1 for s, p in self._positions.items()
            if p.is_open and s == signal.symbol
        )
        if symbol_positions >= self.config.max_positions_per_symbol:
            checks_failed.append("max_positions_per_symbol")
            return self._make_decision(
                signal, False, checks_passed, checks_failed,
                SignalRejectionReason.MAX_POSITIONS_PER_SYMBOL_REACHED,
            )
        checks_passed.append("positions_per_symbol_ok")

        # 5. Max exposure per symbol
        symbol_exposure = sum(
            p.notional_value for s, p in self._positions.items()
            if p.is_open and s == signal.symbol
        )
        max_symbol_exposure = (
            self.config.initial_balance_usdt * self.config.max_exposure_per_symbol_pct
        )
        if symbol_exposure >= max_symbol_exposure:
            checks_failed.append("max_exposure_per_symbol")
            return self._make_decision(
                signal, False, checks_passed, checks_failed,
                SignalRejectionReason.MAX_EXPOSURE_PER_SYMBOL_REACHED,
            )
        checks_passed.append("exposure_per_symbol_ok")

        # 6. Max total exposure
        total_exposure = sum(
            p.notional_value for p in self._positions.values() if p.is_open
        )
        max_total_exposure = (
            self.config.initial_balance_usdt * self.config.max_total_exposure_pct
        )
        if total_exposure >= max_total_exposure:
            checks_failed.append("max_total_exposure")
            return self._make_decision(
                signal, False, checks_passed, checks_failed,
                SignalRejectionReason.MAX_TOTAL_EXPOSURE_REACHED,
            )
        checks_passed.append("total_exposure_ok")

        # 7. Max daily loss
        if abs(self._daily_pnl) >= (
            self.config.initial_balance_usdt * self.config.max_daily_loss_pct
        ):
            checks_failed.append("max_daily_loss")
            return self._make_decision(
                signal, False, checks_passed, checks_failed,
                SignalRejectionReason.MAX_DAILY_LOSS_REACHED,
            )
        checks_passed.append("daily_loss_ok")

        # 8. Max drawdown (check across all positions)
        total_pnl = sum(p.unrealized_pnl for p in self._positions.values())
        if total_pnl < 0:
            drawdown_pct = abs(total_pnl) / self.config.initial_balance_usdt
            if drawdown_pct > self.config.max_drawdown_pct:
                checks_failed.append("max_drawdown")
                return self._make_decision(
                    signal, False, checks_passed, checks_failed,
                    SignalRejectionReason.MAX_DRAWDOWN_REACHED,
                )
        checks_passed.append("drawdown_ok")

        # 9. Entry cooldown
        cooldown_key = f"{signal.strategy_id}:{signal.symbol}"
        last_entry = self._last_entry_time.get(cooldown_key)
        if last_entry is not None:
            elapsed = (datetime.utcnow() - last_entry).total_seconds()
            if elapsed < self.config.entry_cooldown_seconds:
                checks_failed.append("entry_cooldown")
                return self._make_decision(
                    signal, False, checks_passed, checks_failed,
                    SignalRejectionReason.ENTRY_COOLDOWN_ACTIVE,
                )
        checks_passed.append("cooldown_ok")

        # 10. Maximum spread
        if spread_pct > self.config.max_spread_pct:
            checks_failed.append("max_spread")
            return self._make_decision(
                signal, False, checks_passed, checks_failed,
                SignalRejectionReason.SPREAD_TOO_WIDE,
            )
        checks_passed.append("spread_ok")

        # 11. Minimum market depth
        if market_depth_usdt < self.config.min_market_depth_usdt:
            checks_failed.append("min_depth")
            return self._make_decision(
                signal, False, checks_passed, checks_failed,
                SignalRejectionReason.INSUFFICIENT_DEPTH,
            )
        checks_passed.append("depth_ok")

        # 12. Maximum estimated slippage
        if estimated_slippage_pct > self.config.max_estimated_slippage_pct:
            checks_failed.append("max_slippage")
            return self._make_decision(
                signal, False, checks_passed, checks_failed,
                SignalRejectionReason.ESTIMATED_SLIPPAGE_TOO_HIGH,
            )
        checks_passed.append("slippage_ok")

        # 13. Position sizing
        position_size = self._calculate_position_size(current_price)
        if position_size <= 0:
            checks_failed.append("position_size_zero")
            return self._make_decision(
                signal, False, checks_passed, checks_failed,
                SignalRejectionReason.POSITION_SIZE_ZERO,
            )
        checks_passed.append("position_size_ok")

        # All checks passed
        self._active_signal_ids.add(str(signal.signal_id))
        self._last_entry_time[cooldown_key] = datetime.utcnow()

        return RiskDecision(
            signal_id=str(signal.signal_id),
            strategy_id=signal.strategy_id,
            symbol=signal.symbol,
            timeframe=signal.timeframe,
            candle_timestamp=signal.candle_timestamp,
            approved=True,
            position_size=position_size,
            risk_config_version=self.config.version,
            checks_passed=tuple(checks_passed),
            checks_failed=tuple(checks_failed),
        )

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _calculate_position_size(self, current_price: float) -> float:
        """Calculate position size based on configured percentage of balance."""
        if current_price <= 0:
            return 0.0

        # Simple fixed-fraction position sizing
        allocation = self.config.initial_balance_usdt * self.config.position_size_pct
        quantity = allocation / current_price
        return quantity

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def update_positions(self, positions: Dict[str, Position]) -> None:
        """Update the current positions snapshot."""
        self._positions = positions

    def update_daily_pnl(self, pnl: float) -> None:
        """Update the current daily PnL."""
        self._daily_pnl = pnl

    @property
    def kill_switch_engaged(self) -> bool:
        return self.config.kill_switch

    def engage_kill_switch(self) -> None:
        """Dynamically engage the kill switch."""
        object.__setattr__(self.config, 'kill_switch', True)

    def disengage_kill_switch(self) -> None:
        """Dynamically disengage the kill switch."""
        object.__setattr__(self.config, 'kill_switch', False)

    @staticmethod
    def _make_decision(
        signal: StrategySignal,
        approved: bool,
        checks_passed: list[str],
        checks_failed: list[str],
        reason: SignalRejectionReason | None = None,
    ) -> RiskDecision:
        return RiskDecision(
            signal_id=str(signal.signal_id),
            strategy_id=signal.strategy_id,
            symbol=signal.symbol,
            timeframe=signal.timeframe,
            candle_timestamp=signal.candle_timestamp,
            approved=approved,
            position_size=None if not approved else 0.0,
            rejection_reason=reason.value if reason else None,
            risk_config_version="1.0.0",
            checks_passed=tuple(checks_passed),
            checks_failed=tuple(checks_failed),
        )
