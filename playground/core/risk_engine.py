"""Risk engine between strategy evaluation and execution."""

from __future__ import annotations

from datetime import datetime
from typing import Dict

from playground.domain.orders import RiskDecision
from playground.domain.positions import Position
from playground.domain.signals import SignalRejectionReason, StrategySignal
from playground.infrastructure.configuration import RiskEngineConfig
from playground.infrastructure.system_clock import Clock, SystemClock


class RiskEngine:
    """Evaluate signals against position, exposure, loss, and liquidity limits."""

    def __init__(
        self,
        config: RiskEngineConfig | None = None,
        positions: Dict[str, Position] | None = None,
        daily_pnl: float = 0.0,
        active_signal_ids: set[str] | None = None,
        clock: Clock | None = None,
        market_prices: Dict[str, float] | None = None,
    ) -> None:
        self.config = config or RiskEngineConfig()
        self._positions: Dict[str, Position] = {
            symbol: position
            for symbol, position in (positions or {}).items()
            if position.is_open
        }
        self._daily_pnl = daily_pnl
        self._last_entry_time: Dict[str, datetime] = {}
        self._active_signal_ids: set[str] = active_signal_ids or set()
        self._clock = clock or SystemClock()
        self._market_prices: Dict[str, float] = {
            symbol: price
            for symbol, price in (market_prices or {}).items()
            if price > 0
        }

    def evaluate(
        self,
        signal: StrategySignal,
        current_price: float,
        spread_pct: float = 0.0,
        market_depth_usdt: float = 0.0,
        estimated_slippage_pct: float = 0.0,
    ) -> RiskDecision:
        """Evaluate a signal and return an approval or structured rejection."""
        checks_passed: list[str] = []
        checks_failed: list[str] = []

        if current_price > 0:
            self.update_market_price(signal.symbol, current_price)

        if self.config.kill_switch:
            checks_failed.append("kill_switch_engaged")
            return self._make_decision(
                signal,
                False,
                checks_passed,
                checks_failed,
                SignalRejectionReason.KILL_SWITCH_ENGAGED,
            )

        if str(signal.signal_id) in self._active_signal_ids:
            checks_failed.append("duplicate_signal")
            return self._make_decision(
                signal,
                False,
                checks_passed,
                checks_failed,
                SignalRejectionReason.DUPLICATE_SIGNAL,
            )
        checks_passed.append("unique_signal")

        open_positions = sum(1 for position in self._positions.values() if position.is_open)
        if open_positions >= self.config.max_open_positions:
            checks_failed.append("max_open_positions")
            return self._make_decision(
                signal,
                False,
                checks_passed,
                checks_failed,
                SignalRejectionReason.MAX_OPEN_POSITIONS_REACHED,
            )
        checks_passed.append("open_positions_ok")

        symbol_positions = sum(
            1
            for symbol, position in self._positions.items()
            if position.is_open and symbol == signal.symbol
        )
        if symbol_positions >= self.config.max_positions_per_symbol:
            checks_failed.append("max_positions_per_symbol")
            return self._make_decision(
                signal,
                False,
                checks_passed,
                checks_failed,
                SignalRejectionReason.MAX_POSITIONS_PER_SYMBOL_REACHED,
            )
        checks_passed.append("positions_per_symbol_ok")

        position_size = self._calculate_position_size(current_price)
        if position_size <= 0:
            checks_failed.append("position_size_zero")
            return self._make_decision(
                signal,
                False,
                checks_passed,
                checks_failed,
                SignalRejectionReason.POSITION_SIZE_ZERO,
            )

        exposure = self._mark_to_market_exposure(signal.symbol, current_price)
        if exposure is None:
            missing = sorted(
                position.symbol
                for position in self._positions.values()
                if position.is_open
                and position.symbol != signal.symbol
                and self._market_prices.get(position.symbol, 0.0) <= 0
            )
            checks_failed.extend(
                f"missing_mark_price:{symbol}" for symbol in missing
            )
            return self._make_decision(
                signal,
                False,
                checks_passed,
                checks_failed,
                SignalRejectionReason.DATA_CONTINUITY_BROKEN,
            )

        existing_symbol_exposure, existing_total_exposure, total_unrealized = exposure
        proposed_notional = current_price * position_size

        max_symbol_exposure = (
            self.config.initial_balance_usdt
            * self.config.max_exposure_per_symbol_pct
        )
        if existing_symbol_exposure + proposed_notional > max_symbol_exposure:
            checks_failed.append("max_exposure_per_symbol")
            return self._make_decision(
                signal,
                False,
                checks_passed,
                checks_failed,
                SignalRejectionReason.MAX_EXPOSURE_PER_SYMBOL_REACHED,
            )
        checks_passed.append("exposure_per_symbol_ok")

        max_total_exposure = (
            self.config.initial_balance_usdt
            * self.config.max_total_exposure_pct
        )
        if existing_total_exposure + proposed_notional > max_total_exposure:
            checks_failed.append("max_total_exposure")
            return self._make_decision(
                signal,
                False,
                checks_passed,
                checks_failed,
                SignalRejectionReason.MAX_TOTAL_EXPOSURE_REACHED,
            )
        checks_passed.append("total_exposure_ok")

        max_daily_loss = (
            self.config.initial_balance_usdt * self.config.max_daily_loss_pct
        )
        if self._daily_pnl < 0 and abs(self._daily_pnl) >= max_daily_loss:
            checks_failed.append("max_daily_loss")
            return self._make_decision(
                signal,
                False,
                checks_passed,
                checks_failed,
                SignalRejectionReason.MAX_DAILY_LOSS_REACHED,
            )
        checks_passed.append("daily_loss_ok")

        if total_unrealized < 0:
            drawdown_pct = abs(total_unrealized) / self.config.initial_balance_usdt
            if drawdown_pct > self.config.max_drawdown_pct:
                checks_failed.append("max_drawdown")
                return self._make_decision(
                    signal,
                    False,
                    checks_passed,
                    checks_failed,
                    SignalRejectionReason.MAX_DRAWDOWN_REACHED,
                )
        checks_passed.append("drawdown_ok")

        cooldown_key = f"{signal.strategy_id}:{signal.symbol}"
        last_entry = self._last_entry_time.get(cooldown_key)
        if last_entry is not None:
            elapsed = (self._clock.now() - last_entry).total_seconds()
            if elapsed < self.config.entry_cooldown_seconds:
                checks_failed.append("entry_cooldown")
                return self._make_decision(
                    signal,
                    False,
                    checks_passed,
                    checks_failed,
                    SignalRejectionReason.ENTRY_COOLDOWN_ACTIVE,
                )
        checks_passed.append("cooldown_ok")

        if spread_pct > self.config.max_spread_pct:
            checks_failed.append("max_spread")
            return self._make_decision(
                signal,
                False,
                checks_passed,
                checks_failed,
                SignalRejectionReason.SPREAD_TOO_WIDE,
            )
        checks_passed.append("spread_ok")

        if market_depth_usdt < self.config.min_market_depth_usdt:
            checks_failed.append("min_depth")
            return self._make_decision(
                signal,
                False,
                checks_passed,
                checks_failed,
                SignalRejectionReason.INSUFFICIENT_DEPTH,
            )
        checks_passed.append("depth_ok")

        if estimated_slippage_pct > self.config.max_estimated_slippage_pct:
            checks_failed.append("max_slippage")
            return self._make_decision(
                signal,
                False,
                checks_passed,
                checks_failed,
                SignalRejectionReason.ESTIMATED_SLIPPAGE_TOO_HIGH,
            )
        checks_passed.append("slippage_ok")
        checks_passed.append("position_size_ok")

        self._active_signal_ids.add(str(signal.signal_id))
        self._last_entry_time[cooldown_key] = self._clock.now()
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

    def _mark_to_market_exposure(
        self, signal_symbol: str, current_price: float
    ) -> tuple[float, float, float] | None:
        symbol_exposure = 0.0
        total_exposure = 0.0
        total_unrealized = 0.0

        for key, position in self._positions.items():
            if not position.is_open:
                continue
            symbol = position.symbol or key
            mark_price = (
                current_price
                if symbol == signal_symbol
                else self._market_prices.get(symbol, 0.0)
            )
            if mark_price <= 0:
                return None
            market_value = abs(position.quantity) * mark_price
            total_exposure += market_value
            if symbol == signal_symbol:
                symbol_exposure += market_value
            total_unrealized += position.quantity * (
                mark_price - position.avg_entry_price
            )

        return symbol_exposure, total_exposure, total_unrealized

    def _calculate_position_size(self, current_price: float) -> float:
        if current_price <= 0:
            return 0.0
        allocation = (
            self.config.initial_balance_usdt * self.config.position_size_pct
        )
        return allocation / current_price

    def update_positions(self, positions: Dict[str, Position]) -> None:
        """Replace the full position snapshot, including clearing to empty."""
        self._positions = {
            symbol: position
            for symbol, position in positions.items()
            if position.is_open
        }

    def update_market_price(self, symbol: str, price: float) -> None:
        if price > 0:
            self._market_prices[symbol] = price

    def update_market_prices(self, prices: Dict[str, float]) -> None:
        for symbol, price in prices.items():
            self.update_market_price(symbol, price)

    def update_daily_pnl(self, pnl: float) -> None:
        self._daily_pnl = pnl

    @property
    def kill_switch_engaged(self) -> bool:
        return self.config.kill_switch

    def engage_kill_switch(self) -> None:
        object.__setattr__(self.config, "kill_switch", True)

    def disengage_kill_switch(self) -> None:
        object.__setattr__(self.config, "kill_switch", False)

    def _make_decision(
        self,
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
            risk_config_version=self.config.version,
            checks_passed=tuple(checks_passed),
            checks_failed=tuple(checks_failed),
        )
