"""BNB rejection-cluster range-reversion specialist.

Configuration:
- Symbol: BNB/USDT
- Timeframe: 1h
- Direction: long
- Required regime: sideways_range
- Minimum score: 78
- Required rejection reason: range_or_reversion_confirmation_failed

Strategy: enters long when BNB/USDT is in a sideways range and shows
signs of rejection at range lows with mean-reversion patterns.
"""

from __future__ import annotations

from typing import Optional

from playground.domain.market import MarketContext
from playground.domain.signals import (
    Direction, SignalRejection, SignalRejectionReason, StrategySignal,
)
from playground.core.specialist_registry import Strategy, StrategyMeta


class BNBRejectionClusterSpecialist(Strategy):
    """Range-reversion specialist for BNB/USDT on 1h timeframe.

    Looks for rejection patterns at range lows within a sideways regime.
    """

    META = StrategyMeta(
        strategy_id="bnb_range_reversion",
        strategy_version="v1",
        supported_symbols=("BNB-USDT", "BNB/USDT"),
        supported_timeframes=("1h",),
        supported_regimes=("sideways_range",),
        direction=Direction.LONG,
        required_indicators=(
            "rsi_14", "realized_volatility_20", "rolling_low_20",
            "rolling_high_20", "volume_ratio", "candle_range",
            "trend_slope_20",
        ),
        min_score=78.0,
    )

    @property
    def meta(self) -> StrategyMeta:
        return self.META

    def evaluate(self, context: MarketContext) -> StrategySignal | SignalRejection:
        """Evaluate the BNB range-reversion strategy.

        Entry conditions:
        1. Regime is sideways_range
        2. Price is near the lower end of the range
        3. RSI shows oversold (< 40)
        4. Volume is increasing (volume_ratio > 0.8)
        5. Candle shows rejection (close > open, or long lower wick)
        """
        # Prerequisite checks
        prereq = self._check_prerequisites(context)
        if prereq:
            return prereq

        ind = context.indicators
        candle = context.candle

        # Calculate the composite score based on conditions
        score = 0.0
        conditions_met = 0
        total_conditions = 5
        reasons: list[str] = []

        # Condition 1: Near range low (price in lower 30% of range)
        if ind.rolling_high_20 is not None and ind.rolling_low_20 is not None:
            range_span = ind.rolling_high_20 - ind.rolling_low_20
            if range_span > 0:
                position_in_range = (candle.close - ind.rolling_low_20) / range_span
                if position_in_range <= 0.30:
                    conditions_met += 1
                    score += 25.0
                    reasons.append(f"near_range_low(pos={position_in_range:.2f})")

        # Condition 2: RSI oversold (below 40)
        if ind.rsi_14 is not None and ind.rsi_14 < 40.0:
            conditions_met += 1
            score += 20.0 + (40.0 - ind.rsi_14) * 0.5  # More oversold = more points
            reasons.append(f"rsi_oversold(rsi={ind.rsi_14:.1f})")

        # Condition 3: Increasing volume
        if ind.volume_ratio is not None and ind.volume_ratio > 0.8:
            conditions_met += 1
            vol_bonus = min(10.0, (ind.volume_ratio - 0.8) * 50.0)
            score += 15.0 + vol_bonus
            reasons.append(f"volume_surge(ratio={ind.volume_ratio:.2f})")

        # Condition 4: Bullish candle (close > open)
        if candle.close >= candle.open:
            conditions_met += 1
            score += 15.0
            reasons.append("bullish_candle")

        # Condition 5: Rejection pattern - long lower wick relative to body
        body = abs(candle.close - candle.open)
        lower_wick = min(candle.open, candle.close) - candle.low
        if body > 0 and lower_wick > body * 1.5:
            conditions_met += 1
            wick_score = min(15.0, (lower_wick / body) * 5.0)
            score += 10.0 + wick_score
            reasons.append(f"rejection_wick(ratio={lower_wick/body:.2f})")

        # Bonus: Low volatility within sideways regime is favorable
        if ind.realized_volatility_20 is not None and ind.realized_volatility_20 < 0.02:
            score += 5.0
            reasons.append("low_volatility_bonus")

        # Score must meet minimum
        if score < self.META.min_score:
            return self._reject(
                context,
                SignalRejectionReason.RANGE_OR_REVERSION_CONFIRMATION_FAILED,
                detail=f"Score {score:.1f} < {self.META.min_score}. "
                       f"Conditions met: {conditions_met}/{total_conditions}. "
                       f"Reasons: {', '.join(reasons) if reasons else 'none'}",
            )

        # Calculate entry, stop, and target
        entry_price = candle.close
        stop_loss = ind.rolling_low_20 if ind.rolling_low_20 else candle.low * 0.98
        take_profit = (
            (ind.rolling_high_20 + ind.rolling_low_20) / 2.0
            if ind.rolling_high_20 and ind.rolling_low_20
            else entry_price * 1.03
        )

        return self._make_signal(
            context,
            score=score,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "conditions_met": conditions_met,
                "total_conditions": total_conditions,
                "reasons": reasons,
                "position_in_range": (
                    (candle.close - ind.rolling_low_20) / (ind.rolling_high_20 - ind.rolling_low_20)
                    if ind.rolling_high_20 and ind.rolling_low_20 and ind.rolling_high_20 != ind.rolling_low_20
                    else None
                ),
            },
        )
