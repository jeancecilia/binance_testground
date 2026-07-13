"""Deterministic technical indicator engine.

Calculates all indicators from locally stored candles.
No placeholder implementations permitted.
All calculations are deterministic and versioned.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

from playground.domain.market import Candle, IndicatorSnapshot, Symbol, Timeframe
from playground.domain.indicators import IndicatorConfig, InsufficientHistoryRejection


class IndicatorEngineError(Exception):
    """Errors from the indicator engine."""
    pass


class InsufficientHistoryError(IndicatorEngineError):
    """Not enough candles to compute requested indicators."""
    pass


@dataclass(frozen=True)
class IndicatorResult:
    """Bundle of calculated indicator values."""
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi_14: Optional[float] = None
    atr_14: Optional[float] = None
    realized_volatility_20: Optional[float] = None
    rolling_high_20: Optional[float] = None
    rolling_low_20: Optional[float] = None
    drawdown_20: Optional[float] = None
    trend_slope_20: Optional[float] = None
    average_volume_20: Optional[float] = None
    volume_ratio: Optional[float] = None
    candle_range: Optional[float] = None
    relative_candle_range: Optional[float] = None


class IndicatorEngine:
    """Deterministic indicator calculator.

    All calculations operate on a list of Candle objects sorted
    by open_time ascending. The engine requires `minimum_history`
    candles before producing any indicator values.
    """

    def __init__(self, config: IndicatorConfig | None = None) -> None:
        self.config = config or IndicatorConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self, candles: Sequence[Candle],
    ) -> IndicatorResult:
        """Calculate all indicators for the latest candle in the sequence.

        Args:
            candles: Chronologically ordered candles (oldest to newest).

        Returns:
            IndicatorResult with all computed values.

        Raises:
            InsufficientHistoryError: When not enough candles are available.
        """
        if len(candles) < self.config.minimum_history:
            raise InsufficientHistoryError(
                f"Need {self.config.minimum_history} candles, got {len(candles)}"
            )

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]
        latest = candles[-1]

        return IndicatorResult(
            sma_20=self._sma(closes, 20),
            sma_50=self._sma(closes, 50),
            ema_12=self._ema(closes, 12),
            ema_26=self._ema(closes, 26),
            rsi_14=self._rsi(closes, 14),
            atr_14=self._atr(highs, lows, closes, 14),
            realized_volatility_20=self._realized_volatility(closes, 20),
            rolling_high_20=self._rolling_max(highs, 20),
            rolling_low_20=self._rolling_min(lows, 20),
            drawdown_20=self._drawdown(highs, 20),
            trend_slope_20=self._trend_slope(closes, 20),
            average_volume_20=self._sma(volumes, 20),
            volume_ratio=self._volume_ratio(volumes, 20),
            candle_range=latest.candle_range,
            relative_candle_range=latest.relative_range,
        )

    def create_snapshot(
        self, symbol: Symbol, timeframe: Timeframe,
        candles: Sequence[Candle],
    ) -> IndicatorSnapshot:
        """Calculate indicators and return an IndicatorSnapshot domain model."""
        result = self.calculate(candles)
        latest = candles[-1]

        return IndicatorSnapshot(
            symbol=symbol,
            timeframe=timeframe,
            candle_timestamp=latest.open_time,
            indicator_version=self.config.version,
            sma_20=result.sma_20,
            sma_50=result.sma_50,
            ema_12=result.ema_12,
            ema_26=result.ema_26,
            rsi_14=result.rsi_14,
            atr_14=result.atr_14,
            realized_volatility_20=result.realized_volatility_20,
            rolling_high_20=result.rolling_high_20,
            rolling_low_20=result.rolling_low_20,
            drawdown_20=result.drawdown_20,
            trend_slope_20=result.trend_slope_20,
            average_volume_20=result.average_volume_20,
            volume_ratio=result.volume_ratio,
            candle_range=result.candle_range,
            relative_candle_range=result.relative_candle_range,
            history_available=len(candles),
        )

    def check_history_sufficient(
        self, candle_count: int,
    ) -> Optional[InsufficientHistoryRejection]:
        """Check if history is sufficient. Returns rejection if not."""
        if candle_count < self.config.minimum_history:
            return InsufficientHistoryRejection(
                symbol="",
                timeframe="",
                candle_timestamp=datetime.utcnow(),
                available_history=candle_count,
                required_history=self.config.minimum_history,
            )
        return None

    # ------------------------------------------------------------------
    # SMA
    # ------------------------------------------------------------------

    @staticmethod
    def _sma(values: Sequence[float], period: int) -> Optional[float]:
        """Simple Moving Average."""
        if len(values) < period:
            return None
        window = values[-period:]
        return sum(window) / period

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(values: Sequence[float], period: int) -> Optional[float]:
        """Exponential Moving Average.

        Uses SMA as the seed for the first EMA value, then applies
        the standard EMA formula: EMA = price * k + prev_ema * (1 - k)
        where k = 2 / (period + 1).
        """
        if len(values) < period:
            return None

        multiplier = 2.0 / (period + 1.0)
        # Seed EMA with SMA of first `period` values
        ema = sum(values[:period]) / period

        for price in values[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(values: Sequence[float], period: int = 14) -> Optional[float]:
        """Relative Strength Index using Wilder's smoothing.

        RSI = 100 - (100 / (1 + avg_gain / avg_loss))
        """
        if len(values) < period + 1:
            return None

        # Compute price changes
        changes = [values[i] - values[i - 1] for i in range(1, len(values))]

        # Initial average gain/loss (simple average of first `period` changes)
        gains = [c if c > 0 else 0.0 for c in changes[:period]]
        losses = [-c if c < 0 else 0.0 for c in changes[:period]]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        # Wilder's smoothing for remaining changes
        for change in changes[period:]:
            gain = change if change > 0 else 0.0
            loss = -change if change < 0 else 0.0
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------

    @staticmethod
    def _atr(
        highs: Sequence[float], lows: Sequence[float],
        closes: Sequence[float], period: int = 14,
    ) -> Optional[float]:
        """Average True Range using Wilder's smoothing."""
        if len(highs) < period + 1:
            return None

        tr_values: list[float] = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            tr_values.append(tr)

        # Initial ATR = simple average of first `period` TR values
        atr = sum(tr_values[:period]) / period

        # Wilder's smoothing
        for tr in tr_values[period:]:
            atr = (atr * (period - 1) + tr) / period

        return atr

    # ------------------------------------------------------------------
    # Realized volatility
    # ------------------------------------------------------------------

    @staticmethod
    def _realized_volatility(
        values: Sequence[float], period: int = 20,
    ) -> Optional[float]:
        """Annualized realized volatility from log returns."""
        import math as _math

        if len(values) < period + 1:
            return None

        window = values[-(period + 1):]
        log_returns = [
            _math.log(window[i] / window[i - 1])
            for i in range(1, len(window))
        ]

        mean = sum(log_returns) / len(log_returns)
        variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)

        # Assuming 1h candles, 8760 hours/year. Scale appropriately.
        return _math.sqrt(variance * 8760)

    # ------------------------------------------------------------------
    # Rolling high / low
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_max(values: Sequence[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return max(values[-period:])

    @staticmethod
    def _rolling_min(values: Sequence[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        return min(values[-period:])

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    @staticmethod
    def _drawdown(values: Sequence[float], period: int = 20) -> Optional[float]:
        """Maximum drawdown over the period as a negative fraction.

        Returns: e.g. -0.05 means 5% drawdown.
        """
        if len(values) < period:
            return None

        window = values[-period:]
        peak = window[0]
        max_dd = 0.0

        for price in window:
            if price > peak:
                peak = price
            dd = (price - peak) / peak if peak > 0 else 0.0
            if dd < max_dd:
                max_dd = dd

        return max_dd

    # ------------------------------------------------------------------
    # Trend slope
    # ------------------------------------------------------------------

    @staticmethod
    def _trend_slope(values: Sequence[float], period: int = 20) -> Optional[float]:
        """Linear regression slope over the period, normalized by mean price.

        Returns slope as a fraction of mean price (e.g. 0.001 = 0.1% per candle).
        """
        if len(values) < period:
            return None

        window = values[-period:]
        n = len(window)
        mean_price = sum(window) / n

        if mean_price == 0:
            return 0.0

        x_mean = (n - 1) / 2.0
        y_mean = mean_price

        numerator = sum(
            (i - x_mean) * (window[i] - y_mean) for i in range(n)
        )
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return slope / mean_price

    # ------------------------------------------------------------------
    # Volume ratio
    # ------------------------------------------------------------------

    @staticmethod
    def _volume_ratio(values: Sequence[float], period: int = 20) -> Optional[float]:
        """Current volume / average volume over period."""
        if len(values) < period:
            return None

        avg_vol = sum(values[-period:]) / period
        current_vol = values[-1]

        if avg_vol == 0:
            return 1.0

        return current_vol / avg_vol
