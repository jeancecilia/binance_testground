"""Unit tests for the indicator engine.

Validates SMA, EMA, RSI, ATR, volatility, drawdown, and other indicators
against known outputs to ensure determinism and correctness.
"""

from datetime import datetime, timedelta
import math
import pytest

from playground.domain.market import Candle, CandleSource, Symbol, Timeframe
from playground.domain.indicators import IndicatorConfig
from playground.core.indicator_engine import (
    IndicatorEngine, IndicatorResult, InsufficientHistoryError,
)


# ------------------------------------------------------------------
# Helper: create test candles
# ------------------------------------------------------------------

def make_candle(
    open_time: datetime, open_p: float, high: float, low: float,
    close: float, volume: float,
) -> Candle:
    return Candle(
        symbol=Symbol("TEST/USDT"),
        timeframe=Timeframe("1h"),
        open_time=open_time,
        open=open_p,
        high=high,
        low=low,
        close=close,
        volume=volume,
        close_time=open_time + timedelta(hours=1),
        quote_asset_volume=volume * close,
        number_of_trades=100,
        taker_buy_base_volume=volume * 0.5,
        taker_buy_quote_volume=volume * close * 0.5,
        source=CandleSource.HISTORICAL,
    )


def make_candles(prices: list[float], volumes: list[float] | None = None) -> list[Candle]:
    """Create candles from price and volume lists."""
    base_time = datetime(2026, 7, 1, 0, 0, 0)
    candles = []
    vols = volumes or [100.0] * len(prices)
    for i, (price, vol) in enumerate(zip(prices, vols)):
        candles.append(make_candle(
            open_time=base_time + timedelta(hours=i),
            open_p=price * 0.99,
            high=price * 1.02,
            low=price * 0.98,
            close=price,
            volume=vol,
        ))
    return candles


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestSMA:
    def test_sma_simple(self):
        engine = IndicatorEngine(IndicatorConfig(minimum_history=5))
        candles = make_candles([10, 20, 30, 40, 50])
        result = engine.calculate(candles)
        assert result.sma_20 is None  # Only 5 candles, 20-period SMA needs 20
        # 5-period SMA of all 5 values
        sma5 = engine._sma([10, 20, 30, 40, 50], 5)
        assert sma5 == 30.0

    def test_sma_insufficient(self):
        engine = IndicatorEngine(IndicatorConfig(minimum_history=5))
        candles = make_candles([10, 20, 30])
        with pytest.raises(InsufficientHistoryError):
            engine.calculate(candles)

    def test_sma_deterministic(self):
        engine = IndicatorEngine()
        prices = [100.0 + i * 0.5 for i in range(200)]
        candles = make_candles(prices)
        r1 = engine._sma([c.close for c in candles], 20)
        r2 = engine._sma([c.close for c in candles], 20)
        assert r1 == r2


class TestEMA:
    def test_ema_basic(self):
        engine = IndicatorEngine()
        closes = [100.0] * 26  # 26 identical prices
        ema = engine._ema(closes, 12)
        assert ema == pytest.approx(100.0, rel=1e-9)

    def test_ema_insufficient(self):
        engine = IndicatorEngine()
        closes = [100.0] * 10
        ema = engine._ema(closes, 12)
        assert ema is None


class TestRSI:
    def test_rsi_all_gains(self):
        engine = IndicatorEngine()
        prices = [100.0 + i for i in range(20)]  # Steady uptrend
        rsi = engine._rsi(prices, 14)
        assert rsi == 100.0  # All gains, no losses

    def test_rsi_all_losses(self):
        engine = IndicatorEngine()
        prices = [100.0 - i for i in range(20)]  # Steady downtrend
        rsi = engine._rsi(prices, 14)
        assert rsi == 0.0  # All losses, no gains

    def test_rsi_insufficient(self):
        engine = IndicatorEngine()
        rsi = engine._rsi([100.0] * 10, 14)
        assert rsi is None


class TestATR:
    def test_atr_basic(self):
        engine = IndicatorEngine()
        highs = [105.0 + i for i in range(20)]
        lows = [95.0 + i for i in range(20)]
        closes = [100.0 + i for i in range(20)]
        atr = engine._atr(highs, lows, closes, 14)
        assert atr is not None
        assert atr > 0

    def test_atr_constant_range(self):
        engine = IndicatorEngine()
        n = 25
        highs = [105.0] * n
        lows = [95.0] * n
        closes = [100.0] * n
        atr = engine._atr(highs, lows, closes, 14)
        # True range = max(105-95, |105-100|, |95-100|) = 10
        assert atr == pytest.approx(10.0, rel=0.01)


class TestVolatility:
    def test_volatility_flat(self):
        engine = IndicatorEngine()
        prices = [100.0] * 30
        vol = engine._realized_volatility(prices, 20)
        assert vol == 0.0

    def test_volatility_positive(self):
        engine = IndicatorEngine()
        prices = [100.0 + math.sin(i * 0.5) * 5 for i in range(100)]
        vol = engine._realized_volatility(prices, 20)
        assert vol is not None
        assert vol > 0


class TestDrawdown:
    def test_drawdown_zero(self):
        engine = IndicatorEngine()
        prices = [100.0 + i for i in range(30)]  # Always rising
        dd = engine._drawdown(prices, 20)
        assert dd == 0.0

    def test_drawdown_negative(self):
        engine = IndicatorEngine()
        # Peak at 110, drops to 90
        prices = [100.0, 110.0] + [90.0] * 18
        dd = engine._drawdown(prices, 20)
        assert dd is not None
        assert dd < 0
        # Max DD = (90 - 110) / 110 ≈ -0.1818
        assert dd == pytest.approx(-20.0 / 110.0, rel=0.01)


class TestTrendSlope:
    def test_slope_positive(self):
        engine = IndicatorEngine()
        prices = [100.0 + i * 0.5 for i in range(30)]
        slope = engine._trend_slope(prices, 20)
        assert slope is not None
        assert slope > 0

    def test_slope_negative(self):
        engine = IndicatorEngine()
        prices = [100.0 - i * 0.5 for i in range(30)]
        slope = engine._trend_slope(prices, 20)
        assert slope is not None
        assert slope < 0

    def test_slope_flat(self):
        engine = IndicatorEngine()
        prices = [100.0] * 30
        slope = engine._trend_slope(prices, 20)
        assert slope == 0.0


class TestVolumeRatio:
    def test_volume_ratio_normal(self):
        engine = IndicatorEngine()
        volumes = [100.0] * 19 + [200.0]  # avg = 2100/20 = 105, current = 200, ratio = 200/105
        ratio = engine._volume_ratio(volumes, 20)
        # Expected: 200 / 105 ≈ 1.9047619
        assert ratio == pytest.approx(200.0 / 105.0)

    def test_volume_ratio_insufficient(self):
        engine = IndicatorEngine()
        ratio = engine._volume_ratio([100.0] * 10, 20)
        assert ratio is None


class TestEngineIntegration:
    def test_full_calculation(self):
        """Test that the engine produces all indicators without errors."""
        config = IndicatorConfig(minimum_history=50)
        engine = IndicatorEngine(config)

        n = 120
        base = datetime(2026, 7, 1, 0, 0, 0)
        candles = []
        for i in range(n):
            candles.append(make_candle(
                open_time=base + timedelta(hours=i),
                open_p=100.0 + i * 0.1,
                high=100.0 + i * 0.1 + 2.0,
                low=100.0 + i * 0.1 - 2.0,
                close=100.0 + i * 0.1 + 0.5,
                volume=1000.0 + i * 10,
            ))

        result = engine.calculate(candles)

        assert result.sma_20 is not None
        assert result.sma_50 is not None
        assert result.ema_12 is not None
        assert result.ema_26 is not None
        assert result.rsi_14 is not None
        assert result.atr_14 is not None
        assert result.realized_volatility_20 is not None
        assert result.drawdown_20 is not None
        assert result.trend_slope_20 is not None
        assert result.average_volume_20 is not None
        assert result.volume_ratio is not None

    def test_determinism(self):
        """Two identical runs must produce identical results."""
        config = IndicatorConfig(minimum_history=30)
        engine1 = IndicatorEngine(config)
        engine2 = IndicatorEngine(config)

        candles = make_candles([100.0 + i * 0.2 for i in range(100)])

        r1 = engine1.calculate(candles)
        r2 = engine2.calculate(candles)

        for field in [
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi_14', 'atr_14',
            'realized_volatility_20', 'drawdown_20', 'trend_slope_20',
        ]:
            v1 = getattr(r1, field)
            v2 = getattr(r2, field)
            if v1 is None:
                assert v2 is None
            else:
                assert v1 == pytest.approx(v2, rel=1e-12), f"Field {field} differs"
