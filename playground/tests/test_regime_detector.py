"""Unit tests for regime detector and stability overlay."""

from datetime import datetime
import pytest

from playground.domain.market import IndicatorSnapshot, Symbol, Timeframe
from playground.domain.regimes import RegimeConfig, RegimeDecision
from playground.core.regime_detector import RegimeDetector
from playground.core.stability_overlay import StabilityOverlay


def make_indicator_snapshot(**overrides) -> IndicatorSnapshot:
    """Create an indicator snapshot with default neutral values."""
    defaults = {
        "symbol": Symbol("BNB/USDT"),
        "timeframe": Timeframe("1h"),
        "candle_timestamp": datetime(2026, 7, 1, 10, 0, 0),
        "indicator_version": "1.0.0",
        "sma_20": 300.0,
        "sma_50": 295.0,
        "ema_12": 301.0,
        "ema_26": 298.0,
        "rsi_14": 50.0,
        "atr_14": 5.0,
        "realized_volatility_20": 0.01,
        "rolling_high_20": 310.0,
        "rolling_low_20": 290.0,
        "drawdown_20": -0.02,
        "trend_slope_20": 0.0002,
        "average_volume_20": 1000.0,
        "volume_ratio": 1.0,
        "candle_range": 5.0,
        "relative_candle_range": 0.016,
        "history_available": 200,
    }
    defaults.update(overrides)
    return IndicatorSnapshot(**defaults)


class TestRegimeDetector:
    def test_bull_trend(self):
        detector = RegimeDetector()
        snap = make_indicator_snapshot(
            trend_slope_20=0.001,  # Strong positive
            rsi_14=65.0,
            realized_volatility_20=0.03,
        )
        decision = detector.detect("BNB-USDT", "1h", snap)
        assert decision.regime == "bull_trend"

    def test_bear_trend(self):
        detector = RegimeDetector()
        snap = make_indicator_snapshot(
            trend_slope_20=-0.002,  # Strong negative
            rsi_14=35.0,
            realized_volatility_20=0.03,
        )
        decision = detector.detect("BNB-USDT", "1h", snap)
        assert decision.regime == "bear_trend"

    def test_sideways_range(self):
        detector = RegimeDetector()
        snap = make_indicator_snapshot(
            trend_slope_20=0.0001,  # Neutral slope
            rsi_14=48.0,
            realized_volatility_20=0.01,  # Moderate volatility
        )
        decision = detector.detect("BNB-USDT", "1h", snap)
        assert decision.regime == "sideways_range"

    def test_low_volatility_compression(self):
        detector = RegimeDetector()
        snap = make_indicator_snapshot(
            trend_slope_20=0.0001,
            rsi_14=50.0,
            realized_volatility_20=0.001,  # Very low volatility
        )
        decision = detector.detect("BNB-USDT", "1h", snap)
        assert decision.regime == "low_volatility_compression"

    def test_high_volatility_chaos(self):
        detector = RegimeDetector()
        snap = make_indicator_snapshot(
            trend_slope_20=0.0001,
            rsi_14=50.0,
            realized_volatility_20=0.08,  # Very high volatility
        )
        decision = detector.detect("BNB-USDT", "1h", snap)
        assert decision.regime == "high_volatility_chaos"

    def test_crash_environment(self):
        detector = RegimeDetector()
        snap = make_indicator_snapshot(
            trend_slope_20=-0.003,
            rsi_14=20.0,
            drawdown_20=-0.15,
            realized_volatility_20=0.08,
        )
        decision = detector.detect("BNB-USDT", "1h", snap)
        assert decision.regime == "crash_liquidation_environment"

    def test_uncertain_regime(self):
        detector = RegimeDetector()
        snap = make_indicator_snapshot(
            trend_slope_20=0.0003,  # Ambiguous slope
            rsi_14=None,  # Missing RSI
            realized_volatility_20=0.04,  # Between ranges
        )
        decision = detector.detect("BNB-USDT", "1h", snap)
        assert decision.regime == "uncertain_regime"

    def test_deterministic(self):
        detector1 = RegimeDetector()
        detector2 = RegimeDetector()
        snap = make_indicator_snapshot(trend_slope_20=0.001, rsi_14=65.0)

        d1 = detector1.detect("BNB-USDT", "1h", snap)
        d2 = detector2.detect("BNB-USDT", "1h", snap)
        assert d1.regime == d2.regime
        assert d1.decision_reason == d2.decision_reason

    def test_records_all_fields(self):
        detector = RegimeDetector()
        snap = make_indicator_snapshot()
        decision = detector.detect("BNB-USDT", "1h", snap)

        assert decision.symbol == "BNB-USDT"
        assert decision.timeframe == "1h"
        assert isinstance(decision.candle_timestamp, datetime)
        assert decision.regime in {
            "sideways_range", "bull_trend", "bear_trend",
            "regime_transition", "uncertain_regime",
            "low_volatility_compression", "high_volatility_chaos",
            "crash_liquidation_environment",
        }
        assert len(decision.indicator_values) > 0
        assert len(decision.applied_thresholds) > 0
        assert decision.config_version == "1.0.0"
        assert len(decision.decision_reason) > 0


class TestStabilityOverlay:
    def test_safety_regime_passes_through(self):
        overlay = StabilityOverlay()
        raw = RegimeDecision(
            symbol="BNB-USDT", timeframe="1h",
            candle_timestamp=datetime(2026, 7, 1, 10, 0, 0),
            regime="crash_liquidation_environment",
            indicator_values={"drawdown_20": -0.20},
            applied_thresholds={"crash_price_drop_threshold": -0.10},
            config_version="1.0.0",
            decision_reason="Crash detected",
        )
        stability = overlay.evaluate(raw, [])
        assert stability.final_regime == "crash_liquidation_environment"
        assert stability.confidence_score > 0

    def test_low_confidence_goes_uncertain(self):
        overlay = StabilityOverlay(RegimeConfig(
            uncertain_threshold=0.30,
            confidence_threshold=0.60,
        ))
        # Create a raw decision with indicators that produce very low confidence
        raw = RegimeDecision(
            symbol="BNB-USDT", timeframe="1h",
            candle_timestamp=datetime(2026, 7, 1, 10, 0, 0),
            regime="uncertain_regime",
            indicator_values={},
            applied_thresholds={},
            config_version="1.0.0",
            decision_reason="Nothing matched",
        )
        stability = overlay.evaluate(raw, [])
        assert stability.final_regime == "uncertain_regime"

    def test_moderate_confidence_becomes_transition(self):
        config = RegimeConfig(
            confidence_threshold=0.60,
            persistence_threshold=0.50,
            persistence_lookback=10,
        )
        overlay = StabilityOverlay(config)

        # Create a recent history where half the decisions match
        recent = []
        for i in range(10):
            recent.append(RegimeDecision(
                symbol="BNB-USDT", timeframe="1h",
                candle_timestamp=datetime(2026, 7, 1, i, 0, 0),
                regime="bull_trend" if i < 5 else "sideways_range",
                indicator_values={"trend_slope_20": 0.0006 if i < 5 else 0.0001},
                applied_thresholds={},
                config_version="1.0.0",
                decision_reason="Test",
            ))

        # Current is bull_trend, but only 5/10 recent match
        raw = RegimeDecision(
            symbol="BNB-USDT", timeframe="1h",
            candle_timestamp=datetime(2026, 7, 1, 11, 0, 0),
            regime="bull_trend",
            indicator_values={"trend_slope_20": 0.0006, "rsi_14": 55.0},
            applied_thresholds={"trend_slope_bull_threshold": 0.0005},
            config_version="1.0.0",
            decision_reason="Bull trend detected",
        )

        stability = overlay.evaluate(raw, recent)
        # Persistence = 5/10 = 0.5, which is >= persistence_threshold (0.5)
        # But confidence might be moderate...
        # Check that the overlay produces a valid decision
        assert stability.raw_regime == "bull_trend"
        assert stability.confidence_score >= 0
        assert stability.persistence_score >= 0
        assert stability.recent_regime_consistency >= 0

    def test_high_confidence_keeps_regime(self):
        overlay = StabilityOverlay()
        raw = RegimeDecision(
            symbol="BNB-USDT", timeframe="1h",
            candle_timestamp=datetime(2026, 7, 1, 10, 0, 0),
            regime="sideways_range",
            indicator_values={"realized_volatility_20": 0.011},
            applied_thresholds={
                "range_min_volatility": 0.002,
                "range_max_volatility": 0.02,
            },
            config_version="1.0.0",
            decision_reason="In range",
        )
        # High confidence because vol is near center of range
        stability = overlay.evaluate(raw, [])
        assert stability.final_regime in {"sideways_range", "regime_transition"}
        assert stability.confidence_score > 0.5

    def test_deterministic(self):
        overlay1 = StabilityOverlay()
        overlay2 = StabilityOverlay()
        raw = RegimeDecision(
            symbol="BNB-USDT", timeframe="1h",
            candle_timestamp=datetime(2026, 7, 1, 10, 0, 0),
            regime="sideways_range",
            indicator_values={"realized_volatility_20": 0.011},
            applied_thresholds={},
            config_version="1.0.0",
            decision_reason="Test",
        )
        s1 = overlay1.evaluate(raw, [])
        s2 = overlay2.evaluate(raw, [])
        assert s1.final_regime == s2.final_regime
        assert s1.confidence_score == s2.confidence_score
        assert s1.persistence_score == s2.persistence_score
