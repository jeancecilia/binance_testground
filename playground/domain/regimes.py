"""Domain models for regime detection and stability overlay."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional


@dataclass(frozen=True, slots=True)
class RegimeDecision:
    """Raw regime classification for a single candle."""

    symbol: str
    timeframe: str
    candle_timestamp: datetime
    regime: str  # one of the 8 regime labels
    indicator_values: Dict[str, Optional[float]]
    applied_thresholds: Dict[str, float]
    config_version: str
    decision_reason: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_safety_regime(self) -> bool:
        """Safety regimes bypass stability overlay doubt."""
        return self.regime in {
            "crash_liquidation_environment",
            "high_volatility_chaos",
        }


@dataclass(frozen=True, slots=True)
class StabilityDecision:
    """Stabilized regime after confidence and persistence overlay."""

    symbol: str
    timeframe: str
    candle_timestamp: datetime
    raw_regime: str
    final_regime: str
    confidence_score: float
    persistence_score: float
    recent_regime_consistency: float
    decision_reason: str
    stability_config_version: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    """Versioned regime detector configuration."""

    version: str = "1.0.0"

    # Trend detection thresholds
    trend_slope_bull_threshold: float = 0.0005  # positive slope above this = bull
    trend_slope_bear_threshold: float = -0.0005  # negative slope below this = bear
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # Range detection
    range_max_volatility: float = 0.02  # max realized vol for sideways
    range_max_atr_pct: float = 0.015  # max ATR/price for sideways
    range_min_volatility: float = 0.002  # below this = low vol compression

    # Volatility regimes
    high_volatility_threshold: float = 0.05  # realized vol above this = high vol
    crash_price_drop_threshold: float = -0.10  # 10% drop = crash

    # Stability overlay
    confidence_threshold: float = 0.60
    uncertain_threshold: float = 0.30
    persistence_threshold: float = 0.50
    persistence_lookback: int = 10  # candles to check for consistency

    safety_regimes: tuple[str, ...] = (
        "crash_liquidation_environment",
        "high_volatility_chaos",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "trend_slope_bull_threshold": self.trend_slope_bull_threshold,
            "trend_slope_bear_threshold": self.trend_slope_bear_threshold,
            "rsi_overbought": self.rsi_overbought,
            "rsi_oversold": self.rsi_oversold,
            "range_max_volatility": self.range_max_volatility,
            "range_max_atr_pct": self.range_max_atr_pct,
            "range_min_volatility": self.range_min_volatility,
            "high_volatility_threshold": self.high_volatility_threshold,
            "crash_price_drop_threshold": self.crash_price_drop_threshold,
            "confidence_threshold": self.confidence_threshold,
            "uncertain_threshold": self.uncertain_threshold,
            "persistence_threshold": self.persistence_threshold,
            "persistence_lookback": self.persistence_lookback,
        }
