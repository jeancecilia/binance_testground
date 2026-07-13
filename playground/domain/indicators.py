"""Domain models for indicator configuration and results."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any


@dataclass(frozen=True, slots=True)
class IndicatorConfig:
    """Versioned indicator engine configuration."""

    version: str = "1.0.0"
    sma_periods: tuple[int, ...] = (20, 50)
    ema_periods: tuple[int, ...] = (12, 26)
    rsi_period: int = 14
    atr_period: int = 14
    volatility_period: int = 20
    rolling_window: int = 20
    volume_period: int = 20
    trend_slope_period: int = 20
    minimum_history: int = 100  # minimum candles before any indicator is produced

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "sma_periods": list(self.sma_periods),
            "ema_periods": list(self.ema_periods),
            "rsi_period": self.rsi_period,
            "atr_period": self.atr_period,
            "volatility_period": self.volatility_period,
            "rolling_window": self.rolling_window,
            "volume_period": self.volume_period,
            "trend_slope_period": self.trend_slope_period,
            "minimum_history": self.minimum_history,
        }


@dataclass(frozen=True, slots=True)
class InsufficientHistoryRejection:
    """Recorded rejection when not enough candles exist for indicator calculation."""

    symbol: str
    timeframe: str
    candle_timestamp: datetime
    available_history: int
    required_history: int
    reason: str = "insufficient_history"
    created_at: datetime = field(default_factory=datetime.utcnow)
