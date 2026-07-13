"""Raw market-regime classification from locally calculated indicators.

Supports 8 regimes:
- sideways_range, bull_trend, bear_trend
- regime_transition, uncertain_regime
- low_volatility_compression, high_volatility_chaos
- crash_liquidation_environment

Each decision records symbol, timeframe, candle timestamp, indicator values,
selected regime, applied thresholds, config version, and decision reason.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from playground.domain.market import IndicatorSnapshot
from playground.domain.regimes import RegimeConfig, RegimeDecision


class RegimeDetector:
    """Classifies market regime from indicator values using configurable thresholds.

    Deterministic: given the same indicator values and config, always produces
    the same regime classification.
    """

    # All recognized regime labels
    REGIME_SIDEWAYS_RANGE = "sideways_range"
    REGIME_BULL_TREND = "bull_trend"
    REGIME_BEAR_TREND = "bear_trend"
    REGIME_TRANSITION = "regime_transition"
    REGIME_UNCERTAIN = "uncertain_regime"
    REGIME_LOW_VOL_COMPRESSION = "low_volatility_compression"
    REGIME_HIGH_VOL_CHAOS = "high_volatility_chaos"
    REGIME_CRASH_LIQUIDATION = "crash_liquidation_environment"

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, symbol: str, timeframe: str,
        indicators: IndicatorSnapshot,
    ) -> RegimeDecision:
        """Classify the market regime for the given indicator snapshot.

        Evaluates regimes in priority order and returns the first match.
        """
        iv = self._extract_indicator_values(indicators)
        thresholds = self.config.to_dict()
        thresholds = {k: v for k, v in thresholds.items()
                      if isinstance(v, (int, float))}

        # Priority-ordered regime checks
        regime, reason = self._classify(iv)

        return RegimeDecision(
            symbol=symbol,
            timeframe=timeframe,
            candle_timestamp=indicators.candle_timestamp,
            regime=regime,
            indicator_values=iv,
            applied_thresholds=thresholds,
            config_version=self.config.version,
            decision_reason=reason,
        )

    # ------------------------------------------------------------------
    # Classification logic
    # ------------------------------------------------------------------

    def _classify(self, iv: Dict[str, Optional[float]]) -> tuple[str, str]:
        """Priority-ordered regime classification.

        Returns (regime_label, reason).
        """
        # 1. Crash / liquidation environment
        if self._is_crash(iv):
            return self.REGIME_CRASH_LIQUIDATION, (
                f"Drawdown {iv.get('drawdown_20')} below "
                f"threshold {self.config.crash_price_drop_threshold}"
            )

        # 2. High volatility chaos
        if self._is_high_volatility(iv):
            return self.REGIME_HIGH_VOL_CHAOS, (
                f"Realized volatility {iv.get('realized_volatility_20')} above "
                f"threshold {self.config.high_volatility_threshold}"
            )

        # 3. Low volatility compression
        if self._is_low_volatility_compression(iv):
            return self.REGIME_LOW_VOL_COMPRESSION, (
                f"Realized volatility {iv.get('realized_volatility_20')} below "
                f"threshold {self.config.range_min_volatility}"
            )

        # 4. Bull trend
        if self._is_bull_trend(iv):
            return self.REGIME_BULL_TREND, (
                f"Trend slope {iv.get('trend_slope_20')} above "
                f"bull threshold {self.config.trend_slope_bull_threshold}"
            )

        # 5. Bear trend
        if self._is_bear_trend(iv):
            return self.REGIME_BEAR_TREND, (
                f"Trend slope {iv.get('trend_slope_20')} below "
                f"bear threshold {self.config.trend_slope_bear_threshold}"
            )

        # 6. Sideways range
        if self._is_sideways_range(iv):
            return self.REGIME_SIDEWAYS_RANGE, (
                f"Volatility {iv.get('realized_volatility_20')} within range bounds "
                f"[{self.config.range_min_volatility}, {self.config.range_max_volatility}]"
            )

        # 7. Uncertain — nothing matched clearly
        return self.REGIME_UNCERTAIN, (
            f"No regime confidently matched. Slope={iv.get('trend_slope_20')}, "
            f"Vol={iv.get('realized_volatility_20')}, RSI={iv.get('rsi_14')}"
        )

    # ------------------------------------------------------------------
    # Individual regime checks
    # ------------------------------------------------------------------

    def _is_crash(self, iv: Dict[str, Optional[float]]) -> bool:
        dd = iv.get("drawdown_20")
        if dd is None:
            return False
        return dd <= self.config.crash_price_drop_threshold

    def _is_high_volatility(self, iv: Dict[str, Optional[float]]) -> bool:
        vol = iv.get("realized_volatility_20")
        if vol is None:
            return False
        return vol >= self.config.high_volatility_threshold

    def _is_low_volatility_compression(self, iv: Dict[str, Optional[float]]) -> bool:
        vol = iv.get("realized_volatility_20")
        if vol is None:
            return False
        return vol <= self.config.range_min_volatility

    def _is_bull_trend(self, iv: Dict[str, Optional[float]]) -> bool:
        slope = iv.get("trend_slope_20")
        rsi = iv.get("rsi_14")
        if slope is None:
            return False
        # Strong upward slope or sustained bullish RSI
        return (
            slope >= self.config.trend_slope_bull_threshold
            and (rsi is None or rsi >= 50.0)
        )

    def _is_bear_trend(self, iv: Dict[str, Optional[float]]) -> bool:
        slope = iv.get("trend_slope_20")
        rsi = iv.get("rsi_14")
        if slope is None:
            return False
        return (
            slope <= self.config.trend_slope_bear_threshold
            and (rsi is None or rsi <= 50.0)
        )

    def _is_sideways_range(self, iv: Dict[str, Optional[float]]) -> bool:
        vol = iv.get("realized_volatility_20")
        atr = iv.get("atr_14")
        if vol is None:
            return False

        # Volatility must be within range bounds
        in_vol_range = (
            self.config.range_min_volatility < vol < self.config.range_max_volatility
        )

        # ATR check (optional)
        atr_ok = True
        if atr is not None:
            atr_ok = True  # ATR alone doesn't disqualify; it's supplementary

        return in_vol_range and atr_ok

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_indicator_values(
        snapshot: IndicatorSnapshot,
    ) -> Dict[str, Optional[float]]:
        """Extract indicator values into a flat dict for recording."""
        return {
            "sma_20": snapshot.sma_20,
            "sma_50": snapshot.sma_50,
            "ema_12": snapshot.ema_12,
            "ema_26": snapshot.ema_26,
            "rsi_14": snapshot.rsi_14,
            "atr_14": snapshot.atr_14,
            "realized_volatility_20": snapshot.realized_volatility_20,
            "rolling_high_20": snapshot.rolling_high_20,
            "rolling_low_20": snapshot.rolling_low_20,
            "drawdown_20": snapshot.drawdown_20,
            "trend_slope_20": snapshot.trend_slope_20,
            "average_volume_20": snapshot.average_volume_20,
            "volume_ratio": snapshot.volume_ratio,
            "candle_range": snapshot.candle_range,
            "relative_candle_range": snapshot.relative_candle_range,
        }
