"""Stability overlay: applies confidence and persistence checks after raw regime detection.

Decision order (from spec):
1. If raw_regime is a safety regime → pass through
2. If confidence_score < uncertain_threshold → uncertain_regime
3. If confidence_score < confidence_threshold OR persistence_score < persistence_threshold → regime_transition
4. Otherwise → raw_regime
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Dict, Optional, Sequence

from playground.domain.regimes import (
    RegimeConfig, RegimeDecision, StabilityDecision,
)


class StabilityOverlay:
    """Applies confidence and persistence checks to raw regime classifications.

    Deterministic: given the same inputs and config, always produces
    the same stability decision.
    """

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        raw_decision: RegimeDecision,
        recent_decisions: Sequence[RegimeDecision],
    ) -> StabilityDecision:
        """Apply stability overlay to a raw regime decision.

        Args:
            raw_decision: The raw regime classification to stabilize.
            recent_decisions: Previous raw regime decisions, ordered
                              chronologically (oldest to newest). Used
                              to compute persistence and consistency.

        Returns:
            StabilityDecision with the final stabilized regime.
        """
        confidence = self._compute_confidence(raw_decision)
        persistence = self._compute_persistence(raw_decision, recent_decisions)
        consistency = self._compute_consistency(recent_decisions)

        final_regime, reason = self._apply_rules(
            raw_decision.regime, confidence, persistence,
        )

        return StabilityDecision(
            symbol=raw_decision.symbol,
            timeframe=raw_decision.timeframe,
            candle_timestamp=raw_decision.candle_timestamp,
            raw_regime=raw_decision.regime,
            final_regime=final_regime,
            confidence_score=confidence,
            persistence_score=persistence,
            recent_regime_consistency=consistency,
            decision_reason=reason,
            stability_config_version=self.config.version,
        )

    # ------------------------------------------------------------------
    # Decision rules (from spec)
    # ------------------------------------------------------------------

    def _apply_rules(
        self, raw_regime: str, confidence: float, persistence: float,
    ) -> tuple[str, str]:
        """Apply the stability decision tree.

        Returns (final_regime, reason).
        """
        # Rule 1: Safety regimes pass through
        if raw_regime in self.config.safety_regimes:
            return raw_regime, (
                f"Safety regime '{raw_regime}' passed through without stability checks"
            )

        # Rule 2: Below uncertain threshold → uncertain
        if confidence < self.config.uncertain_threshold:
            return "uncertain_regime", (
                f"Confidence {confidence:.3f} < uncertain threshold "
                f"{self.config.uncertain_threshold}"
            )

        # Rule 3: Below confidence or persistence threshold → transition
        if (
            confidence < self.config.confidence_threshold
            or persistence < self.config.persistence_threshold
        ):
            reasons = []
            if confidence < self.config.confidence_threshold:
                reasons.append(
                    f"confidence {confidence:.3f} < {self.config.confidence_threshold}"
                )
            if persistence < self.config.persistence_threshold:
                reasons.append(
                    f"persistence {persistence:.3f} < {self.config.persistence_threshold}"
                )
            return "regime_transition", "; ".join(reasons)

        # Rule 4: Passed all checks → keep raw regime
        return raw_regime, (
            f"Confidence {confidence:.3f} >= {self.config.confidence_threshold} "
            f"and persistence {persistence:.3f} >= {self.config.persistence_threshold}"
        )

    # ------------------------------------------------------------------
    # Confidence score
    # ------------------------------------------------------------------

    def _compute_confidence(self, decision: RegimeDecision) -> float:
        """Compute confidence score for a raw regime decision.

        Based on how clearly the indicator values match the regime thresholds.
        Returns 0.0-1.0 where 1.0 is maximum confidence.
        """
        iv = decision.indicator_values
        regime = decision.regime

        if regime == "sideways_range":
            return self._sideways_confidence(iv)
        elif regime == "bull_trend":
            return self._bull_confidence(iv)
        elif regime == "bear_trend":
            return self._bear_confidence(iv)
        elif regime == "low_volatility_compression":
            return self._low_vol_confidence(iv)
        elif regime == "high_volatility_chaos":
            return self._high_vol_confidence(iv)
        elif regime == "crash_liquidation_environment":
            return self._crash_confidence(iv)
        elif regime == "uncertain_regime":
            return 0.1  # Low confidence by definition
        elif regime == "regime_transition":
            return 0.4  # Moderate-low confidence
        else:
            return 0.3

    def _sideways_confidence(self, iv: Dict[str, Optional[float]]) -> float:
        """Higher confidence when volatility is clearly in the middle of the range."""
        vol = iv.get("realized_volatility_20")
        if vol is None:
            return 0.3

        range_center = (
            self.config.range_min_volatility + self.config.range_max_volatility
        ) / 2.0
        range_half_width = (
            self.config.range_max_volatility - self.config.range_min_volatility
        ) / 2.0

        if range_half_width == 0:
            return 0.5

        # Distance from center, normalized to [0,1] where 0=center, 1=edge
        distance = abs(vol - range_center) / range_half_width
        # Confidence = 1 - distance (clipped to [0,1])
        return max(0.0, min(1.0, 1.0 - distance))

    def _bull_confidence(self, iv: Dict[str, Optional[float]]) -> float:
        slope = iv.get("trend_slope_20")
        rsi = iv.get("rsi_14")
        if slope is None:
            return 0.3

        # How far above threshold
        excess = slope - self.config.trend_slope_bull_threshold
        normalized = min(1.0, excess / (abs(self.config.trend_slope_bull_threshold) + 0.001))

        rsi_factor = 1.0
        if rsi is not None and rsi > 50:
            rsi_factor = min(1.0, (rsi - 50) / 20.0)  # Bonus for RSI 50-70

        return max(0.0, min(1.0, normalized * 0.7 + rsi_factor * 0.3))

    def _bear_confidence(self, iv: Dict[str, Optional[float]]) -> float:
        slope = iv.get("trend_slope_20")
        rsi = iv.get("rsi_14")
        if slope is None:
            return 0.3

        excess = self.config.trend_slope_bear_threshold - slope
        normalized = min(1.0, excess / (abs(self.config.trend_slope_bear_threshold) + 0.001))

        rsi_factor = 1.0
        if rsi is not None and rsi < 50:
            rsi_factor = min(1.0, (50 - rsi) / 20.0)

        return max(0.0, min(1.0, normalized * 0.7 + rsi_factor * 0.3))

    def _low_vol_confidence(self, iv: Dict[str, Optional[float]]) -> float:
        vol = iv.get("realized_volatility_20")
        if vol is None:
            return 0.3
        if vol == 0:
            return 1.0
        ratio = vol / self.config.range_min_volatility
        return max(0.0, min(1.0, 1.0 - ratio))

    def _high_vol_confidence(self, iv: Dict[str, Optional[float]]) -> float:
        vol = iv.get("realized_volatility_20")
        if vol is None:
            return 0.3
        if self.config.high_volatility_threshold == 0:
            return 0.5
        ratio = vol / self.config.high_volatility_threshold
        return min(1.0, ratio / 2.0)  # Max confidence at 2x threshold

    def _crash_confidence(self, iv: Dict[str, Optional[float]]) -> float:
        dd = iv.get("drawdown_20")
        if dd is None:
            return 0.3
        if self.config.crash_price_drop_threshold == 0:
            return 0.5
        ratio = dd / self.config.crash_price_drop_threshold
        return min(1.0, ratio / 2.0)

    # ------------------------------------------------------------------
    # Persistence score
    # ------------------------------------------------------------------

    def _compute_persistence(
        self,
        current: RegimeDecision,
        recent: Sequence[RegimeDecision],
    ) -> float:
        """How persistently the current regime has appeared recently.

        Returns 0.0-1.0 where 1.0 means the same regime has been consistent.
        """
        lookback = self.config.persistence_lookback
        if not recent or lookback < 1:
            return 1.0  # No history → assume persistent

        relevant = list(recent[-lookback:])
        if not relevant:
            return 1.0

        same_count = sum(1 for d in relevant if d.regime == current.regime)
        return same_count / len(relevant)

    # ------------------------------------------------------------------
    # Recent consistency
    # ------------------------------------------------------------------

    def _compute_consistency(
        self, recent: Sequence[RegimeDecision],
    ) -> float:
        """Measure how consistent recent regime classifications have been.

        Returns 0.0-1.0 where 1.0 = all recent decisions are the same regime.
        """
        lookback = self.config.persistence_lookback
        if not recent or lookback < 1:
            return 1.0

        relevant = recent[-lookback:]
        if not relevant:
            return 1.0

        regimes = [d.regime for d in relevant]
        most_common_count = Counter(regimes).most_common(1)[0][1]
        return most_common_count / len(regimes)
