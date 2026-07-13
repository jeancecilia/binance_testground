"""Unit tests for the risk engine."""

from datetime import datetime, timedelta
import pytest

from playground.domain.orders import RiskDecision
from playground.domain.positions import Position
from playground.domain.signals import Direction, SignalId, SignalRejectionReason, StrategySignal
from playground.infrastructure.configuration import RiskEngineConfig
from playground.core.risk_engine import RiskEngine


def make_signal(
    signal_id_str: str = "test:v1:BNB-USDT:1h:2026-07-13T10:00:00Z:long",
    symbol: str = "BNB-USDT",
    entry_price: float = 300.0,
) -> StrategySignal:
    sid = SignalId.parse(signal_id_str)
    return StrategySignal(
        signal_id=sid,
        strategy_id=sid.strategy_id,
        strategy_version=sid.strategy_version,
        symbol=symbol,
        timeframe=sid.timeframe,
        candle_timestamp=sid.candle_timestamp,
        direction=Direction.LONG,
        regime="sideways_range",
        score=85.0,
        entry_price=entry_price,
    )


# Helper: default "safe" market params that pass all liquidity checks
def safe_market():
    return dict(
        spread_pct=0.05,
        market_depth_usdt=5000.0,
        estimated_slippage_pct=0.02,
    )


class TestRiskEngine:
    def test_approves_normal_signal(self):
        engine = RiskEngine()
        signal = make_signal()
        decision = engine.evaluate(signal, current_price=300.0, **safe_market())
        assert decision.approved
        assert decision.position_size is not None
        assert decision.position_size > 0

    def test_kill_switch_rejects(self):
        config = RiskEngineConfig(kill_switch=True)
        engine = RiskEngine(config=config)
        signal = make_signal()
        decision = engine.evaluate(signal, current_price=300.0, **safe_market())
        assert not decision.approved
        assert decision.rejection_reason == SignalRejectionReason.KILL_SWITCH_ENGAGED.value

    def test_duplicate_signal_rejects(self):
        engine = RiskEngine()
        signal = make_signal()
        # First signal passes
        decision1 = engine.evaluate(signal, current_price=300.0, **safe_market())
        assert decision1.approved
        # Same signal again should be rejected
        decision2 = engine.evaluate(signal, current_price=300.0, **safe_market())
        assert not decision2.approved
        assert decision2.rejection_reason == SignalRejectionReason.DUPLICATE_SIGNAL.value

    def test_max_open_positions_rejects(self):
        config = RiskEngineConfig(max_open_positions=1)
        positions = {
            "ETH-USDT": Position(
                symbol="ETH-USDT", quantity=1.0, avg_entry_price=2500.0,
            ),
        }
        engine = RiskEngine(config=config, positions=positions)
        signal = make_signal(symbol="BNB-USDT")
        decision = engine.evaluate(signal, current_price=300.0, **safe_market())
        assert not decision.approved
        assert decision.rejection_reason == SignalRejectionReason.MAX_OPEN_POSITIONS_REACHED.value

    def test_max_positions_per_symbol_rejects(self):
        config = RiskEngineConfig(max_positions_per_symbol=1)
        positions = {
            "BNB-USDT": Position(
                symbol="BNB-USDT", quantity=1.0, avg_entry_price=300.0,
            ),
        }
        engine = RiskEngine(config=config, positions=positions)
        signal = make_signal(symbol="BNB-USDT")
        decision = engine.evaluate(signal, current_price=300.0, **safe_market())
        assert not decision.approved
        assert decision.rejection_reason == SignalRejectionReason.MAX_POSITIONS_PER_SYMBOL_REACHED.value

    def test_max_exposure_rejects(self):
        config = RiskEngineConfig(
            max_positions_per_symbol=5,  # Allow multiple positions so exposure check runs
            max_exposure_per_symbol_pct=0.10,
            initial_balance_usdt=10000.0,
        )
        # Position worth ~2001 > 10% of 10000
        positions = {
            "BNB-USDT": Position(
                symbol="BNB-USDT", quantity=6.67, avg_entry_price=300.0,
            ),
        }
        engine = RiskEngine(config=config, positions=positions)
        signal = make_signal(symbol="BNB-USDT")
        decision = engine.evaluate(signal, current_price=300.0, **safe_market())
        assert not decision.approved
        assert decision.rejection_reason == SignalRejectionReason.MAX_EXPOSURE_PER_SYMBOL_REACHED.value

    def test_spread_too_wide_rejects(self):
        config = RiskEngineConfig(max_spread_pct=0.1)
        engine = RiskEngine(config=config)
        signal = make_signal()
        decision = engine.evaluate(
            signal, current_price=300.0,
            spread_pct=0.5, market_depth_usdt=5000.0, estimated_slippage_pct=0.02,
        )
        assert not decision.approved
        assert decision.rejection_reason == SignalRejectionReason.SPREAD_TOO_WIDE.value

    def test_position_sizing(self):
        config = RiskEngineConfig(
            position_size_pct=0.10,
            initial_balance_usdt=10000.0,
        )
        engine = RiskEngine(config=config)
        signal = make_signal()
        decision = engine.evaluate(signal, current_price=300.0, **safe_market())
        assert decision.approved
        # 10% of 10000 = 1000 USDT / 300 = 3.333...
        assert decision.position_size == pytest.approx(1000.0 / 300.0, rel=0.01)

    def test_all_checks_recorded(self):
        engine = RiskEngine()
        signal = make_signal()
        decision = engine.evaluate(signal, current_price=300.0, **safe_market())
        assert decision.approved
        assert len(decision.checks_passed) > 0
        assert len(decision.checks_failed) == 0
        assert "unique_signal" in decision.checks_passed
        assert "position_size_ok" in decision.checks_passed

    def test_insufficient_depth_rejects(self):
        engine = RiskEngine()
        signal = make_signal()
        decision = engine.evaluate(
            signal, current_price=300.0,
            spread_pct=0.05, market_depth_usdt=100.0, estimated_slippage_pct=0.02,
        )
        assert not decision.approved
        assert decision.rejection_reason == SignalRejectionReason.INSUFFICIENT_DEPTH.value
