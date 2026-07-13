"""Regression tests for reconciliation and mark-to-market risk safety."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime

import pytest

from playground.application.reconciliation import ReconciliationEngine
from playground.core.risk_engine import RiskEngine
from playground.domain.positions import Position
from playground.domain.signals import (
    Direction,
    SignalId,
    SignalRejectionReason,
    StrategySignal,
)
from playground.infrastructure.binance_testnet_broker import BinanceTestnetBroker
from playground.infrastructure.configuration import BinanceConfig, RiskEngineConfig
from playground.infrastructure.sqlite_repository import SQLiteRepository


@pytest.fixture
def repository():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as file:
        path = file.name
    repo = SQLiteRepository(path)
    repo.connect()
    repo.migrate()
    try:
        yield repo
    finally:
        repo.close()
        os.unlink(path)


def trade(
    trade_id: int,
    qty: float,
    price: float,
    is_buyer: bool,
    time_ms: int,
    commission: float = 0.0,
    commission_asset: str = "USDT",
) -> dict:
    return {
        "symbol": "BNBUSDT",
        "id": trade_id,
        "orderId": 1000 + trade_id,
        "price": str(price),
        "qty": str(qty),
        "quoteQty": str(qty * price),
        "commission": str(commission),
        "commissionAsset": commission_asset,
        "time": time_ms,
        "isBuyer": is_buyer,
    }


def account_balance(quantity: float) -> dict:
    return {
        "balances": [
            {"asset": "BNB", "free": str(quantity), "locked": "0"},
            {"asset": "USDT", "free": "10000", "locked": "0"},
        ]
    }


def make_signal(symbol: str = "BNB-USDT") -> StrategySignal:
    signal_id = SignalId(
        strategy_id="test",
        strategy_version="v1",
        symbol=symbol,
        timeframe="1h",
        candle_timestamp=datetime(2026, 7, 13, 10, 0, 0),
        direction=Direction.LONG,
    )
    return StrategySignal(
        signal_id=signal_id,
        strategy_id="test",
        strategy_version="v1",
        symbol=symbol,
        timeframe="1h",
        candle_timestamp=signal_id.candle_timestamp,
        direction=Direction.LONG,
        regime="sideways_range",
        score=90.0,
        entry_price=300.0,
    )


class FakeBroker:
    def __init__(
        self,
        *,
        open_orders: list[dict] | None = None,
        recent_orders: list[dict] | None = None,
        trades: list[dict] | None = None,
        account: dict | None = None,
    ) -> None:
        self.open_orders = open_orders or []
        self.recent_orders = recent_orders or []
        self.trades = trades or []
        self.account = account or account_balance(0.0)

    def get_open_orders(self, symbol: str) -> list[dict]:
        return list(self.open_orders)

    def get_recent_orders(self, symbol: str, limit: int = 50) -> list[dict]:
        return list(self.recent_orders)

    def get_all_trades(self, symbol: str) -> list[dict]:
        return list(self.trades)

    def get_account_info(self) -> dict:
        return self.account


class TestPositionReconstruction:
    def test_buy_sell_ledger_preserves_remaining_cost_basis(self, repository):
        engine = ReconciliationEngine(repository)
        trades = [
            trade(1, 2.0, 100.0, True, 1000),
            trade(2, 1.0, 200.0, True, 2000),
            trade(3, 1.5, 300.0, False, 3000),
        ]

        engine._reconstruct_positions_from_binance(
            "BNB-USDT", account_balance(1.5), trades
        )

        position = repository.get_position("BNB-USDT")
        assert position is not None
        assert position.quantity == pytest.approx(1.5)
        assert position.avg_entry_price == pytest.approx(400.0 / 3.0)
        assert position.realized_pnl == pytest.approx(
            1.5 * (300.0 - 400.0 / 3.0)
        )

    def test_zero_balance_explicitly_closes_stale_position(self, repository):
        repository.upsert_position(
            Position(symbol="BNB-USDT", quantity=2.0, avg_entry_price=100.0)
        )
        engine = ReconciliationEngine(repository)
        trades = [
            trade(1, 2.0, 100.0, True, 1000),
            trade(2, 2.0, 150.0, False, 2000),
        ]

        engine._reconstruct_positions_from_binance(
            "BNB-USDT", account_balance(0.0), trades
        )

        position = repository.get_position("BNB-USDT")
        assert position is not None
        assert position.quantity == 0.0
        assert not position.is_open
        assert position.avg_entry_price == 0.0

    def test_unexplained_balance_blocks_startup(self, repository):
        broker = FakeBroker(account=account_balance(1.0), trades=[])
        engine = ReconciliationEngine(repository, broker=broker)

        result = engine.startup_reconcile("BNB-USDT")

        assert not result.success
        assert not result.can_submit_orders
        assert any("refusing unknown cost basis" in item for item in result.unresolved)

    def test_imported_exchange_order_uses_internal_symbol(self, repository):
        exchange_order = {
            "orderId": 22,
            "clientOrderId": "external-22",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "type": "MARKET",
            "status": "FILLED",
            "origQty": "1",
            "executedQty": "1",
            "cummulativeQuoteQty": "100",
            "price": "0",
        }
        broker = FakeBroker(
            recent_orders=[exchange_order],
            trades=[trade(1, 1.0, 100.0, True, 1000)],
            account=account_balance(1.0),
        )
        engine = ReconciliationEngine(repository, broker=broker)

        result = engine.startup_reconcile("BNB-USDT")

        assert result.success
        order = repository.get_order("external-22")
        assert order is not None
        assert order.symbol == "BNB-USDT"


class TestMarkToMarketRisk:
    def test_total_exposure_uses_current_mark_not_entry_cost(self):
        config = RiskEngineConfig(
            max_open_positions=5,
            max_positions_per_symbol=5,
            max_exposure_per_symbol_pct=1.0,
            max_total_exposure_pct=0.30,
            initial_balance_usdt=10000.0,
        )
        engine = RiskEngine(
            config=config,
            positions={
                "ETH-USDT": Position(
                    symbol="ETH-USDT", quantity=10.0, avg_entry_price=100.0
                )
            },
        )
        engine.update_market_price("ETH-USDT", 400.0)

        decision = engine.evaluate(
            make_signal(),
            current_price=300.0,
            spread_pct=0.05,
            market_depth_usdt=5000.0,
            estimated_slippage_pct=0.02,
        )

        assert not decision.approved
        assert (
            decision.rejection_reason
            == SignalRejectionReason.MAX_TOTAL_EXPOSURE_REACHED.value
        )

    def test_missing_mark_price_fails_closed(self):
        config = RiskEngineConfig(
            max_open_positions=5,
            max_positions_per_symbol=5,
            max_exposure_per_symbol_pct=1.0,
            max_total_exposure_pct=1.0,
        )
        engine = RiskEngine(
            config=config,
            positions={
                "ETH-USDT": Position(
                    symbol="ETH-USDT", quantity=1.0, avg_entry_price=100.0
                )
            },
        )

        decision = engine.evaluate(
            make_signal(),
            current_price=300.0,
            spread_pct=0.05,
            market_depth_usdt=5000.0,
            estimated_slippage_pct=0.02,
        )

        assert not decision.approved
        assert (
            decision.rejection_reason
            == SignalRejectionReason.DATA_CONTINUITY_BROKEN.value
        )
        assert "missing_mark_price:ETH-USDT" in decision.checks_failed

    def test_empty_snapshot_clears_positions(self):
        config = RiskEngineConfig(max_open_positions=1)
        engine = RiskEngine(
            config=config,
            positions={
                "ETH-USDT": Position(
                    symbol="ETH-USDT", quantity=1.0, avg_entry_price=100.0
                )
            },
        )
        engine.update_positions({})

        decision = engine.evaluate(
            make_signal(),
            current_price=300.0,
            spread_pct=0.05,
            market_depth_usdt=5000.0,
            estimated_slippage_pct=0.02,
        )

        assert decision.approved


def test_trade_history_paginates_from_zero(monkeypatch):
    broker = BinanceTestnetBroker(
        BinanceConfig(
            api_key="key",
            api_secret="secret",
            testnet_endpoint="https://testnet.binance.vision",
        )
    )
    broker._validated = True
    calls: list[int] = []

    def fake_request(method: str, endpoint: str, params: dict):
        assert method == "GET"
        calls.append(params["fromId"])
        if params["fromId"] == 0:
            return [
                trade(0, 1.0, 100.0, True, 1000),
                trade(1, 1.0, 110.0, True, 2000),
            ]
        if params["fromId"] == 2:
            return [trade(2, 1.0, 120.0, True, 3000)]
        return []

    monkeypatch.setattr(broker, "_signed_request", fake_request)

    trades = broker.get_all_trades("BNB-USDT", page_size=2)

    assert [int(item["id"]) for item in trades] == [0, 1, 2]
    assert calls == [0, 2]
