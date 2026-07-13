"""Execution pipeline: routes approved risk decisions to the broker.

Handles freshness/liquidity validation and order submission.
In shadow mode, signals are recorded but orders are NOT submitted.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from playground.domain.market import OrderBookSnapshot
from playground.domain.orders import (
    BrokerOrder, OrderRequest, OrderSide, OrderStatus, OrderType, RiskDecision,
    TimeInForce,
)
from playground.domain.positions import Position
from playground.domain.signals import (
    SignalRejectionReason, StrategySignal,
)
from playground.core.risk_engine import RiskEngine
from playground.infrastructure.binance_testnet_broker import BinanceTestnetBroker
from playground.infrastructure.configuration import RuntimeMode
from playground.infrastructure.sqlite_repository import SQLiteRepository

logger = logging.getLogger(__name__)


class ExecutionPipeline:
    """Routes signals through risk → freshness → execution.

    In shadow mode: evaluates risk but does NOT submit orders.
    In testnet mode: evaluates risk and submits approved orders.
    """

    def __init__(
        self,
        repository: SQLiteRepository,
        risk_engine: RiskEngine,
        broker: BinanceTestnetBroker | None = None,
        mode: RuntimeMode = RuntimeMode.SHADOW,
    ) -> None:
        self._repo = repository
        self._risk = risk_engine
        self._broker = broker
        self._mode = mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_signal(
        self, signal: StrategySignal,
        order_book: OrderBookSnapshot | None = None,
    ) -> Optional[RiskDecision]:
        """Process a strategy signal through risk and freshness validation.

        Returns RiskDecision (approved or rejected).
        In shadow mode, stops after risk decision.
        In testnet mode, submits approved orders to the broker.
        """
        # Estimate freshness/liquidity metrics
        spread_pct = order_book.spread_pct if order_book else 0.0
        depth = (
            order_book.depth_at_bid(5) * order_book.mid_price
            if order_book else 0.0
        )
        slippage = self._estimate_slippage(order_book)

        # Run risk evaluation
        decision = self._risk.evaluate(
            signal=signal,
            current_price=signal.entry_price or 0.0,
            spread_pct=spread_pct,
            market_depth_usdt=depth,
            estimated_slippage_pct=slippage,
        )

        # Persist the risk decision
        self._repo.insert_risk_decision(decision)

        if not decision.approved:
            logger.info(
                "Risk rejected", extra={
                    "signal_id": str(signal.signal_id),
                    "rejection_reason": decision.rejection_reason,
                    "checks_failed": list(decision.checks_failed),
                }
            )
            return decision

        logger.info(
            "Risk approved", extra={
                "signal_id": str(signal.signal_id),
                "position_size": decision.position_size,
            }
        )

        # In shadow mode, we stop here (no order submission)
        if self._mode == RuntimeMode.SHADOW:
            return decision

        # In testnet mode, submit the order
        if self._mode == RuntimeMode.TESTNET and self._broker is not None:
            return self._submit_order(signal, decision)

        return decision

    # ------------------------------------------------------------------
    # Order submission (Testnet only)
    # ------------------------------------------------------------------

    def _submit_order(
        self, signal: StrategySignal, decision: RiskDecision,
    ) -> RiskDecision:
        """Submit an approved order to the Testnet broker."""
        if self._broker is None:
            logger.error("Broker not configured for Testnet execution")
            return decision

        if decision.position_size is None or decision.position_size <= 0:
            logger.error(
                "Invalid position size", extra={
                    "signal_id": str(signal.signal_id),
                    "position_size": decision.position_size,
                }
            )
            return decision

        # Create order request with deterministic client_order_id
        order_request = OrderRequest(
            symbol=signal.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=decision.position_size,
            client_order_id=str(signal.signal_id),  # Deterministic ID
        )

        # Persist PENDING intent BEFORE transmission (protects against
        # connection drops where Binance receives the order but we never
        # see the response).
        pending_order = BrokerOrder(
            order_id="",
            client_order_id=order_request.client_order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            price=order_request.price,
            status=OrderStatus.PENDING,
        )
        self._repo.insert_order(pending_order)

        # Submit to broker
        broker_order = self._broker.submit_order(order_request)

        # Update with broker response
        self._repo.update_order_status(
            broker_order.client_order_id,
            broker_order.status,
            broker_order.executed_quantity,
            broker_order.avg_price,
            broker_order.exchange_response,
        )

        logger.info(
            "Order submitted to Testnet", extra={
                "client_order_id": broker_order.client_order_id,
                "exchange_order_id": broker_order.order_id,
                "status": broker_order.status.value,
                "executed_qty": broker_order.executed_quantity,
            }
        )

        return decision

    # ------------------------------------------------------------------
    # Freshness / liquidity checks
    # ------------------------------------------------------------------

    def validate_freshness(
        self, signal: StrategySignal,
        order_book: OrderBookSnapshot | None,
    ) -> Optional[SignalRejectionReason]:
        """Validate market freshness for a signal.

        Returns None if fresh, or a rejection reason if stale.
        """
        # Candle freshness: signal candle must not be too old
        now = datetime.utcnow()
        candle_age_seconds = (now - signal.candle_timestamp).total_seconds()

        # A 1h candle is valid for up to 1.5x its period
        max_age_map = {
            "15m": 15 * 60 * 1.5,
            "1h": 60 * 60 * 1.5,
            "4h": 240 * 60 * 1.5,
            "1d": 1440 * 60 * 1.5,
        }
        max_age = max_age_map.get(signal.timeframe, 3600)

        if candle_age_seconds > max_age:
            return SignalRejectionReason.CANDLE_STALE

        # Order book freshness
        if order_book is not None:
            ob_age = (now - order_book.timestamp).total_seconds()
            if ob_age > 60:  # Order book older than 60 seconds
                return SignalRejectionReason.ORDER_BOOK_STALE

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_slippage(order_book: OrderBookSnapshot | None) -> float:
        """Estimate slippage as a fraction of mid price."""
        if order_book is None:
            return 0.0
        mid = order_book.mid_price
        if mid == 0:
            return 0.0
        return order_book.spread / mid
