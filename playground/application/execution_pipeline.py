"""Execution pipeline: routes approved risk decisions to the broker.

Handles freshness/liquidity validation and order submission.
In shadow mode, signals are recorded but orders are NOT submitted.
Supports both SimulatedBroker (replay) and BinanceTestnetBroker (testnet).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from playground.domain.market import OrderBookSnapshot
from playground.domain.orders import (
    Broker, BrokerOrder, OrderRequest, OrderSide, OrderStatus, OrderType,
    RiskDecision, TimeInForce,
)
from playground.domain.positions import Position
from playground.domain.signals import (
    SignalRejectionReason, StrategySignal,
)
from playground.core.risk_engine import RiskEngine
from playground.infrastructure.configuration import RuntimeMode
from playground.infrastructure.sqlite_repository import SQLiteRepository

logger = logging.getLogger(__name__)


class ExecutionPipeline:
    """Routes signals through risk → freshness → execution.

    In shadow mode: evaluates risk but does NOT submit orders.
    In testnet/replay mode: evaluates risk and submits approved orders.
    """

    def __init__(
        self,
        repository: SQLiteRepository,
        risk_engine: RiskEngine,
        broker: Broker | None = None,
        mode: RuntimeMode = RuntimeMode.SHADOW,
        clock = None,
    ) -> None:
        self._repo = repository
        self._risk = risk_engine
        self._broker = broker
        self._mode = mode
        self._clock = clock

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_signal(
        self, signal: StrategySignal,
        order_book: OrderBookSnapshot | None = None,
        current_price: float | None = None,
    ) -> Optional[RiskDecision]:
        """Process a strategy signal through risk and freshness validation.

        Args:
            signal: The strategy signal to process.
            order_book: Optional order book for liquidity/spread checks.
            current_price: Current market price (required for replay broker).

        Returns RiskDecision (approved or rejected).
        In shadow mode, stops after risk decision.
        In testnet/replay mode, submits approved orders to the broker.
        """
        # Validate freshness before risk evaluation
        freshness_rejection = self.validate_freshness(signal, order_book)
        if freshness_rejection is not None:
            decision = RiskDecision(
                signal_id=str(signal.signal_id),
                strategy_id=signal.strategy_id,
                symbol=signal.symbol,
                timeframe=signal.timeframe,
                candle_timestamp=signal.candle_timestamp,
                approved=False,
                rejection_reason=freshness_rejection.value,
                checks_failed=(freshness_rejection.value,),
            )
            self._repo.insert_risk_decision(decision)
            logger.info("Freshness rejected", extra={
                "signal_id": str(signal.signal_id),
                "reason": freshness_rejection.value,
            })
            return decision

        # Use provided current_price for risk sizing, fall back to entry_price
        price_for_risk = current_price if current_price and current_price > 0 else (signal.entry_price or 0.0)

        # Estimate freshness/liquidity metrics
        spread_pct = order_book.spread_pct if order_book else 0.0
        # BUY orders consume asks; use ask depth for liquidity validation
        depth = (
            order_book.depth_at_ask(5) * order_book.mid_price
            if order_book else 0.0
        )
        slippage = self._estimate_slippage(order_book)

        # Run risk evaluation with current market price
        decision = self._risk.evaluate(
            signal=signal,
            current_price=price_for_risk,
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

        # Submit the order (replay or testnet)
        if self._broker is not None:
            return self._submit_order(signal, decision, current_price or signal.entry_price or 0.0)

        return decision

    # ------------------------------------------------------------------
    # Order submission (Testnet / Replay)
    # ------------------------------------------------------------------

    def _submit_order(
        self, signal: StrategySignal, decision: RiskDecision, current_price: float,
    ) -> RiskDecision:
        """Submit an approved order through the broker."""
        if self._broker is None:
            logger.error("Broker not configured for execution")
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
            client_order_id=str(signal.signal_id),
        )

        # Persist PENDING intent BEFORE transmission
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
        inserted = self._repo.insert_order(pending_order)
        if not inserted:
            logger.error(
                "Duplicate order blocked — client_order_id already exists", extra={
                    "client_order_id": order_request.client_order_id,
                }
            )
            return RiskDecision(
                signal_id=str(signal.signal_id),
                strategy_id=signal.strategy_id,
                symbol=signal.symbol,
                timeframe=signal.timeframe,
                candle_timestamp=signal.candle_timestamp,
                approved=False,
                rejection_reason=SignalRejectionReason.DUPLICATE_SIGNAL.value,
                checks_failed=("duplicate_order",),
            )

        # Submit to broker (shared interface)
        broker_order = self._broker.submit_order(order_request, current_price)

        # Update with broker response (now includes order_id, cumm_quote_qty, price)
        self._repo.update_order_status(
            broker_order.client_order_id,
            broker_order.order_id,
            broker_order.status,
            broker_order.executed_quantity,
            broker_order.cummulative_quote_qty,
            broker_order.avg_price,
            broker_order.price,
            broker_order.exchange_response,
        )

        if broker_order.status == OrderStatus.UNKNOWN:
            logger.warning(
                "Order status UNKNOWN after submission — blocking further orders until reconciliation",
                extra={"client_order_id": broker_order.client_order_id},
            )
            # Engage kill switch until reconciliation resolves
            self._risk.engage_kill_switch()

        logger.info(
            "Order submitted", extra={
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

        Uses the injected clock so replay determinism is preserved.
        Returns None if fresh, or a rejection reason if stale.
        """
        now = self._clock.now() if self._clock else datetime.utcnow()
        candle_age_seconds = (now - signal.candle_timestamp).total_seconds()

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
            if ob_age > 60:
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
