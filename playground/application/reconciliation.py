"""Order reconciliation between local state and Binance Testnet."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Tuple

from playground.domain.market import Symbol
from playground.domain.orders import (
    BrokerOrder,
    Fill,
    OrderSide,
    OrderStatus,
    OrderType,
)
from playground.domain.positions import Position
from playground.infrastructure.binance_testnet_broker import BinanceTestnetBroker
from playground.infrastructure.sqlite_repository import SQLiteRepository

logger = logging.getLogger(__name__)


class PositionReconstructionError(RuntimeError):
    """Raised when exchange history cannot explain the reported balance."""


@dataclass
class ReconciliationResult:
    """Result of a reconciliation run."""

    success: bool
    local_order_count: int = 0
    exchange_order_count: int = 0
    mismatches: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    can_submit_orders: bool = False


class ReconciliationEngine:
    """Reconcile local orders, fills, and positions with Binance Testnet."""

    def __init__(
        self,
        repository: SQLiteRepository,
        broker: BinanceTestnetBroker | None = None,
    ) -> None:
        self._repo = repository
        self._broker = broker

    def startup_reconcile(self, symbol: str) -> ReconciliationResult:
        """Run startup reconciliation for one internal symbol."""
        result = ReconciliationResult(success=True)
        internal_symbol = str(Symbol(symbol))
        local_orders = self._repo.get_open_orders(internal_symbol)
        result.local_order_count = len(local_orders)

        if self._broker is None:
            result.can_submit_orders = False
            return result

        try:
            exchange_open_orders = self._broker.get_open_orders(internal_symbol)
            exchange_recent_orders = self._broker.get_recent_orders(
                internal_symbol, limit=50
            )
            get_all_trades = getattr(self._broker, "get_all_trades", None)
            if callable(get_all_trades):
                exchange_trades = get_all_trades(internal_symbol)
            else:
                exchange_trades = self._broker.get_recent_trades(
                    internal_symbol, limit=1000
                )
            account_info = self._broker.get_account_info()
            result.exchange_order_count = len(exchange_open_orders)

            mismatches = self._compare_orders(
                local_orders, exchange_open_orders, exchange_recent_orders
            )
            result.mismatches = mismatches

            if mismatches:
                repairs, unresolved = self._attempt_repairs(
                    internal_symbol,
                    mismatches,
                    exchange_open_orders,
                    exchange_recent_orders,
                )
                result.repairs = repairs
                result.unresolved = unresolved

            self._reconstruct_positions_from_binance(
                internal_symbol, account_info, exchange_trades
            )

            remaining_unknowns = [
                order
                for order in self._repo.get_open_orders(internal_symbol)
                if order.status
                in {OrderStatus.UNKNOWN, OrderStatus.PENDING_RECONCILIATION}
            ]
            if remaining_unknowns:
                result.unresolved.append(
                    f"{len(remaining_unknowns)} order(s) remain UNKNOWN after "
                    "reconciliation"
                )

            if result.unresolved:
                result.success = False
                result.can_submit_orders = False
                logger.error(
                    "Reconciliation has unresolved mismatches; blocking orders",
                    extra={"unresolved": result.unresolved},
                )
            else:
                result.can_submit_orders = True
                logger.info(
                    "Reconciliation successful; order submission enabled",
                    extra={"repairs": len(result.repairs)},
                )

        except Exception as exc:
            result.success = False
            result.can_submit_orders = False
            result.unresolved.append(f"Reconciliation error: {exc}")
            logger.exception("Reconciliation failed")

        return result

    def _compare_orders(
        self,
        local_orders: list[BrokerOrder],
        exchange_open: list[dict],
        exchange_recent: list[dict],
    ) -> list[str]:
        """Compare local and exchange order state."""
        mismatches: list[str] = []
        local_by_client_id = {o.client_order_id: o for o in local_orders}
        exchange_by_client_id: Dict[str, dict] = {}

        for order in exchange_recent + exchange_open:
            client_id = order.get("clientOrderId", "")
            if client_id:
                exchange_by_client_id[client_id] = order

        for client_id, local_order in local_by_client_id.items():
            if client_id not in exchange_by_client_id:
                mismatches.append(
                    f"Local order {client_id} ({local_order.status.value}) "
                    "not found on exchange"
                )

        for client_id in exchange_by_client_id:
            if client_id not in local_by_client_id:
                mismatches.append(
                    f"Exchange order {client_id} not found locally"
                )

        for client_id in set(local_by_client_id) & set(exchange_by_client_id):
            local_order = local_by_client_id[client_id]
            exchange_order = exchange_by_client_id[client_id]
            exchange_status = self._map_status(exchange_order.get("status", ""))
            if exchange_status is not None and local_order.status != exchange_status:
                mismatches.append(
                    f"Status mismatch for {client_id}: "
                    f"local={local_order.status.value}, "
                    f"exchange={exchange_order.get('status', '')}"
                )

        return mismatches

    def _attempt_repairs(
        self,
        internal_symbol: str,
        mismatches: list[str],
        exchange_open: list[dict],
        exchange_recent: list[dict],
    ) -> Tuple[list[str], list[str]]:
        """Repair recoverable order inconsistencies."""
        repairs: list[str] = []
        unresolved: list[str] = []
        exchange_orders = exchange_recent + exchange_open

        for mismatch in mismatches:
            if "not found on exchange" in mismatch:
                client_id = mismatch.split(" ")[2]
                existing = self._repo.get_order(client_id)
                if existing is None:
                    unresolved.append(mismatch)
                    continue
                self._repo.update_order_status(
                    client_id,
                    existing.order_id,
                    OrderStatus.UNKNOWN,
                    existing.executed_quantity,
                    existing.cummulative_quote_qty,
                    existing.avg_price,
                    existing.price,
                    {
                        "reconciliation_note": (
                            "Order not found on exchange during reconciliation"
                        )
                    },
                )
                repairs.append(
                    f"Marked local order {client_id} as UNKNOWN"
                )
                continue

            if "not found locally" in mismatch:
                client_id = mismatch.split(" ")[2]
                exchange_order = self._find_exchange_order(
                    client_id, exchange_orders
                )
                if exchange_order is None:
                    unresolved.append(mismatch)
                    continue

                mapped = self._broker_order_from_exchange(
                    internal_symbol, exchange_order
                )
                existing = self._repo.get_order(client_id)
                if existing is not None:
                    self._repo.update_order_status(
                        client_id,
                        mapped.order_id,
                        mapped.status,
                        mapped.executed_quantity,
                        mapped.cummulative_quote_qty,
                        mapped.avg_price,
                        mapped.price,
                        mapped.exchange_response,
                    )
                    repairs.append(
                        f"Updated local order {client_id} from exchange data"
                    )
                elif self._repo.insert_order(mapped):
                    repairs.append(
                        f"Created local order {client_id} from exchange data"
                    )
                else:
                    unresolved.append(
                        f"Could not persist exchange order {client_id}"
                    )
                continue

            if "Status mismatch" in mismatch:
                detail = mismatch.split(": ", 1)[1]
                client_id = detail.split(":", 1)[0]
                exchange_order = self._find_exchange_order(
                    client_id, exchange_orders
                )
                if exchange_order is None:
                    unresolved.append(mismatch)
                    continue
                mapped = self._broker_order_from_exchange(
                    internal_symbol, exchange_order
                )
                self._repo.update_order_status(
                    client_id,
                    mapped.order_id,
                    mapped.status,
                    mapped.executed_quantity,
                    mapped.cummulative_quote_qty,
                    mapped.avg_price,
                    mapped.price,
                    mapped.exchange_response,
                )
                repairs.append(
                    f"Updated local order {client_id} to {mapped.status.value}"
                )
                continue

            unresolved.append(mismatch)

        return repairs, unresolved

    @staticmethod
    def _find_exchange_order(
        client_id: str, orders: list[dict]
    ) -> dict | None:
        for order in orders:
            if order.get("clientOrderId") == client_id:
                return order
        return None

    @classmethod
    def _broker_order_from_exchange(
        cls, internal_symbol: str, exchange_order: dict
    ) -> BrokerOrder:
        status = cls._map_status(exchange_order.get("status", ""))
        if status is None:
            status = OrderStatus.UNKNOWN
        side = (
            OrderSide.BUY
            if exchange_order.get("side") == "BUY"
            else OrderSide.SELL
        )
        order_type = (
            OrderType.LIMIT
            if exchange_order.get("type") == "LIMIT"
            else OrderType.MARKET
        )
        executed_qty = float(exchange_order.get("executedQty", 0))
        cumulative_quote = float(
            exchange_order.get("cummulativeQuoteQty", 0)
        )
        avg_price = (
            cumulative_quote / executed_qty if executed_qty > 0 else None
        )
        raw_price = exchange_order.get("price")
        price = float(raw_price) if raw_price not in (None, "", "0", 0) else None

        return BrokerOrder(
            order_id=str(exchange_order.get("orderId", "")),
            client_order_id=exchange_order.get("clientOrderId", ""),
            symbol=str(Symbol(internal_symbol)),
            side=side,
            order_type=order_type,
            quantity=float(exchange_order.get("origQty", 0)),
            price=price,
            status=status,
            executed_quantity=executed_qty,
            cummulative_quote_qty=cumulative_quote,
            avg_price=avg_price,
            exchange_response=exchange_order,
        )

    @staticmethod
    def _map_status(raw_status: str) -> OrderStatus | None:
        return {
            "NEW": OrderStatus.ACCEPTED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }.get(raw_status)

    def _reconstruct_positions_from_binance(
        self,
        symbol: str,
        account_info: dict,
        trades: list[dict],
    ) -> None:
        """Rebuild one spot position from complete chronological trade history.

        The account balance and the trade ledger must agree. Any unexplained
        balance, deposit, withdrawal, incomplete history, or cross-pair asset
        usage causes reconciliation to fail closed instead of inventing a cost
        basis.
        """
        internal_symbol = str(Symbol(symbol))
        exchange_symbol = internal_symbol.replace("-", "")
        base_asset, quote_asset = self._split_symbol(exchange_symbol)
        account_quantity = self._account_balance(account_info, base_asset)
        relevant_trades = [
            trade
            for trade in trades
            if trade.get("symbol") in {exchange_symbol, internal_symbol}
        ]
        relevant_trades.sort(
            key=lambda trade: (
                int(trade.get("time", 0)),
                int(trade.get("id", 0)),
            )
        )

        inventory_qty = 0.0
        average_cost = 0.0
        realized_pnl = 0.0
        commission_quote = 0.0

        for trade in relevant_trades:
            qty = float(trade.get("qty", 0))
            price = float(trade.get("price", 0))
            quote_qty = float(trade.get("quoteQty", qty * price))
            commission = float(trade.get("commission", 0))
            commission_asset = trade.get("commissionAsset", "")
            is_buyer = bool(trade.get("isBuyer"))

            if qty <= 0 or price <= 0:
                raise PositionReconstructionError(
                    f"Invalid trade payload for {internal_symbol}: {trade}"
                )

            if commission_asset == quote_asset:
                commission_quote += commission
            elif commission_asset == base_asset:
                commission_quote += commission * price

            if is_buyer:
                acquired_qty = qty
                acquisition_cost = quote_qty
                if commission_asset == base_asset:
                    acquired_qty -= commission
                elif commission_asset == quote_asset:
                    acquisition_cost += commission

                if acquired_qty <= 0:
                    raise PositionReconstructionError(
                        f"Buy commission consumes full quantity for "
                        f"{internal_symbol}"
                    )
                new_qty = inventory_qty + acquired_qty
                average_cost = (
                    inventory_qty * average_cost + acquisition_cost
                ) / new_qty
                inventory_qty = new_qty
            else:
                disposed_qty = qty
                if commission_asset == base_asset:
                    disposed_qty += commission

                tolerance = max(1e-10, abs(inventory_qty) * 1e-9)
                if disposed_qty > inventory_qty + tolerance:
                    raise PositionReconstructionError(
                        f"Trade history for {internal_symbol} sells "
                        f"{disposed_qty} but only {inventory_qty} is explained"
                    )

                quote_commission = (
                    commission if commission_asset == quote_asset else 0.0
                )
                realized_pnl += qty * (price - average_cost) - quote_commission
                inventory_qty = max(0.0, inventory_qty - disposed_qty)
                if inventory_qty <= tolerance:
                    inventory_qty = 0.0
                    average_cost = 0.0

            fill = Fill(
                fill_id=f"{exchange_symbol}:{trade.get('id', '')}",
                order_id=str(trade.get("orderId", "")),
                client_order_id=trade.get("clientOrderId", ""),
                symbol=internal_symbol,
                side=OrderSide.BUY if is_buyer else OrderSide.SELL,
                quantity=qty,
                price=price,
                commission=commission,
                commission_asset=commission_asset,
                filled_at=datetime.utcfromtimestamp(
                    float(trade.get("time", 0)) / 1000.0
                ),
            )
            self._repo.insert_fill(fill)

        tolerance = max(1e-8, abs(account_quantity) * 1e-6)
        if abs(inventory_qty - account_quantity) > tolerance:
            raise PositionReconstructionError(
                f"Trade history explains {inventory_qty} {base_asset}, but "
                f"account reports {account_quantity}; refusing unknown cost basis"
            )

        existing = self._repo.get_position(internal_symbol)
        position = Position(
            symbol=internal_symbol,
            quantity=account_quantity if account_quantity > tolerance else 0.0,
            avg_entry_price=(average_cost if account_quantity > tolerance else 0.0),
            realized_pnl=realized_pnl,
            total_commission=commission_quote,
            opened_at=(existing.opened_at if existing else datetime.utcnow()),
        )
        self._repo.upsert_position(position)

    @staticmethod
    def _account_balance(account_info: dict, asset: str) -> float:
        for balance in account_info.get("balances", []):
            if balance.get("asset") == asset:
                return float(balance.get("free", 0)) + float(
                    balance.get("locked", 0)
                )
        return 0.0

    @staticmethod
    def _split_symbol(exchange_symbol: str) -> tuple[str, str]:
        for quote in ("USDT", "USDC", "BUSD", "TUSD", "DAI", "BTC", "ETH"):
            if exchange_symbol.endswith(quote) and len(exchange_symbol) > len(quote):
                return exchange_symbol[: -len(quote)], quote
        raise PositionReconstructionError(
            f"Cannot determine base/quote assets for {exchange_symbol}"
        )
