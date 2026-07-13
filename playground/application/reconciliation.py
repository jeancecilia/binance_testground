"""Order reconciliation: startup and periodic reconciliation.

Startup sequence:
1. Load the latest local checkpoint.
2. Load locally known orders and positions.
3. Fetch Testnet open orders.
4. Fetch recent Testnet orders and fills.
5. Fetch Testnet balances and positions.
6. Compare local and exchange state.
7. Repair recoverable local inconsistencies.
8. Record unresolved mismatches.
9. Block new order submission if reconciliation fails.
10. Resume from the first unprocessed completed candle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from playground.domain.orders import (
    BrokerOrder, Fill, OrderSide, OrderStatus, OrderType,
)
from playground.domain.positions import Position
from playground.infrastructure.binance_testnet_broker import BinanceTestnetBroker
from playground.infrastructure.sqlite_repository import SQLiteRepository

logger = logging.getLogger(__name__)


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
    """Handles startup and periodic reconciliation between local state and Testnet."""

    def __init__(
        self,
        repository: SQLiteRepository,
        broker: BinanceTestnetBroker | None = None,
    ) -> None:
        self._repo = repository
        self._broker = broker

    # ------------------------------------------------------------------
    # Startup reconciliation
    # ------------------------------------------------------------------

    def startup_reconcile(self, symbol: str) -> ReconciliationResult:
        """Run the full startup reconciliation sequence.

        Returns ReconciliationResult indicating success/failure and
        whether new order submission is allowed.
        """
        result = ReconciliationResult(success=True)

        # Step 1-2: Load local state for this symbol only
        local_orders = self._repo.get_open_orders(symbol)
        local_positions = {p.symbol: p for p in self._repo.get_all_positions()}
        result.local_order_count = len(local_orders)

        if self._broker is None:
            # No broker = shadow mode, reconciliation not needed
            result.can_submit_orders = False
            return result

        try:
            # Step 3: Fetch Testnet open orders
            exchange_open_orders = self._broker.get_open_orders(symbol)
            result.exchange_order_count = len(exchange_open_orders)

            # Step 4: Fetch recent orders and fills
            exchange_recent_orders = self._broker.get_recent_orders(symbol, limit=50)
            exchange_trades = self._broker.get_recent_trades(symbol, limit=50)

            # Step 5: Fetch Testnet account info
            account_info = self._broker.get_account_info()

            # Step 6-8: Compare and repair
            mismatches = self._compare_orders(local_orders, exchange_open_orders, exchange_recent_orders)
            result.mismatches = mismatches

            if mismatches:
                repairs_done, unresolved = self._attempt_repairs(
                    mismatches, exchange_open_orders, exchange_recent_orders, exchange_trades,
                )
                result.repairs = repairs_done
                result.unresolved = unresolved

            # Step 8b: Reconstruct positions from Binance account data
            self._reconstruct_positions_from_binance(
                symbol, account_info, exchange_trades,
            )

            # Step 9: Re-query DB for UNKNOWN orders after repairs
            refreshed = self._repo.get_open_orders(symbol)
            remaining_unknowns = [
                o for o in refreshed
                if o.status in {OrderStatus.UNKNOWN, OrderStatus.PENDING_RECONCILIATION}
            ]
            if remaining_unknowns:
                result.unresolved.append(
                    f"{len(remaining_unknowns)} order(s) are UNKNOWN after repairs — trading blocked"
                )
            if result.unresolved:
                result.success = False
                result.can_submit_orders = False
                logger.error(
                    "Reconciliation has unresolved mismatches — blocking order submission",
                    extra={"unresolved": result.unresolved},
                )
            else:
                result.can_submit_orders = True
                logger.info(
                    "Reconciliation successful — order submission enabled",
                    extra={"repairs": len(result.repairs)},
                )

        except Exception as e:
            result.success = False
            result.can_submit_orders = False
            result.unresolved.append(f"Reconciliation error: {e}")
            logger.exception("Reconciliation failed with exception")

        return result

    # ------------------------------------------------------------------
    # Comparison logic
    # ------------------------------------------------------------------

    def _compare_orders(
        self,
        local_orders: list[BrokerOrder],
        exchange_open: list[dict],
        exchange_recent: list[dict],
    ) -> list[str]:
        """Compare local and exchange orders, return list of mismatch descriptions."""
        mismatches: list[str] = []

        local_by_client_id = {o.client_order_id: o for o in local_orders}
        exchange_by_client_id: Dict[str, dict] = {}
        for eo in exchange_recent:
            cid = eo.get("clientOrderId", "")
            if cid:
                exchange_by_client_id[cid] = eo
        for eo in exchange_open:
            cid = eo.get("clientOrderId", "")
            if cid:
                exchange_by_client_id[cid] = eo

        # Local orders not on exchange — flag ALL missing orders
        for cid, local in local_by_client_id.items():
            if cid not in exchange_by_client_id:
                mismatches.append(f"Local order {cid} ({local.status.value}) not found on exchange")

        # Exchange orders not locally
        for cid, exch in exchange_by_client_id.items():
            if cid not in local_by_client_id:
                mismatches.append(f"Exchange order {cid} not found locally")

        # Status mismatches
        for cid in set(local_by_client_id.keys()) & set(exchange_by_client_id.keys()):
            local = local_by_client_id[cid]
            exch = exchange_by_client_id[cid]
            exch_status = exch.get("status", "")
            status_map = {
                "NEW": OrderStatus.ACCEPTED,
                "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
                "FILLED": OrderStatus.FILLED,
                "CANCELED": OrderStatus.CANCELLED,
                "REJECTED": OrderStatus.REJECTED,
                "EXPIRED": OrderStatus.EXPIRED,
            }
            mapped = status_map.get(exch_status)
            if mapped and local.status != mapped:
                mismatches.append(
                    f"Status mismatch for {cid}: local={local.status.value}, exchange={exch_status}"
                )

        return mismatches

    # ------------------------------------------------------------------
    # Repair logic
    # ------------------------------------------------------------------

    def _attempt_repairs(
        self,
        mismatches: list[str],
        exchange_open: list[dict],
        exchange_recent: list[dict],
        exchange_trades: list[dict],
    ) -> Tuple[list[str], list[str]]:
        """Attempt to repair recoverable inconsistencies.

        Returns (repairs_made, unresolved).
        """
        repairs: list[str] = []
        unresolved: list[str] = []

        for mismatch in mismatches:
            if "not found on exchange" in mismatch:
                # Local order not on exchange — mark as UNKNOWN
                cid = mismatch.split(" ")[2]
                existing = self._repo.get_order(cid)
                self._repo.update_order_status(
                    cid,
                    existing.order_id if existing else "",
                    OrderStatus.UNKNOWN,
                    existing.executed_quantity if existing else 0.0,
                    existing.cummulative_quote_qty if existing else 0.0,
                    existing.avg_price,
                    existing.price,
                    {"reconciliation_note": "Order not found on exchange during reconciliation"},
                )
                repairs.append(f"Marked local order {cid} as UNKNOWN (not on exchange)")

            elif "not found locally" in mismatch:
                # Exchange has an order we don't know about
                cid = mismatch.split(" ")[2]
                exch_order = None
                for eo in exchange_recent + exchange_open:
                    if eo.get("clientOrderId") == cid:
                        exch_order = eo
                        break

                if exch_order:
                    side = OrderSide.BUY if exch_order.get("side") == "BUY" else OrderSide.SELL
                    order_type = OrderType.LIMIT if exch_order.get("type") == "LIMIT" else OrderType.MARKET
                    status_map = {
                        "NEW": OrderStatus.ACCEPTED,
                        "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
                        "FILLED": OrderStatus.FILLED,
                        "CANCELED": OrderStatus.CANCELLED,
                        "REJECTED": OrderStatus.REJECTED,
                        "EXPIRED": OrderStatus.EXPIRED,
                    }
                    status = status_map.get(exch_order.get("status", ""), OrderStatus.UNKNOWN)

                    # Use upsert: update if local order exists (e.g. UNKNOWN), insert otherwise
                    existing = self._repo.get_order(cid)
                    if existing:
                        self._repo.update_order_status(
                            cid,
                            str(exch_order.get("orderId", "")),
                            status,
                            float(exch_order.get("executedQty", 0)),
                            float(exch_order.get("cummulativeQuoteQty", 0)),
                            float(exch_order.get("price", 0)) if exch_order.get("price") else None,
                            float(exch_order.get("price", 0)) if exch_order.get("price") else None,
                            exch_order,
                        )
                        repairs.append(f"Updated local order {cid} from exchange data (was {existing.status.value})")
                    else:
                        broker_order = BrokerOrder(
                            order_id=str(exch_order.get("orderId", "")),
                            client_order_id=cid,
                            symbol=exch_order.get("symbol", ""),
                            side=side,
                            order_type=order_type,
                            quantity=float(exch_order.get("origQty", 0)),
                            price=float(exch_order.get("price", 0)) if exch_order.get("price") else None,
                            status=status,
                            executed_quantity=float(exch_order.get("executedQty", 0)),
                            cummulative_quote_qty=float(exch_order.get("cummulativeQuoteQty", 0)),
                            exchange_response=exch_order,
                        )
                        self._repo.insert_order(broker_order)
                        repairs.append(f"Created local order {cid} from exchange data")

            elif "Status mismatch" in mismatch:
                # Update local status to match exchange
                parts = mismatch.split(": ")[1]
                cid = parts.split(":")[0]
                exch_status_str = parts.split("exchange=")[1]

                status_map = {
                    "NEW": OrderStatus.ACCEPTED,
                    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
                    "FILLED": OrderStatus.FILLED,
                    "CANCELED": OrderStatus.CANCELLED,
                    "REJECTED": OrderStatus.REJECTED,
                    "EXPIRED": OrderStatus.EXPIRED,
                }
                new_status = status_map.get(exch_status_str, OrderStatus.UNKNOWN)

                exch_data = {}
                for eo in exchange_recent + exchange_open:
                    if eo.get("clientOrderId") == cid:
                        exch_data = eo
                        break

                self._repo.update_order_status(
                    cid,
                    str(exch_data.get("orderId", "")),
                    new_status,
                    float(exch_data.get("executedQty", 0)),
                    float(exch_data.get("cummulativeQuoteQty", 0)),
                    float(exch_data.get("price", 0)) if exch_data.get("price") else None,
                    float(exch_data.get("price", 0)) if exch_data.get("price") else None,
                    exch_data,
                )
                repairs.append(f"Updated local order {cid} status to {new_status.value}")

            else:
                unresolved.append(mismatch)

        return repairs, unresolved

    # ------------------------------------------------------------------
    # Position reconstruction from Binance
    # ------------------------------------------------------------------

    def _reconstruct_positions_from_binance(
        self, symbol: str, account_info: dict, trades: list[dict],
    ) -> None:
        """Reconstruct local positions from Binance account balances and trades.

        This is the authoritative source for Testnet position state.
        """
        from datetime import datetime

        norm_symbol = symbol.replace("/", "").replace("-", "")

        # Parse account balances
        balances = account_info.get("balances", [])
        base_asset = ""
        quote_asset = ""
        for common_quote in ["USDT", "USDC", "BUSD", "BTC", "ETH"]:
            if norm_symbol.endswith(common_quote):
                base_asset = norm_symbol[:-len(common_quote)]
                quote_asset = common_quote
                break

        if not base_asset:
            return

        base_balance = 0.0
        for b in balances:
            if b.get("asset") == base_asset:
                base_balance = float(b.get("free", 0)) + float(b.get("locked", 0))
                break

        # Calculate average entry from recent fills
        total_cost = 0.0
        total_qty = 0.0
        total_commission = 0.0
        for trade in trades:
            if trade.get("symbol") in (norm_symbol, symbol):
                qty = float(trade.get("qty", 0))
                price = float(trade.get("price", 0))
                commission = float(trade.get("commission", 0))
                total_cost += qty * price
                total_qty += qty
                total_commission += commission

                # Persist fills
                fill = Fill(
                    fill_id=str(trade.get("id", "")),
                    order_id=str(trade.get("orderId", "")),
                    client_order_id=trade.get("clientOrderId", ""),
                    symbol=symbol,
                    side=OrderSide.BUY if trade.get("isBuyer") else OrderSide.SELL,
                    quantity=qty,
                    price=price,
                    commission=commission,
                    commission_asset=trade.get("commissionAsset", ""),
                    filled_at=datetime.utcfromtimestamp(
                        trade.get("time", 0) / 1000.0
                    ),
                )
                self._repo.insert_fill(fill)

        # Upsert position
        if abs(base_balance) > 1e-10:
            avg_price = total_cost / total_qty if total_qty > 0 else 0.0
            position = Position(
                symbol=symbol,
                quantity=base_balance,
                avg_entry_price=avg_price,
                total_commission=total_commission,
            )
            self._repo.upsert_position(position)
