"""Simulated broker for replay and local testing.

Supports:
- Market orders
- Configurable fees and slippage
- Partial-fill simulation
- Position tracking
- Realized and unrealized PnL
- Deterministic execution with fixed seed
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from playground.domain.orders import (
    BrokerOrder, Fill, OrderRequest, OrderSide, OrderStatus, OrderType,
)
from playground.domain.positions import Position


@dataclass
class SimulatedBrokerConfig:
    """Configuration for the simulated broker."""

    fee_pct: float = 0.001  # 0.1% fee
    slippage_pct: float = 0.0005  # 0.05% slippage
    enable_partial_fills: bool = False
    partial_fill_pct: float = 0.5  # fill 50% when partial fill triggers
    random_seed: int = 42
    initial_balance_usdt: float = 10000.0


class SimulatedBroker:
    """In-memory simulated broker for replay and local testing.

    Deterministic when using a fixed random seed.
    Tracks positions, fills, and PnL.
    """

    def __init__(self, config: SimulatedBrokerConfig | None = None) -> None:
        self.config = config or SimulatedBrokerConfig()
        self._rng = random.Random(self.config.random_seed)
        self._orders: Dict[str, BrokerOrder] = {}
        self._fills: list[Fill] = []
        self._positions: Dict[str, Position] = {}
        self._balance: float = self.config.initial_balance_usdt
        self._fill_counter = 0

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_order(
        self, order: OrderRequest, current_price: float,
    ) -> BrokerOrder:
        """Submit a market order to the simulated broker.

        Args:
            order: The order request.
            current_price: Current market price for execution.

        Returns:
            BrokerOrder with execution results.
        """
        self._fill_counter += 1

        # Apply slippage
        if order.side == OrderSide.BUY:
            exec_price = current_price * (1.0 + self.config.slippage_pct)
        else:
            exec_price = current_price * (1.0 - self.config.slippage_pct)

        # Calculate fee
        notional = order.quantity * exec_price
        fee = notional * self.config.fee_pct

        # Partial fill simulation
        if self.config.enable_partial_fills:
            fill_ratio = self.config.partial_fill_pct
            fill_qty = order.quantity * fill_ratio
            status = OrderStatus.PARTIALLY_FILLED
        else:
            fill_qty = order.quantity
            status = OrderStatus.FILLED

        # Create order
        order_id = f"SIM-{self._fill_counter}"
        broker_order = BrokerOrder(
            order_id=order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=exec_price,
            status=status,
            executed_quantity=fill_qty,
            cummulative_quote_qty=fill_qty * exec_price,
            avg_price=exec_price,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self._orders[order.client_order_id] = broker_order

        # Create fill
        fill = Fill(
            fill_id=f"FILL-{self._fill_counter}",
            order_id=order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_qty,
            price=exec_price,
            commission=fee,
            commission_asset="USDT",
            filled_at=datetime.utcnow(),
        )
        self._fills.append(fill)

        # Update position
        self._update_position(order, fill_qty, exec_price, fee)

        return broker_order

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _update_position(
        self, order: OrderRequest, quantity: float, price: float, fee: float,
    ) -> None:
        """Update the position for a filled order."""
        symbol = order.symbol
        position = self._positions.get(symbol)

        if position is None:
            position = Position(
                symbol=symbol,
                quantity=0.0,
                avg_entry_price=0.0,
            )

        if order.side == OrderSide.BUY:
            # Increase long position
            total_cost = position.quantity * position.avg_entry_price + quantity * price + fee
            new_qty = position.quantity + quantity
            new_avg = total_cost / new_qty if new_qty > 0 else 0.0

            position = Position(
                symbol=symbol,
                quantity=new_qty,
                avg_entry_price=new_avg,
                realized_pnl=position.realized_pnl,
                total_commission=position.total_commission + fee,
                opened_at=position.opened_at,
                updated_at=datetime.utcnow(),
            )
        else:
            # Decrease position (sell)
            realized_pnl = quantity * (price - position.avg_entry_price) - fee
            new_qty = position.quantity - quantity

            position = Position(
                symbol=symbol,
                quantity=new_qty,
                avg_entry_price=position.avg_entry_price if new_qty > 0 else 0.0,
                realized_pnl=position.realized_pnl + realized_pnl,
                total_commission=position.total_commission + fee,
                opened_at=position.opened_at,
                updated_at=datetime.utcnow(),
            )

        self._positions[symbol] = position
        self._balance -= fee

    def update_unrealized_pnl(self, current_prices: Dict[str, float]) -> None:
        """Update unrealized PnL for all positions based on current prices."""
        for symbol, position in self._positions.items():
            if position.is_open and symbol in current_prices:
                price = current_prices[symbol]
                unrealized = position.quantity * (price - position.avg_entry_price)
                self._positions[symbol] = Position(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    avg_entry_price=position.avg_entry_price,
                    unrealized_pnl=unrealized,
                    realized_pnl=position.realized_pnl,
                    total_commission=position.total_commission,
                    opened_at=position.opened_at,
                    updated_at=datetime.utcnow(),
                )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    def get_order(self, client_order_id: str) -> Optional[BrokerOrder]:
        return self._orders.get(client_order_id)

    def get_fills(self, client_order_id: str) -> list[Fill]:
        return [f for f in self._fills if f.client_order_id == client_order_id]

    def get_all_fills(self) -> list[Fill]:
        return list(self._fills)

    @property
    def total_realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self._positions.values())

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def balance(self) -> float:
        return self._balance

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        """Reset broker state for a fresh replay run."""
        self._rng = random.Random(seed or self.config.random_seed)
        self._orders.clear()
        self._fills.clear()
        self._positions.clear()
        self._balance = self.config.initial_balance_usdt
        self._fill_counter = 0
