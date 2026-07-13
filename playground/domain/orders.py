"""Domain models for orders, fills, and risk decisions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"


class TimeInForce(str, Enum):
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """Domain model for an order intent before submission."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # None for market orders
    client_order_id: str = ""
    time_in_force: TimeInForce = TimeInForce.GTC
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be > 0, got {self.quantity}")
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit orders require a price")
        if self.price is not None and self.price <= 0:
            raise ValueError(f"Price must be > 0, got {self.price}")


@dataclass(frozen=True, slots=True)
class BrokerOrder:
    """An order as tracked by the broker/system."""

    order_id: str  # exchange-assigned order ID
    client_order_id: str  # our deterministic ID
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    status: OrderStatus
    executed_quantity: float = 0.0
    cummulative_quote_qty: float = 0.0
    avg_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    exchange_response: dict = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.executed_quantity

    @property
    def is_active(self) -> bool:
        return self.status in {OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED}


@dataclass(frozen=True, slots=True)
class Fill:
    """A single trade/fill for an order."""

    fill_id: str
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float = 0.0
    commission_asset: str = ""
    filled_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Result of risk engine evaluation for a signal."""

    signal_id: str
    strategy_id: str
    symbol: str
    timeframe: str
    candle_timestamp: datetime
    approved: bool
    position_size: Optional[float] = None  # approved size, None if rejected
    rejection_reason: Optional[str] = None
    risk_config_version: str = "1.0.0"
    checks_passed: tuple[str, ...] = ()
    checks_failed: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=datetime.utcnow)
