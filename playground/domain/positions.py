"""Domain models for positions and engine checkpoints."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass(frozen=True, slots=True)
class Position:
    """A tracked position for a symbol."""

    symbol: str
    quantity: float
    avg_entry_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    opened_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def side(self) -> str:
        return "long" if self.quantity > 0 else "short" if self.quantity < 0 else "flat"

    @property
    def is_open(self) -> bool:
        return abs(self.quantity) > 1e-10

    @property
    def notional_value(self) -> float:
        return abs(self.quantity) * self.avg_entry_price


@dataclass(frozen=True, slots=True)
class EngineCheckpoint:
    """Persisted checkpoint for restart recovery."""

    run_id: str
    symbol: str
    timeframe: str
    last_processed_candle: datetime  # timestamp of the last successfully processed candle
    mode: str  # replay, shadow, testnet
    engine_version: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def checkpoint_key(self) -> str:
        return f"{self.symbol}:{self.timeframe}:{self.mode}"


@dataclass(frozen=True, slots=True)
class EngineRun:
    """Record of an engine run session."""

    run_id: str
    mode: str
    engine_version: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    status: str = "running"  # running, completed, failed, interrupted
