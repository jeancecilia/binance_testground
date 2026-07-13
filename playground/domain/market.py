"""Pure domain models for market data. No infrastructure dependencies."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Symbol(str):
    """Normalized internal symbol representation."""

    def __new__(cls, value: str) -> "Symbol":
        instance = super().__new__(cls, value.upper().replace("/", "-"))
        return instance

    @property
    def exchange_format(self) -> str:
        """Return the symbol in exchange format (e.g. BNBUSDT)."""
        return self.replace("-", "")

    @property
    def display_format(self) -> str:
        """Return the symbol in display format (e.g. BNB/USDT)."""
        return self.replace("-", "/")

    def __repr__(self) -> str:
        return f"Symbol({self.display_format})"


class Timeframe(str):
    """Validated timeframe string (e.g. 1h, 15m, 4h, 1d)."""

    VALID_TIMEFRAMES = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}

    def __new__(cls, value: str) -> "Timeframe":
        if value not in cls.VALID_TIMEFRAMES:
            raise ValueError(f"Invalid timeframe: {value}. Must be one of {cls.VALID_TIMEFRAMES}")
        return super().__new__(cls, value)


class CandleSource(str, Enum):
    """Source of a candle record."""
    HISTORICAL = "historical"
    LIVE = "live"
    BACKFILL = "backfill"
    REPLAY = "replay"


@dataclass(frozen=True, slots=True)
class Candle:
    """Immutable OHLCV candle domain model.

    All numerical values are validated on construction.
    The frozen dataclass ensures immutability for the database layer.
    """

    symbol: Symbol
    timeframe: Timeframe
    open_time: datetime  # UTC candle open timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime  # UTC candle close timestamp
    quote_asset_volume: float
    number_of_trades: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float
    source: CandleSource = CandleSource.HISTORICAL
    ingested_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(
                f"High ({self.high}) must be >= low ({self.low}) for {self.symbol}@{self.open_time}"
            )
        if self.open < self.low or self.open > self.high:
            raise ValueError(
                f"Open ({self.open}) must be between low ({self.low}) and high ({self.high})"
            )
        if self.close < self.low or self.close > self.high:
            raise ValueError(
                f"Close ({self.close}) must be between low ({self.low}) and high ({self.high})"
            )
        if self.volume < 0:
            raise ValueError(f"Volume must be >= 0, got {self.volume}")
        if self.open_time >= self.close_time:
            raise ValueError(f"open_time ({self.open_time}) must be before close_time ({self.close_time})")

    @property
    def is_complete(self) -> bool:
        """A candle is complete if its close_time is in the past."""
        return self.close_time <= datetime.utcnow()

    @property
    def candle_range(self) -> float:
        """High - Low range."""
        return self.high - self.low

    @property
    def relative_range(self) -> float:
        """(High - Low) / Close, guard against zero close."""
        if self.close == 0:
            return 0.0
        return self.candle_range / self.close

    @property
    def direction(self) -> str:
        """'bull' if close >= open, else 'bear'."""
        return "bull" if self.close >= self.open else "bear"


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    """Immutable order-book snapshot."""

    symbol: Symbol
    timestamp: datetime
    bids: tuple[tuple[float, float], ...]  # (price, quantity) pairs, sorted best-to-worst
    asks: tuple[tuple[float, float], ...]  # (price, quantity) pairs, sorted best-to-worst
    ingested_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if not self.bids:
            raise ValueError("Bids must not be empty")
        if not self.asks:
            raise ValueError("Asks must not be empty")

    @property
    def best_bid(self) -> float:
        return self.bids[0][0]

    @property
    def best_ask(self) -> float:
        return self.asks[0][0]

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_pct(self) -> float:
        if self.mid_price == 0:
            return 0.0
        return (self.spread / self.mid_price) * 100.0

    def depth_at_bid(self, levels: int = 5) -> float:
        """Sum of quantity at top N bid levels."""
        return sum(q for _, q in self.bids[:levels])

    def depth_at_ask(self, levels: int = 5) -> float:
        """Sum of quantity at top N ask levels."""
        return sum(q for _, q in self.asks[:levels])


@dataclass(frozen=True, slots=True)
class IndicatorSnapshot:
    """Immutable snapshot of calculated indicators for a candle."""

    symbol: Symbol
    timeframe: Timeframe
    candle_timestamp: datetime
    indicator_version: str
    # Trend indicators
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    # Momentum
    rsi_14: Optional[float] = None
    # Volatility
    atr_14: Optional[float] = None
    realized_volatility_20: Optional[float] = None
    # Price action
    rolling_high_20: Optional[float] = None
    rolling_low_20: Optional[float] = None
    drawdown_20: Optional[float] = None
    trend_slope_20: Optional[float] = None
    # Volume
    average_volume_20: Optional[float] = None
    volume_ratio: Optional[float] = None
    # Candle metrics
    candle_range: Optional[float] = None
    relative_candle_range: Optional[float] = None
    # Metadata
    computed_at: datetime = field(default_factory=datetime.utcnow)
    history_available: int = 0  # number of candles available for computation


@dataclass(frozen=True, slots=True)
class MarketContext:
    """Prepared context passed to strategies.

    Contains everything a strategy needs to make a decision
    without accessing infrastructure.
    """

    symbol: Symbol
    timeframe: Timeframe
    candle: Candle
    indicators: IndicatorSnapshot
    regime: str  # stabilized regime label
    order_book: Optional[OrderBookSnapshot] = None
