"""Validated application configuration. Secrets must not be stored here."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import os


class RuntimeMode(str, Enum):
    REPLAY = "replay"
    SHADOW = "shadow"
    TESTNET = "testnet"


@dataclass(frozen=True)
class MarketDataConfig:
    """Market data ingestion configuration."""

    symbols: tuple[str, ...] = ("BNB/USDT",)
    timeframes: tuple[str, ...] = ("1h", "15m")
    historical_candle_limit: int = 500
    polling_interval_seconds: int = 30
    max_retries: int = 3
    retry_backoff_base_seconds: float = 2.0
    retry_backoff_max_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.historical_candle_limit < 1:
            raise ValueError("historical_candle_limit must be >= 1")
        if self.polling_interval_seconds < 1:
            raise ValueError("polling_interval_seconds must be >= 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        for tf in self.timeframes:
            from playground.domain.market import Timeframe
            Timeframe(tf)  # validates


@dataclass(frozen=True)
class IndicatorEngineConfig:
    """Indicator engine configuration."""

    version: str = "1.0.0"
    sma_periods: tuple[int, ...] = (20, 50)
    ema_periods: tuple[int, ...] = (12, 26)
    rsi_period: int = 14
    atr_period: int = 14
    volatility_period: int = 20
    rolling_window: int = 20
    volume_period: int = 20
    trend_slope_period: int = 20
    minimum_history: int = 100

    def __post_init__(self) -> None:
        if self.minimum_history < 1:
            raise ValueError("minimum_history must be >= 1")
        for period in self.sma_periods + self.ema_periods:
            if period < 2:
                raise ValueError(f"Indicator periods must be >= 2, got {period}")


@dataclass(frozen=True)
class RegimeDetectorConfig:
    """Regime detector and stability overlay configuration."""

    version: str = "1.0.0"
    trend_slope_bull_threshold: float = 0.0005
    trend_slope_bear_threshold: float = -0.0005
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    range_max_volatility: float = 0.02
    range_max_atr_pct: float = 0.015
    range_min_volatility: float = 0.002
    high_volatility_threshold: float = 0.05
    crash_price_drop_threshold: float = -0.10
    confidence_threshold: float = 0.60
    uncertain_threshold: float = 0.30
    persistence_threshold: float = 0.50
    persistence_lookback: int = 10

    def __post_init__(self) -> None:
        if not (0 < self.confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in (0, 1], got {self.confidence_threshold}")
        if not (0 < self.uncertain_threshold <= self.confidence_threshold):
            raise ValueError(f"uncertain_threshold must be in (0, confidence_threshold]")


@dataclass(frozen=True)
class RiskEngineConfig:
    """Risk engine configuration."""

    version: str = "1.0.0"
    max_open_positions: int = 3
    max_positions_per_symbol: int = 1
    max_exposure_per_symbol_pct: float = 0.25
    max_total_exposure_pct: float = 0.50
    position_size_pct: float = 0.10
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.15
    entry_cooldown_seconds: int = 300
    max_spread_pct: float = 0.5
    min_market_depth_usdt: float = 1000.0
    max_estimated_slippage_pct: float = 0.1
    kill_switch: bool = False
    initial_balance_usdt: float = 10000.0

    def __post_init__(self) -> None:
        if self.max_open_positions < 1:
            raise ValueError("max_open_positions must be >= 1")
        if not (0 < self.position_size_pct <= 1.0):
            raise ValueError("position_size_pct must be in (0, 1]")


@dataclass(frozen=True)
class DatabaseConfig:
    """SQLite database configuration."""

    path: str = "playground.db"
    wal_mode: bool = True
    timeout_seconds: float = 5.0


@dataclass(frozen=True)
class BinanceConfig:
    """Binance API configuration. Secrets from environment variables."""

    public_endpoint: str = "https://api.binance.com"
    testnet_endpoint: str = "https://testnet.binance.vision"
    api_key: str = field(default_factory=lambda: os.environ.get("BINANCE_TESTNET_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.environ.get("BINANCE_TESTNET_API_SECRET", ""))
    recv_window_ms: int = 5000

    def __post_init__(self) -> None:
        if "testnet.binance.vision" not in self.testnet_endpoint:
            raise ValueError(
                f"Invalid testnet endpoint: {self.testnet_endpoint}. "
                "Must be a recognized Binance Testnet endpoint."
            )
        if not self.testnet_endpoint.startswith("https://"):
            raise ValueError("Testnet endpoint must use HTTPS")


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"  # json or text
    file_path: Optional[str] = "engine.log"


@dataclass(frozen=True)
class ReplayConfig:
    """Replay mode configuration."""

    dataset_path: str = ""
    dataset_identifier: str = ""
    fee_pct: float = 0.001
    slippage_pct: float = 0.0005
    random_seed: int = 42


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    mode: RuntimeMode = RuntimeMode.SHADOW
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    indicators: IndicatorEngineConfig = field(default_factory=IndicatorEngineConfig)
    regimes: RegimeDetectorConfig = field(default_factory=RegimeDetectorConfig)
    risk: RiskEngineConfig = field(default_factory=RiskEngineConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)

    def validate(self) -> list[str]:
        """Validate the configuration. Returns list of errors (empty = valid)."""
        errors: list[str] = []
        if not self.market_data.symbols:
            errors.append("At least one symbol must be configured")
        if not self.market_data.timeframes:
            errors.append("At least one timeframe must be configured")
        if self.mode == RuntimeMode.REPLAY and not self.replay.dataset_path:
            errors.append("Replay mode requires replay.dataset_path")
        if self.mode == RuntimeMode.TESTNET:
            if not self.binance.api_key:
                errors.append("Testnet mode requires BINANCE_TESTNET_API_KEY environment variable")
            if not self.binance.api_secret:
                errors.append("Testnet mode requires BINANCE_TESTNET_API_SECRET environment variable")
        return errors
