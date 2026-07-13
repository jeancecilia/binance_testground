"""Standalone trading engine entry point.

Supports three runtime modes:
- replay: Historical candles from JSON, simulated broker, deterministic
- shadow: Live Binance data, generates signals, no orders
- testnet: Live Binance data, submits approved orders to Testnet

Usage:
    python -m playground.standalone_run --mode shadow
    python -m playground.standalone_run --mode replay --dataset datasets/july.json
    python -m playground.standalone_run --mode testnet
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import uuid
from datetime import datetime
from typing import Optional

from playground.infrastructure.configuration import (
    AppConfig, BinanceConfig, DatabaseConfig, MarketDataConfig,
    RuntimeMode,
)
from playground.infrastructure.sqlite_repository import SQLiteRepository
from playground.infrastructure.system_clock import SystemClock, ReplayClock, Clock
from playground.infrastructure.binance_market_data import BinanceMarketDataAdapter
from playground.infrastructure.binance_testnet_broker import BinanceTestnetBroker

from playground.core.indicator_engine import IndicatorEngine
from playground.core.regime_detector import RegimeDetector
from playground.core.stability_overlay import StabilityOverlay
from playground.core.specialist_registry import StrategyRegistry
from playground.core.risk_engine import RiskEngine

from playground.strategies.bnb_rejection_specialist import BNBRejectionClusterSpecialist

from playground.application.candle_coordinator import CandleCoordinator
from playground.application.market_pipeline import MarketPipeline
from playground.application.strategy_pipeline import StrategyPipeline
from playground.application.execution_pipeline import ExecutionPipeline
from playground.application.reconciliation import ReconciliationEngine

from playground.replay.json_market_source import JsonMarketSource
from playground.replay.simulated_broker import SimulatedBroker, SimulatedBrokerConfig


# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------

def setup_logging(config) -> None:
    """Configure structured logging."""
    level = getattr(logging, config.logging.level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    if config.logging.file_path:
        logging.basicConfig(
            level=level,
            format=fmt,
            handlers=[
                logging.FileHandler(config.logging.file_path),
                logging.StreamHandler(sys.stdout),
            ],
        )
    else:
        logging.basicConfig(level=level, format=fmt, stream=sys.stdout)


# ------------------------------------------------------------------
# Engine builder
# ------------------------------------------------------------------

class TradingEngine:
    """Top-level trading engine that wires all components together."""

    def __init__(
        self,
        config: AppConfig,
        clock: Clock,
        repository: SQLiteRepository,
        market_source,  # BinanceMarketDataAdapter or JsonMarketSource
        broker=None,  # BinanceTestnetBroker or SimulatedBroker or None
    ) -> None:
        self.config = config
        self.clock = clock
        self.repository = repository
        self.market_source = market_source
        self.broker = broker
        self.run_id = str(uuid.uuid4())
        self._running = False

        # Build core components
        self.indicator_engine = IndicatorEngine()
        self.regime_detector = RegimeDetector()
        self.stability_overlay = StabilityOverlay()

        # Build strategies
        self.registry = StrategyRegistry()
        self.registry.register(BNBRejectionClusterSpecialist())

        # Build risk engine with real configuration and clock
        self.risk_engine = RiskEngine(
            config=config.risk,
            clock=clock,
        )

        # Build pipelines
        self.market_pipeline = MarketPipeline(
            repository=repository,
            market_adapter=market_source if isinstance(market_source, BinanceMarketDataAdapter) else None,
            indicator_engine=self.indicator_engine,
            regime_detector=self.regime_detector,
            stability_overlay=self.stability_overlay,
            clock=clock,
        )

        self.strategy_pipeline = StrategyPipeline(
            registry=self.registry,
            repository=repository,
        )

        self.execution_pipeline = ExecutionPipeline(
            repository=repository,
            risk_engine=self.risk_engine,
            broker=broker,  # Broker ABC — SimulatedBroker or BinanceTestnetBroker
            mode=config.mode,
            clock=clock,
        )

        self.coordinator = CandleCoordinator(
            repository=repository,
            clock=clock,
        )

        self.reconciliation = ReconciliationEngine(
            repository=repository,
            broker=broker if isinstance(broker, BinanceTestnetBroker) else None,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the trading engine."""
        logger = logging.getLogger(__name__)
        logger.info(
            "Engine starting",
            extra={
                "run_id": self.run_id,
                "mode": self.config.mode.value,
                "symbols": list(self.config.market_data.symbols),
                "timeframes": list(self.config.market_data.timeframes),
            },
        )

        self._running = True

        try:
            # Record engine run
            from playground.domain.positions import EngineRun
            self.repository.insert_engine_run(EngineRun(
                run_id=self.run_id,
                mode=self.config.mode.value,
                engine_version="0.1.0",
            ))

            # Reconciliation for Testnet mode
            if self.config.mode == RuntimeMode.TESTNET:
                for symbol in self.config.market_data.symbols:
                    sym_clean = symbol.replace("/", "-")
                    result = self.reconciliation.startup_reconcile(sym_clean)
                    if not result.can_submit_orders:
                        logger.error(
                            "Reconciliation failed — orders blocked",
                            extra={"unresolved": result.unresolved},
                        )
                        return
                # Load persisted positions into risk engine after reconciliation
                self._load_positions_after_reconciliation()

            # Run based on mode
            if self.config.mode == RuntimeMode.REPLAY:
                self._run_replay()
            else:
                self._run_live()

        except KeyboardInterrupt:
            logger.info("Engine interrupted by user")
        except Exception:
            logger.exception("Engine failed")
            raise
        finally:
            self._shutdown()

    def stop(self) -> None:
        """Gracefully stop the engine."""
        self._running = False

    def _shutdown(self) -> None:
        """Clean shutdown."""
        logger = logging.getLogger(__name__)
        logger.info("Engine shutting down", extra={"run_id": self.run_id})

        self.repository.update_engine_run(self.run_id, "completed")
        self.repository.close()

    # ------------------------------------------------------------------
    # Replay mode
    # ------------------------------------------------------------------

    def _run_replay(self) -> None:
        """Run in replay mode with JSON data and simulated broker."""
        logger = logging.getLogger(__name__)
        logger.info("Starting replay mode")

        if not isinstance(self.market_source, JsonMarketSource):
            logger.error("Replay mode requires JsonMarketSource")
            return

        self.market_source.load()
        logger.info(
            f"Loaded {self.market_source.candle_count} candles from "
            f"{self.market_source.dataset_identifier or self.market_source.file_path}"
        )

        # Load all candles into the repository
        for candle in self.market_source.get_all_candles():
            self.repository.insert_candle(candle)

        # Process each candle in chronological order
        self.market_source.reset()
        processed = 0

        for symbol in self.config.market_data.symbols:
            for timeframe in self.config.market_data.timeframes:
                sym = symbol.replace("/", "-")
                candles = self.market_source.get_candles_for_symbol_timeframe(sym, timeframe)

                for candle in candles:
                    if isinstance(self.clock, ReplayClock):
                        self.clock.advance_to(candle.close_time)

                    context = self._process_candle_internal(sym, timeframe, candle)
                    if context:
                        processed += 1

                        # Update simulated broker with current price
                        if isinstance(self.broker, SimulatedBroker):
                            self.broker.update_unrealized_pnl({sym: candle.close})

        logger.info(
            f"Replay complete. Processed {processed} candles.",
            extra={
                "total_pnl": (
                    self.broker.total_realized_pnl + self.broker.total_unrealized_pnl
                    if isinstance(self.broker, SimulatedBroker) else 0
                ),
            },
        )

    # ------------------------------------------------------------------
    # Live mode (shadow / testnet)
    # ------------------------------------------------------------------

    def _run_live(self) -> None:
        """Run in live mode (shadow or testnet)."""
        logger = logging.getLogger(__name__)
        logger.info(f"Starting {self.config.mode.value} mode")

        while self._running:
            for symbol in self.config.market_data.symbols:
                for timeframe in self.config.market_data.timeframes:
                    sym = symbol.replace("/", "-")

                    try:
                        # Ingest new candles
                        self.market_pipeline.ingest_historical(sym, timeframe)

                        # Get pending candles
                        pending = self.coordinator.get_pending_candles(
                            sym, timeframe, self.config.mode.value,
                        )

                        for candle in pending:
                            context = self._process_candle_internal(sym, timeframe, candle)

                            # Only advance checkpoint if processing succeeded
                            if context is not None:
                                self.coordinator.update_checkpoint(
                                    self.run_id, sym, timeframe,
                                    self.config.mode.value,
                                    candle.open_time,
                                )

                    except Exception as e:
                        logger.exception(
                            f"Error processing {sym}@{timeframe}: {e}"
                        )

            # Sleep for polling interval
            self.clock.sleep(self.config.market_data.polling_interval_seconds)

    # ------------------------------------------------------------------
    # Shared processing
    # ------------------------------------------------------------------

    def _process_candle_internal(
        self, symbol: str, timeframe: str, candle,
    ) -> Optional[object]:
        """Process a single candle through the full pipeline."""
        logger = logging.getLogger(__name__)

        try:
            # Market pipeline: indicators → regime → stability
            context = self.market_pipeline.process_candle(symbol, timeframe, candle)
            if context is None:
                return None

            # Strategy pipeline: evaluate strategies
            results = self.strategy_pipeline.evaluate(context)

            # Get order-book snapshot (mode-aware routing)
            order_book = self._get_order_book(symbol, candle)

            # Execution pipeline: risk → order submission
            current_price = candle.close
            for strategy, result in results:
                if hasattr(result, 'signal_id') and not hasattr(result, 'reason'):
                    decision = self.execution_pipeline.process_signal(
                        result, order_book=order_book, current_price=current_price,
                    )
                    # Update risk engine with position state after execution
                    if decision and decision.approved and self.broker is not None:
                        self._sync_positions_to_risk_engine()

            return context

        except Exception as e:
            logger.exception(
                f"Pipeline error for {symbol}@{timeframe} "
                f"candle {candle.open_time}: {e}"
            )
            return None

    def _get_order_book(self, symbol: str, candle):
        """Get order book snapshot, routing per runtime mode.

        - Replay: synthetic order book derived from candle data
        - Shadow: attempt real Binance depth; fall back to synthetic
        - Testnet: real Binance depth only; fail closed if unavailable
        """
        from playground.domain.market import Symbol, OrderBookSnapshot

        if self.config.mode == RuntimeMode.REPLAY:
            return self._build_synthetic_order_book(symbol, candle)

        if self.config.mode == RuntimeMode.TESTNET:
            # Testnet: real depth only
            if isinstance(self.market_source, BinanceMarketDataAdapter):
                try:
                    return self.market_source.fetch_order_book(symbol, depth=5)
                except Exception as e:
                    logger.error(
                        f"Testnet order book unavailable — failing closed: {e}",
                        extra={"symbol": symbol},
                    )
                    return None
            logger.error("Testnet mode requires BinanceMarketDataAdapter for order book")
            return None

        # Shadow: try real, fall back to synthetic
        if isinstance(self.market_source, BinanceMarketDataAdapter):
            try:
                return self.market_source.fetch_order_book(symbol, depth=5)
            except Exception:
                pass
        return self._build_synthetic_order_book(symbol, candle)

    @staticmethod
    def _build_synthetic_order_book(symbol: str, candle):
        """Build a synthetic order book from candle data for replay."""
        from playground.domain.market import Symbol, OrderBookSnapshot
        mid = candle.close
        spread = max(mid * 0.0005, 0.01)
        best_bid = mid - spread / 2
        best_ask = mid + spread / 2

        bids = tuple((best_bid - i * spread * 0.5, candle.volume * 0.2) for i in range(5))
        asks = tuple((best_ask + i * spread * 0.5, candle.volume * 0.2) for i in range(5))
        return OrderBookSnapshot(
            symbol=Symbol(symbol),
            timestamp=candle.close_time,
            bids=bids,
            asks=asks,
        )

    def _sync_positions_to_risk_engine(self):
        """Sync positions into the risk engine from broker or database."""
        positions = {}
        # Simulated broker: in-memory positions
        from playground.replay.simulated_broker import SimulatedBroker
        if isinstance(self.broker, SimulatedBroker):
            for sym, pos in self.broker.get_all_positions().items():
                positions[sym] = pos
        # Testnet / shadow: load from database
        if not positions:
            db_positions = self.repository.get_all_positions()
            for pos in db_positions:
                if pos.is_open:
                    positions[pos.symbol] = pos
        if positions:
            self.risk_engine.update_positions(positions)

    def _load_positions_after_reconciliation(self):
        """After reconciliation, load persisted positions into the risk engine."""
        self._sync_positions_to_risk_engine()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Trading Engine Playground",
    )
    parser.add_argument(
        "--mode", type=str, default="shadow",
        choices=["replay", "shadow", "testnet"],
        help="Runtime mode",
    )
    parser.add_argument(
        "--dataset", type=str, default="",
        help="Path to JSON dataset (replay mode)",
    )
    parser.add_argument(
        "--db", type=str, default="playground.db",
        help="SQLite database path",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build configuration
    mode = RuntimeMode(args.mode)
    config = AppConfig(
        mode=mode,
        database=DatabaseConfig(path=args.db),
        logging=AppConfig().logging,  # use defaults
    )
    # Override log level
    object.__setattr__(config.logging, 'level', args.log_level)

    # Wire --dataset into replay config
    if args.dataset:
        object.__setattr__(config.replay, 'dataset_path', args.dataset)
        object.__setattr__(config.replay, 'dataset_identifier', args.dataset)

    # Validate
    errors = config.validate()
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    setup_logging(config)
    logger = logging.getLogger(__name__)

    # Initialize repository
    repo = SQLiteRepository(config.database.path)
    repo.connect()
    repo.migrate()
    logger.info(f"Database initialized at {config.database.path}")

    # Build based on mode
    if mode == RuntimeMode.REPLAY:
        if not args.dataset:
            logger.error("Replay mode requires --dataset")
            sys.exit(1)

        clock = ReplayClock()
        market_source = JsonMarketSource(args.dataset)
        broker = SimulatedBroker(SimulatedBrokerConfig())

        engine = TradingEngine(
            config=config,
            clock=clock,
            repository=repo,
            market_source=market_source,
            broker=broker,
        )

    elif mode == RuntimeMode.SHADOW:
        clock = SystemClock()
        market_adapter = BinanceMarketDataAdapter()

        engine = TradingEngine(
            config=config,
            clock=clock,
            repository=repo,
            market_source=market_adapter,
            broker=None,
        )

    elif mode == RuntimeMode.TESTNET:
        clock = SystemClock()
        market_adapter = BinanceMarketDataAdapter()
        broker = BinanceTestnetBroker()

        # Validate the Testnet broker before starting
        try:
            broker.validate_endpoint()
            logger.info("Testnet broker validated successfully")
        except Exception as e:
            logger.error(f"Testnet broker validation failed: {e}")
            repo.close()
            sys.exit(1)

        engine = TradingEngine(
            config=config,
            clock=clock,
            repository=repo,
            market_source=market_adapter,
            broker=broker,
        )

    # Handle graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        engine.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start
    engine.start()


if __name__ == "__main__":
    main()
