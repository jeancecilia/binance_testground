"""Standalone trading engine entry point."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import uuid
from typing import Optional

from playground.application.candle_coordinator import CandleCoordinator
from playground.application.execution_pipeline import ExecutionPipeline
from playground.application.market_pipeline import MarketPipeline
from playground.application.reconciliation import ReconciliationEngine
from playground.application.strategy_pipeline import StrategyPipeline
from playground.core.indicator_engine import IndicatorEngine
from playground.core.regime_detector import RegimeDetector
from playground.core.risk_engine import RiskEngine
from playground.core.specialist_registry import StrategyRegistry
from playground.core.stability_overlay import StabilityOverlay
from playground.domain.signals import StrategySignal
from playground.infrastructure.binance_market_data import BinanceMarketDataAdapter
from playground.infrastructure.binance_testnet_broker import BinanceTestnetBroker
from playground.infrastructure.configuration import AppConfig, DatabaseConfig, RuntimeMode
from playground.infrastructure.sqlite_repository import SQLiteRepository
from playground.infrastructure.system_clock import Clock, ReplayClock, SystemClock
from playground.replay.json_market_source import JsonMarketSource
from playground.replay.simulated_broker import SimulatedBroker, SimulatedBrokerConfig
from playground.strategies.bnb_rejection_specialist import BNBRejectionClusterSpecialist


def setup_logging(config: AppConfig) -> None:
    """Configure application logging."""
    level = getattr(logging, config.logging.level.upper(), logging.INFO)
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    if config.logging.file_path:
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[
                logging.FileHandler(config.logging.file_path),
                logging.StreamHandler(sys.stdout),
            ],
        )
    else:
        logging.basicConfig(level=level, format=log_format, stream=sys.stdout)


class TradingEngine:
    """Wire market, strategy, risk, persistence, and broker components."""

    def __init__(
        self,
        config: AppConfig,
        clock: Clock,
        repository: SQLiteRepository,
        market_source,
        broker=None,
    ) -> None:
        self.config = config
        self.clock = clock
        self.repository = repository
        self.market_source = market_source
        self.broker = broker
        self.run_id = str(uuid.uuid4())
        self._running = False

        self.indicator_engine = IndicatorEngine()
        self.regime_detector = RegimeDetector()
        self.stability_overlay = StabilityOverlay()

        self.registry = StrategyRegistry()
        self.registry.register(BNBRejectionClusterSpecialist())

        self.risk_engine = RiskEngine(config=config.risk, clock=clock)
        self.market_pipeline = MarketPipeline(
            repository=repository,
            market_adapter=(
                market_source
                if isinstance(market_source, BinanceMarketDataAdapter)
                else None
            ),
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
            broker=broker,
            mode=config.mode,
            clock=clock,
        )
        self.coordinator = CandleCoordinator(repository=repository, clock=clock)
        self.reconciliation = ReconciliationEngine(
            repository=repository,
            broker=(broker if isinstance(broker, BinanceTestnetBroker) else None),
        )

    def start(self) -> None:
        """Start the configured runtime mode."""
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
            from playground.domain.positions import EngineRun

            self.repository.insert_engine_run(
                EngineRun(
                    run_id=self.run_id,
                    mode=self.config.mode.value,
                    engine_version="0.1.0",
                )
            )

            if self.config.mode == RuntimeMode.TESTNET:
                for symbol in self.config.market_data.symbols:
                    result = self.reconciliation.startup_reconcile(
                        symbol.replace("/", "-")
                    )
                    if not result.can_submit_orders:
                        logger.error(
                            "Reconciliation failed; orders blocked",
                            extra={"unresolved": result.unresolved},
                        )
                        return
                self._load_positions_after_reconciliation()

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
        self._running = False

    def _shutdown(self) -> None:
        logger = logging.getLogger(__name__)
        logger.info("Engine shutting down", extra={"run_id": self.run_id})
        self.repository.update_engine_run(self.run_id, "completed")
        self.repository.close()

    def _run_replay(self) -> None:
        logger = logging.getLogger(__name__)
        if not isinstance(self.market_source, JsonMarketSource):
            logger.error("Replay mode requires JsonMarketSource")
            return

        self.market_source.load()
        logger.info(
            "Loaded %s candles from %s",
            self.market_source.candle_count,
            self.market_source.dataset_identifier or self.market_source.file_path,
        )
        for candle in self.market_source.get_all_candles():
            self.repository.insert_candle(candle)

        self.market_source.reset()
        processed = 0
        for symbol in self.config.market_data.symbols:
            for timeframe in self.config.market_data.timeframes:
                internal_symbol = symbol.replace("/", "-")
                candles = self.market_source.get_candles_for_symbol_timeframe(
                    internal_symbol, timeframe
                )
                for candle in candles:
                    if isinstance(self.clock, ReplayClock):
                        self.clock.advance_to(candle.close_time)
                    context = self._process_candle_internal(
                        internal_symbol, timeframe, candle
                    )
                    if context is not None:
                        processed += 1
                        if isinstance(self.broker, SimulatedBroker):
                            self.broker.update_unrealized_pnl(
                                {internal_symbol: candle.close}
                            )
                            self._sync_positions_to_risk_engine()

        logger.info(
            "Replay complete. Processed %s candles.",
            processed,
            extra={
                "total_pnl": (
                    self.broker.total_realized_pnl
                    + self.broker.total_unrealized_pnl
                    if isinstance(self.broker, SimulatedBroker)
                    else 0.0
                )
            },
        )

    def _run_live(self) -> None:
        logger = logging.getLogger(__name__)
        logger.info("Starting %s mode", self.config.mode.value)

        while self._running:
            for symbol in self.config.market_data.symbols:
                for timeframe in self.config.market_data.timeframes:
                    internal_symbol = symbol.replace("/", "-")
                    try:
                        self.market_pipeline.ingest_historical(
                            internal_symbol, timeframe
                        )
                        pending = self.coordinator.get_pending_candles(
                            internal_symbol,
                            timeframe,
                            self.config.mode.value,
                        )
                        for candle in pending:
                            context = self._process_candle_internal(
                                internal_symbol, timeframe, candle
                            )
                            if context is not None:
                                self.coordinator.update_checkpoint(
                                    self.run_id,
                                    internal_symbol,
                                    timeframe,
                                    self.config.mode.value,
                                    candle.open_time,
                                )
                    except Exception as exc:
                        logger.exception(
                            "Error processing %s@%s: %s",
                            internal_symbol,
                            timeframe,
                            exc,
                        )
            self.clock.sleep(self.config.market_data.polling_interval_seconds)

    def _process_candle_internal(
        self, symbol: str, timeframe: str, candle
    ) -> Optional[object]:
        logger = logging.getLogger(__name__)
        try:
            context = self.market_pipeline.process_candle(
                symbol, timeframe, candle
            )
            if context is None:
                return None

            results = self.strategy_pipeline.evaluate(context)
            order_book = self._get_order_book(symbol, candle)
            current_price = (
                order_book.best_ask
                if order_book is not None and order_book.best_ask > 0
                else candle.close
            )

            self.risk_engine.update_market_price(symbol, current_price)

            for _, result in results:
                if isinstance(result, StrategySignal):
                    decision = self.execution_pipeline.process_signal(
                        result,
                        order_book=order_book,
                        current_price=current_price,
                    )
                    if (
                        decision is not None
                        and decision.approved
                        and self.broker is not None
                    ):
                        self._sync_positions_to_risk_engine()
            return context
        except Exception as exc:
            logger.exception(
                "Pipeline error for %s@%s candle %s: %s",
                symbol,
                timeframe,
                candle.open_time,
                exc,
            )
            return None

    def _get_order_book(self, symbol: str, candle):
        """Route order-book data according to runtime mode."""
        if self.config.mode == RuntimeMode.REPLAY:
            return self._build_synthetic_order_book(symbol, candle)

        if self.config.mode == RuntimeMode.TESTNET:
            if isinstance(self.market_source, BinanceMarketDataAdapter):
                try:
                    return self.market_source.fetch_order_book(symbol, depth=5)
                except Exception as exc:
                    logging.getLogger(__name__).error(
                        "Testnet order book unavailable; failing closed: %s",
                        exc,
                        extra={"symbol": symbol},
                    )
                    return None
            logging.getLogger(__name__).error(
                "Testnet mode requires BinanceMarketDataAdapter"
            )
            return None

        if isinstance(self.market_source, BinanceMarketDataAdapter):
            try:
                return self.market_source.fetch_order_book(symbol, depth=5)
            except Exception:
                pass
        return self._build_synthetic_order_book(symbol, candle)

    @staticmethod
    def _build_synthetic_order_book(symbol: str, candle):
        from playground.domain.market import OrderBookSnapshot, Symbol

        mid = candle.close
        spread = max(mid * 0.0005, 0.01)
        best_bid = mid - spread / 2
        best_ask = mid + spread / 2
        bids = tuple(
            (best_bid - index * spread * 0.5, candle.volume * 0.2)
            for index in range(5)
        )
        asks = tuple(
            (best_ask + index * spread * 0.5, candle.volume * 0.2)
            for index in range(5)
        )
        return OrderBookSnapshot(
            symbol=Symbol(symbol),
            timestamp=candle.close_time,
            bids=bids,
            asks=asks,
        )

    def _sync_positions_to_risk_engine(self) -> None:
        """Replace risk-engine positions, including clearing to an empty set."""
        positions = {}
        if isinstance(self.broker, SimulatedBroker):
            positions = {
                symbol: position
                for symbol, position in self.broker.get_all_positions().items()
                if position.is_open
            }
        else:
            positions = {
                position.symbol: position
                for position in self.repository.get_all_positions()
                if position.is_open
            }
        self.risk_engine.update_positions(positions)

    def _load_positions_after_reconciliation(self) -> None:
        self._sync_positions_to_risk_engine()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Trading Engine Playground"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="shadow",
        choices=["replay", "shadow", "testnet"],
        help="Runtime mode",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Path to JSON dataset (replay mode)",
    )
    parser.add_argument(
        "--db", type=str, default="playground.db", help="SQLite database path"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = RuntimeMode(args.mode)
    config = AppConfig(
        mode=mode,
        database=DatabaseConfig(path=args.db),
        logging=AppConfig().logging,
    )
    object.__setattr__(config.logging, "level", args.log_level)
    if args.dataset:
        object.__setattr__(config.replay, "dataset_path", args.dataset)
        object.__setattr__(config.replay, "dataset_identifier", args.dataset)

    errors = config.validate()
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)

    setup_logging(config)
    logger = logging.getLogger(__name__)
    repository = SQLiteRepository(config.database.path)
    repository.connect()
    repository.migrate()
    logger.info("Database initialized at %s", config.database.path)

    if mode == RuntimeMode.REPLAY:
        if not args.dataset:
            logger.error("Replay mode requires --dataset")
            sys.exit(1)
        clock: Clock = ReplayClock()
        market_source = JsonMarketSource(args.dataset)
        broker = SimulatedBroker(SimulatedBrokerConfig())
    elif mode == RuntimeMode.SHADOW:
        clock = SystemClock()
        market_source = BinanceMarketDataAdapter()
        broker = None
    else:
        clock = SystemClock()
        market_source = BinanceMarketDataAdapter()
        broker = BinanceTestnetBroker()
        try:
            broker.validate_endpoint()
            logger.info("Testnet broker validated successfully")
        except Exception as exc:
            logger.error("Testnet broker validation failed: %s", exc)
            repository.close()
            sys.exit(1)

    engine = TradingEngine(
        config=config,
        clock=clock,
        repository=repository,
        market_source=market_source,
        broker=broker,
    )

    def signal_handler(signum, frame) -> None:
        del frame
        logger.info("Received signal %s, shutting down", signum)
        engine.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    engine.start()


if __name__ == "__main__":
    main()
