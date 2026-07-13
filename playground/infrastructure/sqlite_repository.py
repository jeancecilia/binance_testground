"""SQLite repository with full schema and migrations.

Handles persistence for all domain tables:
candles, order_book_snapshots, indicator_snapshots, regime_decisions,
strategy_evaluations, signals, signal_rejections, risk_decisions,
orders, fills, positions, engine_checkpoints, engine_runs.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Optional, Sequence

from playground.domain.market import Candle, CandleSource, OrderBookSnapshot, Symbol, Timeframe
from playground.domain.indicators import IndicatorConfig
from playground.domain.regimes import RegimeDecision, StabilityDecision, RegimeConfig
from playground.domain.signals import StrategySignal, SignalRejection
from playground.domain.orders import (
    BrokerOrder, Fill, OrderRequest, OrderSide, OrderStatus,
    OrderType, RiskDecision, TimeInForce,
)
from playground.domain.positions import EngineCheckpoint, EngineRun, Position


SCHEMA_VERSION = 1

MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS candles (
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        open_time TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL NOT NULL,
        close_time TEXT NOT NULL,
        quote_asset_volume REAL NOT NULL,
        number_of_trades INTEGER NOT NULL,
        taker_buy_base_volume REAL NOT NULL,
        taker_buy_quote_volume REAL NOT NULL,
        source TEXT NOT NULL DEFAULT 'historical',
        ingested_at TEXT NOT NULL,
        PRIMARY KEY (symbol, timeframe, open_time)
    );

    CREATE TABLE IF NOT EXISTS order_book_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        bids TEXT NOT NULL,
        asks TEXT NOT NULL,
        ingested_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_obs_symbol_ts ON order_book_snapshots(symbol, timestamp);

    CREATE TABLE IF NOT EXISTS indicator_snapshots (
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        candle_timestamp TEXT NOT NULL,
        indicator_version TEXT NOT NULL,
        sma_20 REAL,
        sma_50 REAL,
        ema_12 REAL,
        ema_26 REAL,
        rsi_14 REAL,
        atr_14 REAL,
        realized_volatility_20 REAL,
        rolling_high_20 REAL,
        rolling_low_20 REAL,
        drawdown_20 REAL,
        trend_slope_20 REAL,
        average_volume_20 REAL,
        volume_ratio REAL,
        candle_range REAL,
        relative_candle_range REAL,
        computed_at TEXT NOT NULL,
        history_available INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (symbol, timeframe, candle_timestamp, indicator_version)
    );

    CREATE TABLE IF NOT EXISTS regime_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        candle_timestamp TEXT NOT NULL,
        regime TEXT NOT NULL,
        indicator_values TEXT NOT NULL,
        applied_thresholds TEXT NOT NULL,
        config_version TEXT NOT NULL,
        decision_reason TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_rd_symbol_ts ON regime_decisions(symbol, timeframe, candle_timestamp);

    CREATE TABLE IF NOT EXISTS stability_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        candle_timestamp TEXT NOT NULL,
        raw_regime TEXT NOT NULL,
        final_regime TEXT NOT NULL,
        confidence_score REAL NOT NULL,
        persistence_score REAL NOT NULL,
        recent_regime_consistency REAL NOT NULL,
        decision_reason TEXT NOT NULL,
        stability_config_version TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_sd_symbol_ts ON stability_decisions(symbol, timeframe, candle_timestamp);

    CREATE TABLE IF NOT EXISTS strategy_evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_id TEXT NOT NULL,
        strategy_version TEXT NOT NULL,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        candle_timestamp TEXT NOT NULL,
        direction TEXT NOT NULL,
        regime TEXT NOT NULL,
        result_type TEXT NOT NULL,  -- 'signal' or 'rejection'
        score REAL,
        signal_id TEXT,
        rejection_reason TEXT,
        detail TEXT,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_se_signal ON strategy_evaluations(signal_id);
    CREATE INDEX IF NOT EXISTS idx_se_strategy_ts ON strategy_evaluations(strategy_id, candle_timestamp);

    CREATE TABLE IF NOT EXISTS signals (
        signal_id TEXT PRIMARY KEY,
        strategy_id TEXT NOT NULL,
        strategy_version TEXT NOT NULL,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        candle_timestamp TEXT NOT NULL,
        direction TEXT NOT NULL,
        regime TEXT NOT NULL,
        score REAL NOT NULL,
        entry_price REAL,
        stop_loss REAL,
        take_profit REAL,
        metadata TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS signal_rejections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_id TEXT NOT NULL,
        strategy_version TEXT NOT NULL,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        candle_timestamp TEXT NOT NULL,
        direction TEXT NOT NULL,
        reason TEXT NOT NULL,
        detail TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_sr_strategy_ts ON signal_rejections(strategy_id, candle_timestamp);

    CREATE TABLE IF NOT EXISTS risk_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id TEXT NOT NULL,
        strategy_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        candle_timestamp TEXT NOT NULL,
        approved INTEGER NOT NULL,
        position_size REAL,
        rejection_reason TEXT,
        risk_config_version TEXT NOT NULL,
        checks_passed TEXT NOT NULL DEFAULT '[]',
        checks_failed TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_risk_signal ON risk_decisions(signal_id);

    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT,
        client_order_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        order_type TEXT NOT NULL,
        quantity REAL NOT NULL,
        price REAL,
        status TEXT NOT NULL DEFAULT 'PENDING',
        executed_quantity REAL NOT NULL DEFAULT 0.0,
        cummulative_quote_qty REAL NOT NULL DEFAULT 0.0,
        avg_price REAL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        exchange_response TEXT NOT NULL DEFAULT '{}'
    );

    CREATE TABLE IF NOT EXISTS fills (
        fill_id TEXT PRIMARY KEY,
        order_id TEXT NOT NULL,
        client_order_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        quantity REAL NOT NULL,
        price REAL NOT NULL,
        commission REAL NOT NULL DEFAULT 0.0,
        commission_asset TEXT NOT NULL DEFAULT '',
        filled_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_fills_order ON fills(client_order_id);

    CREATE TABLE IF NOT EXISTS positions (
        symbol TEXT PRIMARY KEY,
        quantity REAL NOT NULL DEFAULT 0.0,
        avg_entry_price REAL NOT NULL DEFAULT 0.0,
        unrealized_pnl REAL NOT NULL DEFAULT 0.0,
        realized_pnl REAL NOT NULL DEFAULT 0.0,
        total_commission REAL NOT NULL DEFAULT 0.0,
        opened_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS engine_checkpoints (
        run_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        last_processed_candle TEXT NOT NULL,
        mode TEXT NOT NULL,
        engine_version TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (symbol, timeframe, mode)
    );

    CREATE TABLE IF NOT EXISTS engine_runs (
        run_id TEXT PRIMARY KEY,
        mode TEXT NOT NULL,
        engine_version TEXT NOT NULL,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        status TEXT NOT NULL DEFAULT 'running'
    );

    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY
    );

    INSERT OR IGNORE INTO schema_version (version) VALUES (1);
    """,
}


class SQLiteRepository:
    """SQLite-backed persistence for all domain records."""

    def __init__(self, db_path: str = "playground.db") -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Repository not connected. Call connect() first.")
        return self._conn

    def migrate(self) -> None:
        """Apply migrations in order."""
        current = self._get_schema_version()
        for version in sorted(MIGRATIONS.keys()):
            if version > current:
                self.conn.executescript(MIGRATIONS[version])
                self.conn.execute(
                    "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                    (version,),
                )
                self.conn.commit()

    def _get_schema_version(self) -> int:
        try:
            row = self.conn.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()
            return row[0] if row and row[0] is not None else 0
        except sqlite3.OperationalError:
            return 0

    # ------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------

    def insert_candle(self, candle: Candle) -> bool:
        """Insert a candle. Returns True if inserted, False if duplicate."""
        try:
            self.conn.execute(
                """INSERT INTO candles (
                    symbol, timeframe, open_time, open, high, low, close,
                    volume, close_time, quote_asset_volume, number_of_trades,
                    taker_buy_base_volume, taker_buy_quote_volume, source, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(candle.symbol),
                    str(candle.timeframe),
                    candle.open_time.isoformat(),
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                    candle.close_time.isoformat(),
                    candle.quote_asset_volume,
                    candle.number_of_trades,
                    candle.taker_buy_base_volume,
                    candle.taker_buy_quote_volume,
                    candle.source.value,
                    candle.ingested_at.isoformat(),
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[Candle]:
        """Fetch candles ordered by open_time ascending."""
        query = "SELECT * FROM candles WHERE symbol = ? AND timeframe = ?"
        params: list = [symbol, timeframe]

        if start_time:
            query += " AND open_time >= ?"
            params.append(start_time.isoformat())
        if end_time:
            query += " AND open_time <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY open_time ASC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_candle(r) for r in rows]

    def candle_exists(self, symbol: str, timeframe: str, open_time: datetime) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM candles WHERE symbol = ? AND timeframe = ? AND open_time = ?",
            (symbol, timeframe, open_time.isoformat()),
        ).fetchone()
        return row is not None

    def get_latest_candle_time(self, symbol: str, timeframe: str) -> Optional[datetime]:
        row = self.conn.execute(
            "SELECT MAX(open_time) FROM candles WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe),
        ).fetchone()
        if row and row[0]:
            return datetime.fromisoformat(row[0])
        return None

    def find_missing_intervals(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> list[datetime]:
        """Return open_times between start and end that have no candle."""
        from datetime import timedelta

        tf_minutes = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360,
            "8h": 480, "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080,
        }
        delta = timedelta(minutes=tf_minutes.get(timeframe, 60))

        existing_rows = self.conn.execute(
            "SELECT open_time FROM candles WHERE symbol = ? AND timeframe = ? AND open_time >= ? AND open_time <= ?",
            (symbol, timeframe, start.isoformat(), end.isoformat()),
        ).fetchall()
        existing = {datetime.fromisoformat(r[0]) for r in existing_rows}

        missing: list[datetime] = []
        current = start
        while current <= end:
            if current not in existing:
                missing.append(current)
            current += delta
        return missing

    @staticmethod
    def _row_to_candle(row: sqlite3.Row) -> Candle:
        return Candle(
            symbol=Symbol(row["symbol"]),
            timeframe=Timeframe(row["timeframe"]),
            open_time=datetime.fromisoformat(row["open_time"]),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            close_time=datetime.fromisoformat(row["close_time"]),
            quote_asset_volume=row["quote_asset_volume"],
            number_of_trades=row["number_of_trades"],
            taker_buy_base_volume=row["taker_buy_base_volume"],
            taker_buy_quote_volume=row["taker_buy_quote_volume"],
            source=CandleSource(row["source"]),
            ingested_at=datetime.fromisoformat(row["ingested_at"]),
        )

    # ------------------------------------------------------------------
    # Order books
    # ------------------------------------------------------------------

    def insert_order_book(self, ob: OrderBookSnapshot) -> None:
        self.conn.execute(
            """INSERT INTO order_book_snapshots (symbol, timestamp, bids, asks, ingested_at)
            VALUES (?, ?, ?, ?, ?)""",
            (
                str(ob.symbol),
                ob.timestamp.isoformat(),
                json.dumps(ob.bids),
                json.dumps(ob.asks),
                ob.ingested_at.isoformat(),
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Indicator snapshots
    # ------------------------------------------------------------------

    def insert_indicator_snapshot(self, snap) -> None:
        from playground.domain.market import IndicatorSnapshot
        self.conn.execute(
            """INSERT OR REPLACE INTO indicator_snapshots (
                symbol, timeframe, candle_timestamp, indicator_version,
                sma_20, sma_50, ema_12, ema_26, rsi_14, atr_14,
                realized_volatility_20, rolling_high_20, rolling_low_20,
                drawdown_20, trend_slope_20, average_volume_20,
                volume_ratio, candle_range, relative_candle_range,
                computed_at, history_available
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(snap.symbol), str(snap.timeframe),
                snap.candle_timestamp.isoformat(), snap.indicator_version,
                snap.sma_20, snap.sma_50, snap.ema_12, snap.ema_26,
                snap.rsi_14, snap.atr_14, snap.realized_volatility_20,
                snap.rolling_high_20, snap.rolling_low_20,
                snap.drawdown_20, snap.trend_slope_20,
                snap.average_volume_20, snap.volume_ratio,
                snap.candle_range, snap.relative_candle_range,
                snap.computed_at.isoformat(), snap.history_available,
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Regime decisions
    # ------------------------------------------------------------------

    def insert_regime_decision(self, rd: RegimeDecision) -> None:
        self.conn.execute(
            """INSERT INTO regime_decisions (
                symbol, timeframe, candle_timestamp, regime,
                indicator_values, applied_thresholds, config_version,
                decision_reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rd.symbol, rd.timeframe, rd.candle_timestamp.isoformat(),
                rd.regime, json.dumps(rd.indicator_values),
                json.dumps(rd.applied_thresholds), rd.config_version,
                rd.decision_reason, rd.created_at.isoformat(),
            ),
        )
        self.conn.commit()

    def insert_stability_decision(self, sd: StabilityDecision) -> None:
        self.conn.execute(
            """INSERT INTO stability_decisions (
                symbol, timeframe, candle_timestamp, raw_regime, final_regime,
                confidence_score, persistence_score, recent_regime_consistency,
                decision_reason, stability_config_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sd.symbol, sd.timeframe, sd.candle_timestamp.isoformat(),
                sd.raw_regime, sd.final_regime, sd.confidence_score,
                sd.persistence_score, sd.recent_regime_consistency,
                sd.decision_reason, sd.stability_config_version,
                sd.created_at.isoformat(),
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Strategy evaluations
    # ------------------------------------------------------------------

    def insert_strategy_evaluation(
        self, strategy_id: str, strategy_version: str, symbol: str,
        timeframe: str, candle_timestamp: datetime, direction: str,
        regime: str, result_type: str, score: Optional[float],
        signal_id: Optional[str], rejection_reason: Optional[str],
        detail: str,
    ) -> None:
        self.conn.execute(
            """INSERT INTO strategy_evaluations (
                strategy_id, strategy_version, symbol, timeframe,
                candle_timestamp, direction, regime, result_type,
                score, signal_id, rejection_reason, detail, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                strategy_id, strategy_version, symbol, timeframe,
                candle_timestamp.isoformat(), direction, regime, result_type,
                score, signal_id, rejection_reason, detail,
                datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def insert_signal(self, signal: StrategySignal) -> bool:
        """Insert a signal. Returns False if duplicate."""
        try:
            self.conn.execute(
                """INSERT INTO signals (
                    signal_id, strategy_id, strategy_version, symbol, timeframe,
                    candle_timestamp, direction, regime, score, entry_price,
                    stop_loss, take_profit, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(signal.signal_id), signal.strategy_id,
                    signal.strategy_version, signal.symbol, signal.timeframe,
                    signal.candle_timestamp.isoformat(), signal.direction.value,
                    signal.regime, signal.score, signal.entry_price,
                    signal.stop_loss, signal.take_profit,
                    json.dumps(signal.metadata), signal.created_at.isoformat(),
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def signal_exists(self, signal_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM signals WHERE signal_id = ?", (signal_id,)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Signal rejections
    # ------------------------------------------------------------------

    def insert_signal_rejection(self, rejection: SignalRejection) -> None:
        self.conn.execute(
            """INSERT INTO signal_rejections (
                strategy_id, strategy_version, symbol, timeframe,
                candle_timestamp, direction, reason, detail, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rejection.strategy_id, rejection.strategy_version,
                rejection.symbol, rejection.timeframe,
                rejection.candle_timestamp.isoformat(),
                rejection.direction.value, rejection.reason.value,
                rejection.detail, rejection.created_at.isoformat(),
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Risk decisions
    # ------------------------------------------------------------------

    def insert_risk_decision(self, rd: RiskDecision) -> None:
        self.conn.execute(
            """INSERT INTO risk_decisions (
                signal_id, strategy_id, symbol, timeframe, candle_timestamp,
                approved, position_size, rejection_reason, risk_config_version,
                checks_passed, checks_failed, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rd.signal_id, rd.strategy_id, rd.symbol, rd.timeframe,
                rd.candle_timestamp.isoformat(), int(rd.approved),
                rd.position_size, rd.rejection_reason,
                rd.risk_config_version,
                json.dumps(list(rd.checks_passed)),
                json.dumps(list(rd.checks_failed)),
                rd.created_at.isoformat(),
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def insert_order(self, order: BrokerOrder) -> bool:
        """Insert order. Returns False if duplicate client_order_id."""
        try:
            self.conn.execute(
                """INSERT INTO orders (
                    order_id, client_order_id, symbol, side, order_type,
                    quantity, price, status, executed_quantity,
                    cummulative_quote_qty, avg_price, created_at, updated_at,
                    exchange_response
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    order.order_id, order.client_order_id, order.symbol,
                    order.side.value, order.order_type.value,
                    order.quantity, order.price, order.status.value,
                    order.executed_quantity, order.cummulative_quote_qty,
                    order.avg_price, order.created_at.isoformat(),
                    order.updated_at.isoformat(),
                    json.dumps(order.exchange_response),
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def update_order_status(
        self, client_order_id: str, order_id: str, status: OrderStatus,
        executed_qty: float, cumm_quote_qty: float,
        avg_price: Optional[float], price: Optional[float],
        exchange_response: dict,
    ) -> None:
        self.conn.execute(
            """UPDATE orders SET order_id = ?, status = ?, executed_quantity = ?,
            cummulative_quote_qty = ?, avg_price = ?, price = ?,
            updated_at = ?, exchange_response = ?
            WHERE client_order_id = ?""",
            (
                order_id, status.value, executed_qty, cumm_quote_qty,
                avg_price, price,
                datetime.utcnow().isoformat(), json.dumps(exchange_response),
                client_order_id,
            ),
        )
        self.conn.commit()

    def get_order(self, client_order_id: str) -> Optional[BrokerOrder]:
        row = self.conn.execute(
            "SELECT * FROM orders WHERE client_order_id = ?",
            (client_order_id,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_order(row)

    def get_open_orders(self) -> list[BrokerOrder]:
        rows = self.conn.execute(
            "SELECT * FROM orders WHERE status IN ('PENDING','SUBMITTED','ACCEPTED','PARTIALLY_FILLED')"
        ).fetchall()
        return [self._row_to_order(r) for r in rows]

    @staticmethod
    def _row_to_order(row: sqlite3.Row) -> BrokerOrder:
        return BrokerOrder(
            order_id=row["order_id"] or "",
            client_order_id=row["client_order_id"],
            symbol=row["symbol"],
            side=OrderSide(row["side"]),
            order_type=OrderType(row["order_type"]),
            quantity=row["quantity"],
            price=row["price"],
            status=OrderStatus(row["status"]),
            executed_quantity=row["executed_quantity"],
            cummulative_quote_qty=row["cummulative_quote_qty"],
            avg_price=row["avg_price"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            exchange_response=json.loads(row["exchange_response"]),
        )

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------

    def insert_fill(self, fill: Fill) -> bool:
        try:
            self.conn.execute(
                """INSERT INTO fills (
                    fill_id, order_id, client_order_id, symbol, side,
                    quantity, price, commission, commission_asset, filled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fill.fill_id, fill.order_id, fill.client_order_id,
                    fill.symbol, fill.side.value, fill.quantity, fill.price,
                    fill.commission, fill.commission_asset,
                    fill.filled_at.isoformat(),
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def upsert_position(self, position: Position) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO positions (
                symbol, quantity, avg_entry_price, unrealized_pnl,
                realized_pnl, total_commission, opened_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                position.symbol, position.quantity, position.avg_entry_price,
                position.unrealized_pnl, position.realized_pnl,
                position.total_commission,
                position.opened_at.isoformat(), position.updated_at.isoformat(),
            ),
        )
        self.conn.commit()

    def get_position(self, symbol: str) -> Optional[Position]:
        row = self.conn.execute(
            "SELECT * FROM positions WHERE symbol = ?", (symbol,)
        ).fetchone()
        if not row:
            return None
        return Position(
            symbol=row["symbol"],
            quantity=row["quantity"],
            avg_entry_price=row["avg_entry_price"],
            unrealized_pnl=row["unrealized_pnl"],
            realized_pnl=row["realized_pnl"],
            total_commission=row["total_commission"],
            opened_at=datetime.fromisoformat(row["opened_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def get_all_positions(self) -> list[Position]:
        rows = self.conn.execute("SELECT * FROM positions").fetchall()
        return [
            Position(
                symbol=r["symbol"], quantity=r["quantity"],
                avg_entry_price=r["avg_entry_price"],
                unrealized_pnl=r["unrealized_pnl"],
                realized_pnl=r["realized_pnl"],
                total_commission=r["total_commission"],
                opened_at=datetime.fromisoformat(r["opened_at"]),
                updated_at=datetime.fromisoformat(r["updated_at"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def upsert_checkpoint(self, checkpoint: EngineCheckpoint) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO engine_checkpoints (
                run_id, symbol, timeframe, last_processed_candle, mode,
                engine_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                checkpoint.run_id, checkpoint.symbol, checkpoint.timeframe,
                checkpoint.last_processed_candle.isoformat(),
                checkpoint.mode, checkpoint.engine_version,
                checkpoint.created_at.isoformat(),
            ),
        )
        self.conn.commit()

    def get_checkpoint(
        self, symbol: str, timeframe: str, mode: str
    ) -> Optional[EngineCheckpoint]:
        row = self.conn.execute(
            "SELECT * FROM engine_checkpoints WHERE symbol = ? AND timeframe = ? AND mode = ?",
            (symbol, timeframe, mode),
        ).fetchone()
        if not row:
            return None
        return EngineCheckpoint(
            run_id=row["run_id"],
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            last_processed_candle=datetime.fromisoformat(row["last_processed_candle"]),
            mode=row["mode"],
            engine_version=row["engine_version"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ------------------------------------------------------------------
    # Engine runs
    # ------------------------------------------------------------------

    def insert_engine_run(self, run: EngineRun) -> None:
        self.conn.execute(
            """INSERT INTO engine_runs (run_id, mode, engine_version, started_at, status)
            VALUES (?, ?, ?, ?, ?)""",
            (run.run_id, run.mode, run.engine_version, run.started_at.isoformat(), run.status),
        )
        self.conn.commit()

    def update_engine_run(self, run_id: str, status: str, ended_at: Optional[datetime] = None) -> None:
        self.conn.execute(
            "UPDATE engine_runs SET status = ?, ended_at = ? WHERE run_id = ?",
            (status, (ended_at or datetime.utcnow()).isoformat(), run_id),
        )
        self.conn.commit()
