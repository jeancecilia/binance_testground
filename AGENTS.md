# Build Standalone Trading Engine Playground

**Ticket Type:** Epic
**Priority:** High
**Status:** Ready for Development

## Summary

Build a standalone trading-engine playground that independently performs market-data ingestion, local persistence, indicator calculation, regime detection, strategy evaluation, risk validation, simulated trading, and Binance Testnet execution.

The application must be self-contained and runnable on a developer machine or isolated server.

## Objective

Create an autonomous trading application that:

1. Fetches raw market data directly from Binance.
2. Stores its own market and execution data locally.
3. Calculates technical indicators from raw candle data.
4. Determines market regimes using configurable rules.
5. Evaluates registered trading strategies.
6. Applies risk, freshness, liquidity, and idempotency controls.
7. Supports shadow and Binance Testnet execution.
8. Restores its state safely after a restart.
9. Supports deterministic historical replay.

## Runtime Modes

The application must support three explicit modes:

### Replay

* Reads historical candles from local JSON or CSV files.
* Uses a simulated clock.
* Uses a simulated broker.
* Produces deterministic results.

### Shadow

* Reads live public Binance market data.
* Generates and records signals.
* Does not submit orders.

### Testnet

* Reads live public Binance market data.
* Evaluates strategies and risk controls.
* Sends approved orders exclusively to Binance Testnet.

There is no live-money execution mode in this application.

## Functional Requirements

### 1. Market-data ingestion

Implement a Binance public market-data adapter.

It must:

* Fetch historical OHLCV candles.
* Fetch newly completed candles.
* Support at least `1h` and `15m` timeframes.
* Exclude incomplete candles from strategy evaluation.
* Normalize exchange symbols into an internal symbol format.
* Retry temporary failures using bounded backoff.
* Detect and backfill missing candle intervals.
* Validate candle ordering and numerical values.
* Persist ingestion timestamps and source metadata.

Initial symbol:

```text
BNB/USDT
```

Symbols and timeframes must be configurable.

### 2. Local database

Use SQLite as the initial local persistence layer.

Required tables:

* `candles`
* `order_book_snapshots`
* `indicator_snapshots`
* `regime_decisions`
* `strategy_evaluations`
* `signals`
* `signal_rejections`
* `risk_decisions`
* `orders`
* `fills`
* `positions`
* `engine_checkpoints`
* `engine_runs`

Requirements:

* Raw market data must be immutable.
* Duplicate candles must not be inserted.
* Derived records must include configuration or engine versions.
* State required for restart recovery must be persisted.
* Database migrations must be included.

### 3. Domain models

Create standalone domain models for:

* Candle
* Order-book snapshot
* Indicator snapshot
* Market context
* Regime decision
* Strategy signal
* Risk decision
* Order request
* Broker order
* Fill
* Position
* Engine checkpoint

Domain models must not contain database, HTTP, environment-variable, or exchange-specific logic.

### 4. Indicator engine

Calculate indicators directly from locally stored candles.

Minimum indicators:

* SMA
* EMA
* RSI
* ATR
* Realized volatility
* Rolling high and low
* Drawdown
* Trend slope
* Average volume
* Volume ratio
* Candle range
* Relative candle range

Requirements:

* Calculations must be deterministic.
* Required history length must be explicit.
* Invalid values must not reach regime or strategy logic.
* Insufficient history must produce a recorded safe rejection.
* Indicator settings must be configurable and versioned.
* Placeholder implementations are not permitted.

### 5. Regime detector

Implement raw market-regime classification using locally calculated indicators.

Supported regimes:

* `sideways_range`
* `bull_trend`
* `bear_trend`
* `regime_transition`
* `uncertain_regime`
* `low_volatility_compression`
* `high_volatility_chaos`
* `crash_liquidation_environment`

Each raw regime decision must record:

* Symbol
* Timeframe
* Candle timestamp
* Indicator values
* Selected regime
* Applied thresholds
* Configuration version
* Decision reason

### 6. Stability overlay

Apply confidence and persistence checks after raw regime detection.

The overlay must calculate:

* Confidence score
* Persistence score
* Recent regime consistency
* Final stabilized regime
* Decision reason

Required decision order:

```python
if raw_regime in safety_regimes:
    final_regime = raw_regime
elif confidence_score < uncertain_threshold:
    final_regime = "uncertain_regime"
elif (
    confidence_score < confidence_threshold
    or persistence_score < persistence_threshold
):
    final_regime = "regime_transition"
else:
    final_regime = raw_regime
```

All thresholds must be configurable.

### 7. Strategy interface and registry

Create a standard strategy interface and registry.

Each strategy must declare:

* Strategy ID
* Strategy version
* Supported symbols
* Supported timeframes
* Supported regimes
* Direction
* Required indicators
* Entry conditions
* Exit rules
* Risk configuration

Initial strategy:

**BNB rejection-cluster range-reversion specialist**

Configuration:

* Symbol: `BNB/USDT`
* Timeframe: `1h`
* Direction: long
* Required regime: `sideways_range`
* Minimum score: `78`
* Required rejection reason: `range_or_reversion_confirmation_failed`

Strategies must receive a prepared market context and return either a signal or a structured rejection.

Strategies must not:

* Call exchange APIs.
* Read from the database directly.
* Access environment variables.
* Submit orders.
* Read the system clock directly.

### 8. Candle-close coordinator

Implement a coordinator that processes newly completed candles.

It must:

* Determine the latest completed candle.
* Compare it with the stored checkpoint.
* Skip candles already processed.
* Process missed candles in chronological order.
* Avoid evaluating a currently forming candle.
* Update the checkpoint after successful processing.
* Recover safely after interruption.

Fixed sleep intervals may be used for polling, but they must not determine whether a strategy is evaluated.

### 9. Signal idempotency

Generate a deterministic identifier for every signal.

The identifier must include:

```text
strategy_id
strategy_version
symbol
timeframe
candle_timestamp
direction
```

Example:

```text
bnb_range_reversion:v1:BNB-USDT:1h:2026-07-13T10:00:00Z:long
```

Requirements:

* The same strategy and candle must not generate multiple executable signals.
* Scheduler repetition must not create duplicate orders.
* Application restarts must not create duplicate orders.
* The signal ID should be used as the broker client-order ID where supported.

### 10. Freshness and liquidity validation

Before approving an order, validate:

* Candle freshness
* Order-book freshness
* Bid-ask spread
* Available depth
* Estimated slippage
* Data continuity
* Exchange connectivity

Failed checks must create a structured rejection with a machine-readable reason code.

### 11. Risk engine

Implement a separate risk engine between strategy evaluation and execution.

Minimum controls:

* Maximum open positions
* Maximum positions per symbol
* Maximum exposure per symbol
* Maximum total exposure
* Position sizing
* Maximum daily loss
* Maximum drawdown
* Entry cooldown
* Maximum spread
* Minimum market depth
* Maximum estimated slippage
* One entry per strategy per candle
* Kill switch

Every strategy signal must result in either:

* An approved risk decision, or
* A persisted risk rejection.

Strategies must never call the broker directly.

### 12. Simulated broker

Implement a simulated broker for replay and local testing.

It must support:

* Market orders
* Configurable fees
* Configurable slippage
* Partial-fill simulation where enabled
* Position tracking
* Realized and unrealized PnL
* Order status transitions
* Deterministic execution when using a fixed configuration and seed

### 13. Binance Testnet broker

Implement a Binance Testnet execution adapter.

Requirements:

* Read credentials from environment variables.
* Validate the configured endpoint during startup.
* Reject any endpoint that is not explicitly recognized as Testnet.
* Submit approved orders.
* Use deterministic client-order IDs.
* Persist order intent before transmission.
* Persist exchange responses.
* Handle accepted, rejected, filled, partially filled, cancelled, and unknown states.
* Redact credentials and signatures from logs.
* Fail closed when credentials or endpoint configuration are invalid.

### 14. Order reconciliation

Implement startup and periodic reconciliation.

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

### 15. Deterministic replay

Implement historical replay using a replaceable market source and clock.

Example:

```python
engine = TradingEngine(
    market_source=JsonMarketSource("datasets/july_2026.json"),
    clock=ReplayClock(),
    broker=SimulatedBroker(),
    repository=SQLiteRepository("replay.db"),
)
```

Replay must use the same:

* Indicator engine
* Regime detector
* Stability overlay
* Strategy modules
* Risk engine
* Signal idempotency logic

Replay metadata must include:

* Dataset identifier
* Engine version
* Indicator configuration version
* Regime configuration version
* Strategy version
* Risk configuration version
* Fee model
* Slippage model
* Random seed

Two runs with identical inputs and configuration must produce identical decisions and results.

### 16. Configuration

Provide validated configuration for:

* Runtime mode
* Symbols
* Timeframes
* Polling interval
* Historical candle limit
* Indicator periods
* Regime thresholds
* Stability thresholds
* Strategy parameters
* Risk limits
* SQLite path
* Binance public endpoint
* Binance Testnet endpoint
* Logging level

Invalid configuration must prevent startup.

Secrets must not be stored in committed configuration files.

### 17. Logging and audit trail

Use structured logs.

Required events:

* Engine startup
* Engine shutdown
* Market-data fetch
* Missing-candle recovery
* Indicator calculation
* Raw regime decision
* Stability decision
* Strategy evaluation
* Signal creation
* Signal rejection
* Risk approval
* Risk rejection
* Order creation
* Order submission
* Broker response
* Fill processing
* Position update
* Reconciliation result
* Retry
* Exception

Logs should include, where applicable:

* Engine run ID
* Symbol
* Timeframe
* Candle timestamp
* Strategy ID
* Signal ID
* Order ID

## Proposed Project Structure

```text
playground/
├── domain/
│   ├── market.py
│   ├── indicators.py
│   ├── regimes.py
│   ├── signals.py
│   ├── orders.py
│   └── positions.py
├── core/
│   ├── indicator_engine.py
│   ├── regime_detector.py
│   ├── stability_overlay.py
│   ├── specialist_registry.py
│   └── risk_engine.py
├── application/
│   ├── candle_coordinator.py
│   ├── market_pipeline.py
│   ├── strategy_pipeline.py
│   ├── execution_pipeline.py
│   └── reconciliation.py
├── strategies/
│   └── bnb_rejection_specialist.py
├── infrastructure/
│   ├── binance_market_data.py
│   ├── binance_testnet_broker.py
│   ├── sqlite_repository.py
│   ├── configuration.py
│   └── system_clock.py
├── replay/
│   ├── json_market_source.py
│   ├── replay_clock.py
│   └── simulated_broker.py
├── migrations/
├── tests/
└── standalone_run.py
```

## Acceptance Criteria

### Standalone operation

* The application can be installed and run as its own software project.
* SQLite is created and initialized automatically.
* Market data is obtained from the configured market source.
* All calculations are performed inside the application.
* All runtime state is stored locally.
* Replay, shadow, and Testnet modes can be selected explicitly.

### Market data

* Completed candles are ingested and persisted.
* Duplicate candles are rejected.
* Missing candle intervals are detected and backfilled.
* Incomplete candles cannot trigger strategy evaluation.
* Restarting the application does not unnecessarily reprocess candles.

### Indicators and regimes

* Indicator implementations contain no placeholder values.
* Unit tests validate calculations against known outputs.
* Raw regime decisions are deterministic.
* Stability decisions are deterministic.
* `uncertain_regime` is reachable.
* All decisions store their evidence and configuration version.

### Strategies

* The BNB specialist runs only for the configured symbol, timeframe, and regime.
* Every strategy evaluation is recorded.
* Accepted signals and rejected evaluations are distinguishable.
* Strategies do not access infrastructure directly.
* Reprocessing a candle does not create a duplicate signal.

### Risk and execution

* Every submitted order has an approved risk decision.
* Stale or illiquid conditions reject execution.
* Exposure and loss limits block additional orders when breached.
* The kill switch blocks new orders.
* Testnet orders can only be submitted to the validated Testnet endpoint.
* Invalid endpoint configuration prevents startup.

### Recovery

* The engine restores its latest checkpoint after restart.
* Known orders and positions are reconstructed.
* Testnet state is reconciled before new orders are allowed.
* Restarting after order submission does not create a duplicate order.
* Unresolved reconciliation errors block execution.

### Replay

* Local historical data can replace live ingestion.
* The simulated broker can replace Testnet execution.
* Core trading logic remains unchanged between runtime modes.
* Identical replay inputs produce identical outputs.
* Replay produces signals, rejections, orders, fills, positions, and PnL records.

## Required Tests

### Unit tests

* SMA
* EMA
* RSI
* ATR
* Volatility
* Drawdown
* Raw regime rules
* Stability overlay
* Signal-ID generation
* Position sizing
* Risk limits
* Candle-close detection
* Configuration validation

### Integration tests

* Binance candle normalization
* SQLite migrations
* Candle persistence
* Duplicate candle prevention
* Missing-candle backfill
* Strategy evaluation
* Idempotent signal processing
* Simulated execution
* Mocked Testnet order submission
* Restart recovery
* Order reconciliation
* Replay determinism

### Failure tests

* Market API timeout
* Rate limiting
* Invalid candle payload
* Missing historical data
* Stale market data
* Invalid Testnet credentials
* Invalid Testnet endpoint
* SQLite lock
* Interruption during order submission
* Exchange order missing locally
* Local position inconsistent with Testnet
* Insufficient indicator history

## Out of Scope

* Live-money trading
* Multi-exchange support
* High-frequency trading
* Tick-level execution
* User-facing dashboard
* Automatic strategy optimization
* Automatic strategy discovery
* Automatic promotion from Testnet to live execution
* Mobile or web application interfaces

## Definition of Done

The epic is complete when:

* The application runs in replay, shadow, and Testnet modes.
* It independently ingests and stores market data.
* Indicators and regimes are calculated locally.
* The BNB strategy executes end to end.
* Risk decisions are required before every order.
* Duplicate signals and orders are prevented.
* Restart recovery and Testnet reconciliation work.
* Historical replay is deterministic.
* Testnet-only enforcement is covered by automated tests.
* Unit, integration, and failure test suites pass.
* Installation, configuration, runtime operation, replay, and recovery procedures are documented.
