"""Unit tests for configuration validation."""

import os
import pytest

from playground.infrastructure.configuration import (
    AppConfig, BinanceConfig, DatabaseConfig, IndicatorEngineConfig,
    MarketDataConfig, RegimeDetectorConfig, RiskEngineConfig,
    RuntimeMode,
)


class TestMarketDataConfig:
    def test_defaults(self):
        config = MarketDataConfig()
        assert "BNB/USDT" in config.symbols
        assert "1h" in config.timeframes
        assert config.historical_candle_limit == 500

    def test_invalid_limit(self):
        with pytest.raises(ValueError):
            MarketDataConfig(historical_candle_limit=0)

    def test_invalid_timeframe(self):
        with pytest.raises(ValueError):
            MarketDataConfig(timeframes=("invalid",))


class TestIndicatorEngineConfig:
    def test_defaults(self):
        config = IndicatorEngineConfig()
        assert config.minimum_history == 100
        assert 20 in config.sma_periods

    def test_invalid_minimum_history(self):
        with pytest.raises(ValueError):
            IndicatorEngineConfig(minimum_history=0)

    def test_invalid_period(self):
        with pytest.raises(ValueError):
            IndicatorEngineConfig(sma_periods=(1,))


class TestRegimeDetectorConfig:
    def test_defaults(self):
        config = RegimeDetectorConfig()
        assert 0 < config.confidence_threshold <= 1.0
        assert 0 < config.uncertain_threshold <= config.confidence_threshold

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            RegimeDetectorConfig(confidence_threshold=1.5)

    def test_uncertain_exceeds_confidence(self):
        with pytest.raises(ValueError):
            RegimeDetectorConfig(
                confidence_threshold=0.5,
                uncertain_threshold=0.7,
            )


class TestRiskEngineConfig:
    def test_defaults(self):
        config = RiskEngineConfig()
        assert config.max_open_positions == 3
        assert 0 < config.position_size_pct <= 1.0

    def test_invalid_max_positions(self):
        with pytest.raises(ValueError):
            RiskEngineConfig(max_open_positions=0)

    def test_invalid_position_size(self):
        with pytest.raises(ValueError):
            RiskEngineConfig(position_size_pct=0)
        with pytest.raises(ValueError):
            RiskEngineConfig(position_size_pct=1.5)


class TestBinanceConfig:
    def test_default_testnet_endpoint(self):
        config = BinanceConfig()
        assert "testnet.binance.vision" in config.testnet_endpoint

    def test_rejects_non_testnet_endpoint(self):
        with pytest.raises(ValueError):
            BinanceConfig(testnet_endpoint="https://api.binance.com")

    def test_rejects_http(self):
        with pytest.raises(ValueError):
            BinanceConfig(testnet_endpoint="http://testnet.binance.vision")


class TestAppConfig:
    def test_shadow_mode_no_credentials_required(self):
        config = AppConfig(mode=RuntimeMode.SHADOW)
        errors = config.validate()
        assert len(errors) == 0

    def test_testnet_mode_requires_credentials(self):
        # Ensure env vars are not set
        old_key = os.environ.pop("BINANCE_TESTNET_API_KEY", None)
        old_secret = os.environ.pop("BINANCE_TESTNET_API_SECRET", None)

        try:
            config = AppConfig(
                mode=RuntimeMode.TESTNET,
                binance=BinanceConfig(api_key="", api_secret=""),
            )
            errors = config.validate()
            assert len(errors) > 0
        finally:
            if old_key:
                os.environ["BINANCE_TESTNET_API_KEY"] = old_key
            if old_secret:
                os.environ["BINANCE_TESTNET_API_SECRET"] = old_secret

    def test_replay_mode_requires_dataset(self):
        config = AppConfig(mode=RuntimeMode.REPLAY)
        errors = config.validate()
        # Empty dataset path should be flagged
        assert any("dataset" in e.lower() for e in errors)

    def test_no_symbols_is_error(self):
        config = AppConfig(
            market_data=MarketDataConfig(symbols=()),
        )
        errors = config.validate()
        assert any("symbol" in e.lower() for e in errors)
