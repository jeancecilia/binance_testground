"""Unit tests for signal ID generation and idempotency."""

from datetime import datetime
import pytest

from playground.domain.signals import Direction, SignalId


class TestSignalId:
    def test_signal_id_format(self):
        sid = SignalId(
            strategy_id="bnb_range_reversion",
            strategy_version="v1",
            symbol="BNB-USDT",
            timeframe="1h",
            candle_timestamp=datetime(2026, 7, 13, 10, 0, 0),
            direction=Direction.LONG,
        )
        expected = "bnb_range_reversion:v1:BNB-USDT:1h:2026-07-13T10:00:00Z:long"
        assert str(sid) == expected

    def test_signal_id_deterministic(self):
        ts = datetime(2026, 7, 13, 10, 0, 0)
        sid1 = SignalId("s1", "v1", "BNB-USDT", "1h", ts, Direction.LONG)
        sid2 = SignalId("s1", "v1", "BNB-USDT", "1h", ts, Direction.LONG)
        assert str(sid1) == str(sid2)
        assert hash(sid1) == hash(sid2)

    def test_signal_id_unique_per_candle(self):
        ts1 = datetime(2026, 7, 13, 10, 0, 0)
        ts2 = datetime(2026, 7, 13, 11, 0, 0)
        sid1 = SignalId("s1", "v1", "BNB-USDT", "1h", ts1, Direction.LONG)
        sid2 = SignalId("s1", "v1", "BNB-USDT", "1h", ts2, Direction.LONG)
        assert str(sid1) != str(sid2)

    def test_signal_id_unique_per_strategy(self):
        ts = datetime(2026, 7, 13, 10, 0, 0)
        sid1 = SignalId("s1", "v1", "BNB-USDT", "1h", ts, Direction.LONG)
        sid2 = SignalId("s2", "v1", "BNB-USDT", "1h", ts, Direction.LONG)
        assert str(sid1) != str(sid2)

    def test_signal_id_parse_roundtrip(self):
        original = SignalId(
            strategy_id="bnb_range_reversion",
            strategy_version="v1",
            symbol="BNB-USDT",
            timeframe="1h",
            candle_timestamp=datetime(2026, 7, 13, 10, 0, 0),
            direction=Direction.LONG,
        )
        parsed = SignalId.parse(str(original))
        assert parsed.strategy_id == original.strategy_id
        assert parsed.strategy_version == original.strategy_version
        assert parsed.symbol == original.symbol
        assert parsed.timeframe == original.timeframe
        assert parsed.candle_timestamp == original.candle_timestamp
        assert parsed.direction == original.direction

    def test_signal_id_parse_invalid(self):
        with pytest.raises(ValueError):
            SignalId.parse("too:few:parts")

    def test_signal_id_string_representation(self):
        sid = SignalId(
            strategy_id="test_strategy",
            strategy_version="v2",
            symbol="ETH-USDT",
            timeframe="4h",
            candle_timestamp=datetime(2026, 7, 13, 12, 0, 0),
            direction=Direction.SHORT,
        )
        s = str(sid)
        assert s == "test_strategy:v2:ETH-USDT:4h:2026-07-13T12:00:00Z:short"
