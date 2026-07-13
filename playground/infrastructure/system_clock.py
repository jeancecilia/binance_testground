"""System clock abstraction for testability and replay."""

from abc import ABC, abstractmethod
from datetime import datetime
import time


class Clock(ABC):
    """Abstract clock interface for testability."""

    @abstractmethod
    def now(self) -> datetime:
        """Return the current time in UTC."""
        ...

    @abstractmethod
    def sleep(self, seconds: float) -> None:
        """Sleep for the given number of seconds."""
        ...

    @abstractmethod
    def timestamp(self) -> float:
        """Return the current Unix timestamp."""
        ...


class SystemClock(Clock):
    """Real system clock."""

    def now(self) -> datetime:
        return datetime.utcnow()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def timestamp(self) -> float:
        return time.time()


class ReplayClock(Clock):
    """Simulated clock for replay mode.

    The clock is advanced manually by the replay coordinator
    to the timestamp of each candle being processed.
    """

    def __init__(self, start_time: datetime | None = None) -> None:
        self._current = start_time or datetime.utcnow()
        self._start_timestamp = time.time()

    @property
    def current(self) -> datetime:
        return self._current

    def advance_to(self, dt: datetime) -> None:
        """Advance the simulated clock to a specific time."""
        if dt < self._current:
            raise ValueError(
                f"Cannot rewind clock from {self._current.isoformat()} to {dt.isoformat()}"
            )
        self._current = dt

    def now(self) -> datetime:
        return self._current

    def sleep(self, seconds: float) -> None:
        """In replay mode, sleep is effectively a no-op or minimal sleep."""
        self._current = datetime.utcfromtimestamp(
            self._current.timestamp() + seconds
        )

    def timestamp(self) -> float:
        return self._current.timestamp()
