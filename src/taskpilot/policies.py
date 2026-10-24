"""Reliability policies: :class:`Retry` and :class:`Cache`."""

from __future__ import annotations

import hashlib
import pickle
import random
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional, Tuple, Type, Union

from .errors import FlowDefinitionError

ExcTypes = Union[Type[BaseException], Tuple[Type[BaseException], ...]]


class Retry:
    """Retry a failing step with exponential backoff.

    ``Retry(3)`` means *up to 3 attempts in total*. Waits grow as
    ``backoff * factor ** (attempt - 1)``, capped at ``max_delay``, with a
    little random jitter so many retries do not hit a server at once.

    Example::

        @flow.step(retry=Retry(5, backoff=0.5, on=(ConnectionError,)))
        def fetch(url): ...
    """

    __slots__ = ("attempts", "backoff", "factor", "max_delay", "jitter", "on", "when")

    def __init__(
        self,
        attempts: int = 3,
        *,
        backoff: float = 0.1,
        factor: float = 2.0,
        max_delay: float = 30.0,
        jitter: float = 0.1,
        on: ExcTypes = Exception,
        when: Optional[Callable[[BaseException], bool]] = None,
    ):
        if not isinstance(attempts, int) or attempts < 1:
            raise FlowDefinitionError(f"Retry attempts must be an int >= 1, got {attempts!r}")
        if backoff < 0 or factor < 1 or max_delay < 0 or jitter < 0:
            raise FlowDefinitionError("Retry backoff/max_delay/jitter must be >= 0 and factor >= 1")
        self.attempts = attempts
        self.backoff = backoff
        self.factor = factor
        self.max_delay = max_delay
        self.jitter = jitter
        self.on = on
        self.when = when

    @classmethod
    def coerce(cls, value: Any) -> "Retry":
        """Turn ``None`` / ``int`` / ``Retry`` into a :class:`Retry`."""
        if value is None or value is False:
            return _NO_RETRY
        if isinstance(value, Retry):
            return value
        if isinstance(value, bool):
            return cls(3)
        if isinstance(value, int):
            return cls(value)
        raise FlowDefinitionError(f"retry= must be an int or Retry(...), got {value!r}")

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        if attempt >= self.attempts or not isinstance(exc, self.on):
            return False
        return self.when is None or bool(self.when(exc))

    def delay(self, attempt: int) -> float:
        base = min(self.max_delay, self.backoff * (self.factor ** (attempt - 1)))
        if self.jitter and base:
            base += random.uniform(0, base * self.jitter)
        return base

    def __repr__(self) -> str:
        return f"Retry({self.attempts}, backoff={self.backoff}, factor={self.factor})"


_NO_RETRY = Retry(1)


_MISSING = object()


class Cache:
    """Thread-safe LRU memo cache with optional time-to-live.

    Results are keyed by the step's input values, so the same inputs give the
    stored result instantly instead of running the step again.

    ``cache=True`` uses ``Cache()``; ``cache=60`` means ``Cache(ttl=60)``.
    """

    def __init__(self, ttl: Optional[float] = None, maxsize: int = 1024):
        if maxsize < 1:
            raise FlowDefinitionError("Cache maxsize must be >= 1")
        self.ttl = ttl
        self.maxsize = maxsize
        self._data: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @classmethod
    def coerce(cls, value: Any) -> Optional["Cache"]:
        if value is None or value is False:
            return None
        if isinstance(value, Cache):
            return value
        if value is True:
            return cls()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return cls(ttl=float(value))
        raise FlowDefinitionError(f"cache= must be True, a TTL in seconds or Cache(...), got {value!r}")

    @staticmethod
    def make_key(args: dict) -> Optional[str]:
        """Stable key for a set of arguments, or ``None`` if they can't be hashed."""
        items = sorted(args.items())
        try:
            raw = pickle.dumps(items, protocol=4)
        except Exception:
            try:
                raw = repr(items).encode()
            except Exception:
                return None
        return hashlib.blake2b(raw, digest_size=16).hexdigest()

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return _MISSING
            stamp, value = entry
            if self.ttl is not None and time.monotonic() - stamp > self.ttl:
                del self._data[key]
                self.misses += 1
                return _MISSING
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Cache(ttl={self.ttl}, maxsize={self.maxsize}, size={len(self)})"


MISSING = _MISSING
