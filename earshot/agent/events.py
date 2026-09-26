"""In-process pub/sub for trace events that drive the demo UI.

Events live only in memory and go to subscribers (the UI, the CLI trace).
Never put raw audio or embeddings in an event.
"""

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {**self.data, "type": self.type, "ts": self.ts, "latency_ms": self.latency_ms}


Subscriber = Callable[[Event], None]


class EventBus:
    def __init__(self):
        self._subs: list[Subscriber] = []

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        """Returns an unsubscribe function."""
        self._subs.append(fn)
        return lambda: self._subs.remove(fn) if fn in self._subs else None

    def publish(self, type: str, /, latency_ms: float | None = None, **data: Any) -> Event:
        event = Event(type, data, latency_ms=latency_ms)
        for fn in list(self._subs):
            try:
                fn(event)
            except Exception:  # a broken UI subscriber must never break a decision
                log.exception("event subscriber failed on %s", type)
        return event

    @contextmanager
    def step(self, type: str, /, **data: Any) -> Iterator[dict[str, Any]]:
        """Time a block and publish it. Add result fields to the yielded dict."""
        start = time.perf_counter()
        extra: dict[str, Any] = {}
        yield extra
        self.publish(type, latency_ms=(time.perf_counter() - start) * 1000, **data, **extra)

    def led(self, state: str) -> None:
        """idle | listening | working | done | amber | refused | off"""
        self.publish("led", state=state)


bus = EventBus()
