"""Events let you watch a run: logging, tracing, progress bars, streaming UIs."""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Dict, List, Optional

#: Event types the engine emits by itself.
EVENT_TYPES = (
    "flow_start",
    "flow_end",
    "step_start",
    "step_end",
    "step_retry",
    "step_skip",
    "step_error",
    "loop_round",
    "step_wait",
    "log",
)


class Event:
    """Something that happened during a run."""

    __slots__ = ("type", "flow", "run_id", "step", "data", "time")

    def __init__(self, type: str, flow: str, run_id: str, step: Optional[str], data: Dict[str, Any], at: Optional[float] = None):
        self.type = type
        self.flow = flow
        self.run_id = run_id
        self.step = step
        self.data = data
        self.time = at if at is not None else time.time()

    def __getattr__(self, item: str) -> Any:
        # Lets you write event.value / event.error / event.duration.
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError(item) from None

    def __repr__(self) -> str:
        where = f" step={self.step!r}" if self.step else ""
        return f"Event({self.type!r}{where})"


Handler = Callable[[Event], Any]


class EventBus:
    """Keeps handlers per event type. ``"*"`` listens to everything."""

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Handler]] = {}

    def add(self, type: str, handler: Handler) -> None:
        self._handlers.setdefault(type, []).append(handler)

    def remove(self, type: str, handler: Handler) -> None:
        handlers = self._handlers.get(type, [])
        if handler in handlers:
            handlers.remove(handler)

    def handlers_for(self, type: str) -> List[Handler]:
        return self._handlers.get(type, []) + self._handlers.get("*", [])

    def __bool__(self) -> bool:
        return any(self._handlers.values())


def _safe_print(text: str) -> None:
    try:
        print(text, file=sys.stderr, flush=True)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"), file=sys.stderr, flush=True)


def _fmt_ms(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"


def console_printer(event: Event) -> None:
    """The handler behind ``verbose=True``. Prints a compact live trace to stderr."""
    t, d = event.type, event.data
    if t == "flow_start":
        _safe_print(f"> {event.flow}  run={event.run_id}")
    elif t == "step_end":
        extra = " (cached)" if d.get("cached") else " (resumed)" if d.get("resumed") else ""
        if d.get("fallback"):
            extra += " (fallback)"
        mark = "  ↳" if d.get("dynamic") else "  ✓"
        _safe_print(f"{mark} {event.step:<20} {_fmt_ms(d.get('duration', 0.0))}{extra}")
    elif t == "step_retry":
        _safe_print(f"  ↻ {event.step:<20} retry {d['attempt']}/{d['attempts']} in {d['delay']:.2f}s: {d['error']!r}")
    elif t == "step_skip":
        _safe_print(f"  - {event.step:<20} skipped: {d.get('reason', '')}")
    elif t == "step_error":
        _safe_print(f"  ✗ {event.step:<20} failed: {d['error']!r}")
    elif t == "loop_round":
        state = f"done ({d.get('reason')})" if d.get("done") else "again"
        _safe_print(f"  ⟳ loop {d.get('loop')} round {d.get('round')}: {state}")
    elif t == "step_wait":
        q = f": {d.get('question')}" if d.get("question") else ""
        _safe_print(f"  ? {event.step:<20} waiting for '{d.get('key')}'{q}")
    elif t == "log":
        _safe_print(f"    · {event.step}: {d.get('message', '')}")
    elif t == "flow_end":
        r = d.get("result")
        summary = f"({r.counts()})" if r is not None else ""
        r_status = r.status if r is not None else ("ok" if d.get("ok") else "failed")
        _safe_print(f"■ {event.flow} {r_status} in {_fmt_ms(d.get('duration', 0.0))} {summary}")
