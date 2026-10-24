"""Tracing: turn every run into a tree of timed spans.

::

    from taskpilot.tracing import MemoryTracer, JSONLTracer, ConsoleTracer, OpenTelemetryTracer

    flow.trace(ConsoleTracer())            # print a timing tree after each run
    flow.trace(JSONLTracer("traces.jsonl"))  # one span per line, OpenTelemetry-like fields
    flow.trace(OpenTelemetryTracer())      # real OTel spans, if `opentelemetry-api` is installed

A run is one *trace*: the flow is the root span, each step (and each
``ctx.call``) is a child span with its status, attempts, cache hit and error.
Retries, loop rounds, waits and ``ctx.log`` messages become span events.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

from .events import Event

__all__ = ["Span", "Tracer", "MemoryTracer", "JSONLTracer", "ConsoleTracer", "OpenTelemetryTracer"]


def _new_id(nbytes: int) -> str:
    return os.urandom(nbytes).hex()


class Span:
    """One timed piece of work (a run, a step, or a dynamic call)."""

    __slots__ = ("trace_id", "span_id", "parent_id", "name", "kind", "start", "end", "status", "attributes", "events")

    def __init__(self, trace_id: str, name: str, kind: str, parent_id: Optional[str], start: float):
        self.trace_id = trace_id
        self.span_id = _new_id(8)
        self.parent_id = parent_id
        self.name = name
        self.kind = kind
        self.start = start
        self.end: Optional[float] = None
        self.status = "unset"
        self.attributes: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []

    @property
    def duration(self) -> float:
        return (self.end or self.start) - self.start

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "duration_ms": round(self.duration * 1000, 3),
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }

    def __repr__(self) -> str:
        return f"Span({self.name!r}, {self.kind}, {self.status}, {self.duration * 1000:.2f}ms)"


def _safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    text = repr(value)
    return text if len(text) <= 200 else text[:197] + "..."


class Tracer:
    """Builds spans from run events. Subclass and override :meth:`export`.

    ``export(spans)`` receives every span of a finished run (root first).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open: Dict[Tuple[str, Any], Span] = {}
        self._roots: Dict[str, Span] = {}
        self._runs: Dict[str, List[Span]] = {}

    # The flow calls this for every event (it is registered with flow.on("*")).
    def __call__(self, event: Event) -> None:
        with self._lock:
            done = self._handle(event)
        if done is not None:
            self.export(done)

    def _handle(self, e: Event) -> Optional[List[Span]]:
        run, t, d = e.run_id, e.type, e.data
        if t == "flow_start":
            root = Span(run.ljust(32, "0")[:32], e.flow, "flow", None, e.time)
            root.attributes.update({"flow": e.flow, "run_id": run})
            self._roots[run] = root
            self._runs[run] = [root]
            return None
        root = self._roots.get(run)
        if root is None:
            return None
        key = (run, e.step, d.get("call_id"))
        if t == "step_start":
            parent = root
            if d.get("dynamic") and d.get("parent"):
                parent = self._open.get((run, d["parent"], None), root)
            span = Span(root.trace_id, e.step or "?", "call" if d.get("dynamic") else "step", parent.span_id, e.time)
            if d.get("dynamic"):
                span.attributes["parent_step"] = d.get("parent")
            self._open[key] = span
            self._runs[run].append(span)
        elif t in ("step_end", "step_skip", "step_error", "step_wait"):
            span = self._open.pop(key, None)
            if span is None:  # e.g. skipped before it started
                span = Span(root.trace_id, e.step or "?", "step", root.span_id, e.time)
                self._runs[run].append(span)
            span.end = e.time
            span.status = {"step_end": "ok", "step_skip": "skipped", "step_error": "error", "step_wait": "waiting"}[t]
            for k in ("attempts", "cached", "fallback", "resumed", "reason", "key", "question"):
                if d.get(k) not in (None, False, ""):
                    span.attributes[k] = _safe(d[k])
            if t == "step_error":
                err = d.get("error")
                span.attributes["error.type"] = type(err).__name__
                span.attributes["error.message"] = _safe(str(err))
        elif t != "flow_end":  # retries, logs, loop rounds and custom events
            target = self._open.get(key) or self._open.get((run, e.step, None)) or root
            attrs = {k: _safe(v) for k, v in d.items() if k not in ("call_id",)}
            target.events.append({"name": t, "time": e.time, "attributes": attrs})
        if t == "flow_end":
            root.end = e.time
            result = d.get("result")
            root.status = result.status if result is not None else ("ok" if d.get("ok") else "error")
            if result is not None:
                root.attributes["steps"] = len(result.steps)
                root.attributes["counts"] = result.counts()
            spans = self._runs.pop(run, [])
            self._roots.pop(run, None)
            for k in [k for k in self._open if k[0] == run]:
                s = self._open.pop(k)
                s.end, s.status = e.time, "cancelled"
            return spans
        return None

    def export(self, spans: List[Span]) -> None:  # pragma: no cover - meant to be overridden
        pass


class MemoryTracer(Tracer):
    """Keeps finished spans in memory (``tracer.spans``, ``tracer.traces``)."""

    def __init__(self, keep: int = 1000):
        super().__init__()
        self.keep = keep
        self.traces: List[List[Span]] = []

    def export(self, spans: List[Span]) -> None:
        with self._lock:
            self.traces.append(spans)
            del self.traces[:-self.keep]

    @property
    def spans(self) -> List[Span]:
        return [s for trace in self.traces for s in trace]

    def last(self) -> List[Span]:
        return self.traces[-1] if self.traces else []


class JSONLTracer(Tracer):
    """Appends one JSON object per span to a file (easy to ship anywhere)."""

    def __init__(self, path: str):
        super().__init__()
        self.path = os.fspath(path)
        self._file_lock = threading.Lock()

    def export(self, spans: List[Span]) -> None:
        lines = [json.dumps(s.to_dict(), default=str) for s in spans]
        with self._file_lock:
            folder = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(folder, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")


class ConsoleTracer(Tracer):
    """Prints a timing tree of each run to stderr."""

    def __init__(self, stream: Any = None):
        super().__init__()
        self.stream = stream

    def export(self, spans: List[Span]) -> None:
        children: Dict[Optional[str], List[Span]] = {}
        for s in spans:
            children.setdefault(s.parent_id, []).append(s)
        lines: List[str] = []

        def walk(span: Span, depth: int) -> None:
            mark = {"ok": "✓", "error": "✗", "skipped": "-", "waiting": "?", "paused": "?"}.get(span.status, "·")
            extra = ""
            if span.attributes.get("attempts", 1) and span.attributes.get("attempts", 1) > 1:
                extra += f" x{span.attributes['attempts']}"
            if span.attributes.get("cached"):
                extra += " cached"
            lines.append(f"{'  ' * depth}{mark} {span.name:<{max(1, 24 - 2 * depth)}} {span.duration * 1000:9.2f}ms{extra}")
            for c in sorted(children.get(span.span_id, []), key=lambda x: x.start):
                walk(c, depth + 1)

        for root in children.get(None, []):
            walk(root, 0)
        text = "\n".join(lines)
        stream = self.stream or sys.stderr
        try:
            print(text, file=stream, flush=True)
        except UnicodeEncodeError:
            print(text.encode("ascii", "replace").decode("ascii"), file=stream, flush=True)


class OpenTelemetryTracer(Tracer):
    """Sends spans to OpenTelemetry. Needs ``pip install opentelemetry-api`` (optional).

    Spans are created after the run finishes with their real start and end
    times, so they show up in Jaeger, Honeycomb, Azure Monitor, etc.
    """

    def __init__(self, tracer: Any = None, service: str = "taskpilot"):
        super().__init__()
        try:
            from opentelemetry import trace as otel_trace  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "OpenTelemetryTracer needs the optional 'opentelemetry-api' package: pip install opentelemetry-api"
            ) from None
        self._otel = otel_trace
        self._tracer = tracer or otel_trace.get_tracer(service)

    def export(self, spans: List[Span]) -> None:
        made: Dict[str, Any] = {}
        status_code = self._otel.StatusCode
        for s in spans:
            parent = made.get(s.parent_id or "")
            ctx = self._otel.set_span_in_context(parent) if parent is not None else None
            otel_span = self._tracer.start_span(f"{s.kind}:{s.name}", context=ctx, start_time=int(s.start * 1e9),
                                                attributes={k: v for k, v in s.attributes.items() if isinstance(v, (str, bool, int, float))})
            for ev in s.events:
                otel_span.add_event(ev["name"], {k: v for k, v in ev["attributes"].items() if isinstance(v, (str, bool, int, float))},
                                    timestamp=int(ev["time"] * 1e9))
            if s.status in ("error", "failed"):
                otel_span.set_status(self._otel.Status(status_code.ERROR))
            made[s.span_id] = otel_span
        for s in reversed(spans):
            made[s.span_id].end(end_time=int((s.end or s.start) * 1e9))
