"""What a run gives back: :class:`RunResult` made of :class:`StepResult` s."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

from .errors import did_you_mean

SUCCESS = "success"
SKIPPED = "skipped"
FAILED = "failed"
PAUSED = "paused"
PENDING = "pending"


class StepResult:
    """Outcome of one step in one run."""

    __slots__ = (
        "name", "status", "value", "error", "attempts", "duration",
        "cached", "resumed", "fallback", "reason", "released", "parent",
    )

    def __init__(self, name: str):
        self.name = name
        self.status = PENDING
        self.value: Any = None
        self.error: Optional[BaseException] = None
        self.attempts = 0
        self.duration = 0.0
        self.cached = False
        self.resumed = False
        self.fallback = False
        self.reason = ""
        self.released = False
        #: for dynamic calls: the step that called it.
        self.parent = ""

    @property
    def ok(self) -> bool:
        return self.status == SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "value": None if self.released else self.value,
            "error": None if self.error is None else f"{type(self.error).__name__}: {self.error}",
            "attempts": self.attempts,
            "duration": self.duration,
            "cached": self.cached,
            "resumed": self.resumed,
            "fallback": self.fallback,
            "reason": self.reason,
        }

    def __repr__(self) -> str:
        return f"StepResult({self.name!r}, {self.status}, {self.duration * 1000:.2f}ms)"


class RunResult:
    """Everything that happened in one run of a flow.

    * ``result.output`` – the final answer (see below)
    * ``result["step"]`` – the value of any step
    * ``result.ok`` – ``True`` when no step failed
    * ``result.steps["step"]`` – full :class:`StepResult` (status, time, error...)

    ``output`` is the value of the *final* steps (steps nothing depends on)
    that succeeded: one such step gives its value directly, several give a
    ``{name: value}`` dict, none gives ``None``.
    """

    def __init__(self, flow: str, run_id: str, inputs: Dict[str, Any], order: Tuple[str, ...], sinks: Tuple[str, ...]):
        self.flow = flow
        self.run_id = run_id
        self.inputs = inputs
        self.order = order
        self.sinks = sinks
        self.steps: Dict[str, StepResult] = {name: StepResult(name) for name in order}
        self.state: Dict[str, Any] = {}
        self.duration = 0.0
        self.error: Optional[BaseException] = None
        #: loop step -> its value in every round (oldest first).
        self.history: Dict[str, List[Any]] = {}
        #: loop name -> {"rounds": n, "stopped_by": "until" | "max_iterations" | ...}
        self.loops: Dict[str, Dict[str, Any]] = {}
        #: questions this run is waiting on (see ``ctx.wait_for``).
        self.pending: List[Any] = []
        #: answers given so far (``flow.resume(..., key=answer)``).
        self.answers: Dict[str, Any] = {}
        #: every dynamic step call (ctx.call / ctx.map), in finishing order.
        self.calls: List[StepResult] = []
        #: steps whose values can be reused when resuming.
        self.restorable: frozenset = frozenset()

    # ----------------------------------------------------------- accessors
    @property
    def ok(self) -> bool:
        """True when the run finished and no step failed (a paused run is not ok yet)."""
        return self.error is None and not self.pending and not any(s.status == FAILED for s in self.steps.values())

    @property
    def paused(self) -> bool:
        """True when the run stopped to wait for a person (see :attr:`pending`)."""
        return bool(self.pending)

    @property
    def status(self) -> str:
        """``"ok"``, ``"paused"`` or ``"failed"``."""
        if self.error is not None or any(s.status == FAILED for s in self.steps.values()):
            return "failed"
        return "paused" if self.pending else "ok"

    def save(self, path: str) -> str:
        """Save a paused run to a file so it can be resumed later (even in another process)."""
        from .human import save_paused

        return save_paused(self, path)

    @property
    def output(self) -> Any:
        done = [n for n in self.sinks if self.steps[n].status == SUCCESS]
        if not done:
            return None
        if len(done) == 1:
            return self.steps[done[0]].value
        return {n: self.steps[n].value for n in done}

    @property
    def values(self) -> Dict[str, Any]:
        """``{step: value}`` for every step that succeeded and still holds its value."""
        return {n: s.value for n, s in self.steps.items() if s.status == SUCCESS and not s.released}

    @property
    def failed(self) -> List[str]:
        return [n for n, s in self.steps.items() if s.status == FAILED]

    @property
    def skipped(self) -> List[str]:
        return [n for n, s in self.steps.items() if s.status == SKIPPED]

    @property
    def succeeded(self) -> List[str]:
        return [n for n, s in self.steps.items() if s.status == SUCCESS]

    @property
    def timings(self) -> Dict[str, float]:
        return {n: s.duration for n, s in self.steps.items()}

    def get(self, name: str, default: Any = None) -> Any:
        s = self.steps.get(name)
        if s is None or s.status != SUCCESS or s.released:
            return default
        return s.value

    def __getitem__(self, name: str) -> Any:
        s = self.steps.get(name)
        if s is None:
            raise KeyError(f"No step named '{name}' in flow '{self.flow}'.{did_you_mean(name, self.steps)}")
        if s.released:
            raise KeyError(f"Step '{name}' value was released to save memory (lean=True). Mark it keep=True to keep it.")
        if s.status != SUCCESS:
            why = f": {s.reason}" if s.reason else ""
            raise KeyError(f"Step '{name}' has no value (status: {s.status}{why})")
        return s.value

    def __contains__(self, name: object) -> bool:
        return name in self.steps and self.steps[name].status == SUCCESS  # type: ignore[index]

    def __iter__(self) -> Iterator[str]:
        return iter(self.steps)

    def __bool__(self) -> bool:
        return self.ok

    # ----------------------------------------------------------- reporting
    def counts(self) -> str:
        c = {SUCCESS: 0, SKIPPED: 0, FAILED: 0, PAUSED: 0, PENDING: 0}
        for s in self.steps.values():
            c[s.status] += 1
        parts = [f"{c[SUCCESS]} ok"]
        if c[SKIPPED]:
            parts.append(f"{c[SKIPPED]} skipped")
        if c[FAILED]:
            parts.append(f"{c[FAILED]} failed")
        if c[PAUSED]:
            parts.append(f"{c[PAUSED]} waiting")
        if c[PENDING]:
            parts.append(f"{c[PENDING]} not run")
        return ", ".join(parts)

    def summary(self) -> str:
        """A readable table of every step: status, time, attempts, notes."""
        width = max([len(n) for n in self.steps] + [4])
        lines = [f"Flow '{self.flow}' run {self.run_id}: {self.status.upper()} in {self.duration * 1000:.2f}ms ({self.counts()})"]
        lines.append(f"  {'step':<{width}}  {'status':<8} {'time':>10}  tries  notes")
        for n in self.order:
            s = self.steps[n]
            notes = []
            if s.cached:
                notes.append("cached")
            if s.resumed:
                notes.append("resumed")
            if s.fallback:
                notes.append("fallback")
            if s.reason:
                notes.append(s.reason)
            if s.error is not None and s.status == FAILED:
                notes.append(f"{type(s.error).__name__}: {s.error}")
            lines.append(f"  {n:<{width}}  {s.status:<8} {s.duration * 1000:>8.2f}ms  {s.attempts:>5}  {'; '.join(notes)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flow": self.flow,
            "run_id": self.run_id,
            "ok": self.ok,
            "status": self.status,
            "pending": [{"key": r.key, "step": r.step, "question": r.question} for r in self.pending],
            "duration": self.duration,
            "error": None if self.error is None else f"{type(self.error).__name__}: {self.error}",
            "steps": {n: s.to_dict() for n, s in self.steps.items()},
        }

    def __repr__(self) -> str:
        return f"RunResult(flow={self.flow!r}, ok={self.ok}, {self.counts()}, {self.duration * 1000:.2f}ms)"
