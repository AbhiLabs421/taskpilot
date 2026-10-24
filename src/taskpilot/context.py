"""The :class:`Context` object a step receives when it asks for ``ctx``."""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional


class RunInfo:
    """Data shared by every step of one run."""

    __slots__ = ("flow", "run_id", "inputs", "state", "lock", "emit", "answers", "runner")

    def __init__(self, flow: str, run_id: str, inputs: Dict[str, Any], state: Dict[str, Any], emit: Callable,
                 answers: Optional[Dict[str, Any]] = None):
        self.answers: Dict[str, Any] = answers if answers is not None else {}
        self.runner: Any = None
        self.flow = flow
        self.run_id = run_id
        self.inputs = inputs
        self.state = state
        self.lock = threading.RLock()
        self.emit = emit


class Context:
    """What a step sees about the run it is part of.

    Ask for it by adding a parameter named ``ctx`` (or ``context``)::

        @flow.step
        def score(answer, ctx):
            ctx.set("seen", ctx.get("seen", 0) + 1)   # shared, thread-safe state
            ctx.log("scoring", attempt=ctx.attempt)
            return len(answer)

    Attributes:
        step: name of the running step.
        attempt: 1 for the first try, 2 for the first retry, ...
        iteration: loop counter for ``repeat_until`` steps (starts at 1).
        previous: the previous iteration's result (``None`` on the first).
        item: index of the item being processed in a ``map`` step.
    """

    __slots__ = ("_run", "step", "attempt", "iteration", "previous", "item")

    def __init__(self, run: RunInfo, step: str, item: Optional[int] = None):
        self._run = run
        self.step = step
        self.attempt = 1
        self.iteration = 1
        self.previous: Any = None
        self.item = item

    @property
    def flow(self) -> str:
        return self._run.flow

    @property
    def run_id(self) -> str:
        return self._run.run_id

    @property
    def inputs(self) -> Dict[str, Any]:
        return self._run.inputs

    @property
    def state(self) -> Dict[str, Any]:
        """The shared state dict. Prefer :meth:`get`/:meth:`set`/:meth:`update` from threads."""
        return self._run.state

    def get(self, key: str, default: Any = None) -> Any:
        with self._run.lock:
            return self._run.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._run.lock:
            self._run.state[key] = value

    def update(self, key: str, fn: Callable[[Any], Any], default: Any = None) -> Any:
        """Atomically replace ``state[key]`` with ``fn(old)`` and return the new value."""
        with self._run.lock:
            new = fn(self._run.state.get(key, default))
            self._run.state[key] = new
            return new

    def emit(self, event: str, **data: Any) -> None:
        """Send a custom event to ``flow.on(event, ...)`` listeners and streams."""
        self._run.emit(event, self.step, data)

    def wait_for(self, key: str, question: Optional[str] = None, **data: Any) -> Any:
        """Pause the run until a person answers ``key``; return the answer.

        The first time, this stops the step and the run returns with
        ``result.paused == True``. After ``flow.resume(result, key=answer)``
        the step runs again and this call returns ``answer``. Anything in
        ``data`` is shown to the person in ``result.pending``.
        """
        if not isinstance(key, str) or not key:
            raise ValueError("wait_for(key): key must be a non-empty string")
        answers = self._run.answers
        if key in answers:
            return answers[key]
        from .human import Request, WaitingForInput

        raise WaitingForInput(Request(key, self.step, question, data))

    # ---------------------------------------------------- dynamic steps
    def _runner(self) -> Any:
        runner = self._run.runner
        if runner is None:
            raise RuntimeError("This run has finished; ctx.call/ctx.map only work while the step runs.")
        return runner

    def call(self, step: Any, **kwargs: Any) -> Any:
        """Run a dynamic step (or any function) now and return its result.

        ``step`` is the name of a step defined with ``dynamic=True`` (its
        retry, timeout, cache and fallback apply) or a plain function.
        Keyword arguments fill its parameters; anything missing comes from
        the run inputs or the parameter defaults. Use :meth:`acall` in async steps.
        """
        return self._runner().call_sync(step, kwargs, self.step)

    def map(self, step: Any, items: Any, *, max_concurrency: Optional[int] = None) -> list:
        """Run a dynamic step once per item, in parallel; results keep the item order.

        Each item is a dict of keyword arguments, or a plain value that goes to
        the step's first parameter (like Python's ``map``).
        """
        return self._runner().map_sync(step, items, self.step, max_concurrency)

    async def acall(self, step: Any, **kwargs: Any) -> Any:
        """Async version of :meth:`call` for ``async def`` steps."""
        return await self._runner().call_async(step, kwargs, self.step)

    async def amap(self, step: Any, items: Any, *, max_concurrency: Optional[int] = None) -> list:
        """Async version of :meth:`map` for ``async def`` steps."""
        return await self._runner().map_async(step, items, self.step, max_concurrency)

    def answered(self, key: str) -> bool:
        """Whether ``key`` already has an answer (``wait_for`` would not pause)."""
        return key in self._run.answers

    def log(self, message: str, **data: Any) -> None:
        """Shortcut for ``emit("log", message=..., **data)``."""
        data["message"] = message
        self._run.emit("log", self.step, data)

    def __repr__(self) -> str:
        return f"Context(flow={self.flow!r}, step={self.step!r}, attempt={self.attempt})"
