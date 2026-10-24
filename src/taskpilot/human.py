"""Human-in-the-loop: a step pauses the run until a person answers.

::

    @flow.step
    def approval(draft, ctx):
        return ctx.wait_for("approve", question="Publish this?", draft=draft)

    result = flow.run(topic="AI")          # result.paused is True
    result.pending[0].question             # 'Publish this?'
    result.save("run.json")                # optional: resume later, anywhere
    flow.resume("run.json", approve=True)  # finished steps are not run again

The waiting step runs again from its first line when resumed, and this time
``ctx.wait_for`` returns the answer. Keep side effects after the
``wait_for`` call (or in a later step).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

from .checkpoint import _decode, _encode
from .errors import TaskPilotError

_VERSION = 1


class Request:
    """A question a paused run is waiting on."""

    __slots__ = ("key", "step", "question", "data", "time")

    def __init__(self, key: str, step: str, question: Optional[str] = None, data: Optional[Dict[str, Any]] = None, at: Optional[float] = None):
        self.key = key
        self.step = step
        self.question = question
        self.data = data or {}
        self.time = at if at is not None else time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "step": self.step, "question": self.question, "data": _encode(self.data), "time": self.time}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Request":
        return cls(d["key"], d["step"], d.get("question"), _decode(d["data"]) if d.get("data") else {}, d.get("time"))

    def __repr__(self) -> str:
        q = f", {self.question!r}" if self.question else ""
        return f"Request({self.key!r}, step={self.step!r}{q})"


class WaitingForInput(TaskPilotError):
    """Raised by ``ctx.wait_for`` to pause the step. Not an error: never retried."""

    def __init__(self, request: Request):
        self.request = request
        super().__init__(f"Step '{request.step}' is waiting for '{request.key}'")


def save_paused(result: Any, path: str) -> str:
    """Write everything needed to resume ``result`` into ``path`` (atomic)."""
    from .result import SUCCESS

    def enc(what: str, value: Any) -> Dict[str, Any]:
        try:
            return _encode(value)
        except Exception as exc:
            raise TaskPilotError(
                f"Can't save {what} ({type(value).__name__}): {exc}. Return JSON or picklable values, "
                f"or resume in memory with flow.resume(result, ...)."
            ) from None

    values = {}
    for name, sr in result.steps.items():
        if sr.status == SUCCESS and not sr.released and name in result.restorable:
            values[name] = enc(f"the value of step '{name}'", sr.value)
    data = {
        "taskpilot_paused": _VERSION,
        "flow": result.flow,
        "run_id": result.run_id,
        "inputs": enc("the inputs", result.inputs),
        "state": enc("ctx.state", result.state),
        "answers": enc("the answers", result.answers),
        "values": values,
        "loops": sorted(result.loops),
        "pending": [r.to_dict() for r in result.pending],
    }
    path = os.fspath(path)
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".taskpilot-", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


class PausedRun:
    """What :meth:`Flow.resume` needs, read from a :class:`RunResult` or a file."""

    def __init__(self, flow: str, inputs: Dict[str, Any], state: Dict[str, Any], answers: Dict[str, Any],
                 values: Dict[str, Any], loops: List[str], pending: List[Request]):
        self.flow = flow
        self.inputs = inputs
        self.state = state
        self.answers = answers
        self.values = values
        self.loops = loops
        self.pending = pending

    @classmethod
    def from_result(cls, result: Any) -> "PausedRun":
        from .result import SUCCESS

        values = {n: sr.value for n, sr in result.steps.items()
                  if sr.status == SUCCESS and not sr.released and n in result.restorable}
        return cls(result.flow, dict(result.inputs), dict(result.state), dict(result.answers), values,
                   list(result.loops), list(result.pending))

    @classmethod
    def load(cls, path: str) -> "PausedRun":
        try:
            with open(os.fspath(path), "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            raise TaskPilotError(f"Cannot read paused run from {path}: {exc}") from None
        if not isinstance(data, dict) or data.get("taskpilot_paused") != _VERSION:
            raise TaskPilotError(f"{path} is not a paused taskpilot run (save one with result.save(path))")
        return cls(
            data["flow"],
            _decode(data["inputs"]),
            _decode(data["state"]),
            _decode(data["answers"]),
            {n: _decode(v) for n, v in data["values"].items()},
            list(data.get("loops", [])),
            [Request.from_dict(r) for r in data.get("pending", [])],
        )
