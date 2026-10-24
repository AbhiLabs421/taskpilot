"""Every error taskpilot raises, in one place.

All errors inherit from :class:`TaskPilotError`, so ``except TaskPilotError`` catches
anything the framework itself raises. Messages are written to tell you *what*
went wrong and *how* to fix it.
"""

from __future__ import annotations

import difflib
from typing import Any, Iterable, Optional


class TaskPilotError(Exception):
    """Base class for all taskpilot errors."""


def did_you_mean(name: str, choices: Iterable[str]) -> str:
    """Return a ``" Did you mean 'x'?"`` hint, or an empty string."""
    close = difflib.get_close_matches(name, list(choices), n=1, cutoff=0.6)
    return f" Did you mean '{close[0]}'?" if close else ""


class FlowDefinitionError(TaskPilotError):
    """The flow is wired incorrectly (duplicate step, cycle, bad option...)."""


class CycleError(FlowDefinitionError):
    """Steps depend on each other in a loop, so no order can run them."""

    def __init__(self, cycle: list):
        self.cycle = list(cycle)
        path = " -> ".join(self.cycle)
        super().__init__(
            f"Cycle detected: {path}. A step cannot (directly or indirectly) "
            f"depend on itself. Use `repeat_until=` on a single step for loops."
        )


class MissingInputError(TaskPilotError):
    """A step needs a value that is neither a step nor a run input."""

    def __init__(self, name: str, needed_by: list, known: Iterable[str] = ()):
        self.name = name
        self.needed_by = list(needed_by)
        users = ", ".join(f"'{s}'" for s in self.needed_by)
        super().__init__(
            f"Missing input '{name}' (needed by {users}). Pass it as "
            f"flow.run({name}=...), give the parameter a default value, or add "
            f"a step named '{name}'.{did_you_mean(name, known)}"
        )


class StepError(TaskPilotError):
    """A step raised an exception (after all retries were used up)."""

    def __init__(self, step: str, cause: BaseException, attempts: int = 1, result: Any = None):
        self.step = step
        self.cause = cause
        self.attempts = attempts
        #: The partial :class:`~taskpilot.RunResult` of the run that failed.
        self.result = result
        tries = f" after {attempts} attempts" if attempts > 1 else ""
        super().__init__(f"Step '{step}' failed{tries}: {type(cause).__name__}: {cause}")


class StepTimeout(TaskPilotError, TimeoutError):
    """A single step attempt took longer than its ``timeout``."""

    def __init__(self, step: str, seconds: float):
        self.step = step
        self.seconds = seconds
        super().__init__(f"Step '{step}' timed out after {seconds:g}s")


class FlowTimeout(TaskPilotError, TimeoutError):
    """The whole run took longer than the flow's ``timeout``."""

    def __init__(self, flow: str, seconds: float, result: Any = None):
        self.flow = flow
        self.seconds = seconds
        self.result = result
        super().__init__(f"Flow '{flow}' timed out after {seconds:g}s")


class LLMError(TaskPilotError):
    """An LLM provider returned an error or an unusable response."""

    def __init__(self, message: str, status: Optional[int] = None, body: str = ""):
        self.status = status
        self.body = body
        super().__init__(message)


class VerificationError(TaskPilotError):
    """An answer still failed its checks after all fixes (``strict=True``)."""

    def __init__(self, output: str, problems: list):
        self.output = output
        self.problems = list(problems)
        super().__init__("Answer failed verification: " + "; ".join(self.problems[:3]))


class CheckFailed(TaskPilotError):
    """A step's result was rejected by its ``check=`` function."""

    def __init__(self, step: str, reason: str):
        self.step = step
        self.reason = reason
        super().__init__(f"Step '{step}' result was rejected: {reason}")


class ToolError(TaskPilotError):
    """A tool was defined incorrectly."""
